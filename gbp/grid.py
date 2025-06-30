from .gbp import Gaussian, Variable, Factor
from .gbp import update_variable, update_factor, factor_energy, tree_stack, l2
from .factor import  h_fn, h2_fn, h3_fn, h4_fn, h5_fn
from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np



def generate_grid_slam_data(H=16, W=16, dx=1.0, dy=1.0, prior_noise_std=0.05, odom_noise_std=0.05, seed=0):
    """
    Generate 2D SLAM data over a regular H x W grid.

    Each variable is a node located at position (j*dx, i*dy), where i is row index and j is column index.
    Relative pose measurements (between factors) are added between horizontal and vertical neighbors.
    Each variable also receives a weak but accurate prior to ensure the global graph is well-constrained.

    Args:
        H: number of rows in the grid
        W: number of columns in the grid
        dx: horizontal spacing between grid points
        dy: vertical spacing between grid points
        odom_noise_std: standard deviation of noise added to relative measurements (between factors)
        prior_std: standard deviation for prior factors (should be much larger than odom_noise_std to make priors weak)
        seed: random seed for reproducibility

    Returns:
        positions: (N, 2) array of ground-truth positions (N = H * W)
        prior_meas: list of (i, z) where i is variable index and z is its true position
        between_meas: list of (i, j, z) where (i, j) is a measurement edge and z is the noisy relative pose from i to j
    """
    np.random.seed(seed)
    N = H * W  # total number of variables

    # Step 1: Generate ground-truth positions on the grid
    positions = []
    for i in range(H):
        for j in range(W):
            x = j * dx
            y = -i * dy
            positions.append([x, y])
    positions = np.array(positions)  # shape (N, 2)

    # Step 2: Add weak but accurate prior for each variable
    prior_meas = []
    for idx, pos in enumerate(positions):
        noise = np.random.randn(2) * prior_noise_std
        z = (pos + noise).tolist()
        prior_meas.append((idx, z))  # accurate measurement with weak information will be set in z_Lam later

    # Step 3: Add noisy relative pose measurements (between factors)
    between_meas = []
    for i in range(H):
        for j in range(W):
            idx = i * W + j  # flat index of (i, j)

            # Horizontal neighbor (i, j) -> (i, j+1)
            if j < W - 1:
                nbr = i * W + (j + 1)
                rel = positions[nbr] - positions[idx]  # ideal relative translation
                noise = np.random.randn(2) * odom_noise_std
                z = (rel + noise).tolist()
                between_meas.append((idx, nbr, z))

            # Vertical neighbor (i, j) -> (i+1, j)
            if i < H - 1:
                nbr = (i + 1) * W + j
                rel = positions[nbr] - positions[idx]
                noise = np.random.randn(2) * odom_noise_std
                z = (rel + noise).tolist()
                between_meas.append((idx, nbr, z))

    return positions, prior_meas, between_meas



def build_pose_slam_graph(N, prior_meas, between_meas, prior_std=0.05, odom_std=0.05, Ni_v=10, D=2):
    """
    Build a 2D pose-SLAM factor graph with:
    - N variable nodes (each 2D position)
    - Prior measurements (with strong precision)
    - Between measurements (with moderate precision from noise_std)
    
    Parameters:
    - N: number of variables
    - prior_meas: list of (i, z) where z is the prior measurement at node i
    - between_meas: list of (i, j, z) where z is relative measurement from i to j
    - noise_std: standard deviation for between measurements (list or array of length D)
    - Ni_v: number of factor connections per variable (default 5)
    - D: dimension of each variable (default 2)
    
    Returns:
    - varis: Variable object
    - prior_facs: Factor object for priors
    - between_facs: Factor object for between factors
    """

    # === Step 1: Initialize Variable nodes ===
    var_ids = jnp.arange(N, dtype=jnp.int32)
    belief = Gaussian(jnp.zeros((N, D)), jnp.tile(jnp.eye(D), (N, 1, 1)))  # initial mean 0, covariance I
    msgs = Gaussian(jnp.zeros((N, Ni_v, D)), jnp.zeros((N, Ni_v, D, D)))  # messages (eta, Lambda) to each factor port
    adj_factor_idx = -jnp.ones((N, Ni_v), dtype=jnp.int32)  # -1 indicates no connected factor at this port

    varis = Variable(var_ids, belief, msgs, adj_factor_idx)

    # === Step 2: Build Prior Factors (strong precision for anchoring the graph) ===
    prior_factor_id = []
    prior_z = []
    prior_z_Lam = []
    prior_threshold = []
    prior_adj_var_id = []
    prior_adj_var_idx = []

    fac_counter = 0  # global factor ID counter

    for (i, z) in prior_meas:
        prior_factor_id.append(fac_counter)
        prior_z.append(jnp.array(z))

        # Very weak prior: large noise variance -> small precision
        prior_z_Lam.append(jnp.eye(D) / prior_std)  # shape (D, D)

        prior_threshold.append(1.0)
        prior_adj_var_id.append([i])     # only connected to variable i
        prior_adj_var_idx.append([0])     # use port 0 for prior

        varis.adj_factor_idx = varis.adj_factor_idx.at[i, 0].set(fac_counter)
        fac_counter += 1

    prior_facs = Factor(
        factor_id=jnp.array(prior_factor_id),
        z=jnp.stack(prior_z),
        z_Lam=jnp.stack(prior_z_Lam),
        threshold=jnp.array(prior_threshold),
        potential=None,
        adj_var_id=jnp.array(prior_adj_var_id),
        adj_var_idx=jnp.array(prior_adj_var_idx),
    )

    # === Step 3: Build Between Factors (relative pose measurements) ===
    between_factor_id = []
    between_z = []
    between_z_Lam = []
    between_threshold = []
    between_adj_var_id = []
    between_adj_var_idx = []

    for (i, j, z) in between_meas:
        between_factor_id.append(fac_counter)
        between_z.append(jnp.array(z))

        # Between-factor noise: use provided noise_std to compute precision
        between_z_Lam.append(jnp.diag(1.0 / (jnp.ones(D)*odom_std)   ))  # shape (D, D)

        between_threshold.append(1.0)

        # Assign first empty port >=1 to variable i and j
        port_i = int(jnp.argmax(varis.adj_factor_idx[i, 1:] == -1)) + 1
        port_j = int(jnp.argmax(varis.adj_factor_idx[j, 1:] == -1)) + 1
        varis.adj_factor_idx = varis.adj_factor_idx.at[i, port_i].set(fac_counter)
        varis.adj_factor_idx = varis.adj_factor_idx.at[j, port_j].set(fac_counter)

        between_adj_var_id.append([i, j])
        between_adj_var_idx.append([port_i, port_j])
        fac_counter += 1

    between_facs = Factor(
        factor_id=jnp.array(between_factor_id),
        z=jnp.stack(between_z),
        z_Lam=jnp.stack(between_z_Lam),
        threshold=jnp.array(between_threshold),
        potential=None,
        adj_var_id=jnp.array(between_adj_var_id),
        adj_var_idx=jnp.array(between_adj_var_idx),
    )

    return varis, prior_facs, between_facs



def gbp_solve(varis, prior_facs, between_facs, num_iters=50, visualize=False, prior_h=h_fn, between_h=h2_fn):
    """
    Run GBP on a fine grid.

    Parameters:
    - varis: Variable objects to be inferred 
    - prior_facs: unary factors (priors on single variables)
    - between_facs: binary factors (between connected variable pairs)
    - num_iters: number of message passing iterations
    - visualize: whether to record energy and variable estimates during optimization
    - prior_h: measurement function for prior factors (default: h_fn)
    - between_h: measurement function for between factors (default: h2_fn)

    Returns:
    - varis: updated variables after inference
    - prior_facs: updated prior factors
    - between_facs: updated between factors
    - energy_log: array of total energy at each iteration (empty if visualize=False)
    - positions_log: array of linearization points per iteration (empty if visualize=False)
    """

    energy_log = []
    positions_log = []
    
    # First iteration: update variables with only prior factors
    varis, vtof_msgs, linpoints = update_variable(varis)
    prior_facs, varis = update_factor(prior_facs, varis, vtof_msgs, linpoints, prior_h, l2)

    if visualize:
        # Linearization points and energy computation
        prior_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
            prior_facs, linpoints[prior_facs.adj_var_id[:, 0]], prior_h
        ))

        between_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
            between_facs, linpoints[between_facs.adj_var_id], between_h
        ))

        energy = prior_energy + between_energy
        energy_log.append(energy)

        positions_log.append(linpoints)


    for i in range(num_iters-1):
        # Step 1: Variable update
        varis, vtof_msgs, linpoints = update_variable(varis)


        # Step 2: Factor update, both prior and between factors
        prior_facs, varis = update_factor(prior_facs, varis, vtof_msgs, linpoints, prior_h, l2)
        between_facs, varis = update_factor(between_facs, varis, vtof_msgs, linpoints, between_h, l2)

        # Step 3: Linearlization points and energy computation
        if visualize:
            
            prior_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
                prior_facs, linpoints[prior_facs.adj_var_id[:, 0]], prior_h
            ))

            between_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
                between_facs, linpoints[between_facs.adj_var_id], between_h
            ))

            energy = prior_energy + between_energy
            energy_log.append(energy)

            positions_log.append(linpoints)
        

    return varis, prior_facs, between_facs, np.array(energy_log), np.array(positions_log)



def build_coarse_slam_graph(
    varis_fine: Variable,
    prior_facs_fine: Factor,
    between_facs_fine: Factor,
    H: int, W: int,
    stride: int = 2,
    prior_std: float = 1.0,
    between_std: float = 0.1,
) -> Tuple[Variable, Factor, Factor]:
    D = 2
    patch_map: Dict[int, List[int]] = {}
    fine_to_patch: Dict[int, Tuple[int, int]] = {}
    coarse_var_id = 0

    coarse_beliefs = []
    coarse_msgs_eta = []
    coarse_msgs_Lam = []
    coarse_adj_factor_idx = []


    # === 1. Build Coarse Variables ===
    for i in range(0, H - 1, stride):
        for j in range(0, W - 1, stride):
            v00 = i * W + j
            v01 = v00 + 1
            v10 = v00 + W
            v11 = v10 + 1
            patch = [v00, v01, v10, v11]
            patch_map[coarse_var_id] = patch
            for k, vid in enumerate(patch):
                fine_to_patch[vid] = (coarse_var_id, k)

            Ni_v = 15
            eta = jnp.zeros((8))  # 8D for coarse variable
            Lam = jnp.zeros((8, 8))
            coarse_beliefs.append(Gaussian(eta, Lam))


            coarse_msgs_eta.append(jnp.zeros((Ni_v, 8)))
            coarse_msgs_Lam.append(jnp.zeros((Ni_v, 8, 8)))
            coarse_adj_factor_idx.append(-jnp.ones(Ni_v, dtype=jnp.int32))
            coarse_var_id += 1

    varis_coarse = Variable(
        var_id=jnp.arange(len(patch_map)),
        belief=tree_stack(coarse_beliefs, axis=0),
        msgs=Gaussian(jnp.stack(coarse_msgs_eta), jnp.stack(coarse_msgs_Lam)),
        adj_factor_idx=jnp.stack(coarse_adj_factor_idx),
    )

    # === 2. Build Coarse Priors ===
    fine_between_dict = {
        (int(i), int(j)): k for k, (i, j) in enumerate(between_facs_fine.adj_var_id)
    }

    fine_between_dict.update({(int(j), int(i)): k for k, (i, j) in enumerate(between_facs_fine.adj_var_id)})

    prior_ids, prior_zs, prior_zLams = [], [], []
    adj_var_ids, adj_var_idxs = [], []
    factor_id_counter = 0

    for patch_id, patch in patch_map.items():
        residuals = []
        precisions = []

        for v in patch:
            mask = (prior_facs_fine.adj_var_id[:, 0] == v)
            if jnp.any(mask):
                i = jnp.argmax(mask)
                z_i = prior_facs_fine.z[i]
                z_Lam_i = prior_facs_fine.z_Lam[i]
            else:
                z_i = jnp.zeros((D,))
                z_Lam_i = (1. / (prior_std ** 2)) * jnp.eye(D)
            residuals.append(z_i)
            precisions.append(z_Lam_i)

        edge_indices = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for i, j in edge_indices:
            a, b = patch[i], patch[j]
            key = (a, b)
            if key in fine_between_dict:
                k = fine_between_dict[key]
                a_k, b_k = between_facs_fine.adj_var_id[k]
                z = between_facs_fine.z[k]
                z_Lam = between_facs_fine.z_Lam[k]
                if a_k == b:
                    z = -z
                
                residuals.append(z)
                precisions.append(z_Lam)
            else:
                residuals.append(jnp.zeros((D,)))
                precisions.append((1. / (between_std ** 2)) * jnp.eye(D))

        z = jnp.concatenate(residuals)
        z_Lam = jax.scipy.linalg.block_diag(*precisions)


        prior_ids.append(factor_id_counter)
        prior_zs.append(z)
        prior_zLams.append(z_Lam)
        adj_var_ids.append(jnp.array([patch_id]))
        adj_var_idxs.append(jnp.array([0]))

        varis_coarse.adj_factor_idx = varis_coarse.adj_factor_idx.at[patch_id, 0].set(factor_id_counter)
        factor_id_counter += 1


    prior_facs_coarse = Factor(
        factor_id=jnp.array(prior_ids, dtype=jnp.int32),
        z=jnp.stack(prior_zs),
        z_Lam=jnp.stack(prior_zLams),
        threshold=jnp.ones((len(prior_ids),)),
        potential=None,
        adj_var_id=jnp.stack(adj_var_ids),
        adj_var_idx=jnp.stack(adj_var_idxs),
    )

    # === Build Horizontal & Vertical Between Factors Separately ===
    horizontal_ids, horizontal_zs, horizontal_zLams = [], [], []
    horizontal_adj_ids, horizontal_adj_idxs = [], []

    vertical_ids, vertical_zs, vertical_zLams = [], [], []
    vertical_adj_ids, vertical_adj_idxs = [], []

    height = H // stride
    width = W // stride

    for row in range(height):
        for col in range(width):
            patch_i = row * width + col

            # Horizontal neighbor
            if col < width - 1:
                patch_j = patch_i + 1
                pi_patch, pj_patch = patch_map[patch_i], patch_map[patch_j]

                fine_pairs = [(pi_patch[1], pj_patch[0]), (pi_patch[3], pj_patch[2])]

                residuals = []
                precisions = []
                for a, b in fine_pairs:
                    key = (a, b)
                    if key in fine_between_dict:
                        k = fine_between_dict[key]
                        a_k, b_k = between_facs_fine.adj_var_id[k]
                        z = between_facs_fine.z[k]
                        z_Lam = between_facs_fine.z_Lam[k]
                        if a_k == b:
                            z = -z
                    else:
                        z = jnp.zeros((D,))
                        z_Lam = (1. / (between_std ** 2)) * jnp.eye(D)
                    residuals.append(z)
                    precisions.append(z_Lam)

                z = jnp.concatenate(residuals)
                z_Lam = jax.scipy.linalg.block_diag(*precisions)

                adj_id = jnp.array([patch_i, patch_j])
                port_i = int(jnp.argmax(varis_coarse.adj_factor_idx[patch_i] == -1))
                port_j = int(jnp.argmax(varis_coarse.adj_factor_idx[patch_j] == -1))
                adj_idx = jnp.array([port_i, port_j])

                horizontal_ids.append(factor_id_counter)
                horizontal_zs.append(z)
                horizontal_zLams.append(z_Lam)
                horizontal_adj_ids.append(adj_id)
                horizontal_adj_idxs.append(adj_idx)

                
                varis_coarse.adj_factor_idx = varis_coarse.adj_factor_idx.at[patch_i, port_i].set(factor_id_counter)
                varis_coarse.adj_factor_idx = varis_coarse.adj_factor_idx.at[patch_j, port_j].set(factor_id_counter)

                factor_id_counter += 1

            # Vertical neighbor
            if row < height - 1:
                patch_j = patch_i + width
                pi_patch, pj_patch = patch_map[patch_i], patch_map[patch_j]
                fine_pairs = [(pi_patch[2], pj_patch[0]), (pi_patch[3], pj_patch[1])]

                residuals = []
                precisions = []
                for a, b in fine_pairs:
                    key = (a, b)
                    if key in fine_between_dict:
                        k = fine_between_dict[key]
                        a_k, b_k = between_facs_fine.adj_var_id[k]
                        z = between_facs_fine.z[k]
                        z_Lam = between_facs_fine.z_Lam[k]
                        if a_k == b:
                            z = -z
                    else:
                        z = jnp.zeros((D,))
                        z_Lam = (1. / (between_std ** 2)) * jnp.eye(D)
                    residuals.append(z)
                    precisions.append(z_Lam)

                z = jnp.concatenate(residuals)
                z_Lam = jax.scipy.linalg.block_diag(*precisions)

                adj_id = jnp.array([patch_i, patch_j])
                port_i = int(jnp.argmax(varis_coarse.adj_factor_idx[patch_i] == -1))
                port_j = int(jnp.argmax(varis_coarse.adj_factor_idx[patch_j] == -1))
                adj_idx = jnp.array([port_i, port_j])

                vertical_ids.append(factor_id_counter)
                vertical_zs.append(z)
                vertical_zLams.append(z_Lam)
                vertical_adj_ids.append(adj_id)
                vertical_adj_idxs.append(adj_idx)

                varis_coarse.adj_factor_idx = varis_coarse.adj_factor_idx.at[patch_i, port_i].set(factor_id_counter)
                varis_coarse.adj_factor_idx = varis_coarse.adj_factor_idx.at[patch_j, port_j].set(factor_id_counter)


                factor_id_counter += 1

    horizontal_between_facs = Factor(
        factor_id=jnp.array(horizontal_ids, dtype=jnp.int32),
        z=jnp.stack(horizontal_zs),
        z_Lam=jnp.stack(horizontal_zLams),
        threshold=jnp.ones((len(horizontal_ids),)),
        potential=None,
        adj_var_id=jnp.stack(horizontal_adj_ids),
        adj_var_idx=jnp.stack(horizontal_adj_idxs),
    )

    vertical_between_facs = Factor(
        factor_id=jnp.array(vertical_ids, dtype=jnp.int32),
        z=jnp.stack(vertical_zs),
        z_Lam=jnp.stack(vertical_zLams),
        threshold=jnp.ones((len(vertical_ids),)),
        potential=None,
        adj_var_id=jnp.stack(vertical_adj_ids),
        adj_var_idx=jnp.stack(vertical_adj_idxs),
    )

    return varis_coarse, prior_facs_coarse, horizontal_between_facs, vertical_between_facs



def gbp_solve_coarse(varis, prior_facs, horizontal_between_facs, vertical_between_facs, num_iters=50, visualize=False, prior_h=h3_fn, between_h=[h4_fn,h5_fn]):
    energy_log = []
    positions_log = []

    
    # Initialize variable with only priors factors 
    varis, vtof_msgs, linpoints = update_variable(varis)
    prior_facs, varis = update_factor(prior_facs, varis, vtof_msgs, linpoints, prior_h, l2)
    if visualize:
        # Linearization points and energy computation
        prior_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
            prior_facs, linpoints[prior_facs.adj_var_id[:, 0]], prior_h
        ))

        horizontal_between_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
            horizontal_between_facs, linpoints[horizontal_between_facs.adj_var_id], between_h[0]
        ))

        vertical_between_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
            vertical_between_facs, linpoints[vertical_between_facs.adj_var_id], between_h[1]
        ))


        energy = prior_energy + horizontal_between_energy + vertical_between_energy
        energy_log.append(energy)

        positions_log.append(linpoints)


    for i in range(num_iters-1):
        # Step 1: Variable update
        varis, vtof_msgs, linpoints = update_variable(varis)

        # Step 2: Factor update
        prior_facs, varis = update_factor(prior_facs, varis, vtof_msgs, linpoints, prior_h, l2)

        horizontal_between_facs, varis = update_factor(horizontal_between_facs, varis, vtof_msgs, 
                                                       linpoints, between_h[0], l2)

        vertical_between_facs, varis = update_factor(vertical_between_facs, varis, vtof_msgs, 
                                                       linpoints, between_h[1], l2)
        
        if visualize:
            # Step 3: Linearization points and energy computation
            prior_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
                prior_facs, linpoints[prior_facs.adj_var_id[:, 0]], prior_h
            ))
    
            horizontal_between_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
                horizontal_between_facs, linpoints[horizontal_between_facs.adj_var_id], between_h[0]
            ))
    
            vertical_between_energy = jnp.sum(jax.vmap(factor_energy, in_axes=(0, 0, None))(
                vertical_between_facs, linpoints[vertical_between_facs.adj_var_id], between_h[1]
            ))

            energy = prior_energy + horizontal_between_energy + vertical_between_energy

            energy_log.append(energy)
            positions_log.append(linpoints)

        
    return varis, prior_facs, horizontal_between_facs, vertical_between_facs, \
            np.array(energy_log), np.array(positions_log)



def build_coarser_slam_graph(
    varis_coarse: Variable,
    prior_facs_coarse: Factor,
    horizontal_between_facs: Factor,
    vertical_between_facs: Factor,
    H: int, W: int,
    stride: int = 2,
):
    
    patch_map: Dict[int, List[int]] = {}
    coarse_to_patch: Dict[int, Tuple[int, int]] = {}
    coarser_var_id = 0


    coarser_beliefs = []
    coarser_msgs_eta = []
    coarser_msgs_Lam = []
    coarser_adj_factor_idx = []

    H = H//2
    W = W//2

    # === 1. Build Coarse Variables ===
    for i in range(0, H - 1, stride):
        for j in range(0, W - 1, stride):
            v00 = i * W + j
            v01 = v00 + 1
            v10 = v00 + W
            v11 = v10 + 1
            patch = [v00, v01, v10, v11]
            patch_map[coarser_var_id] = patch
            for k, vid in enumerate(patch):
                coarse_to_patch[vid] = (coarser_var_id, k)

            Ni_v = 20
            eta = jnp.zeros((32))  # 32D for coarse variable
            Lam = jnp.zeros((32, 32))
            coarser_beliefs.append(Gaussian(eta, Lam))

            coarser_msgs_eta.append(jnp.zeros((Ni_v, 32)))
            coarser_msgs_Lam.append(jnp.zeros((Ni_v, 32, 32)))
            coarser_adj_factor_idx.append(-jnp.ones(Ni_v, dtype=jnp.int32))
            coarser_var_id += 1


    varis_coarser = Variable(
        var_id=jnp.arange(len(patch_map)),
        belief=tree_stack(coarser_beliefs, axis=0),
        msgs=Gaussian(jnp.stack(coarser_msgs_eta), jnp.stack(coarser_msgs_Lam)),
        adj_factor_idx=jnp.stack(coarser_adj_factor_idx),
    )


    # === 2. Build Coarser Priors ===
    horizontal_between_facs_dict = {
        (int(i), int(j)): k for k, (i, j) in enumerate(horizontal_between_facs.adj_var_id)
    }
    horizontal_between_facs_dict.update(
        {(int(j), int(i)): k for k, (i, j) in enumerate(horizontal_between_facs.adj_var_id)})

    vertical_between_facs_dict = {
        (int(i), int(j)): k for k, (i, j) in enumerate(vertical_between_facs.adj_var_id)
    }
    vertical_between_facs_dict.update(
        {(int(j), int(i)): k for k, (i, j) in enumerate(vertical_between_facs.adj_var_id)})
    
    #print("horizontal_between_facs_dict",horizontal_between_facs_dict)

    prior_ids, prior_zs, prior_zLams = [], [], []
    adj_var_ids, adj_var_idxs = [], []
    factor_id_counter = 0

    for patch_id, patch in patch_map.items():
        residuals = []
        precisions = []

        for v in patch:
            mask = (prior_facs_coarse.adj_var_id[:, 0] == v)

            i = jnp.argmax(mask)
            z_i = prior_facs_coarse.z[i]
            z_Lam_i = prior_facs_coarse.z_Lam[i]

            residuals.append(z_i)
            precisions.append(z_Lam_i)

        edge_indices = [(0, 1), (0, 2), (1, 3), (2, 3)]
        for i, j in edge_indices:
            a, b = patch[i], patch[j]

            #print("ijab:",i,j,a,b)
            key = (a, b)

            if key in horizontal_between_facs_dict:
                k = horizontal_between_facs_dict[key]
                z = horizontal_between_facs.z[k]
                z_Lam = horizontal_between_facs.z_Lam[k]
                
            elif key in vertical_between_facs_dict:
                k = vertical_between_facs_dict[key]
                z = vertical_between_facs.z[k]
                z_Lam = vertical_between_facs.z_Lam[k]
                
            residuals.append(z)
            precisions.append(z_Lam)


        z = jnp.concatenate(residuals)
        z_Lam = jax.scipy.linalg.block_diag(*precisions)


        prior_ids.append(factor_id_counter)
        prior_zs.append(z)
        prior_zLams.append(z_Lam)
        adj_var_ids.append(jnp.array([patch_id]))
        adj_var_idxs.append(jnp.array([0]))

        varis_coarser.adj_factor_idx = varis_coarser.adj_factor_idx.at[patch_id, 0].set(factor_id_counter)
        factor_id_counter += 1


    prior_facs_coarser = Factor(
        factor_id=jnp.array(prior_ids, dtype=jnp.int32),
        z=jnp.stack(prior_zs),
        z_Lam=jnp.stack(prior_zLams),
        threshold=jnp.ones((len(prior_ids),)),
        potential=None,
        adj_var_id=jnp.stack(adj_var_ids),
        adj_var_idx=jnp.stack(adj_var_idxs),
    )


    # === Build Horizontal & Vertical Between Factors Separately ===
    horizontal_ids, horizontal_zs, horizontal_zLams = [], [], []
    horizontal_adj_ids, horizontal_adj_idxs = [], []

    vertical_ids, vertical_zs, vertical_zLams = [], [], []
    vertical_adj_ids, vertical_adj_idxs = [], []

    height = H // stride
    width = W // stride

    for row in range(height):
        for col in range(width):
            patch_i = row * width + col

            # Horizontal neighbor
            if col < width - 1:
                patch_j = patch_i + 1
                pi_patch, pj_patch = patch_map[patch_i], patch_map[patch_j]

                # pi_patch 0,1,8,9, pj_patch 2,3,10,11 
                coarse_pairs = [(pi_patch[1], pj_patch[0]), (pi_patch[3], pj_patch[2])]

                residuals = []
                precisions = []
                for a, b in coarse_pairs:
                    key = (a, b)
                    k = horizontal_between_facs_dict[key]

                    z = horizontal_between_facs.z[k]
                    z_Lam = horizontal_between_facs.z_Lam[k]


                    residuals.append(z)
                    precisions.append(z_Lam)


                z = jnp.concatenate(residuals)
                z_Lam = jax.scipy.linalg.block_diag(*precisions)

                adj_id = jnp.array([patch_i, patch_j])
                port_i = int(jnp.argmax(varis_coarser.adj_factor_idx[patch_i] == -1))
                port_j = int(jnp.argmax(varis_coarser.adj_factor_idx[patch_j] == -1))
                adj_idx = jnp.array([port_i, port_j])


                horizontal_ids.append(factor_id_counter)
                horizontal_zs.append(z)
                horizontal_zLams.append(z_Lam)
                horizontal_adj_ids.append(adj_id)
                horizontal_adj_idxs.append(adj_idx)

                varis_coarser.adj_factor_idx = varis_coarser.adj_factor_idx.at[patch_i, port_i].set(factor_id_counter)
                varis_coarser.adj_factor_idx = varis_coarser.adj_factor_idx.at[patch_j, port_j].set(factor_id_counter)

                factor_id_counter += 1


            # Vertical neighbor
            if row < height - 1:
                patch_j = patch_i + width
                pi_patch, pj_patch = patch_map[patch_i], patch_map[patch_j]
                coarse_pairs = [(pi_patch[2], pj_patch[0]), (pi_patch[3], pj_patch[1])]

                residuals = []
                precisions = []
                for a, b in coarse_pairs:
                    key = (a, b)
                    k = vertical_between_facs_dict[key]

                    a_k = vertical_between_facs.adj_var_id[k, 0]  # coarse variable id
                    #print("vertical, k, key=(a,b)",k,key)
                    z = vertical_between_facs.z[k]
                    z_Lam = vertical_between_facs.z_Lam[k]
                    if a_k == b:
                            z = -z


                    residuals.append(z)
                    precisions.append(z_Lam)

                z = jnp.concatenate(residuals)
                z_Lam = jax.scipy.linalg.block_diag(*precisions)

                adj_id = jnp.array([patch_i, patch_j])
                port_i = int(jnp.argmax(varis_coarser.adj_factor_idx[patch_i] == -1))
                port_j = int(jnp.argmax(varis_coarser.adj_factor_idx[patch_j] == -1))
                adj_idx = jnp.array([port_i, port_j])

            
                #print("vertical", factor_id_counter)
                vertical_ids.append(factor_id_counter)
                vertical_zs.append(z)
                vertical_zLams.append(z_Lam)
                vertical_adj_ids.append(adj_id)
                vertical_adj_idxs.append(adj_idx)

                varis_coarser.adj_factor_idx = varis_coarser.adj_factor_idx.at[patch_i, port_i].set(factor_id_counter)
                varis_coarser.adj_factor_idx = varis_coarser.adj_factor_idx.at[patch_j, port_j].set(factor_id_counter)

                factor_id_counter += 1

    horizontal_coarser_facs = Factor(
        factor_id=jnp.array(horizontal_ids, dtype=jnp.int32),
        z=jnp.stack(horizontal_zs),
        z_Lam=jnp.stack(horizontal_zLams),
        threshold=jnp.ones((len(horizontal_ids),)),
        potential=None,
        adj_var_id=jnp.stack(horizontal_adj_ids),
        adj_var_idx=jnp.stack(horizontal_adj_idxs),
    )

    vertical_coarser_facs = Factor(
        factor_id=jnp.array(vertical_ids, dtype=jnp.int32),
        z=jnp.stack(vertical_zs),
        z_Lam=jnp.stack(vertical_zLams),
        threshold=jnp.ones((len(vertical_ids),)),
        potential=None,
        adj_var_id=jnp.stack(vertical_adj_ids),
        adj_var_idx=jnp.stack(vertical_adj_idxs),
    )

    return varis_coarser, prior_facs_coarser, horizontal_coarser_facs, vertical_coarser_facs

