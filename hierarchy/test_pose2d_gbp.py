# test_pose2d_gbp.py
# Minimal 2D pose-graph SLAM test for the GBP implementation in gbp.py

import numpy as np
from gbp.gbp import FactorGraph, VariableNode, Factor

def make_circle_gt(n=20, radius=5.0):
    """Generate ground-truth 2D poses on a circle."""
    thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = radius * np.cos(thetas)
    ys = radius * np.sin(thetas)
    return np.stack([xs, ys], axis=1)  # (n,2)

def h_between(x):
    """
    Linear measurement model: z = x_j - x_i, where
    x = [xi, yi, xj, yj]^T
    """
    return np.array([x[2] - x[0], x[3] - x[1]], dtype=float)

# Constant Jacobian for the linear model above
J_BETWEEN = [[-1.0, 0.0, 1.0, 0.0],
             [ 0.0,-1.0, 0.0, 1.0]]

def build_graph(gt_xy, odo_noise_std=0.10, close_loop=True, seed=0):
    """
    Build a simple 2D pose-graph with between factors (odometry-style).
    Anchors pose 0 with a strong prior.
    """
    rng = np.random.default_rng(seed)
    N = gt_xy.shape[0]

    fg = FactorGraph(
        nonlinear_factors=False,   # linear model; no relinearisation needed
        eta_damping=0.0
    )

    # --- Create variables (each is 2-DoF: x, y)
    for i in range(N):
        v = VariableNode(i, dofs=2)
        v.type = "pose"  # avoid any multigrid "multi*" checks
        v.GT = gt_xy[i].copy()  # used by error printing in FactorGraph (optional)

        # Small positive prior for numerical stability on all nodes
        tiny = 1e-6
        v.prior.lam = np.diag([tiny, tiny]).astype(float)
        v.prior.eta = v.prior.lam @ np.zeros(2, dtype=float)

        # Strong prior to anchor the first node at its GT
        if i == 0:
            w = 1e6
            v.prior.lam = np.diag([w, w]).astype(float)
            v.prior.eta = v.prior.lam @ v.GT

        fg.var_nodes.append(v)

    fg.n_var_nodes = len(fg.var_nodes)

    # --- Create between factors for consecutive pairs
    fid = 0
    sigma = odo_noise_std
    for i in range(N - 1):
        z_ij = (gt_xy[i+1] - gt_xy[i]) + rng.normal(0.0, sigma, size=2)
        f = Factor(
            factor_id=fid,
            adj_var_nodes=[fg.var_nodes[i], fg.var_nodes[i+1]],
            measurement=z_ij,
            gauss_noise_std=sigma,
            meas_fn=h_between,      # not used when jac_fn is a list, but set for completeness
            jac_fn=J_BETWEEN,
        )
        f.type = "between"
        fg.factors.append(f)
        fid += 1

    # Optional loop closure to make a ring
    if close_loop:
        z_cl = (gt_xy[0] - gt_xy[-1]) + rng.normal(0.0, sigma, size=2)
        f = Factor(
            factor_id=fid,
            adj_var_nodes=[fg.var_nodes[-1], fg.var_nodes[0]],
            measurement=z_cl,
            gauss_noise_std=sigma,
            meas_fn=h_between,
            jac_fn=J_BETWEEN,
        )
        f.type = "between"
        fg.factors.append(f)
        fid += 1

    fg.n_factor_nodes = len(fg.factors)

    # Wire adjacency (so variables see their incident factors)
    for f in fg.factors:
        for v in f.adj_var_nodes:
            v.adj_factors.append(f)

    # Initial beliefs from priors
    for v in fg.var_nodes:
        v.update_belief()

    # Compute initial factor parameters (use current beliefs as linearisation points)
    for f in fg.factors:
        f.compute_factor()

    return fg

def avg_pos_err(fg):
    errs = []
    for v in fg.var_nodes:
        if hasattr(v, "GT"):
            errs.append(np.linalg.norm(v.mu - v.GT))
    return float(np.mean(errs)) if errs else np.nan

def main():
    np.set_printoptions(precision=4, suppress=True)
    N = 20
    gt = make_circle_gt(N, radius=5.0)

    fg = build_graph(gt_xy=gt, odo_noise_std=0.10, close_loop=True, seed=42)

    iters = 100
    print("Running synchronous GBP...")
    for k in range(1, iters + 1):
        # One synchronous iteration: compute all messages, then update beliefs
        fg.synchronous_iteration()
        # (Optional) update residuals if you want to inspect energy
        fg.update_all_residuals()

        if k % 10 == 0 or k == 1:
            err = avg_pos_err(fg)
            print(f"[iter {k:3d}] avg |mu - GT| = {err:.4f}")

    # Show a few estimates vs GT
    print("\nFirst 5 poses (mu vs GT):")
    for i in range(min(5, fg.n_var_nodes)):
        print(f"  {i:02d}: mu={fg.var_nodes[i].mu}, GT={fg.var_nodes[i].GT}")

if __name__ == "__main__":
    main()
