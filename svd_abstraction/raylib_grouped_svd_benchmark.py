from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import scipy.linalg


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOCAL_RAYLIB_ROOT = SCRIPT_DIR / "raylib_gbp_local"

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(LOCAL_RAYLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_RAYLIB_ROOT))

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(1, str(WORKSPACE_ROOT))

from svd_abstraction.grouping import groups_from_grid
from svd_abstraction.grouping import groups_from_kmeans
from svd_abstraction.grouping import groups_from_order
from svd_abstraction.raylib_baware_interp_analysis import collect_fixed_lam_errors
from svd_abstraction.raylib_baware_interp_analysis import odom_tiny_init_base
from svd_abstraction.raylib_fanaskov_twolevel_experiment import fanaskov_edge_solve
from svd_abstraction.raylib_fanaskov_twolevel_experiment import relative_error_vec
from svd_abstraction.raylib_local_eta_prolongation_validation import build_slam_graph
from svd_abstraction.raylib_local_eta_prolongation_validation import exact_mean
from svd_abstraction.raylib_local_eta_prolongation_validation import mean_vector
from svd_abstraction.raylib_same_hierarchy_jacobi_experiment import build_transfer_operators
from svd_abstraction.raylib_variance_freeze_experiment import eta_only_iteration
from svd_abstraction.raylib_variance_freeze_experiment import lam_state


def base_nodes(graph) -> list[dict]:
    nodes = []
    for var in graph.multigrid_vars[0]:
        if getattr(var, "type", None) == "dead":
            continue
        gt = np.asarray(var.GT, dtype=float)
        nodes.append(
            {
                "data": {"id": str(int(var.variableID))},
                "position": {"x": float(gt[0]), "y": float(gt[1])},
            }
        )
    return nodes


def var_slices(graph) -> dict[int, slice]:
    mapping = {}
    offset = 0
    for var in graph.multigrid_vars[0]:
        if getattr(var, "type", None) == "dead":
            continue
        dofs = int(var.dofs)
        mapping[int(var.variableID)] = slice(offset, offset + dofs)
        offset += dofs
    return mapping


def ordered_base_ids(graph) -> list[int]:
    return [
        int(var.variableID)
        for var in graph.multigrid_vars[0]
        if getattr(var, "type", None) != "dead"
    ]


def loop_and_degree_stats(graph, loop_sep_min: int = 2) -> tuple[dict[int, int], dict[int, int]]:
    degree = {vid: 0 for vid in ordered_base_ids(graph)}
    loop_touch = {vid: 0 for vid in degree}
    for factor in graph.multigrid_factors[0]:
        if getattr(factor, "type", None) == "dead":
            continue
        if len(factor.adj_vIDs) != 2:
            continue
        i, j = [int(v) for v in factor.adj_vIDs]
        degree[i] += 1
        degree[j] += 1
        if abs(i - j) >= loop_sep_min:
            loop_touch[i] += 1
            loop_touch[j] += 1
    return degree, loop_touch


def weighted_order_groups(ids: list[int], weights: dict[int, float], target_groups: int) -> list[list[int]]:
    if not ids:
        return []
    target_groups = max(1, min(int(target_groups), len(ids)))
    total_weight = float(sum(weights.get(i, 1.0) for i in ids))
    target_weight = total_weight / target_groups

    groups: list[list[int]] = []
    current: list[int] = []
    current_weight = 0.0
    remaining_groups = target_groups
    remaining_ids = len(ids)

    for idx, var_id in enumerate(ids):
        current.append(var_id)
        current_weight += float(weights.get(var_id, 1.0))
        remaining_ids = len(ids) - idx - 1
        # Keep enough ids for the remaining groups and cut once we reached the
        # target local weight.
        if len(current) > 0 and remaining_groups > 1:
            enough_left = remaining_ids >= (remaining_groups - 1)
            if enough_left and current_weight >= target_weight:
                groups.append(current)
                current = []
                current_weight = 0.0
                remaining_groups -= 1

    if current:
        groups.append(current)
    return groups


def group_list(
    graph,
    method: str,
    group_size: int,
    gx: int,
    gy: int,
    kmeans_k: int,
    target_groups: int | None,
    loop_window: int,
    loop_boost: float,
    degree_boost: float,
    loop_sep_min: int,
) -> list[list[int]]:
    nodes = base_nodes(graph)
    ids = ordered_base_ids(graph)
    if method == "order":
        return groups_from_order(nodes, group_size=group_size, tail_heavy=True)
    if method == "grid":
        return groups_from_grid(nodes, gx=gx, gy=gy)
    if method == "kmeans":
        return groups_from_kmeans(nodes, k=kmeans_k, seed=0)
    if method in {"loop_aware", "degree_aware"}:
        degree, loop_touch = loop_and_degree_stats(graph, loop_sep_min=loop_sep_min)
        if target_groups is None:
            target_groups = len(groups_from_order(nodes, group_size=group_size, tail_heavy=True))
        weights = {vid: 1.0 for vid in ids}
        if method == "loop_aware":
            for vid in ids:
                base_weight = 1.0 + loop_boost * loop_touch.get(vid, 0)
                local_max = loop_touch.get(vid, 0)
                for offset in range(1, loop_window + 1):
                    local_max = max(
                        local_max,
                        loop_touch.get(vid - offset, 0),
                        loop_touch.get(vid + offset, 0),
                    )
                weights[vid] = 1.0 + loop_boost * local_max
        else:
            for vid in ids:
                weights[vid] = 1.0 + degree_boost * max(0, degree.get(vid, 0) - 2)
        return weighted_order_groups(ids, weights=weights, target_groups=target_groups)
    raise ValueError(f"Unknown grouping method: {method}")


def blockdiag_belief_info(graph, group: list[int]) -> np.ndarray:
    blocks = []
    base_vars = {int(var.variableID): var for var in graph.multigrid_vars[0] if getattr(var, "type", None) != "dead"}
    for var_id in group:
        blocks.append(np.asarray(base_vars[var_id].belief.lam, dtype=float))
    return scipy.linalg.block_diag(*blocks)


def blockdiag_belief_cov(graph, group: list[int]) -> np.ndarray:
    blocks = []
    base_vars = {int(var.variableID): var for var in graph.multigrid_vars[0] if getattr(var, "type", None) != "dead"}
    for var_id in group:
        lam = np.asarray(base_vars[var_id].belief.lam, dtype=float)
        blocks.append(np.linalg.inv(lam))
    return scipy.linalg.block_diag(*blocks)


def group_boundary_vars(graph, group: list[int]) -> tuple[list[int], list[int]]:
    group_set = {int(v) for v in group}
    boundary = []
    interior = []
    base_vars = {int(var.variableID): var for var in graph.multigrid_vars[0] if getattr(var, "type", None) != "dead"}
    for var_id in group:
        var = base_vars[int(var_id)]
        touches_outside = False
        for factor in var.adj_factors:
            for adj_vid in factor.adj_vIDs:
                if int(adj_vid) not in group_set:
                    touches_outside = True
                    break
            if touches_outside:
                break
        if touches_outside:
            boundary.append(int(var_id))
        else:
            interior.append(int(var_id))
    return boundary, interior


def local_boundary_modes(
    graph,
    a0: np.ndarray,
    group: list[int],
    slices: dict[int, slice],
    r_reduced: int,
    ridge: float = 1e-10,
) -> np.ndarray:
    boundary_vars, interior_vars = group_boundary_vars(graph, group)
    if not boundary_vars or not interior_vars:
        full_indices = []
        for var_id in group:
            sl = slices[int(var_id)]
            full_indices.extend(range(sl.start, sl.stop))
        full_indices = np.array(full_indices, dtype=int)
        a_sub = a0[np.ix_(full_indices, full_indices)]
        eigvals, eigvecs = np.linalg.eigh(a_sub)
        return eigvecs[:, : min(int(r_reduced), a_sub.shape[0])]

    boundary_idx = []
    for var_id in boundary_vars:
        sl = slices[int(var_id)]
        boundary_idx.extend(range(sl.start, sl.stop))
    boundary_idx = np.array(boundary_idx, dtype=int)

    interior_idx = []
    for var_id in interior_vars:
        sl = slices[int(var_id)]
        interior_idx.extend(range(sl.start, sl.stop))
    interior_idx = np.array(interior_idx, dtype=int)

    a_bb = a0[np.ix_(boundary_idx, boundary_idx)]
    a_bi = a0[np.ix_(boundary_idx, interior_idx)]
    a_ib = a0[np.ix_(interior_idx, boundary_idx)]
    a_ii = a0[np.ix_(interior_idx, interior_idx)]
    a_ii_reg = a_ii + ridge * np.eye(a_ii.shape[0], dtype=float)

    try:
        solve_ii = np.linalg.solve(a_ii_reg, a_ib)
    except np.linalg.LinAlgError:
        solve_ii = np.linalg.pinv(a_ii_reg) @ a_ib
    schur = a_bb - a_bi @ solve_ii
    schur = 0.5 * (schur + schur.T)

    eigvals, eigvecs = np.linalg.eigh(schur)
    n_modes = min(int(r_reduced), schur.shape[0])
    boundary_modes = eigvecs[:, :n_modes]
    interior_modes = -solve_ii @ boundary_modes

    full_indices = []
    for var_id in group:
        sl = slices[int(var_id)]
        full_indices.extend(range(sl.start, sl.stop))
    full_indices = np.array(full_indices, dtype=int)

    local_pos = {idx: pos for pos, idx in enumerate(full_indices.tolist())}
    basis = np.zeros((full_indices.shape[0], n_modes), dtype=float)
    for row, global_idx in enumerate(boundary_idx.tolist()):
        basis[local_pos[global_idx], :] = boundary_modes[row, :]
    for row, global_idx in enumerate(interior_idx.tolist()):
        basis[local_pos[global_idx], :] = interior_modes[row, :]

    q, _ = np.linalg.qr(basis, mode="reduced")
    return q[:, :n_modes]


def build_grouped_svd_basis(
    graph,
    a0: np.ndarray,
    groups: list[list[int]],
    r_reduced: int,
    basis_source: str,
) -> np.ndarray:
    slices = var_slices(graph)
    total_dim = a0.shape[0]
    if basis_source == "joint_covariance":
        source_matrix = np.linalg.inv(a0)
    elif basis_source == "joint_information":
        source_matrix = a0
    else:
        source_matrix = None

    total_reduced = 0
    local_bases: list[np.ndarray] = []
    full_indices_per_group: list[np.ndarray] = []

    for group in groups:
        full_indices = []
        for var_id in group:
            sl = slices[int(var_id)]
            full_indices.extend(range(sl.start, sl.stop))
        full_indices_arr = np.array(full_indices, dtype=int)
        full_indices_per_group.append(full_indices_arr)

        if basis_source == "joint_covariance" or basis_source == "joint_information":
            block = source_matrix[np.ix_(full_indices_arr, full_indices_arr)]
        elif basis_source == "local_sub_covariance":
            a_sub = a0[np.ix_(full_indices_arr, full_indices_arr)]
            block = np.linalg.inv(a_sub)
        elif basis_source == "local_sub_information":
            block = a0[np.ix_(full_indices_arr, full_indices_arr)]
        elif basis_source == "belief_covariance":
            block = blockdiag_belief_cov(graph, group)
        elif basis_source == "belief_information":
            block = blockdiag_belief_info(graph, group)
        elif basis_source == "local_boundary_modes":
            basis_local = local_boundary_modes(
                graph=graph,
                a0=a0,
                group=group,
                slices=slices,
                r_reduced=r_reduced,
            )
            local_bases.append(basis_local)
            total_reduced += basis_local.shape[1]
            continue
        else:
            raise ValueError(f"Unknown basis_source: {basis_source}")

        full_dim = block.shape[0]
        r_local = min(int(r_reduced), full_dim)
        eigvals, eigvecs = np.linalg.eigh(block)
        if basis_source.endswith("covariance"):
            order = np.argsort(eigvals)[::-1]
        else:
            order = np.argsort(eigvals)
        basis_local = eigvecs[:, order[:r_local]]
        local_bases.append(basis_local)
        total_reduced += r_local

    p = np.zeros((total_dim, total_reduced), dtype=float)
    offset = 0
    for full_indices, basis_local in zip(full_indices_per_group, local_bases):
        r_local = basis_local.shape[1]
        p[np.ix_(full_indices, np.arange(offset, offset + r_local))] = basis_local
        offset += r_local

    return p


def projection_residual(vec: np.ndarray, basis: np.ndarray) -> float:
    coeffs, *_ = np.linalg.lstsq(basis, vec, rcond=None)
    fit = basis @ coeffs
    return float(np.linalg.norm(vec - fit) / max(np.linalg.norm(vec), 1e-15))


def projection_residuals(vectors: dict[int, np.ndarray], basis: np.ndarray) -> dict[int, float]:
    return {step: projection_residual(vec, basis) for step, vec in vectors.items()}


def ideal_coarse_correction(a0: np.ndarray, b0: np.ndarray, x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    residual = b0 - a0 @ x
    ac = basis.T @ a0 @ basis
    rc = basis.T @ residual
    yc = np.linalg.solve(ac, rc)
    return x + basis @ yc


def run_two_level_subspace_fanaskov(
    basis: np.ndarray,
    mu_star: np.ndarray,
    a0: np.ndarray,
    b0: np.ndarray,
    x0: np.ndarray,
    max_cycles: int,
    tol: float,
    base_sweeps: int,
    coarse_sweeps: int,
) -> tuple[int | None, list[float]]:
    ac = basis.T @ a0 @ basis
    x = x0.copy()
    relerrs = [relative_error_vec(x, mu_star)]
    conv = 0 if relerrs[-1] < tol else None

    for cyc in range(1, max_cycles + 1):
        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a=a0, b=residual, n_sweeps=base_sweeps, mode="parallel")

        residual = b0 - a0 @ x
        coarse_rhs = basis.T @ residual
        coarse_err = fanaskov_edge_solve(a=ac, b=coarse_rhs, n_sweeps=coarse_sweeps, mode="parallel")
        x = x + basis @ coarse_err

        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a=a0, b=residual, n_sweeps=base_sweeps, mode="parallel")

        rel = relative_error_vec(x, mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = cyc
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, relerrs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--grouping",
        type=str,
        default="order",
        choices=["order", "grid", "kmeans", "loop_aware", "degree_aware"],
    )
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--gx", type=int, default=8)
    parser.add_argument("--gy", type=int, default=4)
    parser.add_argument("--kmeans-k", type=int, default=26)
    parser.add_argument("--target-groups", type=int, default=None)
    parser.add_argument("--loop-window", type=int, default=2)
    parser.add_argument("--loop-boost", type=float, default=3.0)
    parser.add_argument("--degree-boost", type=float, default=1.0)
    parser.add_argument("--loop-sep-min", type=int, default=2)
    parser.add_argument(
        "--basis-source",
        type=str,
        default="joint_covariance",
        choices=[
            "joint_covariance",
            "joint_information",
            "local_sub_covariance",
            "local_sub_information",
            "belief_covariance",
            "belief_information",
            "local_boundary_modes",
        ],
    )
    parser.add_argument("--r-reduced", type=int, default=2)
    parser.add_argument("--base-sweeps", type=int, default=1)
    parser.add_argument("--coarse-sweeps", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--late-step", type=int, default=2000)
    parser.add_argument("--sample-steps", type=int, nargs="+", default=[100, 500, 1000, 2000])
    parser.add_argument("--compare-raylib-p", action="store_true")
    args = parser.parse_args()

    graph = build_slam_graph(n=args.n, seed=args.seed, prior_prop=args.prior_prop)
    mu_star = exact_mean(graph)
    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)

    # Freeze base lam using the original base dynamics.
    prev = lam_state(graph, [0])
    freeze_step = None
    for step in range(1, 1000):
        graph.synchronous_iteration(level=0)
        curr = lam_state(graph, [0])
        delta = float(np.max(np.abs(curr - prev)))
        if delta < args.variance_threshold:
            freeze_step = step
            break
        prev = curr
    if freeze_step is None:
        raise RuntimeError("Base lam did not settle within 1000 iterations")

    groups = group_list(
        graph,
        method=args.grouping,
        group_size=args.group_size,
        gx=args.gx,
        gy=args.gy,
        kmeans_k=args.kmeans_k,
        target_groups=args.target_groups,
        loop_window=args.loop_window,
        loop_boost=args.loop_boost,
        degree_boost=args.degree_boost,
        loop_sep_min=args.loop_sep_min,
    )
    p_grouped = build_grouped_svd_basis(
        graph,
        a0=a0,
        groups=groups,
        r_reduced=args.r_reduced,
        basis_source=args.basis_source,
    )

    odom_tiny_init_base(graph, n=args.n)
    x0 = mean_vector(graph)
    errors = collect_fixed_lam_errors(graph, mu_star, args.sample_steps)
    late_error = errors[max(args.sample_steps)]
    x_late = mu_star + late_error

    proj_res = projection_residuals(errors, p_grouped)
    x0_corr = ideal_coarse_correction(a0, b0, x0, p_grouped)
    xlate_corr = ideal_coarse_correction(a0, b0, x_late, p_grouped)
    two_conv, two_relerrs = run_two_level_subspace_fanaskov(
        basis=p_grouped,
        mu_star=mu_star,
        a0=a0,
        b0=b0,
        x0=x0,
        max_cycles=args.max_cycles,
        tol=args.tol,
        base_sweeps=args.base_sweeps,
        coarse_sweeps=args.coarse_sweeps,
    )

    print(f"n={args.n} seed={args.seed} prior_prop={args.prior_prop}")
    print(
        "grouping="
        f"{args.grouping} group_size={args.group_size} gx={args.gx} gy={args.gy} "
        f"kmeans_k={args.kmeans_k} target_groups={args.target_groups} "
        f"loop_window={args.loop_window} loop_boost={args.loop_boost} degree_boost={args.degree_boost} "
        f"loop_sep_min={args.loop_sep_min}"
    )
    print(f"basis_source={args.basis_source} r_reduced={args.r_reduced}")
    print(f"freeze_step={freeze_step}")
    print(f"num_groups={len(groups)} coarse_dim={p_grouped.shape[1]} nnzP={int(np.count_nonzero(np.abs(p_grouped) > 0.0))}")
    print("projection_residuals")
    for step in args.sample_steps:
        print(f"  step {step}: {proj_res[step]}")

    print("ideal_correction")
    print(f"  from_odom_tiny_before {relative_error_vec(x0, mu_star)}")
    print(f"  from_odom_tiny_after {relative_error_vec(x0_corr, mu_star)}")
    print(f"  from_late_before {relative_error_vec(x_late, mu_star)}")
    print(f"  from_late_after {relative_error_vec(xlate_corr, mu_star)}")

    print(f"twolevel_conv {two_conv}")
    for point in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]:
        if point < len(two_relerrs):
            print(f"twolevel {point} {two_relerrs[point]}")

    if args.compare_raylib_p:
        graph_compare = build_slam_graph(n=args.n, seed=args.seed, prior_prop=args.prior_prop)
        from svd_abstraction.raylib_local_eta_prolongation_validation import build_hierarchy
        from svd_abstraction.raylib_same_hierarchy_jacobi_experiment import build_transfer_operators

        build_hierarchy(graph_compare)
        _, p_raylib = build_transfer_operators(graph_compare, coarse_level=1)
        raylib_proj = projection_residuals(errors, p_raylib)
        print(f"raylib_coarse_dim {p_raylib.shape[1]}")
        print("raylib_projection_residuals")
        for step in args.sample_steps:
            print(f"  step {step}: {raylib_proj[step]}")


if __name__ == "__main__":
    main()
