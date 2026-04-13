from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouping import groups_from_grid
from svd_abstraction.grouping import groups_from_kmeans
from svd_abstraction.grouping import groups_from_order
from svd_abstraction.pose_graph import build_noisy_pose_graph
from svd_abstraction.pose_graph import make_slam_like_graph
from svd_abstraction.residual_abstraction import SVDResidualHierarchy


def mean_vector(graph):
    return np.concatenate([var.mu for var in graph.var_nodes[: graph.n_var_nodes]])


def relative_error(graph, x_star):
    denom = np.linalg.norm(x_star)
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(mean_vector(graph) - x_star) / denom)


def ordered_ids(graph) -> list[int]:
    return [int(var.variableID) for var in graph.var_nodes[: graph.n_var_nodes] if getattr(var, "active", True)]


def loop_and_degree_stats(graph, loop_sep_min: int = 2) -> tuple[dict[int, int], dict[int, int]]:
    degree = {vid: 0 for vid in ordered_ids(graph)}
    loop_touch = {vid: 0 for vid in degree}
    for factor in graph.factors[: graph.n_factor_nodes]:
        if not getattr(factor, "active", True):
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

    for idx, var_id in enumerate(ids):
        current.append(var_id)
        current_weight += float(weights.get(var_id, 1.0))
        remaining_ids = len(ids) - idx - 1
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
    nodes,
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
    if method == "order":
        return groups_from_order(nodes, group_size=group_size, tail_heavy=True)
    if method == "grid":
        return groups_from_grid(nodes, gx=gx, gy=gy)
    if method == "kmeans":
        return groups_from_kmeans(nodes, k=kmeans_k, seed=0)
    if method in {"loop_aware", "degree_aware"}:
        ids = ordered_ids(graph)
        degree, loop_touch = loop_and_degree_stats(graph, loop_sep_min=loop_sep_min)
        if target_groups is None:
            target_groups = len(groups_from_order(nodes, group_size=group_size, tail_heavy=True))
        weights = {vid: 1.0 for vid in ids}
        if method == "loop_aware":
            for vid in ids:
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


def build_graphs(n, step_size, loop_prob, loop_radius, prior_prop, prior_sigma, odom_sigma, seed):
    nodes, edges = make_slam_like_graph(
        N=n,
        step_size=step_size,
        loop_prob=loop_prob,
        loop_radius=loop_radius,
        prior_prop=prior_prop,
        seed=seed,
    )
    exact_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )
    base_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )
    mg_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )
    return nodes, edges, exact_graph, base_graph, mg_graph


def run_base_until_converged(graph, x_star, tol, max_iters):
    history = [relative_error(graph, x_star)]
    conv = 0 if history[-1] < tol else None
    for it in range(1, max_iters + 1):
        graph.synchronous_iteration()
        err = relative_error(graph, x_star)
        history.append(err)
        if conv is None and err < tol:
            conv = it
            break
        if not np.isfinite(err) or err > 1e12:
            break
    return conv, history


def run_grouped_vcycles(
    graph,
    groups,
    group_size,
    x_star,
    tol,
    max_cycles,
    r_reduced,
    basis_source,
    eta_assignment_mode,
    pre_smooth,
    post_smooth,
    upward_coarse_sweeps,
    downward_coarse_sweeps,
    num_levels,
    top_level_solver,
    base_scheduler,
    coarse_scheduler,
    fixed_lam,
):
    hierarchy = SVDResidualHierarchy(
        base_graph=graph,
        groups=groups,
        group_size=group_size,
        num_levels=num_levels,
        r_reduced=r_reduced,
        basis_source=basis_source,
        freeze_basis=True,
        eta_assignment_mode=eta_assignment_mode,
        absolute_system=True,
    )
    history = [relative_error(graph, x_star)]
    conv = 0 if history[-1] < tol else None
    last_stats = None
    for cycle in range(1, max_cycles + 1):
        last_stats = hierarchy.v_cycle(
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            upward_coarse_sweeps=upward_coarse_sweeps,
            downward_coarse_sweeps=downward_coarse_sweeps,
            base_scheduler=base_scheduler,
            coarse_scheduler=coarse_scheduler,
            fixed_lam=fixed_lam,
            top_level_solver=top_level_solver,
        )
        err = relative_error(graph, x_star)
        history.append(err)
        if conv is None and err < tol:
            conv = cycle
            break
        if not np.isfinite(err) or err > 1e12:
            break
    return conv, history, last_stats


def print_points(name, conv, history, points):
    print(f"{name} conv {conv}")
    for point in points:
        if point < len(history):
            print(f"{name} {point} {history[point]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--step-size", type=float, default=25.0)
    parser.add_argument("--loop-prob", type=float, default=0.05)
    parser.add_argument("--loop-radius", type=float, default=50.0)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grouping", type=str, default="order", choices=["order", "grid", "kmeans", "loop_aware", "degree_aware"])
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--gx", type=int, default=8)
    parser.add_argument("--gy", type=int, default=4)
    parser.add_argument("--kmeans-k", type=int, default=26)
    parser.add_argument("--target-groups", type=int, default=None)
    parser.add_argument("--loop-window", type=int, default=2)
    parser.add_argument("--loop-boost", type=float, default=3.0)
    parser.add_argument("--degree-boost", type=float, default=1.0)
    parser.add_argument("--loop-sep-min", type=int, default=2)
    parser.add_argument("--r-reduced", type=int, default=2)
    parser.add_argument(
        "--basis-source",
        type=str,
        default="joint_covariance",
        choices=["joint_covariance", "joint_information", "belief_covariance", "belief_information"],
    )
    parser.add_argument(
        "--eta-assignment-mode",
        type=str,
        default="all_in_prior",
        choices=["all_in_prior", "projected_terms"],
    )
    parser.add_argument("--num-levels", type=int, default=2)
    parser.add_argument("--pre-smooth", type=int, default=1)
    parser.add_argument("--post-smooth", type=int, default=0)
    parser.add_argument("--upward-coarse-sweeps", type=int, default=1)
    parser.add_argument("--downward-coarse-sweeps", type=int, default=1)
    parser.add_argument("--base-scheduler", type=str, default="sync", choices=["sync", "residual"])
    parser.add_argument("--coarse-scheduler", type=str, default="sync", choices=["sync", "residual"])
    parser.add_argument("--fixed-lam", action="store_true")
    parser.add_argument("--top-level-solver", type=str, default="iterative", choices=["iterative", "direct"])
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-base-iters", type=int, default=2000)
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--points", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000])
    args = parser.parse_args()

    nodes, _, exact_graph, base_graph, mg_graph = build_graphs(
        n=args.n,
        step_size=args.step_size,
        loop_prob=args.loop_prob,
        loop_radius=args.loop_radius,
        prior_prop=args.prior_prop,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
    )
    x_star, _ = exact_graph.joint_distribution_cov_absolute()

    groups = group_list(
        nodes=nodes,
        graph=mg_graph,
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

    base_conv, base_hist = run_base_until_converged(
        base_graph,
        x_star=x_star,
        tol=args.tol,
        max_iters=args.max_base_iters,
    )
    mg_conv, mg_hist, mg_stats = run_grouped_vcycles(
        mg_graph,
        groups=groups,
        group_size=args.group_size,
        x_star=x_star,
        tol=args.tol,
        max_cycles=args.max_cycles,
        r_reduced=args.r_reduced,
        basis_source=args.basis_source,
        eta_assignment_mode=args.eta_assignment_mode,
        pre_smooth=args.pre_smooth,
        post_smooth=args.post_smooth,
        upward_coarse_sweeps=args.upward_coarse_sweeps,
        downward_coarse_sweeps=args.downward_coarse_sweeps,
        num_levels=args.num_levels,
        top_level_solver=args.top_level_solver,
        base_scheduler=args.base_scheduler,
        coarse_scheduler=args.coarse_scheduler,
        fixed_lam=args.fixed_lam,
    )

    print(
        f"grouping={args.grouping} group_size={args.group_size} gx={args.gx} gy={args.gy} "
        f"kmeans_k={args.kmeans_k} target_groups={args.target_groups} loop_window={args.loop_window} "
        f"loop_boost={args.loop_boost} degree_boost={args.degree_boost} loop_sep_min={args.loop_sep_min}"
    )
    print(
        f"basis_source={args.basis_source} r_reduced={args.r_reduced} num_levels={args.num_levels} "
        f"pre={args.pre_smooth} post={args.post_smooth} up={args.upward_coarse_sweeps} "
        f"down={args.downward_coarse_sweeps} base_scheduler={args.base_scheduler} "
        f"coarse_scheduler={args.coarse_scheduler} fixed_lam={args.fixed_lam} "
        f"top_level_solver={args.top_level_solver}"
    )
    print(f"num_groups={len(groups)}")
    print_points("base_gbp", base_conv, base_hist, args.points)
    print_points("grouped_svd_gbp", mg_conv, mg_hist, args.points)
    if mg_stats is not None:
        print(
            "last_cycle_stats "
            f"before={mg_stats.residual_norm_before} after={mg_stats.residual_norm_after} "
            f"coarse={mg_stats.coarse_residual_norm} corr={mg_stats.correction_norm}"
        )


if __name__ == "__main__":
    main()
