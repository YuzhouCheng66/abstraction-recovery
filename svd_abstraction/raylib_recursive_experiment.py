"""Standalone recursive multigrid experiments for raylib_gbp.

This script keeps raylib's base GBP implementation intact and evaluates an
experimental recursive refreshed-coarse schedule on the same linear toy graph
used elsewhere in `svd_abstraction`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from contextlib import contextmanager

import numpy as np


WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOCAL_RAYLIB_ROOT = SCRIPT_DIR / "raylib_gbp_local"
EXTERNAL_RAYLIB_ROOT = pathlib.Path("/home/yuzhou/Desktop/raylib_gbp")
RAYLIB_ROOT = EXTERNAL_RAYLIB_ROOT if EXTERNAL_RAYLIB_ROOT.exists() else LOCAL_RAYLIB_ROOT

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(RAYLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(RAYLIB_ROOT))

from amg import functions as amg_fnc
from gbp.factors import linear_displacement
from gbp.gbp import Factor
from gbp.gbp import FactorGraph
from gbp.gbp import VariableNode

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(1, str(WORKSPACE_ROOT))

from svd_abstraction.pose_graph import make_grid_pose_graph
from svd_abstraction.pose_graph import make_slam_like_graph


def build_raylib_graph(
    nodes,
    edges,
    prior_sigma=6.0,
    odom_sigma=3.0,
    tiny_prior=1e-12,
    seed=0,
):
    rng = np.random.default_rng(seed)
    graph = FactorGraph(nonlinear_factors=False, eta_damping=0.0)
    eye2 = np.eye(2, dtype=float)

    prior_noises = {}
    odom_noises = {}
    for edge in edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]
        if dst not in {"prior", "anchor"}:
            odom_noises[(int(src), int(dst))] = rng.normal(0.0, odom_sigma, size=2)
        elif dst == "prior":
            prior_noises[int(src)] = rng.normal(0.0, prior_sigma, size=2)

    var_nodes = []
    for i, node in enumerate(nodes):
        var = VariableNode(i, 2)
        var.GT = np.array([node["position"]["x"], node["position"]["y"]], dtype=float)
        var.type = "base"
        var.prior.lam = tiny_prior * eye2
        var.prior.eta = np.zeros(2, dtype=float)
        var_nodes.append(var)

    factors = []
    factor_id = 0
    for edge in edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]

        if dst not in {"prior", "anchor"}:
            i = int(src)
            j = int(dst)
            var_i = var_nodes[i]
            var_j = var_nodes[j]
            measurement = (var_j.GT - var_i.GT) + odom_noises[(i, j)]
            factor = Factor(
                factor_id,
                [var_i, var_j],
                measurement,
                odom_sigma,
                linear_displacement.meas_fn,
                linear_displacement.jac_fn,
                loss=None,
                mahalanobis_threshold=2,
            )
            factor.type = "odometry"
            factor.compute_factor(linpoint=np.r_[var_i.GT, var_j.GT], update_self=True)
            factors.append(factor)
            var_i.adj_factors.append(factor)
            var_j.adj_factors.append(factor)
            factor_id += 1
            continue

        if dst == "prior":
            i = int(src)
            var = var_nodes[i]
            measurement = var.GT + prior_noises[i]
            lam = eye2 / (prior_sigma**2)
            var.prior.lam = var.prior.lam + lam
            var.prior.eta = var.prior.eta + lam @ measurement

    anchor_var = var_nodes[0]
    anchor_lam = eye2 / ((1e-4) ** 2)
    anchor_var.prior.lam = anchor_var.prior.lam + anchor_lam
    anchor_var.prior.eta = anchor_var.prior.eta + anchor_lam @ anchor_var.GT

    graph.var_nodes = var_nodes.copy()
    graph.factors = factors.copy()
    graph.n_var_nodes = len(var_nodes)
    graph.n_factor_nodes = len(factors)
    graph.multigrid_vars[0].extend(var_nodes)
    graph.multigrid_factors[0].extend(factors)

    for var in graph.var_nodes[: graph.n_var_nodes]:
        var.update_belief()

    return graph


def mean_vector(graph):
    base_vars = [var for var in graph.var_nodes[: graph.n_var_nodes] if var.type != "multigrid"]
    return np.concatenate([var.mu for var in base_vars])


def level_vars(graph, level):
    return [var for var in graph.multigrid_vars[level] if var.type != "dead"]


def residual_vector_level(graph, level):
    vars = level_vars(graph, level)
    if not vars:
        return np.zeros(0, dtype=float)
    return np.concatenate([var.compute_residual() for var in vars])


def correction_vector_to_child(graph, level):
    child_level = level - 1
    child_vars = level_vars(graph, child_level)
    child_slices = {}
    offset = 0
    for var in child_vars:
        child_slices[var.variableID] = slice(offset, offset + var.dofs)
        offset += var.dofs

    delta = np.zeros(offset, dtype=float)
    for coarse_var in level_vars(graph, level):
        for coeff, child_var in zip(
            coarse_var.multigrid.interpolation_coefficients,
            coarse_var.multigrid.interpolation_vars,
        ):
            sl = child_slices.get(child_var.variableID)
            if sl is None:
                continue
            delta[sl] += coeff @ coarse_var.mu
    return delta


def relative_error(graph, x_star):
    denom = np.linalg.norm(x_star)
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(mean_vector(graph) - x_star) / denom)


def run_base_until_converged(graph, x_star, tol=1e-6, max_iters=2000):
    err = relative_error(graph, x_star)
    for iteration in range(1, max_iters + 1):
        graph.synchronous_iteration(level=0)
        err = relative_error(graph, x_star)
        if err < tol:
            return iteration, err
    return max_iters, err


@contextmanager
def force_total_levels(max_total_levels=None):
    if max_total_levels is None:
        yield
        return

    original = amg_fnc.coarsen_graph

    def wrapped(graph, vars):
        if len(graph.multigrid_vars) >= max_total_levels:
            return
        return original(graph, vars)

    amg_fnc.coarsen_graph = wrapped
    try:
        yield
    finally:
        amg_fnc.coarsen_graph = original


def build_multigrid_graph(nodes, edges, max_total_levels=None, args=None):
    graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    graph.enable_second_pass_coarse_match = not args.disable_second_pass_coarse_match
    graph.multigrid_split_mode = args.split_mode
    for var in graph.multigrid_vars[0]:
        var.multigrid.theta = args.theta
        var.multigrid.interp_mode = args.interp_mode
    with force_total_levels(max_total_levels=max_total_levels):
        amg_fnc.coarsen_graph(graph, graph.multigrid_vars[0].copy())
    return graph


def recursive_refreshed_level(
    graph,
    level,
    max_level,
    coarse_micro,
    correction_log=None,
    post_iter_after_child=False,
):
    # Match the successful two-level refreshed schedule:
    # refresh from the child, iterate once on this level, recurse upward if needed,
    # then immediately prolongate the correction downward.
    for _ in range(coarse_micro):
        graph.update_all_residual_etas(level=level)
        graph.update_all_beliefs(level=level)
        graph.synchronous_iteration(level=level)

        if level < max_level:
            recursive_refreshed_level(
                graph,
                level + 1,
                max_level,
                coarse_micro,
                correction_log=correction_log,
                post_iter_after_child=post_iter_after_child,
            )
            if post_iter_after_child:
                graph.synchronous_iteration(level=level)

        graph.update_all_residuals(level=level)
        if correction_log is not None:
            child_before = residual_vector_level(graph, level - 1)
            corr = correction_vector_to_child(graph, level)
            graph.prolongate_corrections(level=level)
            child_after = residual_vector_level(graph, level - 1)
            denom = max(np.linalg.norm(child_before), 1e-15)
            correction_log.append(
                {
                    "level": level,
                    "target_level": level - 1,
                    "res_before": float(np.linalg.norm(child_before)),
                    "res_after": float(np.linalg.norm(child_after)),
                    "ratio": float(np.linalg.norm(child_after) / denom),
                    "correction_norm": float(np.linalg.norm(corr)),
                }
            )
        else:
            graph.prolongate_corrections(level=level)


def recursive_refreshed_cycle(graph, coarse_micro, correction_log=None, post_iter_after_child=False):
    graph.synchronous_iteration(level=0)
    max_level = len(graph.multigrid_vars) - 1
    if max_level >= 1:
        recursive_refreshed_level(
            graph,
            level=1,
            max_level=max_level,
            coarse_micro=coarse_micro,
            correction_log=correction_log,
            post_iter_after_child=post_iter_after_child,
        )


def run_recursive_refreshed_until_converged(
    graph,
    x_star,
    coarse_micro,
    post_iter_after_child=False,
    tol=1e-6,
    max_cycles=500,
):
    err = relative_error(graph, x_star)
    for cycle in range(1, max_cycles + 1):
        recursive_refreshed_cycle(
            graph,
            coarse_micro=coarse_micro,
            post_iter_after_child=post_iter_after_child,
        )
        err = relative_error(graph, x_star)
        if err < tol:
            return cycle, err
    return max_cycles, err


def recursive_standard_level(graph, level, max_level, correction_log=None):
    graph.update_all_residual_etas(level=level)
    graph.update_all_beliefs(level=level)
    graph.synchronous_iteration(level=level)

    if level < max_level:
        recursive_standard_level(graph, level + 1, max_level, correction_log=correction_log)

    graph.synchronous_iteration(level=level)

    if correction_log is not None:
        child_before = residual_vector_level(graph, level - 1)
        corr = correction_vector_to_child(graph, level)
        graph.prolongate_corrections(level=level)
        child_after = residual_vector_level(graph, level - 1)
        denom = max(np.linalg.norm(child_before), 1e-15)
        correction_log.append(
            {
                "level": level,
                "target_level": level - 1,
                "res_before": float(np.linalg.norm(child_before)),
                "res_after": float(np.linalg.norm(child_after)),
                "ratio": float(np.linalg.norm(child_after) / denom),
                "correction_norm": float(np.linalg.norm(corr)),
            }
        )
    else:
        graph.prolongate_corrections(level=level)


def recursive_standard_cycle(graph, correction_log=None):
    graph.synchronous_iteration(level=0)
    max_level = len(graph.multigrid_vars) - 1
    if max_level >= 1:
        recursive_standard_level(graph, level=1, max_level=max_level, correction_log=correction_log)


def run_recursive_standard_until_converged(graph, x_star, tol=1e-6, max_cycles=500):
    err = relative_error(graph, x_star)
    for cycle in range(1, max_cycles + 1):
        recursive_standard_cycle(graph)
        err = relative_error(graph, x_star)
        if err < tol:
            return cycle, err
    return max_cycles, err


def summarize_correction_logs(cycle_logs):
    grouped = {}
    for cycle_log in cycle_logs:
        for entry in cycle_log:
            key = (entry["level"], entry["target_level"])
            grouped.setdefault(key, []).append(entry)

    summaries = []
    for (level, target_level), entries in sorted(grouped.items(), reverse=True):
        ratios = np.array([entry["ratio"] for entry in entries], dtype=float)
        correction_norms = np.array([entry["correction_norm"] for entry in entries], dtype=float)
        summaries.append(
            {
                "level": level,
                "target_level": target_level,
                "count": len(entries),
                "avg_ratio": float(np.mean(ratios)),
                "median_ratio": float(np.median(ratios)),
                "frac_hurt": float(np.mean(ratios > 1.0)),
                "avg_correction_norm": float(np.mean(correction_norms)),
                "first_entry": entries[0],
            }
        )
    return summaries


def refreshed_sweeps_per_cycle(num_coarse_levels, coarse_micro):
    sweeps = 0
    for _ in range(num_coarse_levels):
        sweeps = coarse_micro * (1 + sweeps)
    return sweeps


def standard_sweeps_per_cycle(num_coarse_levels):
    return 2 * num_coarse_levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-type",
        type=str,
        default="slam",
        choices=["slam", "grid"],
    )
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--grid-nx", type=int, default=16)
    parser.add_argument("--grid-ny", type=int, default=16)
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--grid-shortcut-prob", type=float, default=0.0)
    parser.add_argument("--grid-shortcut-min-sep", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=15.0)
    parser.add_argument("--loop-prob", type=float, default=0.08)
    parser.add_argument("--loop-radius", type=float, default=40.0)
    parser.add_argument("--prior-prop", type=float, default=0.15)
    parser.add_argument("--prior-sigma", type=float, default=6.0)
    parser.add_argument("--odom-sigma", type=float, default=3.0)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument(
        "--interp-mode",
        type=str,
        default="extended_if_needed",
        choices=["direct", "extended_if_needed", "extended_all"],
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="rs",
        choices=["rs", "pmis", "pmis2"],
    )
    parser.add_argument("--coarse-micro", type=int, default=2)
    parser.add_argument(
        "--disable-second-pass-coarse-match",
        action="store_true",
        help="Skip the extra coarse promotion pass and rely on interpolation to handle uncovered strong-fine links.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="refreshed",
        choices=["refreshed", "refreshed_post", "standard"],
    )
    parser.add_argument("--analysis-cycles", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-base-iters", type=int, default=2000)
    parser.add_argument("--max-v-cycles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.graph_type == "grid":
        nodes, edges = make_grid_pose_graph(
            nx=args.grid_nx,
            ny=args.grid_ny,
            spacing=args.grid_spacing,
            prior_prop=args.prior_prop,
            shortcut_prob=args.grid_shortcut_prob,
            shortcut_min_sep=args.grid_shortcut_min_sep,
            seed=args.seed,
        )
    else:
        nodes, edges = make_slam_like_graph(
            N=args.N,
            step_size=args.step_size,
            loop_prob=args.loop_prob,
            loop_radius=args.loop_radius,
            prior_prop=args.prior_prop,
            seed=args.seed,
        )

    exact_graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    x_star, _ = exact_graph.joint_distribution_cov()

    base_graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    base_iters, base_err = run_base_until_converged(
        base_graph,
        x_star,
        tol=args.tol,
        max_iters=args.max_base_iters,
    )

    results = []
    for label, max_levels in (("natural", None), ("two_level", 2)):
        graph = build_multigrid_graph(nodes, edges, max_total_levels=max_levels, args=args)
        sizes = [len(level_vars) for level_vars in graph.multigrid_vars]
        if args.schedule in {"refreshed", "refreshed_post"}:
            cycles, err = run_recursive_refreshed_until_converged(
                graph,
                x_star,
                coarse_micro=args.coarse_micro,
                post_iter_after_child=(args.schedule == "refreshed_post"),
                tol=args.tol,
                max_cycles=args.max_v_cycles,
            )
        else:
            cycles, err = run_recursive_standard_until_converged(
                graph,
                x_star,
                tol=args.tol,
                max_cycles=args.max_v_cycles,
            )
        dist = np.linalg.norm(mean_vector(graph) - mean_vector(base_graph))
        results.append((label, sizes, cycles, err, dist))

    correction_summaries = None
    if args.schedule in {"standard", "refreshed", "refreshed_post"}:
        graph = build_multigrid_graph(nodes, edges, max_total_levels=None, args=args)
        cycle_logs = []
        for _ in range(args.analysis_cycles):
            correction_log = []
            if args.schedule == "standard":
                recursive_standard_cycle(graph, correction_log=correction_log)
            else:
                recursive_refreshed_cycle(
                    graph,
                    coarse_micro=args.coarse_micro,
                    correction_log=correction_log,
                    post_iter_after_child=(args.schedule == "refreshed_post"),
                )
            cycle_logs.append(correction_log)
        correction_summaries = summarize_correction_logs(cycle_logs)

    if args.graph_type == "grid":
        print(
            f"Toy example: grid pose graph {args.grid_nx}x{args.grid_ny} "
            f"with {len(nodes)} variables and {len(edges)} factors "
            f"(shortcut_prob={args.grid_shortcut_prob:.3f}, min_sep={args.grid_shortcut_min_sep})"
        )
    else:
        print(f"Toy example: linear pose graph with {len(nodes)} variables and {len(edges)} factors")
    if args.schedule == "refreshed":
        print(
            f"Recursive refreshed schedule: base=1, coarse_micro={args.coarse_micro}; "
            f"tol={args.tol:.1e}"
        )
    elif args.schedule == "refreshed_post":
        print(
            "Recursive refreshed schedule with local post-solve after child correction: "
            f"base=1, coarse_micro={args.coarse_micro}; tol={args.tol:.1e}"
        )
    else:
        print(f"Recursive standard V-cycle schedule: base=1, each coarse level pre=1 post=1; tol={args.tol:.1e}")
    print(
        f"Coarsening mode: split_mode={args.split_mode}; "
        f"interp_mode={args.interp_mode}; "
        f"Splitting config: theta={args.theta:.2f}; "
        f"second_pass_coarse_match={'off' if args.disable_second_pass_coarse_match else 'on'}"
    )
    print(
        f"Base GBP converged in {base_iters} sweeps; "
        f"final relative error to exact solve = {base_err:.6e}"
    )
    for label, sizes, cycles, err, dist in results:
        num_coarse_levels = len(sizes) - 1
        if args.schedule == "refreshed":
            coarse_sweeps = cycles * refreshed_sweeps_per_cycle(num_coarse_levels, args.coarse_micro)
        else:
            coarse_sweeps = cycles * standard_sweeps_per_cycle(num_coarse_levels)
        print(
            f"{label}: sizes={sizes}; cycles={cycles}; base_sweeps={cycles}; "
            f"coarse_total_sweeps={coarse_sweeps}; "
            f"relerr={err:.6e}; dist_to_base={dist:.6e}"
        )

    if correction_summaries is not None:
        print(f"Correction analysis over first {args.analysis_cycles} cycles on the natural hierarchy:")
        for summary in correction_summaries:
            first = summary["first_entry"]
            print(
                f"level {summary['level']} -> {summary['target_level']}: "
                f"avg_ratio={summary['avg_ratio']:.6f}; "
                f"median_ratio={summary['median_ratio']:.6f}; "
                f"frac_hurt={summary['frac_hurt']:.3f}; "
                f"avg_corr_norm={summary['avg_correction_norm']:.6e}; "
                f"first={first['res_before']:.6e}->{first['res_after']:.6e}"
            )


if __name__ == "__main__":
    main()
