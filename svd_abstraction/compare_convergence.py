"""Compare pure base GBP convergence against SVD residual V-cycles."""

import argparse
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

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


def run_base_until_converged(graph, x_star, tol, max_iters):
    history = []
    for it in range(1, max_iters + 1):
        graph.synchronous_iteration()
        err = relative_error(graph, x_star)
        history.append(err)
        if err < tol:
            return it, err, history
    return max_iters, history[-1], history


def run_vcycle_until_converged(
    graph,
    nodes,
    x_star,
    tol,
    max_cycles,
    group_size,
    r_reduced,
    basis_source,
    warmup,
    num_levels,
):
    hierarchy = SVDResidualHierarchy(
        base_graph=graph,
        groups=groups_from_order(nodes, group_size=group_size, tail_heavy=True),
        group_size=group_size,
        num_levels=num_levels,
        r_reduced=r_reduced,
        basis_source=basis_source,
        freeze_basis=True,
    )

    # Align the actual V-cycle schedule with the raylib AMG comparison:
    # one base sweep per cycle, then one upward coarse sweep and one downward coarse sweep.
    pre_smooth = 1
    post_smooth = 0
    upward_coarse_sweeps = 1
    downward_coarse_sweeps = 1

    base_sweeps = 0
    total_sweeps = 0
    if warmup > 0:
        hierarchy.warmup(iterations=warmup, scheduler="sync")
        base_sweeps += warmup
        total_sweeps += warmup
    hierarchy.build_hierarchy()
    actual_levels = hierarchy.total_levels()

    history = []
    last_stats = None
    for cycle in range(1, max_cycles + 1):
        last_stats = hierarchy.v_cycle(
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            upward_coarse_sweeps=upward_coarse_sweeps,
            downward_coarse_sweeps=downward_coarse_sweeps,
            scheduler="sync",
        )
        base_sweeps += pre_smooth + post_smooth
        total_sweeps += pre_smooth + post_smooth + (actual_levels - 1) * (
            upward_coarse_sweeps + downward_coarse_sweeps
        )
        err = relative_error(graph, x_star)
        history.append(err)
        if err < tol:
            return cycle, actual_levels, base_sweeps, total_sweeps, err, history, last_stats
    return max_cycles, actual_levels, base_sweeps, total_sweeps, history[-1], history, last_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=15.0)
    parser.add_argument("--loop-prob", type=float, default=0.08)
    parser.add_argument("--loop-radius", type=float, default=40.0)
    parser.add_argument("--prior-prop", type=float, default=0.15)
    parser.add_argument("--prior-sigma", type=float, default=6.0)
    parser.add_argument("--odom-sigma", type=float, default=3.0)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--r-reduced", type=int, default=2)
    parser.add_argument("--basis-source", type=str, default="belief_covariance")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-levels", type=int, default=2)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-base-iters", type=int, default=2000)
    parser.add_argument("--max-v-cycles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    nodes, edges = make_slam_like_graph(
        N=args.N,
        step_size=args.step_size,
        loop_prob=args.loop_prob,
        loop_radius=args.loop_radius,
        prior_prop=args.prior_prop,
        seed=args.seed,
    )

    exact_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    x_star, _ = exact_graph.joint_distribution_cov()

    base_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    vcycle_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )

    base_iters, base_err, _ = run_base_until_converged(
        base_graph,
        x_star,
        tol=args.tol,
        max_iters=args.max_base_iters,
    )
    v_cycles, actual_levels, v_base_sweeps, v_total_sweeps, v_err, _, v_stats = run_vcycle_until_converged(
        vcycle_graph,
        nodes,
        x_star,
        tol=args.tol,
        max_cycles=args.max_v_cycles,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        basis_source=args.basis_source,
        warmup=args.warmup,
        num_levels=args.num_levels,
    )

    base_final = mean_vector(base_graph)
    vcycle_final = mean_vector(vcycle_graph)

    print(f"Toy example: linear pose graph with {len(nodes)} variables and {len(edges)} factors")
    print(
        f"Basis source: {args.basis_source}; group_size={args.group_size}; "
        f"r={args.r_reduced}; warmup={args.warmup}; requested_levels={args.num_levels}; actual_levels={actual_levels}"
    )
    print(f"Base GBP converged in {base_iters} synchronous sweeps; final relative error to exact solve = {base_err:.6e}")
    print(
        "V-cycle converged in "
        f"{v_cycles} cycles / {v_base_sweeps} base smoothing sweeps / "
        f"{v_total_sweeps} total sweeps; final relative error to exact solve = {v_err:.6e}"
    )
    print(f"Distance between converged base and V-cycle means = {np.linalg.norm(base_final - vcycle_final):.6e}")
    if v_stats is not None:
        print(f"Last V-cycle residual before/after = {v_stats.residual_norm_before:.6e} -> {v_stats.residual_norm_after:.6e}")


if __name__ == "__main__":
    main()
