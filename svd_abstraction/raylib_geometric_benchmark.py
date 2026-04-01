"""Benchmark raylib vanilla/AMG/geometric hierarchies under the same V-cycle semantics.

This script does not modify raylib. It compares:
* base raylib GBP
* raylib AMG hierarchy
* raylib with an independently constructed geometric hierarchy

The goal is to reuse raylib's own persistent coarse graphs/messages and
`vcycle_step()` rather than external recursive schedules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import sys

import numpy as np
import scipy.linalg

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.pose_graph import make_grid_pose_graph
from svd_abstraction.pose_graph import make_slam_like_graph
from svd_abstraction.raylib_geometric_builder import build_geometric_multigrid_graph
from svd_abstraction.raylib_recursive_experiment import build_multigrid_graph
from svd_abstraction.raylib_recursive_experiment import build_raylib_graph


@dataclass
class AMGArgs:
    prior_sigma: float
    odom_sigma: float
    seed: int
    theta: float
    split_mode: str
    interp_mode: str
    disable_second_pass_coarse_match: bool


def base_mean_vector(graph) -> np.ndarray:
    base_vars = [var for var in graph.var_nodes[: graph.n_var_nodes] if var.type != "multigrid"]
    return np.concatenate([var.mu for var in base_vars])


def exact_map_from_graph(graph) -> np.ndarray:
    eta, lam = graph.joint_distribution_inf()
    stabilized = 0.5 * (lam + lam.T)
    chol, lower = scipy.linalg.cho_factor(stabilized, lower=False, check_finite=False)
    return scipy.linalg.cho_solve((chol, lower), eta, check_finite=False)


def relative_error(graph, x_star: np.ndarray) -> float:
    denom = np.linalg.norm(x_star)
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(base_mean_vector(graph) - x_star) / denom)


def live_level_sizes(graph) -> list[int]:
    return [len([var for var in level if var.type != "dead"]) for level in graph.multigrid_vars]


def run_base_until_converged(graph, x_star: np.ndarray, tol: float, max_iters: int):
    err = relative_error(graph, x_star)
    for it in range(1, max_iters + 1):
        graph.synchronous_iteration(level=0)
        err = relative_error(graph, x_star)
        if err < tol:
            return it, err
    return max_iters, err


def run_vcycle_until_converged(
    graph,
    x_star: np.ndarray,
    tol: float,
    max_cycles: int,
    top_level_solver: str = "iterative",
):
    err = relative_error(graph, x_star)
    for cycle in range(1, max_cycles + 1):
        graph.vcycle_step(top_level_solver=top_level_solver)
        err = relative_error(graph, x_star)
        if err < tol:
            return cycle, err
    return max_cycles, err


def make_problem(args):
    if args.graph_type == "chain":
        nodes, edges = make_slam_like_graph(
            N=args.n,
            step_size=args.step_size,
            loop_prob=0.0,
            loop_radius=1.0,
            prior_prop=0.0,
            seed=args.seed,
        )
        kind = "chain"
        nx = args.n
        ny = 1
    elif args.graph_type == "slam":
        nodes, edges = make_slam_like_graph(
            N=args.n,
            step_size=args.step_size,
            loop_prob=args.loop_prob,
            loop_radius=args.loop_radius,
            prior_prop=args.prior_prop,
            seed=args.seed,
        )
        kind = "chain"
        nx = args.n
        ny = 1
    else:
        nodes, edges = make_grid_pose_graph(
            nx=args.grid_nx,
            ny=args.grid_ny,
            spacing=args.grid_spacing,
            prior_prop=args.prior_prop,
            shortcut_prob=args.grid_shortcut_prob,
            shortcut_min_sep=args.grid_shortcut_min_sep,
            seed=args.seed,
        )
        kind = "grid"
        nx = args.grid_nx
        ny = args.grid_ny
    return nodes, edges, kind, nx, ny


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-type", choices=["chain", "slam", "grid"], default="chain")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--grid-nx", type=int, default=32)
    parser.add_argument("--grid-ny", type=int, default=32)
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--grid-shortcut-prob", type=float, default=0.0)
    parser.add_argument("--grid-shortcut-min-sep", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=25.0)
    parser.add_argument("--loop-prob", type=float, default=0.05)
    parser.add_argument("--loop-radius", type=float, default=50.0)
    parser.add_argument("--prior-prop", type=float, default=0.02)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--split-mode", type=str, default="pmis2")
    parser.add_argument(
        "--interp-mode",
        type=str,
        default="extended_if_needed",
        choices=["direct", "extended_if_needed", "extended_all"],
    )
    parser.add_argument("--disable-second-pass-coarse-match", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-base-iters", type=int, default=4000)
    parser.add_argument("--max-cycles", type=int, default=1200)
    parser.add_argument("--top-level-solver", choices=["iterative", "direct"], default="iterative")
    args = parser.parse_args()

    nodes, edges, kind, nx, ny = make_problem(args)

    base_graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    x_star = exact_map_from_graph(base_graph)

    print(f"Problem: {args.graph_type}, nodes={len(nodes)}, edges={len(edges)}")
    print(f"Exact MAP norm: {np.linalg.norm(x_star):.6e}")

    base_iters, base_err = run_base_until_converged(base_graph, x_star, tol=args.tol, max_iters=args.max_base_iters)
    print(f"vanilla: iterations={base_iters}, rel_error={base_err:.6e}")

    amg_args = AMGArgs(
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
        theta=args.theta,
        split_mode=args.split_mode,
        interp_mode=args.interp_mode,
        disable_second_pass_coarse_match=args.disable_second_pass_coarse_match,
    )

    for label, max_levels in (("raylib_amg_two_level", 2), ("raylib_amg_multilevel", None)):
        try:
            graph = build_multigrid_graph(nodes, edges, max_total_levels=max_levels, args=amg_args)
            cycles, err = run_vcycle_until_converged(
                graph,
                x_star,
                tol=args.tol,
                max_cycles=args.max_cycles,
                top_level_solver=args.top_level_solver,
            )
            print(f"{label}: levels={live_level_sizes(graph)}, cycles={cycles}, rel_error={err:.6e}")
        except Exception as exc:
            print(f"{label}: FAILED: {type(exc).__name__}: {exc}")

    for label, max_levels in (("raylib_geom_two_level", 2), ("raylib_geom_multilevel", None)):
        try:
            graph = build_geometric_multigrid_graph(
                nodes,
                edges,
                kind=kind,
                nx=nx,
                ny=ny,
                max_total_levels=max_levels,
                prior_sigma=args.prior_sigma,
                odom_sigma=args.odom_sigma,
                seed=args.seed,
            )
            cycles, err = run_vcycle_until_converged(
                graph,
                x_star,
                tol=args.tol,
                max_cycles=args.max_cycles,
                top_level_solver=args.top_level_solver,
            )
            print(f"{label}: levels={live_level_sizes(graph)}, cycles={cycles}, rel_error={err:.6e}")
        except Exception as exc:
            print(f"{label}: FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
