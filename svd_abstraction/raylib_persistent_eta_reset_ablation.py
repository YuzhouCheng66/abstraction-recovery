"""Ablations for eta-side reset on top of raylib's persistent multigrid hierarchy.

This keeps raylib itself untouched. We build the usual persistent hierarchy
through the existing helper code, then replace the per-level iterative updates
inside a V-cycle with variants that preserve lam-side information while
resetting eta/mu-side state in a self-consistent way.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.pose_graph import make_grid_pose_graph
from svd_abstraction.pose_graph import make_slam_like_graph
from svd_abstraction.raylib_recursive_experiment import build_multigrid_graph
from svd_abstraction.raylib_recursive_experiment import build_raylib_graph
from svd_abstraction.raylib_recursive_experiment import relative_error


def level_live_vars(graph, level: int):
    return [var for var in graph.multigrid_vars[level] if getattr(var, "type", "") != "dead"]


def reset_level_eta_keep_lam(graph, level: int, mode: str = "prior_eta") -> None:
    """Preserve lam-side state on a raylib level, reset eta/mu-side coherently."""
    for var in level_live_vars(graph, level):
        if not getattr(var, "active", True):
            continue

        if mode == "zero_all_eta":
            new_eta = np.zeros_like(var.belief.eta)
        elif mode == "prior_eta":
            new_eta = np.array(var.prior.eta, copy=True)
        else:
            raise ValueError(f"Unknown eta reset mode: {mode}")

        var.belief.eta = new_eta
        var.Sigma = 1.0 / np.diagonal(var.belief.lam)
        var.mu = var.Sigma * var.belief.eta

        for factor in var.adj_factors:
            belief_ix = factor.adj_var_nodes.index(var)
            factor.adj_beliefs[belief_ix].eta = np.array(var.belief.eta, copy=True)
            factor.adj_beliefs[belief_ix].lam = np.array(var.belief.lam, copy=True)
            factor.messages[belief_ix].eta[:] = 0.0


def vcycle_step_eta_reset(
    graph,
    top_level_solver: str = "iterative",
    top_level_ridge: float = 1e-10,
    reset_stage: str = "none",
    eta_mode: str = "prior_eta",
) -> None:
    """Raylib V-cycle variant with eta-side reset on persistent coarse levels."""
    if top_level_solver not in {"iterative", "direct"}:
        raise ValueError(f"Unknown top_level_solver: {top_level_solver}")
    if reset_stage not in {"none", "up", "down", "both"}:
        raise ValueError(f"Unknown reset_stage: {reset_stage}")

    top_level = len(graph.multigrid_vars) - 1

    graph.synchronous_iteration(level=0)

    for level in range(1, len(graph.multigrid_vars)):
        graph.update_all_residual_etas(level=level)
        if reset_stage in {"up", "both"}:
            reset_level_eta_keep_lam(graph, level, mode=eta_mode)
        else:
            graph.update_all_beliefs(level=level)

        if level == top_level and top_level_solver == "direct":
            graph.direct_solve_level(level, ridge=top_level_ridge)
        else:
            graph.synchronous_iteration(level=level)
        graph.update_all_residuals(level=level)

    for level in range(len(graph.multigrid_vars) - 1, 0, -1):
        if reset_stage in {"down", "both"}:
            reset_level_eta_keep_lam(graph, level, mode=eta_mode)
        if not (level == top_level and top_level_solver == "direct"):
            graph.synchronous_iteration(level=level)
        graph.prolongate_corrections(level=level)


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
    elif args.graph_type == "slam":
        nodes, edges = make_slam_like_graph(
            N=args.n,
            step_size=args.step_size,
            loop_prob=args.loop_prob,
            loop_radius=args.loop_radius,
            prior_prop=args.prior_prop,
            seed=args.seed,
        )
    elif args.graph_type == "grid":
        nodes, edges = make_grid_pose_graph(
            nx=args.grid_nx,
            ny=args.grid_ny,
            spacing=args.grid_spacing,
            shortcut_prob=args.grid_shortcut_prob,
            shortcut_min_sep=args.grid_shortcut_min_sep,
            seed=args.seed,
        )
    else:
        raise ValueError(args.graph_type)
    return nodes, edges


def run_ablation(args):
    nodes, edges = make_problem(args)

    base_graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    x_star, _ = base_graph.joint_distribution_cov()

    mg_args = SimpleNamespace(
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
        theta=args.theta,
        split_mode=args.split_mode,
        interp_mode=args.interp_mode,
        disable_second_pass_coarse_match=args.disable_second_pass_coarse_match,
    )
    max_total_levels = None if args.max_total_levels is not None and args.max_total_levels <= 0 else args.max_total_levels
    graph = build_multigrid_graph(nodes, edges, max_total_levels=max_total_levels, args=mg_args)

    errs = [relative_error(graph, x_star)]
    for _ in range(args.cycles):
        if args.reset_stage == "none":
            graph.vcycle_step(top_level_solver=args.top_level_solver)
        else:
            vcycle_step_eta_reset(
                graph,
                top_level_solver=args.top_level_solver,
                reset_stage=args.reset_stage,
                eta_mode=args.eta_mode,
            )
        errs.append(relative_error(graph, x_star))

    levels = [len(level_live_vars(graph, level)) for level in range(len(graph.multigrid_vars))]
    return {
        "levels": levels,
        "errors": errs,
        "final_error": errs[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-type", choices=["chain", "slam", "grid"], default="chain")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--step-size", type=float, default=25.0)
    parser.add_argument("--loop-prob", type=float, default=0.05)
    parser.add_argument("--loop-radius", type=float, default=50.0)
    parser.add_argument("--prior-prop", type=float, default=0.02)
    parser.add_argument("--grid-nx", type=int, default=16)
    parser.add_argument("--grid-ny", type=int, default=16)
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--grid-shortcut-prob", type=float, default=0.0)
    parser.add_argument("--grid-shortcut-min-sep", type=int, default=3)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--split-mode", type=str, default="pmis2")
    parser.add_argument("--interp-mode", type=str, default="extended_if_needed")
    parser.add_argument("--disable-second-pass-coarse-match", action="store_true")
    parser.add_argument("--max-total-levels", type=int, default=2, help="Use 0 for unrestricted multilevel")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--top-level-solver", choices=["iterative", "direct"], default="iterative")
    parser.add_argument("--reset-stage", choices=["none", "up", "down", "both"], default="none")
    parser.add_argument("--eta-mode", choices=["prior_eta", "zero_all_eta"], default="prior_eta")
    args = parser.parse_args()

    result = run_ablation(args)
    print("levels", result["levels"])
    print("errors", [float(v) for v in result["errors"]])
    print("final_error", float(result["final_error"]))


if __name__ == "__main__":
    main()
