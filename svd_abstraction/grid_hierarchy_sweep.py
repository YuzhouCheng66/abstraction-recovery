"""Structured sweeps for grid-Gaussian hierarchy experiments.

This script keeps three research lines in one place:
1. same-system comparisons between geometric multigrid and raylib GBP
2. split/interpolation/levels sweeps on grid hierarchies
3. shortcut-degradation sweeps from regular grids toward nonlocal graphs
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.grid_same_system_benchmark import system_from_grid_graph
from svd_abstraction.poisson_multigrid_benchmark import build_rectangular_levels_from_matrix
from svd_abstraction.poisson_multigrid_benchmark import run_multigrid
from svd_abstraction.pose_graph import make_grid_pose_graph
from svd_abstraction.raylib_recursive_experiment import build_multigrid_graph
from svd_abstraction.raylib_recursive_experiment import build_raylib_graph
from svd_abstraction.raylib_recursive_experiment import recursive_standard_cycle
from svd_abstraction.raylib_recursive_experiment import relative_error
from svd_abstraction.raylib_recursive_experiment import run_recursive_standard_until_converged
from svd_abstraction.raylib_recursive_experiment import summarize_correction_logs


@dataclass
class SweepConfig:
    prior_sigma: float = 1.0
    odom_sigma: float = 1.0
    theta: float = 0.25
    seed: int = 0
    split_mode: str = "pmis2"
    interp_mode: str = "extended_if_needed"
    disable_second_pass_coarse_match: bool = True


def raylib_args(cfg: SweepConfig) -> SimpleNamespace:
    return SimpleNamespace(
        prior_sigma=cfg.prior_sigma,
        odom_sigma=cfg.odom_sigma,
        theta=cfg.theta,
        seed=cfg.seed,
        split_mode=cfg.split_mode,
        interp_mode=cfg.interp_mode,
        disable_second_pass_coarse_match=cfg.disable_second_pass_coarse_match,
    )


def correction_summary(graph, analysis_cycles: int = 20):
    cycle_logs = []
    for _ in range(analysis_cycles):
        correction_log = []
        recursive_standard_cycle(graph, correction_log=correction_log)
        cycle_logs.append(correction_log)
    return summarize_correction_logs(cycle_logs)


def eval_raylib_grid(
    nx: int,
    ny: int,
    spacing: float,
    prior_prop: float,
    shortcut_prob: float,
    shortcut_min_sep: int,
    cfg: SweepConfig,
    max_levels: int | None,
    max_cycles: int,
    tol: float,
):
    nodes, edges = make_grid_pose_graph(
        nx=nx,
        ny=ny,
        spacing=spacing,
        prior_prop=prior_prop,
        shortcut_prob=shortcut_prob,
        shortcut_min_sep=shortcut_min_sep,
        seed=cfg.seed,
    )
    exact_graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=cfg.prior_sigma,
        odom_sigma=cfg.odom_sigma,
        tiny_prior=1e-12,
        seed=cfg.seed,
    )
    x_star, _ = exact_graph.joint_distribution_cov()
    graph = build_multigrid_graph(nodes, edges, max_total_levels=max_levels, args=raylib_args(cfg))
    sizes = [len(level_vars) for level_vars in graph.multigrid_vars]
    cycles, err = run_recursive_standard_until_converged(
        graph,
        x_star,
        tol=tol,
        max_cycles=max_cycles,
    )
    return {
        "sizes": sizes,
        "cycles": cycles,
        "rel_error": err,
        "num_nodes": len(nodes),
        "num_factors": len(edges),
    }


def run_same_system_mode(args):
    print("== Same-System Benchmark ==")
    print("grid  split  interp              geom_two    geom_multi  ray_two     ray_multi")
    for grid_n in args.grid_sizes:
        for split in args.split_modes:
            for interp in args.interp_modes:
                cfg = SweepConfig(
                    prior_sigma=args.prior_sigma,
                    odom_sigma=args.odom_sigma,
                    theta=args.theta,
                    seed=args.seed,
                    split_mode=split,
                    interp_mode=interp,
                    disable_second_pass_coarse_match=not args.enable_second_pass,
                )
                nodes, edges, eta, lam, x_star = system_from_grid_graph(
                    nx=grid_n,
                    ny=grid_n,
                    spacing=args.grid_spacing,
                    prior_prop=args.prior_prop,
                    prior_sigma=args.prior_sigma,
                    odom_sigma=args.odom_sigma,
                    seed=args.seed,
                    shortcut_prob=args.shortcut_prob,
                    shortcut_min_sep=args.shortcut_min_sep,
                )
                geom_two = build_rectangular_levels_from_matrix(
                    grid_n,
                    grid_n,
                    lam,
                    block_dofs=2,
                    max_levels=2,
                )
                geom_multi = build_rectangular_levels_from_matrix(
                    grid_n,
                    grid_n,
                    lam,
                    block_dofs=2,
                    max_levels=None,
                )
                two = run_multigrid(
                    geom_two,
                    eta,
                    x_star,
                    omega=args.omega,
                    pre_sweeps=args.pre,
                    post_sweeps=args.post,
                    tol=args.tol,
                    max_cycles=args.max_cycles,
                )
                multi = run_multigrid(
                    geom_multi,
                    eta,
                    x_star,
                    omega=args.omega,
                    pre_sweeps=args.pre,
                    post_sweeps=args.post,
                    tol=args.tol,
                    max_cycles=args.max_cycles,
                )
                ray_two = eval_raylib_grid(
                    grid_n,
                    grid_n,
                    args.grid_spacing,
                    args.prior_prop,
                    args.shortcut_prob,
                    args.shortcut_min_sep,
                    cfg,
                    max_levels=2,
                    max_cycles=args.max_cycles,
                    tol=args.tol,
                )
                ray_multi = eval_raylib_grid(
                    grid_n,
                    grid_n,
                    args.grid_spacing,
                    args.prior_prop,
                    args.shortcut_prob,
                    args.shortcut_min_sep,
                    cfg,
                    max_levels=None,
                    max_cycles=args.max_cycles,
                    tol=args.tol,
                )
                geom_two_err = two["error_history"][-1] / max(np.linalg.norm(x_star), 1e-15)
                geom_multi_err = multi["error_history"][-1] / max(np.linalg.norm(x_star), 1e-15)
                print(
                    f"{grid_n:4d}  {split:5s}  {interp:18s}  "
                    f"{geom_two_err:9.2e}  {geom_multi_err:10.2e}  "
                    f"{ray_two['rel_error']:9.2e}  {ray_multi['rel_error']:10.2e}"
                )


def run_hierarchy_mode(args):
    print("== Grid Hierarchy Sweep ==")
    print("grid  split  interp              natural(err,size)               two_level(err,size)            winner")
    for grid_n in args.grid_sizes:
        for split in args.split_modes:
            for interp in args.interp_modes:
                cfg = SweepConfig(
                    prior_sigma=args.prior_sigma,
                    odom_sigma=args.odom_sigma,
                    theta=args.theta,
                    seed=args.seed,
                    split_mode=split,
                    interp_mode=interp,
                    disable_second_pass_coarse_match=not args.enable_second_pass,
                )
                natural = eval_raylib_grid(
                    grid_n,
                    grid_n,
                    args.grid_spacing,
                    args.prior_prop,
                    args.shortcut_prob,
                    args.shortcut_min_sep,
                    cfg,
                    max_levels=None,
                    max_cycles=args.max_cycles,
                    tol=args.tol,
                )
                two = eval_raylib_grid(
                    grid_n,
                    grid_n,
                    args.grid_spacing,
                    args.prior_prop,
                    args.shortcut_prob,
                    args.shortcut_min_sep,
                    cfg,
                    max_levels=2,
                    max_cycles=args.max_cycles,
                    tol=args.tol,
                )
                winner = "natural" if natural["rel_error"] < two["rel_error"] else "two_level"
                print(
                    f"{grid_n:4d}  {split:5s}  {interp:18s}  "
                    f"{natural['rel_error']:9.2e} {natural['sizes']!s:24s}  "
                    f"{two['rel_error']:9.2e} {two['sizes']!s:24s}  {winner}"
                )


def run_shortcut_mode(args):
    print("== Shortcut Degradation Sweep ==")
    print("p_shortcut  natural_err  two_err     nat_sizes              two_sizes              level1_hurt")
    for shortcut_prob in args.shortcut_probs:
        cfg = SweepConfig(
            prior_sigma=args.prior_sigma,
            odom_sigma=args.odom_sigma,
            theta=args.theta,
            seed=args.seed,
            split_mode=args.split_modes[0],
            interp_mode=args.interp_modes[0],
            disable_second_pass_coarse_match=not args.enable_second_pass,
        )
        natural = eval_raylib_grid(
            args.grid_sizes[0],
            args.grid_sizes[0],
            args.grid_spacing,
            args.prior_prop,
            shortcut_prob,
            args.shortcut_min_sep,
            cfg,
            max_levels=None,
            max_cycles=args.max_cycles,
            tol=args.tol,
        )
        two = eval_raylib_grid(
            args.grid_sizes[0],
            args.grid_sizes[0],
            args.grid_spacing,
            args.prior_prop,
            shortcut_prob,
            args.shortcut_min_sep,
            cfg,
            max_levels=2,
            max_cycles=args.max_cycles,
            tol=args.tol,
        )

        nodes, edges = make_grid_pose_graph(
            nx=args.grid_sizes[0],
            ny=args.grid_sizes[0],
            spacing=args.grid_spacing,
            prior_prop=args.prior_prop,
            shortcut_prob=shortcut_prob,
            shortcut_min_sep=args.shortcut_min_sep,
            seed=args.seed,
        )
        graph = build_multigrid_graph(nodes, edges, max_total_levels=None, args=raylib_args(cfg))
        summaries = correction_summary(graph, analysis_cycles=args.analysis_cycles)
        level1_hurt = np.nan
        for entry in summaries:
            if entry["level"] == 1 and entry["target_level"] == 0:
                level1_hurt = entry["frac_hurt"]
                break
        print(
            f"{shortcut_prob:10.3f}  {natural['rel_error']:10.2e}  {two['rel_error']:10.2e}  "
            f"{str(natural['sizes']):22s}  {str(two['sizes']):22s}  {level1_hurt:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["same_system", "hierarchy", "shortcut", "all"], default="all")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[15, 31])
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--split-modes", nargs="+", default=["pmis2"])
    parser.add_argument("--interp-modes", nargs="+", default=["extended_if_needed"])
    parser.add_argument("--enable-second-pass", action="store_true")
    parser.add_argument("--shortcut-prob", type=float, default=0.0)
    parser.add_argument("--shortcut-probs", type=float, nargs="+", default=[0.0, 0.01, 0.02])
    parser.add_argument("--shortcut-min-sep", type=int, default=4)
    parser.add_argument("--omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--analysis-cycles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode in {"same_system", "all"}:
        run_same_system_mode(args)
        print()
    if args.mode in {"hierarchy", "all"}:
        run_hierarchy_mode(args)
        print()
    if args.mode in {"shortcut", "all"}:
        run_shortcut_mode(args)


if __name__ == "__main__":
    main()
