"""Compare geometric multigrid and raylib GBP on the same grid-Gaussian system."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import sys
from time import perf_counter

import numpy as np
from scipy import sparse

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.poisson_multigrid_benchmark import build_rectangular_levels_from_matrix
from svd_abstraction.poisson_multigrid_benchmark import run_jacobi
from svd_abstraction.poisson_multigrid_benchmark import run_multigrid
from svd_abstraction.pose_graph import make_grid_pose_graph
from svd_abstraction.raylib_recursive_experiment import build_multigrid_graph
from svd_abstraction.raylib_recursive_experiment import build_raylib_graph
from svd_abstraction.raylib_recursive_experiment import relative_error
from svd_abstraction.raylib_recursive_experiment import run_base_until_converged
from svd_abstraction.raylib_recursive_experiment import run_recursive_standard_until_converged


@dataclass
class RaylibArgs:
    prior_sigma: float
    odom_sigma: float
    seed: int
    theta: float
    split_mode: str
    interp_mode: str
    disable_second_pass_coarse_match: bool


def system_from_grid_graph(
    nx: int,
    ny: int,
    spacing: float,
    prior_prop: float,
    prior_sigma: float,
    odom_sigma: float,
    seed: int,
    shortcut_prob: float = 0.0,
    shortcut_min_sep: int = 4,
):
    nodes, edges = make_grid_pose_graph(
        nx=nx,
        ny=ny,
        spacing=spacing,
        prior_prop=prior_prop,
        shortcut_prob=shortcut_prob,
        shortcut_min_sep=shortcut_min_sep,
        seed=seed,
    )
    exact_graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )
    eta, lam = exact_graph.joint_distribution_inf()
    x_star, _ = exact_graph.joint_distribution_cov()
    return nodes, edges, eta, sparse.csr_matrix(lam), x_star


def run_raylib_family(nodes, edges, x_star, args: RaylibArgs, tol: float, max_base_iters: int, max_v_cycles: int):
    base_graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    t0 = perf_counter()
    base_iters, base_err = run_base_until_converged(
        base_graph,
        x_star,
        tol=tol,
        max_iters=max_base_iters,
    )
    base_time = perf_counter() - t0

    results = {
        "raylib_base": {
            "iterations": base_iters,
            "rel_error": base_err,
            "elapsed_time": base_time,
        }
    }

    for label, max_levels in (("raylib_two_level", 2), ("raylib_multilevel", None)):
        mg_graph = build_multigrid_graph(
            nodes,
            edges,
            max_total_levels=max_levels,
            args=args,
        )
        sizes = [len(level_vars) for level_vars in mg_graph.multigrid_vars]
        t0 = perf_counter()
        cycles, err = run_recursive_standard_until_converged(
            mg_graph,
            x_star,
            tol=tol,
            max_cycles=max_v_cycles,
        )
        elapsed = perf_counter() - t0
        results[label] = {
            "iterations": cycles,
            "rel_error": err,
            "elapsed_time": elapsed,
            "sizes": sizes,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=15, help="Use an odd grid so geometric coarsening is exact.")
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--grid-shortcut-prob", type=float, default=0.0)
    parser.add_argument("--grid-shortcut-min-sep", type=int, default=4)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-jacobi", type=int, default=10000)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--max-base-iters", type=int, default=5000)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--split-mode", choices=["rs", "pmis", "pmis2"], default="pmis2")
    parser.add_argument(
        "--interp-mode",
        choices=["direct", "extended_if_needed", "extended_all"],
        default="extended_all",
    )
    parser.add_argument("--disable-second-pass-coarse-match", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    nodes, edges = make_grid_pose_graph(
        nx=args.grid_n,
        ny=args.grid_n,
        spacing=args.grid_spacing,
        prior_prop=args.prior_prop,
        shortcut_prob=args.grid_shortcut_prob,
        shortcut_min_sep=args.grid_shortcut_min_sep,
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
    eta, lam = exact_graph.joint_distribution_inf()
    lam = sparse.csr_matrix(lam)
    x_star, _ = exact_graph.joint_distribution_cov()

    geom_two = build_rectangular_levels_from_matrix(
        args.grid_n,
        args.grid_n,
        lam,
        block_dofs=2,
        max_levels=2,
    )
    geom_multi = build_rectangular_levels_from_matrix(
        args.grid_n,
        args.grid_n,
        lam,
        block_dofs=2,
        max_levels=None,
    )

    jacobi = run_jacobi(
        geom_multi[0],
        eta,
        x_star,
        omega=args.omega,
        tol=args.tol,
        max_iters=args.max_jacobi,
    )
    geom_two_result = run_multigrid(
        geom_two,
        eta,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )
    geom_multi_result = run_multigrid(
        geom_multi,
        eta,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )

    raylib_args = RaylibArgs(
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
        theta=args.theta,
        split_mode=args.split_mode,
        interp_mode=args.interp_mode,
        disable_second_pass_coarse_match=args.disable_second_pass_coarse_match,
    )
    raylib_results = run_raylib_family(
        nodes,
        edges,
        x_star,
        args=raylib_args,
        tol=args.tol,
        max_base_iters=args.max_base_iters,
        max_v_cycles=args.max_cycles,
    )

    print(
        f"Same-system benchmark: {args.grid_n}x{args.grid_n} grid Gaussian with "
        f"{len(nodes)} variables and {len(edges)} factors "
        f"(shortcut_prob={args.grid_shortcut_prob:.3f}, min_sep={args.grid_shortcut_min_sep})"
    )
    print(
        f"Raylib hierarchy: split_mode={args.split_mode}; interp_mode={args.interp_mode}; "
        f"second_pass={'off' if args.disable_second_pass_coarse_match else 'on'}"
    )
    print(f"Geometric MG sizes (two-level): {[(lvl.nx, lvl.ny) for lvl in geom_two]}")
    print(f"Geometric MG sizes (multilevel): {[(lvl.nx, lvl.ny) for lvl in geom_multi]}")

    print(
        f"Jacobi(base): iterations={jacobi['iterations']}; rel_error={jacobi['error_history'][-1] / max(np.linalg.norm(x_star), 1e-15):.6e}; "
        f"time={jacobi['elapsed_time']:.3f}s"
    )
    print(
        f"Geom two-level: cycles={geom_two_result['iterations']}; rel_error={geom_two_result['error_history'][-1] / max(np.linalg.norm(x_star), 1e-15):.6e}; "
        f"time={geom_two_result['elapsed_time']:.3f}s"
    )
    print(
        f"Geom multilevel: cycles={geom_multi_result['iterations']}; rel_error={geom_multi_result['error_history'][-1] / max(np.linalg.norm(x_star), 1e-15):.6e}; "
        f"time={geom_multi_result['elapsed_time']:.3f}s"
    )
    print(
        f"Raylib base GBP: sweeps={raylib_results['raylib_base']['iterations']}; "
        f"rel_error={raylib_results['raylib_base']['rel_error']:.6e}; "
        f"time={raylib_results['raylib_base']['elapsed_time']:.3f}s"
    )
    print(
        f"Raylib two-level: sizes={raylib_results['raylib_two_level']['sizes']}; "
        f"cycles={raylib_results['raylib_two_level']['iterations']}; "
        f"rel_error={raylib_results['raylib_two_level']['rel_error']:.6e}; "
        f"time={raylib_results['raylib_two_level']['elapsed_time']:.3f}s"
    )
    print(
        f"Raylib multilevel: sizes={raylib_results['raylib_multilevel']['sizes']}; "
        f"cycles={raylib_results['raylib_multilevel']['iterations']}; "
        f"rel_error={raylib_results['raylib_multilevel']['rel_error']:.6e}; "
        f"time={raylib_results['raylib_multilevel']['elapsed_time']:.3f}s"
    )


if __name__ == "__main__":
    main()
