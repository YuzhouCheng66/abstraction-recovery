"""Reference-aligned GBP benchmark on a long linear chain pose graph.

This benchmark uses the same linear chain system in three ways:
  * scalar channel (exact for the current linear xy chain)
  * full 2D block system
  * optional raylib hierarchy baseline

For the GBP part, the base solver uses direct synchronous block GBP whose
message updates are aligned against `svd_abstraction/gbp/gbp.py`. The geometric
multigrid runs use the same operator and the same geometric 1D hierarchy, but
swap the smoother to "fresh direct GBP on the current defect equation".
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.block_gabp import build_block_gabp_level
from svd_abstraction.block_gabp import block_gabp_full_sweep
from svd_abstraction.block_gabp import block_gabp_mean
from svd_abstraction.block_gabp import initialize_block_gabp_state
from svd_abstraction.block_gabp import run_block_gabp_direct
from svd_abstraction.block_gabp import run_multigrid_with_block_gabp_direct_fresh
from svd_abstraction.chain_same_system_benchmark import RaylibArgs
from svd_abstraction.chain_same_system_benchmark import build_line_levels_from_matrix
from svd_abstraction.chain_same_system_benchmark import extract_scalar_channel
from svd_abstraction.chain_same_system_benchmark import run_raylib_family
from svd_abstraction.chain_same_system_benchmark import system_from_chain
from svd_abstraction.gbp_from_operator import build_reference_gbp_graph_from_operator
from svd_abstraction.gbp_from_operator import graph_mean_vector


def relative_error(result: dict[str, object]) -> float:
    errors = result["error_history"]
    return float(errors[-1] / max(errors[0], 1e-15))


def alignment_check(n: int = 8) -> dict[str, float]:
    """Small sweep-level check against the reference GBP core."""
    _, _, eta, lam, _ = system_from_chain(n, 25.0, 1.0, 1.0, 0)
    level = build_block_gabp_level(lam, block_dofs=2)
    state = initialize_block_gabp_state(level)
    graph = build_reference_gbp_graph_from_operator(eta, lam, block_dofs=2)

    diffs = []
    for _ in range(10):
        block_gabp_full_sweep(level, state, eta)
        x_helper = block_gabp_mean(level, state.p_msg, state.h_msg, eta)
        graph.synchronous_iteration()
        x_ref = graph_mean_vector(graph)
        diffs.append(float(np.linalg.norm(x_helper - x_ref)))

    return {
        "max_diff": max(diffs),
        "tail_diff": diffs[-1],
    }


def summarize(name: str, result: dict[str, object]) -> str:
    rel = relative_error(result)
    return (
        f"{name}: iterations={result['iterations']}, "
        f"rel_error={rel:.3e}, "
        f"final_residual={result['residual_history'][-1]:.3e}, "
        f"time={result['elapsed_time']:.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--step-size", type=float, default=25.0)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-base-iters", type=int, default=4000)
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["scalar", "block", "all"], default="all")
    parser.add_argument("--include-raylib", action="store_true")
    parser.add_argument("--skip-alignment-check", action="store_true")
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--split-mode", choices=["rs", "pmis", "pmis2"], default="pmis2")
    parser.add_argument(
        "--interp-mode",
        choices=["direct", "extended_if_needed", "extended_all"],
        default="extended_if_needed",
    )
    parser.add_argument("--disable-second-pass-coarse-match", action="store_true", default=True)
    args = parser.parse_args()

    if not args.skip_alignment_check:
        check = alignment_check()
        print(
            "Alignment check (2D block, n=8): "
            f"max_diff={check['max_diff']:.3e}, tail_diff={check['tail_diff']:.3e}"
        )
        print("")

    nodes, edges, eta, lam, x_star_full = system_from_chain(
        n=args.n,
        step_size=args.step_size,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
    )

    print(f"Reference-aligned chain benchmark: n={args.n}, edges={len(edges)}")
    print("")

    if args.mode in ("scalar", "all"):
        b_x, a_x = extract_scalar_channel(eta, lam, 0)
        x_star_x = x_star_full[0::2]
        scalar_two_levels = build_line_levels_from_matrix(args.n, a_x, block_dofs=1, max_levels=2)
        scalar_multi_levels = build_line_levels_from_matrix(args.n, a_x, block_dofs=1, max_levels=None)
        print(f"Scalar geometric two-level sizes: {[lvl.n for lvl in scalar_two_levels]}")
        print(f"Scalar geometric multilevel sizes: {[lvl.n for lvl in scalar_multi_levels]}")
        scalar_base = run_block_gabp_direct(
            build_block_gabp_level(a_x, block_dofs=1),
            b_x,
            x_star_x,
            tol=args.tol,
            max_iters=args.max_base_iters,
        )
        print(summarize("Scalar GBP(base)", scalar_base))
        scalar_two = run_multigrid_with_block_gabp_direct_fresh(
            scalar_two_levels,
            block_dofs=1,
            b=b_x,
            x_star=x_star_x,
            pre_sweeps=args.pre,
            post_sweeps=args.post,
            tol=args.tol,
            max_cycles=args.max_cycles,
        )
        print(summarize("Scalar GBP(two-level MG)", scalar_two))
        scalar_multi = run_multigrid_with_block_gabp_direct_fresh(
            scalar_multi_levels,
            block_dofs=1,
            b=b_x,
            x_star=x_star_x,
            pre_sweeps=args.pre,
            post_sweeps=args.post,
            tol=args.tol,
            max_cycles=args.max_cycles,
        )
        print(summarize("Scalar GBP(multilevel MG)", scalar_multi))
        print("")

    if args.mode in ("block", "all"):
        block_two_levels = build_line_levels_from_matrix(args.n, lam, block_dofs=2, max_levels=2)
        block_multi_levels = build_line_levels_from_matrix(args.n, lam, block_dofs=2, max_levels=None)
        print(f"Block geometric two-level sizes: {[lvl.n for lvl in block_two_levels]}")
        print(f"Block geometric multilevel sizes: {[lvl.n for lvl in block_multi_levels]}")
        block_base = run_block_gabp_direct(
            build_block_gabp_level(lam, block_dofs=2),
            eta,
            x_star_full,
            tol=args.tol,
            max_iters=args.max_base_iters,
        )
        print(summarize("Block2 GBP(base)", block_base))
        block_two = run_multigrid_with_block_gabp_direct_fresh(
            block_two_levels,
            block_dofs=2,
            b=eta,
            x_star=x_star_full,
            pre_sweeps=args.pre,
            post_sweeps=args.post,
            tol=args.tol,
            max_cycles=args.max_cycles,
        )
        print(summarize("Block2 GBP(two-level MG)", block_two))
        block_multi = run_multigrid_with_block_gabp_direct_fresh(
            block_multi_levels,
            block_dofs=2,
            b=eta,
            x_star=x_star_full,
            pre_sweeps=args.pre,
            post_sweeps=args.post,
            tol=args.tol,
            max_cycles=args.max_cycles,
        )
        print(summarize("Block2 GBP(multilevel MG)", block_multi))
        print("")

    if args.include_raylib:
        ray_args = RaylibArgs(
            prior_sigma=args.prior_sigma,
            odom_sigma=args.odom_sigma,
            seed=args.seed,
            theta=args.theta,
            split_mode=args.split_mode,
            interp_mode=args.interp_mode,
            disable_second_pass_coarse_match=args.disable_second_pass_coarse_match,
        )
        ray = run_raylib_family(
            nodes,
            edges,
            x_star_full,
            args=ray_args,
            tol=args.tol,
            max_base_iters=args.max_base_iters,
            max_v_cycles=args.max_cycles,
        )
        print("")
        print(
            f"Raylib(base): iterations={ray['raylib_base']['iterations']}, "
            f"rel_error={ray['raylib_base']['rel_error']:.3e}"
        )
        print(
            f"Raylib(two-level): sizes={ray['raylib_two_level']['sizes']}, "
            f"iterations={ray['raylib_two_level']['iterations']}, "
            f"rel_error={ray['raylib_two_level']['rel_error']:.3e}"
        )
        print(
            f"Raylib(multilevel): sizes={ray['raylib_multilevel']['sizes']}, "
            f"iterations={ray['raylib_multilevel']['iterations']}, "
            f"rel_error={ray['raylib_multilevel']['rel_error']:.3e}"
        )


if __name__ == "__main__":
    main()
