"""Reference-aligned smoother benchmark on grid Gaussian systems.

This keeps the geometric hierarchy fixed and swaps only the smoother:
  * weighted Jacobi
  * direct synchronous GBP aligned to svd_abstraction/gbp/gbp.py

The default path benchmarks the full 2D block system. A scalar-channel mode is
also available because the current linear xy grid Gaussian decouples exactly.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.block_gabp import build_block_gabp_level
from svd_abstraction.block_gabp import run_block_gabp_direct
from svd_abstraction.block_gabp import run_multigrid_with_block_gabp_direct_fresh
from svd_abstraction.grid_same_system_benchmark import system_from_grid_graph
from svd_abstraction.poisson_multigrid_benchmark import build_rectangular_levels_from_matrix
from svd_abstraction.poisson_multigrid_benchmark import run_jacobi
from svd_abstraction.poisson_multigrid_benchmark import run_multigrid


def extract_scalar_channel(
    eta: np.ndarray,
    lam: sparse.spmatrix | np.ndarray,
    channel: int,
    block_dofs: int = 2,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    lam_csr = sparse.csr_matrix(lam)
    idx = np.arange(channel, lam_csr.shape[0], block_dofs)
    return eta[idx], lam_csr[idx][:, idx].tocsr()


def relative_error(result: dict[str, object]) -> float:
    errors = result["error_history"]
    return float(errors[-1] / max(errors[0], 1e-15))


def summarize(name: str, result: dict[str, object], init_norm: float) -> str:
    if "error_history" in result:
        rel = result["error_history"][-1] / max(init_norm, 1e-15)
        residual = result["residual_history"][-1]
    else:
        rel = result["rel_error"]
        residual = float("nan")
    return (
        f"{name}: iterations={result['iterations']}, "
        f"rel_error={rel:.3e}, "
        f"residual={residual:.3e}, "
        f"time={result.get('elapsed_time', float('nan')):.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=31)
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
    parser.add_argument("--max-jacobi", type=int, default=5000)
    parser.add_argument("--max-gabp", type=int, default=5000)
    parser.add_argument("--max-cycles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["scalar", "block", "all"], default="block")
    args = parser.parse_args()

    _, _, eta, lam, x_star = system_from_grid_graph(
        nx=args.grid_n,
        ny=args.grid_n,
        spacing=args.grid_spacing,
        prior_prop=args.prior_prop,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
        shortcut_prob=args.grid_shortcut_prob,
        shortcut_min_sep=args.grid_shortcut_min_sep,
    )

    print(
        f"Reference-aligned grid benchmark: {args.grid_n}x{args.grid_n}, "
        f"shortcut_prob={args.grid_shortcut_prob:.3f}, min_sep={args.grid_shortcut_min_sep}"
    )
    print(f"pre={args.pre}, post={args.post}, omega={args.omega:.3f}")
    print("")

    if args.mode in ("scalar", "all"):
        b_x, a_x = extract_scalar_channel(eta, lam, 0)
        x_star_x = spla.spsolve(a_x, b_x)
        scalar_two = build_rectangular_levels_from_matrix(args.grid_n, args.grid_n, a_x, block_dofs=1, max_levels=2)
        scalar_multi = build_rectangular_levels_from_matrix(args.grid_n, args.grid_n, a_x, block_dofs=1, max_levels=None)
        init_scalar = np.linalg.norm(x_star_x)
        print(f"Scalar sizes (two-level): {[(lvl.nx, lvl.ny) for lvl in scalar_two]}")
        print(f"Scalar sizes (multilevel): {[(lvl.nx, lvl.ny) for lvl in scalar_multi]}")
        jac_base = run_jacobi(scalar_multi[0], b_x, x_star_x, omega=args.omega, tol=args.tol, max_iters=args.max_jacobi)
        jac_two = run_multigrid(scalar_two, b_x, x_star_x, omega=args.omega, pre_sweeps=args.pre, post_sweeps=args.post, tol=args.tol, max_cycles=args.max_cycles)
        jac_multi = run_multigrid(scalar_multi, b_x, x_star_x, omega=args.omega, pre_sweeps=args.pre, post_sweeps=args.post, tol=args.tol, max_cycles=args.max_cycles)
        gabp_base = run_block_gabp_direct(build_block_gabp_level(a_x, 1), b_x, x_star_x, tol=args.tol, max_iters=args.max_gabp)
        gabp_two = run_multigrid_with_block_gabp_direct_fresh(scalar_two, 1, b_x, x_star_x, pre_sweeps=args.pre, post_sweeps=args.post, tol=args.tol, max_cycles=args.max_cycles)
        gabp_multi = run_multigrid_with_block_gabp_direct_fresh(scalar_multi, 1, b_x, x_star_x, pre_sweeps=args.pre, post_sweeps=args.post, tol=args.tol, max_cycles=args.max_cycles)
        print(summarize("Scalar Jacobi(base)", jac_base, init_scalar))
        print(summarize("Scalar Jacobi(two-level)", jac_two, init_scalar))
        print(summarize("Scalar Jacobi(multilevel)", jac_multi, init_scalar))
        print(summarize("Scalar GBP(base)", gabp_base, init_scalar))
        print(summarize("Scalar GBP(two-level)", gabp_two, init_scalar))
        print(summarize("Scalar GBP(multilevel)", gabp_multi, init_scalar))
        print("")

    if args.mode in ("block", "all"):
        block_two = build_rectangular_levels_from_matrix(args.grid_n, args.grid_n, lam, block_dofs=2, max_levels=2)
        block_multi = build_rectangular_levels_from_matrix(args.grid_n, args.grid_n, lam, block_dofs=2, max_levels=None)
        init_block = np.linalg.norm(x_star)
        print(f"Block sizes (two-level): {[(lvl.nx, lvl.ny) for lvl in block_two]}")
        print(f"Block sizes (multilevel): {[(lvl.nx, lvl.ny) for lvl in block_multi]}")
        jac_base = run_jacobi(block_multi[0], eta, x_star, omega=args.omega, tol=args.tol, max_iters=args.max_jacobi)
        jac_two = run_multigrid(block_two, eta, x_star, omega=args.omega, pre_sweeps=args.pre, post_sweeps=args.post, tol=args.tol, max_cycles=args.max_cycles)
        jac_multi = run_multigrid(block_multi, eta, x_star, omega=args.omega, pre_sweeps=args.pre, post_sweeps=args.post, tol=args.tol, max_cycles=args.max_cycles)
        gabp_base = run_block_gabp_direct(build_block_gabp_level(lam, 2), eta, x_star, tol=args.tol, max_iters=args.max_gabp)
        gabp_two = run_multigrid_with_block_gabp_direct_fresh(block_two, 2, eta, x_star, pre_sweeps=args.pre, post_sweeps=args.post, tol=args.tol, max_cycles=args.max_cycles)
        gabp_multi = run_multigrid_with_block_gabp_direct_fresh(block_multi, 2, eta, x_star, pre_sweeps=args.pre, post_sweeps=args.post, tol=args.tol, max_cycles=args.max_cycles)
        print(summarize("Block Jacobi(base)", jac_base, init_block))
        print(summarize("Block Jacobi(two-level)", jac_two, init_block))
        print(summarize("Block Jacobi(multilevel)", jac_multi, init_block))
        print(summarize("Block GBP(base)", gabp_base, init_block))
        print(summarize("Block GBP(two-level)", gabp_two, init_block))
        print(summarize("Block GBP(multilevel)", gabp_multi, init_block))
        print("")


if __name__ == "__main__":
    main()
