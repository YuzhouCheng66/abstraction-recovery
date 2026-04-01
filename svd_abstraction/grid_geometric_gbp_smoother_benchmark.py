"""Compare Jacobi and GaBP smoothers on the same geometric multigrid hierarchy.

We intentionally keep the geometric restriction/prolongation and Galerkin
coarse operators fixed, and only swap the smoother:
  * weighted Jacobi
  * scalar synchronous GaBP

For the current grid-Gaussian model, x/y channels are exactly decoupled, so we
benchmark one scalar channel as a clean apples-to-apples comparison.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.grid_same_system_benchmark import system_from_grid_graph
from svd_abstraction.poisson_multigrid_benchmark import build_rectangular_levels_from_matrix
from svd_abstraction.poisson_multigrid_benchmark import run_jacobi
from svd_abstraction.poisson_multigrid_benchmark import run_multigrid
from svd_abstraction.scalar_gabp import build_scalar_gabp_level
from svd_abstraction.scalar_gabp import run_multigrid_with_gabp
from svd_abstraction.scalar_gabp import run_scalar_gabp


def extract_scalar_channel(
    eta: np.ndarray,
    lam: sparse.spmatrix | np.ndarray,
    channel: int,
    block_dofs: int = 2,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Extract one scalar channel from a block-separable grid Gaussian system."""
    if channel < 0 or channel >= block_dofs:
        raise ValueError("channel must be within the block dof range")

    lam_csr = sparse.csr_matrix(lam)
    idx = np.arange(channel, lam_csr.shape[0], block_dofs)
    return eta[idx], lam_csr[idx][:, idx].tocsr()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=31, help="Use an odd grid so geometric coarsening is exact.")
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--grid-shortcut-prob", type=float, default=0.0)
    parser.add_argument("--grid-shortcut-min-sep", type=int, default=4)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--channel", choices=["x", "y"], default="x")
    parser.add_argument("--omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-jacobi", type=int, default=5000)
    parser.add_argument("--max-gabp", type=int, default=5000)
    parser.add_argument("--max-cycles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, _, eta, lam, _ = system_from_grid_graph(
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

    channel_idx = 0 if args.channel == "x" else 1
    b, a = extract_scalar_channel(eta, lam, channel_idx)
    x_star = spla.spsolve(a, b)
    initial_error = np.linalg.norm(x_star)
    if initial_error == 0.0:
        initial_error = 1.0

    two_levels = build_rectangular_levels_from_matrix(
        args.grid_n,
        args.grid_n,
        a,
        block_dofs=1,
        max_levels=2,
    )
    multilevel = build_rectangular_levels_from_matrix(
        args.grid_n,
        args.grid_n,
        a,
        block_dofs=1,
        max_levels=None,
    )

    jacobi_base = run_jacobi(
        two_levels[0],
        b,
        x_star,
        omega=args.omega,
        tol=args.tol,
        max_iters=args.max_jacobi,
    )
    jacobi_two = run_multigrid(
        two_levels,
        b,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )
    jacobi_multi = run_multigrid(
        multilevel,
        b,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )

    gabp_base = run_scalar_gabp(
        build_scalar_gabp_level(a),
        b,
        x_star,
        tol=args.tol,
        max_iters=args.max_gabp,
    )
    gabp_two = run_multigrid_with_gabp(
        two_levels,
        b,
        x_star,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )
    gabp_multi = run_multigrid_with_gabp(
        multilevel,
        b,
        x_star,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )

    def summarize(name: str, result: dict[str, object], is_mg: bool) -> None:
        rel_error = result["error_history"][-1] / initial_error
        residual = result["residual_history"][-1]
        time_str = f"time={result.get('elapsed_time', float('nan')):.3f}s" if "elapsed_time" in result else ""
        print(
            f"{name}: iterations={result['iterations']}, rel_error={rel_error:.3e}, "
            f"residual={residual:.3e}{', ' + time_str if time_str else ''}"
        )
        if is_mg:
            print(
                f"  smoothing_work={result['smoothing_work']:.2f}, "
                f"coarse_solves={result['coarse_solves']:.0f}"
            )

    print(
        f"Grid Gaussian scalar-channel benchmark: {args.grid_n}x{args.grid_n}, "
        f"channel={args.channel}, shortcut_prob={args.grid_shortcut_prob:.3f}, "
        f"min_sep={args.grid_shortcut_min_sep}"
    )
    print(f"Geometric sizes (two-level): {[(lvl.nx, lvl.ny) for lvl in two_levels]}")
    print(f"Geometric sizes (multilevel): {[(lvl.nx, lvl.ny) for lvl in multilevel]}")
    print(f"pre={args.pre}, post={args.post}, omega={args.omega:.3f}")
    print("")
    summarize("Jacobi(base)", jacobi_base, is_mg=False)
    summarize("Jacobi(two-level)", jacobi_two, is_mg=True)
    summarize("Jacobi(multilevel)", jacobi_multi, is_mg=True)
    summarize("GaBP(base)", gabp_base, is_mg=False)
    summarize("GaBP(two-level)", gabp_two, is_mg=True)
    summarize("GaBP(multilevel)", gabp_multi, is_mg=True)


if __name__ == "__main__":
    main()
