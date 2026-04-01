"""Compare Jacobi and GaBP smoothers on textbook geometric multigrid for Poisson."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from scipy.sparse import linalg as spla

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.poisson_multigrid_benchmark import build_levels
from svd_abstraction.poisson_multigrid_benchmark import run_jacobi
from svd_abstraction.poisson_multigrid_benchmark import run_multigrid
from svd_abstraction.poisson_multigrid_benchmark import sample_rhs
from svd_abstraction.scalar_gabp import build_scalar_gabp_level
from svd_abstraction.scalar_gabp import run_multigrid_with_gabp
from svd_abstraction.scalar_gabp import run_scalar_gabp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=255, help="Interior grid width; must be 2^k - 1.")
    parser.add_argument("--rhs", choices=["ones", "random", "smooth"], default="random")
    parser.add_argument("--omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-jacobi", type=int, default=5000)
    parser.add_argument("--max-gabp", type=int, default=5000)
    parser.add_argument("--max-cycles", type=int, default=500)
    args = parser.parse_args()

    all_levels = build_levels(args.n)
    two_levels = build_levels(args.n, max_levels=2)

    b = sample_rhs(args.n, args.rhs)
    x_star = spla.spsolve(all_levels[0].a, b)
    initial_error = np.linalg.norm(x_star)
    if initial_error == 0.0:
        initial_error = 1.0

    jacobi = run_jacobi(
        all_levels[0],
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
        all_levels,
        b,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )

    gabp = run_scalar_gabp(
        build_scalar_gabp_level(all_levels[0].a),
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
        all_levels,
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
        print(
            f"{name}: iterations={result['iterations']}, rel_error={rel_error:.3e}, "
            f"residual={residual:.3e}, time={result['elapsed_time']:.3f}s"
        )
        if is_mg:
            print(
                f"  smoothing_work={result['smoothing_work']:.2f}, "
                f"coarse_solves={result['coarse_solves']:.0f}"
            )

    print(f"Poisson smoother benchmark: {args.n}x{args.n}, rhs={args.rhs}")
    print(f"Two-level sizes: {[lvl.n for lvl in two_levels]}")
    print(f"Multilevel sizes: {[lvl.n for lvl in all_levels]}")
    print(f"pre={args.pre}, post={args.post}, omega={args.omega:.3f}")
    print("")
    summarize("Jacobi(base)", jacobi, is_mg=False)
    summarize("Jacobi(two-level)", jacobi_two, is_mg=True)
    summarize("Jacobi(multilevel)", jacobi_multi, is_mg=True)
    summarize("GaBP(base)", gabp, is_mg=False)
    summarize("GaBP(two-level)", gabp_two, is_mg=True)
    summarize("GaBP(multilevel)", gabp_multi, is_mg=True)


if __name__ == "__main__":
    main()
