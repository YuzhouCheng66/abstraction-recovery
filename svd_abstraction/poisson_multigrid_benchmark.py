"""Classical geometric multigrid benchmark on a 2D Poisson model problem.

This script gives us a clean textbook baseline:
  * model problem: 2D Poisson equation on a regular grid
  * smoother: weighted Jacobi
  * transfer: full-weighting restriction + bilinear prolongation
  * coarse operator: Galerkin product A_c = R A P

It is meant to separate "toy problem pathology" from hierarchy / transfer
issues in the current pose-graph GBP experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def poisson_2d_matrix(n: int) -> sparse.csr_matrix:
    """Return the 5-point Laplacian on an n x n interior grid."""
    if n <= 0:
        raise ValueError("n must be positive")

    e = np.ones(n, dtype=float)
    t = sparse.diags(
        diagonals=[-e[:-1], 4.0 * e, -e[:-1]],
        offsets=[-1, 0, 1],
        format="csr",
    )
    i = sparse.eye(n, format="csr")
    off = sparse.diags([-e[:-1], -e[:-1]], [-1, 1], shape=(n, n), format="csr")
    return sparse.kron(i, t, format="csr") + sparse.kron(off, i, format="csr")


def restriction_1d(n_fine: int) -> sparse.csr_matrix:
    """Full-weighting restriction from n_fine to (n_fine - 1) / 2."""
    if n_fine <= 1 or (n_fine - 1) % 2 != 0:
        raise ValueError("n_fine must be of the form 2*k + 1")

    n_coarse = (n_fine - 1) // 2
    rows = []
    cols = []
    data = []
    for i_coarse in range(n_coarse):
        i_fine = 2 * i_coarse + 1
        for offset, weight in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            j_fine = i_fine + offset
            if 0 <= j_fine < n_fine:
                rows.append(i_coarse)
                cols.append(j_fine)
                data.append(weight)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_coarse, n_fine))


def prolongation_2d(n_fine: int) -> sparse.csr_matrix:
    """Bilinear prolongation using the transpose relation P = 4 R^T."""
    r1d = restriction_1d(n_fine)
    r2d = sparse.kron(r1d, r1d, format="csr")
    return (4.0 * r2d.transpose()).tocsr()


def restriction_2d_rect(nx_fine: int, ny_fine: int) -> sparse.csr_matrix:
    """Full-weighting restriction on a rectangular node grid."""
    rx = restriction_1d(nx_fine)
    ry = restriction_1d(ny_fine)
    return sparse.kron(ry, rx, format="csr")


def prolongation_2d_rect(nx_fine: int, ny_fine: int) -> sparse.csr_matrix:
    """Bilinear prolongation on a rectangular node grid."""
    return (4.0 * restriction_2d_rect(nx_fine, ny_fine).transpose()).tocsr()


@dataclass
class Level:
    n: int
    a: sparse.csr_matrix
    r: sparse.csr_matrix | None
    p: sparse.csr_matrix | None
    diag_inv: np.ndarray
    nx: int | None = None
    ny: int | None = None
    block_dofs: int = 1


def build_levels(n: int, max_levels: int | None = None, min_coarse_n: int = 1) -> list[Level]:
    if n <= 0:
        raise ValueError("n must be positive")
    if (n + 1) & n != 0:
        raise ValueError("n must be of the form 2^k - 1, e.g. 31, 63, 127")

    levels: list[Level] = []
    n_level = n
    level_count = 0
    while True:
        a = poisson_2d_matrix(n_level).tocsr()
        diag_inv = 1.0 / a.diagonal()
        levels.append(Level(n=n_level, a=a, r=None, p=None, diag_inv=diag_inv, nx=n_level, ny=n_level))
        level_count += 1

        if n_level <= min_coarse_n:
            break
        if max_levels is not None and level_count >= max_levels:
            break

        n_coarse = (n_level - 1) // 2
        if n_coarse < 1:
            break

        p = prolongation_2d(n_level)
        r = 0.25 * p.transpose()
        levels[-1].p = p.tocsr()
        levels[-1].r = r.tocsr()
        n_level = n_coarse

    for idx in range(len(levels) - 1):
        fine = levels[idx]
        coarse = levels[idx + 1]
        coarse.a = (fine.r @ fine.a @ fine.p).tocsr()
        coarse.diag_inv = 1.0 / coarse.a.diagonal()

    return levels


def build_rectangular_levels_from_matrix(
    nx: int,
    ny: int,
    a: sparse.spmatrix | np.ndarray,
    block_dofs: int = 1,
    max_levels: int | None = None,
    min_coarse_n: int = 1,
) -> list[Level]:
    """Build geometric Galerkin levels for a rectangular node grid.

    The fine operator is supplied externally; transfer operators are geometric.
    The unknown ordering is assumed to be node-major with optional block dofs.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if (nx - 1) % 2 != 0 or (ny - 1) % 2 != 0:
        raise ValueError("nx and ny must be odd so they can be halved geometrically")
    if block_dofs <= 0:
        raise ValueError("block_dofs must be positive")

    a0 = sparse.csr_matrix(a)
    expected_dim = nx * ny * block_dofs
    if a0.shape != (expected_dim, expected_dim):
        raise ValueError(f"Expected matrix shape {(expected_dim, expected_dim)}, got {a0.shape}")

    levels: list[Level] = []
    nx_level = nx
    ny_level = ny
    a_level = a0
    level_count = 0

    while True:
        levels.append(
            Level(
                n=nx_level,
                a=a_level.tocsr(),
                r=None,
                p=None,
                diag_inv=1.0 / a_level.diagonal(),
                nx=nx_level,
                ny=ny_level,
                block_dofs=block_dofs,
            )
        )
        level_count += 1

        if min(nx_level, ny_level) <= min_coarse_n:
            break
        if max_levels is not None and level_count >= max_levels:
            break
        if (nx_level - 1) % 2 != 0 or (ny_level - 1) % 2 != 0:
            break

        nx_coarse = (nx_level - 1) // 2
        ny_coarse = (ny_level - 1) // 2
        if nx_coarse < 1 or ny_coarse < 1:
            break

        p_scalar = prolongation_2d_rect(nx_level, ny_level)
        r_scalar = 0.25 * p_scalar.transpose()
        if block_dofs == 1:
            p = p_scalar.tocsr()
            r = r_scalar.tocsr()
        else:
            eye = sparse.eye(block_dofs, format="csr")
            p = sparse.kron(p_scalar, eye, format="csr")
            r = sparse.kron(r_scalar, eye, format="csr")

        levels[-1].p = p.tocsr()
        levels[-1].r = r.tocsr()
        a_level = (r @ a_level @ p).tocsr()
        nx_level = nx_coarse
        ny_level = ny_coarse

    return levels


def weighted_jacobi(level: Level, x: np.ndarray, b: np.ndarray, omega: float, iterations: int) -> np.ndarray:
    for _ in range(iterations):
        x = x + omega * level.diag_inv * (b - level.a @ x)
    return x


def v_cycle(
    levels: list[Level],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    omega: float,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
) -> np.ndarray:
    level = levels[level_idx]
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / levels[0].a.shape[0])

    if level_idx == len(levels) - 1:
        stats["coarse_solves"] += 1
        return spla.spsolve(level.a, b)

    x = weighted_jacobi(level, x, b, omega, pre_sweeps)
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle(
        levels,
        level_idx + 1,
        coarse_error,
        coarse_rhs,
        omega,
        pre_sweeps,
        post_sweeps,
        stats,
    )
    x = x + level.p @ coarse_error
    x = weighted_jacobi(level, x, b, omega, post_sweeps)
    return x


def run_jacobi(
    level: Level,
    b: np.ndarray,
    x_star: np.ndarray,
    omega: float,
    tol: float,
    max_iters: int,
) -> dict[str, object]:
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - level.a @ x))]
    error_history = [float(initial_error)]
    t0 = perf_counter()

    for it in range(1, max_iters + 1):
        x = weighted_jacobi(level, x, b, omega, iterations=1)
        residual_history.append(float(np.linalg.norm(b - level.a @ x)))
        error = float(np.linalg.norm(x - x_star))
        error_history.append(error)
        if error / max(initial_error, 1e-15) < tol:
            return {
                "iterations": it,
                "residual_history": residual_history,
                "error_history": error_history,
                "x": x,
                "elapsed_time": perf_counter() - t0,
            }

    return {
        "iterations": max_iters,
        "residual_history": residual_history,
        "error_history": error_history,
        "x": x,
        "elapsed_time": perf_counter() - t0,
    }


def run_multigrid(
    levels: list[Level],
    b: np.ndarray,
    x_star: np.ndarray,
    omega: float,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
) -> dict[str, object]:
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle(
            levels,
            0,
            x,
            b,
            omega=omega,
            pre_sweeps=pre_sweeps,
            post_sweeps=post_sweeps,
            stats=stats,
        )
        residual_history.append(float(np.linalg.norm(b - levels[0].a @ x)))
        error = float(np.linalg.norm(x - x_star))
        error_history.append(error)
        if error / max(initial_error, 1e-15) < tol:
            return {
                "iterations": cycle,
                "residual_history": residual_history,
                "error_history": error_history,
                "x": x,
                "smoothing_work": stats["smoothing_work"],
                "coarse_solves": stats["coarse_solves"],
                "elapsed_time": perf_counter() - t0,
            }

    return {
        "iterations": max_cycles,
        "residual_history": residual_history,
        "error_history": error_history,
        "x": x,
        "smoothing_work": stats["smoothing_work"],
        "coarse_solves": stats["coarse_solves"],
        "elapsed_time": perf_counter() - t0,
    }


def sample_rhs(n: int, mode: str) -> np.ndarray:
    m = n * n
    if mode == "ones":
        return np.ones(m, dtype=float)
    if mode == "random":
        rng = np.random.default_rng(0)
        return rng.standard_normal(m)
    if mode == "smooth":
        x = np.linspace(1.0 / (n + 1), n / (n + 1), n)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        return np.sin(np.pi * xx).ravel() * np.sin(np.pi * yy).ravel()
    raise ValueError(f"Unknown rhs mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=127, help="Interior grid width; must be 2^k - 1.")
    parser.add_argument("--rhs", choices=["ones", "random", "smooth"], default="smooth")
    parser.add_argument("--omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-jacobi", type=int, default=5000)
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
    two_level = run_multigrid(
        two_levels,
        b,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )
    multilevel = run_multigrid(
        all_levels,
        b,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )

    def summarize(name: str, result: dict[str, object], is_mg: bool) -> None:
        final_error = result["error_history"][-1] / initial_error
        final_residual = result["residual_history"][-1]
        print(
            f"{name}: iterations={result['iterations']}, "
            f"rel_error={final_error:.3e}, residual={final_residual:.3e}, "
            f"time={result['elapsed_time']:.3f}s"
        )
        if is_mg:
            print(
                f"  smoothing_work={result['smoothing_work']:.2f}, "
                f"coarse_solves={result['coarse_solves']:.0f}"
            )

    print(f"Poisson grid: {args.n} x {args.n}")
    print(f"RHS mode: {args.rhs}; omega={args.omega:.3f}; pre={args.pre}; post={args.post}")
    print(f"Two-level sizes: {[lvl.n for lvl in two_levels]}")
    print(f"Multilevel sizes: {[lvl.n for lvl in all_levels]}")
    summarize("Jacobi", jacobi, is_mg=False)
    summarize("Two-level V-cycle", two_level, is_mg=True)
    summarize("Multilevel V-cycle", multilevel, is_mg=True)

    sample_points = [1, 2, 5, 10, 20, 50, 100]
    print("\nRelative error history:")
    for k in sample_points:
        if k < len(jacobi["error_history"]) and k < len(two_level["error_history"]) and k < len(multilevel["error_history"]):
            print(
                f"  k={k:3d}: "
                f"Jacobi={jacobi['error_history'][k] / initial_error:.3e}, "
                f"Two-level={two_level['error_history'][k] / initial_error:.3e}, "
                f"Multilevel={multilevel['error_history'][k] / initial_error:.3e}"
            )


if __name__ == "__main__":
    main()
