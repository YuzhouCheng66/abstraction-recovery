"""Scalar Gaussian BP helpers for attractive linear systems.

This module implements synchronous scalar GaBP on SPD M-matrices, i.e.
systems whose off-diagonal entries are non-positive. It is intentionally
lightweight so we can use it as a drop-in smoother inside geometric
multigrid experiments on grid-Gaussian systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass
class ScalarGaBPLevel:
    a: sparse.csr_matrix
    diag: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    aij: np.ndarray
    rev: np.ndarray


@dataclass
class ScalarGaBPState:
    p_msg: np.ndarray
    h_msg: np.ndarray


def build_scalar_gabp_level(a: sparse.spmatrix | np.ndarray, tol: float = 1e-12) -> ScalarGaBPLevel:
    """Build directed-edge GaBP data for a scalar attractive system."""
    a_csr = sparse.csr_matrix(a)
    if a_csr.shape[0] != a_csr.shape[1]:
        raise ValueError("GaBP requires a square system matrix.")

    diag = a_csr.diagonal().astype(float, copy=True)
    rows, cols = a_csr.nonzero()

    directed: list[tuple[int, int, float]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i == j:
            continue
        value = float(a_csr[i, j])
        if value > tol:
            raise ValueError(f"GaBP helper expects non-positive off-diagonals, got A[{i},{j}]={value}.")
        directed.append((i, j, value))

    key_to_idx = {(i, j): idx for idx, (i, j, _) in enumerate(directed)}
    src = np.array([i for i, _, _ in directed], dtype=np.int32)
    dst = np.array([j for _, j, _ in directed], dtype=np.int32)
    aij = np.array([value for _, _, value in directed], dtype=float)
    rev = np.array([key_to_idx[(j, i)] for i, j, _ in directed], dtype=np.int32)

    return ScalarGaBPLevel(
        a=a_csr,
        diag=diag,
        src=src,
        dst=dst,
        aij=aij,
        rev=rev,
    )


def gabp_correction(level: ScalarGaBPLevel, rhs: np.ndarray, sweeps: int) -> np.ndarray:
    """Approximate the correction e in A e = rhs using synchronous GaBP."""
    if sweeps <= 0:
        return np.zeros_like(rhs)

    n_edges = level.src.size
    p_msg = np.zeros(n_edges, dtype=float)
    h_msg = np.zeros(n_edges, dtype=float)

    for _ in range(sweeps):
        in_p = np.bincount(level.dst, weights=p_msg, minlength=level.diag.size)
        in_h = np.bincount(level.dst, weights=h_msg, minlength=level.diag.size)

        excl_p = level.diag[level.src] + in_p[level.src] - p_msg[level.rev]
        excl_h = rhs[level.src] + in_h[level.src] - h_msg[level.rev]

        p_msg = -(level.aij * level.aij) / excl_p
        h_msg = -(level.aij * excl_h) / excl_p

    in_p = np.bincount(level.dst, weights=p_msg, minlength=level.diag.size)
    in_h = np.bincount(level.dst, weights=h_msg, minlength=level.diag.size)
    return (rhs + in_h) / (level.diag + in_p)


def initialize_gabp_state(level: ScalarGaBPLevel) -> ScalarGaBPState:
    """Create a zero-message synchronous GaBP state."""
    n_edges = level.src.size
    return ScalarGaBPState(
        p_msg=np.zeros(n_edges, dtype=float),
        h_msg=np.zeros(n_edges, dtype=float),
    )


def gabp_sweep(level: ScalarGaBPLevel, state: ScalarGaBPState, rhs: np.ndarray) -> None:
    """Advance synchronous GaBP by one message-passing sweep in place."""
    in_p = np.bincount(level.dst, weights=state.p_msg, minlength=level.diag.size)
    in_h = np.bincount(level.dst, weights=state.h_msg, minlength=level.diag.size)

    excl_p = level.diag[level.src] + in_p[level.src] - state.p_msg[level.rev]
    excl_h = rhs[level.src] + in_h[level.src] - state.h_msg[level.rev]

    state.p_msg = -(level.aij * level.aij) / excl_p
    state.h_msg = -(level.aij * excl_h) / excl_p


def gabp_mean(level: ScalarGaBPLevel, state: ScalarGaBPState, rhs: np.ndarray) -> np.ndarray:
    """Read out the current mean estimate from a synchronous GaBP state."""
    in_p = np.bincount(level.dst, weights=state.p_msg, minlength=level.diag.size)
    in_h = np.bincount(level.dst, weights=state.h_msg, minlength=level.diag.size)
    return (rhs + in_h) / (level.diag + in_p)


def run_scalar_gabp(
    level: ScalarGaBPLevel,
    b: np.ndarray,
    x_star: np.ndarray,
    tol: float,
    max_iters: int,
) -> dict[str, object]:
    """Run plain synchronous scalar GaBP with persistent messages."""
    state = initialize_gabp_state(level)
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - level.a @ x))]
    error_history = [float(initial_error)]
    t0 = perf_counter()

    for it in range(1, max_iters + 1):
        gabp_sweep(level, state, b)
        x = gabp_mean(level, state, b)
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


def gabp_smooth(level, gabp_level: ScalarGaBPLevel, x: np.ndarray, b: np.ndarray, sweeps: int) -> np.ndarray:
    """Residual-correction smoother using a few GaBP sweeps."""
    residual = b - level.a @ x
    correction = gabp_correction(gabp_level, residual, sweeps=sweeps)
    return x + correction


def v_cycle_with_gabp(
    levels,
    gabp_levels: list[ScalarGaBPLevel],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
) -> np.ndarray:
    """Generic V-cycle with GaBP used as the level smoother."""
    level = levels[level_idx]
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / levels[0].a.shape[0])

    if level_idx == len(levels) - 1:
        stats["coarse_solves"] += 1
        return spla.spsolve(level.a, b)

    x = gabp_smooth(level, gabp_levels[level_idx], x, b, pre_sweeps)
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_with_gabp(
        levels,
        gabp_levels,
        level_idx + 1,
        coarse_error,
        coarse_rhs,
        pre_sweeps,
        post_sweeps,
        stats,
    )
    x = x + level.p @ coarse_error
    x = gabp_smooth(level, gabp_levels[level_idx], x, b, post_sweeps)
    return x


def run_multigrid_with_gabp(
    levels,
    b: np.ndarray,
    x_star: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
) -> dict[str, object]:
    """Run a geometric V-cycle while swapping the smoother to GaBP."""
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    gabp_levels = [build_scalar_gabp_level(level.a) for level in levels[:-1]]
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle_with_gabp(
            levels,
            gabp_levels,
            0,
            x,
            b,
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
