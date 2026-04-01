"""Block Gaussian BP helpers for attractive block linear systems."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass
class BlockGaBPLevel:
    a: sparse.csr_matrix
    block_dofs: int
    n_nodes: int
    diag_blocks: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    aij: np.ndarray
    rev: np.ndarray


@dataclass
class BlockGaBPSmootherState:
    p_msg: np.ndarray
    h_msg: np.ndarray


def build_block_gabp_level(
    a: sparse.spmatrix | np.ndarray,
    block_dofs: int,
    tol: float = 1e-12,
) -> BlockGaBPLevel:
    """Extract a block pairwise GaBP representation from a sparse operator."""
    a_csr = sparse.csr_matrix(a)
    if a_csr.shape[0] != a_csr.shape[1]:
        raise ValueError("GaBP requires a square system matrix.")
    if a_csr.shape[0] % block_dofs != 0:
        raise ValueError("Matrix dimension must be divisible by block_dofs.")

    bsr = a_csr.tobsr(blocksize=(block_dofs, block_dofs))
    n_nodes = a_csr.shape[0] // block_dofs

    diag_blocks = np.zeros((n_nodes, block_dofs, block_dofs), dtype=float)
    directed: list[tuple[int, int, np.ndarray]] = []

    for i in range(n_nodes):
        start = bsr.indptr[i]
        end = bsr.indptr[i + 1]
        for pos in range(start, end):
            j = int(bsr.indices[pos])
            block = np.asarray(bsr.data[pos], dtype=float)
            if i == j:
                diag_blocks[i] = block
                continue
            if np.any(block > tol):
                raise ValueError(f"Expected attractive block system; found positive off-diagonal block at {(i, j)}.")
            directed.append((i, j, block))

    key_to_idx = {(i, j): idx for idx, (i, j, _) in enumerate(directed)}
    src = np.array([i for i, _, _ in directed], dtype=np.int32)
    dst = np.array([j for _, j, _ in directed], dtype=np.int32)
    aij = np.stack([block for _, _, block in directed], axis=0) if directed else np.zeros((0, block_dofs, block_dofs))
    rev = np.array([key_to_idx[(j, i)] for i, j, _ in directed], dtype=np.int32) if directed else np.zeros(0, dtype=np.int32)

    return BlockGaBPLevel(
        a=a_csr,
        block_dofs=block_dofs,
        n_nodes=n_nodes,
        diag_blocks=diag_blocks,
        src=src,
        dst=dst,
        aij=aij,
        rev=rev,
    )


def _sum_matrix_messages(level: BlockGaBPLevel, p_msg: np.ndarray) -> np.ndarray:
    sums = np.zeros((level.n_nodes, level.block_dofs, level.block_dofs), dtype=float)
    for edge_idx, dst in enumerate(level.dst.tolist()):
        sums[dst] += p_msg[edge_idx]
    return sums


def _sum_vector_messages(level: BlockGaBPLevel, h_msg: np.ndarray) -> np.ndarray:
    sums = np.zeros((level.n_nodes, level.block_dofs), dtype=float)
    for edge_idx, dst in enumerate(level.dst.tolist()):
        sums[dst] += h_msg[edge_idx]
    return sums


def initialize_block_gabp_state(level: BlockGaBPLevel) -> BlockGaBPSmootherState:
    n_edges = level.src.size
    d = level.block_dofs
    return BlockGaBPSmootherState(
        p_msg=np.zeros((n_edges, d, d), dtype=float),
        h_msg=np.zeros((n_edges, d), dtype=float),
    )


def block_gabp_full_sweep(level: BlockGaBPLevel, state: BlockGaBPSmootherState, rhs: np.ndarray) -> None:
    """One synchronous GBP sweep updating both precision and eta messages."""
    rhs_blocks = rhs.reshape(level.n_nodes, level.block_dofs)
    in_p = _sum_matrix_messages(level, state.p_msg)
    in_h = _sum_vector_messages(level, state.h_msg)
    new_p = np.empty_like(state.p_msg)
    new_h = np.empty_like(state.h_msg)

    for edge_idx in range(level.src.size):
        src = int(level.src[edge_idx])
        rev_idx = int(level.rev[edge_idx])
        excl_p = level.diag_blocks[src] + in_p[src] - state.p_msg[rev_idx]
        excl_h = rhs_blocks[src] + in_h[src] - state.h_msg[rev_idx]
        solved_a = np.linalg.solve(excl_p, level.aij[edge_idx])
        solved_h = np.linalg.solve(excl_p, excl_h)
        new_p[edge_idx] = -level.aij[edge_idx].T @ solved_a
        new_h[edge_idx] = -level.aij[edge_idx].T @ solved_h

    state.p_msg = new_p
    state.h_msg = new_h


def block_gabp_precision_sweep(level: BlockGaBPLevel, p_msg: np.ndarray) -> np.ndarray:
    """One synchronous sweep for precision messages."""
    in_p = _sum_matrix_messages(level, p_msg)
    new_p = np.empty_like(p_msg)
    for edge_idx in range(level.src.size):
        src = int(level.src[edge_idx])
        rev_idx = int(level.rev[edge_idx])
        excl = level.diag_blocks[src] + in_p[src] - p_msg[rev_idx]
        solved = np.linalg.solve(excl, level.aij[edge_idx])
        new_p[edge_idx] = -level.aij[edge_idx].T @ solved
    return new_p


def converge_block_gabp_precision(
    level: BlockGaBPLevel,
    tol: float = 1e-10,
    max_iters: int = 500,
) -> tuple[np.ndarray, int]:
    """Iterate precision messages to convergence once for a fixed operator."""
    p_msg = np.zeros((level.src.size, level.block_dofs, level.block_dofs), dtype=float)
    for it in range(1, max_iters + 1):
        new_p = block_gabp_precision_sweep(level, p_msg)
        delta = 0.0 if p_msg.size == 0 else float(np.max(np.abs(new_p - p_msg)))
        p_msg = new_p
        if delta < tol:
            return p_msg, it
    return p_msg, max_iters


def block_gabp_eta_sweep(level: BlockGaBPLevel, p_msg: np.ndarray, h_msg: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """One synchronous sweep for eta/mean messages with fixed precision messages."""
    rhs_blocks = rhs.reshape(level.n_nodes, level.block_dofs)
    in_p = _sum_matrix_messages(level, p_msg)
    in_h = _sum_vector_messages(level, h_msg)
    new_h = np.empty_like(h_msg)
    for edge_idx in range(level.src.size):
        src = int(level.src[edge_idx])
        rev_idx = int(level.rev[edge_idx])
        excl_p = level.diag_blocks[src] + in_p[src] - p_msg[rev_idx]
        excl_h = rhs_blocks[src] + in_h[src] - h_msg[rev_idx]
        solved = np.linalg.solve(excl_p, excl_h)
        new_h[edge_idx] = -level.aij[edge_idx].T @ solved
    return new_h


def block_gabp_mean(level: BlockGaBPLevel, p_msg: np.ndarray, h_msg: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Read out current block means from fixed P and H messages."""
    rhs_blocks = rhs.reshape(level.n_nodes, level.block_dofs)
    in_p = _sum_matrix_messages(level, p_msg)
    in_h = _sum_vector_messages(level, h_msg)
    x_blocks = np.empty_like(rhs_blocks)
    for node_idx in range(level.n_nodes):
        x_blocks[node_idx] = np.linalg.solve(level.diag_blocks[node_idx] + in_p[node_idx], rhs_blocks[node_idx] + in_h[node_idx])
    return x_blocks.reshape(-1)


def run_block_gabp(
    level: BlockGaBPLevel,
    b: np.ndarray,
    x_star: np.ndarray,
    tol: float,
    max_iters: int,
    precision_tol: float = 1e-10,
    precision_max_iters: int = 500,
) -> dict[str, object]:
    """Run synchronous block GaBP with preconverged precision messages."""
    p_msg, precision_iters = converge_block_gabp_precision(level, tol=precision_tol, max_iters=precision_max_iters)
    h_msg = np.zeros((level.src.size, level.block_dofs), dtype=float)
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - level.a @ x))]
    error_history = [float(initial_error)]
    t0 = perf_counter()

    for it in range(1, max_iters + 1):
        h_msg = block_gabp_eta_sweep(level, p_msg, h_msg, b)
        x = block_gabp_mean(level, p_msg, h_msg, b)
        residual_history.append(float(np.linalg.norm(b - level.a @ x)))
        error = float(np.linalg.norm(x - x_star))
        error_history.append(error)
        if error / max(initial_error, 1e-15) < tol:
            return {
                "iterations": it,
                "residual_history": residual_history,
                "error_history": error_history,
                "x": x,
                "precision_iterations": precision_iters,
                "elapsed_time": perf_counter() - t0,
            }

    return {
        "iterations": max_iters,
        "residual_history": residual_history,
        "error_history": error_history,
        "x": x,
        "precision_iterations": precision_iters,
        "elapsed_time": perf_counter() - t0,
    }


def run_block_gabp_direct(
    level: BlockGaBPLevel,
    b: np.ndarray,
    x_star: np.ndarray,
    tol: float,
    max_iters: int,
) -> dict[str, object]:
    """Run direct synchronous block GBP matching the reference message schedule."""
    state = initialize_block_gabp_state(level)
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - level.a @ x))]
    error_history = [float(initial_error)]
    t0 = perf_counter()

    for it in range(1, max_iters + 1):
        block_gabp_full_sweep(level, state, b)
        x = block_gabp_mean(level, state.p_msg, state.h_msg, b)
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


def initialize_block_gabp_smoother(
    level: BlockGaBPLevel,
    precision_tol: float = 1e-10,
    precision_max_iters: int = 500,
) -> BlockGaBPSmootherState:
    p_msg, _ = converge_block_gabp_precision(level, tol=precision_tol, max_iters=precision_max_iters)
    return BlockGaBPSmootherState(
        p_msg=p_msg,
        h_msg=np.zeros((level.src.size, level.block_dofs), dtype=float),
    )


def block_gabp_smooth(
    level,
    gabp_level: BlockGaBPLevel,
    state: BlockGaBPSmootherState,
    x: np.ndarray,
    b: np.ndarray,
    sweeps: int,
) -> np.ndarray:
    """Residual-correction smoother with persistent block GaBP mean messages."""
    if sweeps <= 0:
        return x
    rhs = b - level.a @ x
    for _ in range(sweeps):
        state.h_msg = block_gabp_eta_sweep(gabp_level, state.p_msg, state.h_msg, rhs)
    correction = block_gabp_mean(gabp_level, state.p_msg, state.h_msg, rhs)
    return x + correction


def block_gabp_smooth_direct(
    level,
    gabp_level: BlockGaBPLevel,
    state: BlockGaBPSmootherState,
    x: np.ndarray,
    b: np.ndarray,
    sweeps: int,
) -> np.ndarray:
    """Residual-correction smoother using direct synchronous block GBP sweeps."""
    if sweeps <= 0:
        return x
    rhs = b - level.a @ x
    for _ in range(sweeps):
        block_gabp_full_sweep(gabp_level, state, rhs)
    correction = block_gabp_mean(gabp_level, state.p_msg, state.h_msg, rhs)
    return x + correction


def block_gabp_smooth_direct_lamside_reuse(
    level,
    gabp_level: BlockGaBPLevel,
    state: BlockGaBPSmootherState,
    x: np.ndarray,
    b: np.ndarray,
    sweeps: int,
) -> np.ndarray:
    """Residual-correction smoother with lam-side reuse and fresh eta-side.

    The precision / variance side (`p_msg`) is carried across smoothing calls
    and keeps evolving as a warm start. The eta / mean side (`h_msg`) is reset
    for each new defect equation, matching the intended "lam-side reuse"
    ablation under a true V-cycle.
    """
    if sweeps <= 0:
        return x
    rhs = b - level.a @ x
    state.h_msg.fill(0.0)
    for _ in range(sweeps):
        block_gabp_full_sweep(gabp_level, state, rhs)
    correction = block_gabp_mean(gabp_level, state.p_msg, state.h_msg, rhs)
    return x + correction


def block_gabp_smooth_direct_fresh(
    level,
    gabp_level: BlockGaBPLevel,
    x: np.ndarray,
    b: np.ndarray,
    sweeps: int,
) -> np.ndarray:
    """Residual-correction smoother using fresh direct synchronous GBP sweeps.

    Each smoothing call solves the current defect equation approximately with
    zero-initialized messages. This matches the reference GBP core on that
    residual equation without carrying stale message state across different
    right-hand sides.
    """
    if sweeps <= 0:
        return x
    rhs = b - level.a @ x
    state = initialize_block_gabp_state(gabp_level)
    for _ in range(sweeps):
        block_gabp_full_sweep(gabp_level, state, rhs)
    correction = block_gabp_mean(gabp_level, state.p_msg, state.h_msg, rhs)
    return x + correction


def v_cycle_with_block_gabp(
    levels,
    gabp_levels: list[BlockGaBPLevel],
    states: list[BlockGaBPSmootherState],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
) -> np.ndarray:
    """Generic V-cycle with block GaBP used as the level smoother."""
    level = levels[level_idx]
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / levels[0].a.shape[0])

    if level_idx == len(levels) - 1:
        stats["coarse_solves"] += 1
        return spla.spsolve(level.a, b)

    x = block_gabp_smooth(level, gabp_levels[level_idx], states[level_idx], x, b, pre_sweeps)
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_with_block_gabp(
        levels,
        gabp_levels,
        states,
        level_idx + 1,
        coarse_error,
        coarse_rhs,
        pre_sweeps,
        post_sweeps,
        stats,
    )
    x = x + level.p @ coarse_error
    x = block_gabp_smooth(level, gabp_levels[level_idx], states[level_idx], x, b, post_sweeps)
    return x


def v_cycle_with_block_gabp_direct(
    levels,
    gabp_levels: list[BlockGaBPLevel],
    states: list[BlockGaBPSmootherState],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
) -> np.ndarray:
    """Generic V-cycle with direct synchronous block GBP used as the smoother."""
    level = levels[level_idx]
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / levels[0].a.shape[0])

    if level_idx == len(levels) - 1:
        stats["coarse_solves"] += 1
        return spla.spsolve(level.a, b)

    x = block_gabp_smooth_direct(level, gabp_levels[level_idx], states[level_idx], x, b, pre_sweeps)
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_with_block_gabp_direct(
        levels,
        gabp_levels,
        states,
        level_idx + 1,
        coarse_error,
        coarse_rhs,
        pre_sweeps,
        post_sweeps,
        stats,
    )
    x = x + level.p @ coarse_error
    x = block_gabp_smooth_direct(level, gabp_levels[level_idx], states[level_idx], x, b, post_sweeps)
    return x


def v_cycle_with_block_gabp_direct_lamside_reuse(
    levels,
    gabp_levels: list[BlockGaBPLevel],
    states: list[BlockGaBPSmootherState],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
) -> np.ndarray:
    """V-cycle using direct synchronous GBP with lam-side reuse."""
    level = levels[level_idx]
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / levels[0].a.shape[0])

    if level_idx == len(levels) - 1:
        stats["coarse_solves"] += 1
        return spla.spsolve(level.a, b)

    x = block_gabp_smooth_direct_lamside_reuse(
        level, gabp_levels[level_idx], states[level_idx], x, b, pre_sweeps
    )
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_with_block_gabp_direct_lamside_reuse(
        levels,
        gabp_levels,
        states,
        level_idx + 1,
        coarse_error,
        coarse_rhs,
        pre_sweeps,
        post_sweeps,
        stats,
    )
    x = x + level.p @ coarse_error
    x = block_gabp_smooth_direct_lamside_reuse(
        level, gabp_levels[level_idx], states[level_idx], x, b, post_sweeps
    )
    return x


def v_cycle_with_block_gabp_direct_fresh(
    levels,
    gabp_levels: list[BlockGaBPLevel],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
) -> np.ndarray:
    """Generic V-cycle with fresh direct synchronous block GBP as smoother."""
    level = levels[level_idx]
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / levels[0].a.shape[0])

    if level_idx == len(levels) - 1:
        stats["coarse_solves"] += 1
        return spla.spsolve(level.a, b)

    x = block_gabp_smooth_direct_fresh(level, gabp_levels[level_idx], x, b, pre_sweeps)
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_with_block_gabp_direct_fresh(
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
    x = block_gabp_smooth_direct_fresh(level, gabp_levels[level_idx], x, b, post_sweeps)
    return x


def run_multigrid_with_block_gabp(
    levels,
    block_dofs: int,
    b: np.ndarray,
    x_star: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
    precision_tol: float = 1e-10,
    precision_max_iters: int = 500,
) -> dict[str, object]:
    """Run a geometric V-cycle while swapping the smoother to block GaBP."""
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    gabp_levels = [build_block_gabp_level(level.a, block_dofs=block_dofs) for level in levels[:-1]]
    states = [
        initialize_block_gabp_smoother(level, precision_tol=precision_tol, precision_max_iters=precision_max_iters)
        for level in gabp_levels
    ]
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle_with_block_gabp(
            levels,
            gabp_levels,
            states,
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


def run_multigrid_with_block_gabp_direct_fresh(
    levels,
    block_dofs: int,
    b: np.ndarray,
    x_star: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
) -> dict[str, object]:
    """Run a geometric V-cycle using fresh direct synchronous block GBP smoothing."""
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    gabp_levels = [build_block_gabp_level(level.a, block_dofs=block_dofs) for level in levels[:-1]]
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle_with_block_gabp_direct_fresh(
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


def run_multigrid_with_block_gabp_direct_lamside_reuse(
    levels,
    block_dofs: int,
    b: np.ndarray,
    x_star: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
) -> dict[str, object]:
    """Run geometric MG with direct GBP and lam-side reuse across smoothers."""
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    gabp_levels = [build_block_gabp_level(level.a, block_dofs=block_dofs) for level in levels[:-1]]
    states = [initialize_block_gabp_state(level) for level in gabp_levels]
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle_with_block_gabp_direct_lamside_reuse(
            levels,
            gabp_levels,
            states,
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


def run_multigrid_with_block_gabp_direct(
    levels,
    block_dofs: int,
    b: np.ndarray,
    x_star: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
) -> dict[str, object]:
    """Run a geometric V-cycle while using direct synchronous block GBP as smoother."""
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    gabp_levels = [build_block_gabp_level(level.a, block_dofs=block_dofs) for level in levels[:-1]]
    states = [initialize_block_gabp_state(level) for level in gabp_levels]
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle_with_block_gabp_direct(
            levels,
            gabp_levels,
            states,
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
