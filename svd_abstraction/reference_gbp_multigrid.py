"""Geometric multigrid integrations that use the reference GBP core exactly."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg
from scipy.sparse import linalg as spla

from svd_abstraction.gbp_from_operator import build_reference_gbp_graph_from_operator
from svd_abstraction.gbp_from_operator import decompose_block_operator
from svd_abstraction.gbp_from_operator import distribute_eta_to_pairwise_lstsq
from svd_abstraction.gbp_from_operator import graph_mean_vector


@dataclass
class ReferenceGBPLevelState:
    level: object
    block_dofs: int
    graph: object
    pair_eta_mode: str = "unary"


def _stack_message_lam(graph) -> np.ndarray:
    blocks = []
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg in factor.messages:
            blocks.append(np.array(msg.lam, copy=True))
    if not blocks:
        return np.zeros((0, 0, 0), dtype=float)
    return np.stack(blocks, axis=0)


def _copy_graph_lam_side(dst_graph, src_graph) -> None:
    for dst_factor, src_factor in zip(
        dst_graph.factors[: dst_graph.n_factor_nodes],
        src_graph.factors[: src_graph.n_factor_nodes],
    ):
        for dst_msg, src_msg in zip(dst_factor.messages, src_factor.messages):
            dst_msg.lam = np.array(src_msg.lam, copy=True)
        for dst_adj, src_adj in zip(dst_factor.adj_beliefs, src_factor.adj_beliefs):
            dst_adj.lam = np.array(src_adj.lam, copy=True)

    for dst_var, src_var in zip(
        dst_graph.var_nodes[: dst_graph.n_var_nodes],
        src_graph.var_nodes[: src_graph.n_var_nodes],
    ):
        dst_var.belief.lam = np.array(src_var.belief.lam, copy=True)
        dst_var.Sigma = np.array(src_var.Sigma, copy=True)


def _set_graph_rhs(
    graph,
    eta_new: np.ndarray,
    lam_fixed,
    block_dofs: int,
    pair_eta_mode: str = "unary",
) -> None:
    unary_eta, _, pair_weights = decompose_block_operator(eta_new, lam_fixed, block_dofs)
    if pair_eta_mode == "unary":
        pair_eta = [np.zeros(block_dofs, dtype=float) for _ in pair_weights]
    elif pair_eta_mode == "incidence_lstsq":
        unary_eta, pair_eta = distribute_eta_to_pairwise_lstsq(unary_eta, pair_weights, block_dofs)
    else:
        raise ValueError(f"Unknown pair_eta_mode: {pair_eta_mode}")
    for var, local_eta in zip(graph.var_nodes[: graph.n_var_nodes], unary_eta):
        var.prior.eta = np.array(local_eta, copy=True)
    for factor, factor_eta_edge in zip(graph.factors[: graph.n_factor_nodes], pair_eta):
        factor.factor.eta = np.concatenate([-factor_eta_edge, factor_eta_edge])


def _reset_graph_all_state(graph) -> None:
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg, adj in zip(factor.messages, factor.adj_beliefs):
            msg.eta[:] = 0.0
            msg.lam[:] = 0.0
            adj.eta[:] = 0.0
            adj.lam[:] = 0.0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        var.belief.eta[:] = 0.0
        var.belief.lam[:] = 0.0
        var.mu[:] = 0.0
        var.Sigma[:] = 0.0
    graph.update_all_beliefs()


def _reset_graph_eta_only_keep_lam(graph) -> None:
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg, adj in zip(factor.messages, factor.adj_beliefs):
            msg.eta[:] = 0.0
            adj.eta[:] = 0.0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        var.belief.eta[:] = 0.0
        var.mu[:] = 0.0


def _reset_graph_eta_keep_lam_with_prior_eta(graph) -> None:
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg in factor.messages:
            msg.eta[:] = 0.0

    for var in graph.var_nodes[: graph.n_var_nodes]:
        var.belief.eta = np.array(var.prior.eta, copy=True)
        try:
            chol, lower = scipy.linalg.cho_factor(var.belief.lam, lower=False, check_finite=False)
            var.mu = scipy.linalg.cho_solve((chol, lower), var.belief.eta)
            var.Sigma = scipy.linalg.cho_solve((chol, lower), np.eye(var.dofs))
        except np.linalg.LinAlgError:
            var.mu = np.linalg.solve(var.belief.lam, var.belief.eta)
            var.Sigma = np.linalg.inv(var.belief.lam)

    for var in graph.var_nodes[: graph.n_var_nodes]:
        for factor in var.adj_factors:
            belief_ix = factor.adj_var_nodes.index(var)
            factor.adj_beliefs[belief_ix].eta = np.array(var.belief.eta, copy=True)
            factor.adj_beliefs[belief_ix].lam = np.array(var.belief.lam, copy=True)


def _reset_graph_eta_keep_lam_rebuild_beliefs(graph) -> None:
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg, adj in zip(factor.messages, factor.adj_beliefs):
            msg.eta[:] = 0.0
            adj.eta[:] = 0.0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        var.belief.eta[:] = 0.0
        var.mu[:] = 0.0
    graph.update_all_beliefs()


def preconverge_graph_lam_side(graph, tol: float = 1e-10, max_iters: int = 500) -> int:
    """Iterate a zero-RHS reference graph until message lam converges."""
    prev = _stack_message_lam(graph)
    if prev.size == 0:
        return 0
    for it in range(1, max_iters + 1):
        graph.synchronous_iteration()
        cur = _stack_message_lam(graph)
        delta = float(np.max(np.abs(cur - prev)))
        prev = cur
        if delta < tol:
            return it
    return max_iters


def build_reference_gbp_level_states(
    levels,
    block_dofs: int,
    preconverge_lam: bool = False,
    lam_tol: float = 1e-10,
    lam_max_iters: int = 500,
    include_top: bool = False,
    pair_eta_mode: str = "unary",
) -> list[ReferenceGBPLevelState]:
    states: list[ReferenceGBPLevelState] = []
    level_iter = levels if include_top else levels[:-1]
    for level in level_iter:
        graph = build_reference_gbp_graph_from_operator(
            np.zeros(level.a.shape[0], dtype=float),
            level.a,
            block_dofs=block_dofs,
            pair_eta_mode=pair_eta_mode,
        )
        if preconverge_lam:
            preconverge_graph_lam_side(graph, tol=lam_tol, max_iters=lam_max_iters)
        states.append(
            ReferenceGBPLevelState(
                level=level,
                block_dofs=block_dofs,
                graph=graph,
                pair_eta_mode=pair_eta_mode,
            )
        )
    return states


def build_reference_gbp_level_states_raylib_style(
    levels,
    block_dofs: int,
    level0_eta: np.ndarray,
    preconverge_lam: bool = False,
    lam_tol: float = 1e-10,
    lam_max_iters: int = 500,
    pair_eta_mode: str = "incidence_lstsq",
) -> list[ReferenceGBPLevelState]:
    """Build persistent level graphs with raylib-style semantics.

    Level 0 is a persistent absolute-state graph built on the true RHS.
    Levels >= 1 are persistent correction graphs built on zero RHS and later
    receive restricted residuals through prior.eta updates.
    """
    states: list[ReferenceGBPLevelState] = []
    for level_idx, level in enumerate(levels):
        eta = level0_eta if level_idx == 0 else np.zeros(level.a.shape[0], dtype=float)
        graph = build_reference_gbp_graph_from_operator(
            eta,
            level.a,
            block_dofs=block_dofs,
            pair_eta_mode=pair_eta_mode,
        )
        if preconverge_lam:
            lam_graph = build_reference_gbp_graph_from_operator(
                np.zeros(level.a.shape[0], dtype=float),
                level.a,
                block_dofs=block_dofs,
                pair_eta_mode=pair_eta_mode,
            )
            preconverge_graph_lam_side(lam_graph, tol=lam_tol, max_iters=lam_max_iters)
            _copy_graph_lam_side(graph, lam_graph)
            graph.update_all_beliefs()
        states.append(
            ReferenceGBPLevelState(
                level=level,
                block_dofs=block_dofs,
                graph=graph,
                pair_eta_mode=pair_eta_mode,
            )
        )
    return states


def smooth_reference_gbp_fresh(level, x: np.ndarray, b: np.ndarray, sweeps: int, block_dofs: int) -> np.ndarray:
    """Fresh-reference GBP smoother on the current defect equation.

    This is step-wise identical to running the reference `FactorGraph`
    synchronous GBP core on the defect equation of this level.
    """
    if sweeps <= 0:
        return x
    rhs = b - level.a @ x
    graph = build_reference_gbp_graph_from_operator(rhs, level.a, block_dofs=block_dofs)
    for _ in range(sweeps):
        graph.synchronous_iteration()
    return x + graph_mean_vector(graph)


def smooth_reference_gbp_reuse(
    state: ReferenceGBPLevelState,
    x: np.ndarray,
    b: np.ndarray,
    sweeps: int,
    mode: str = "reset_all",
    fixed_lam: bool = False,
) -> np.ndarray:
    """Reference GBP smoother with a persistent per-level FactorGraph.

    Modes:
      * reset_all: reuse graph object but reset all message/belief state
      * keep_lam_side: preserve lam-side state, clear eta/mu-side state
      * keep_lam_side_prior_eta: preserve lam-side state, zero message eta,
        but reinitialize belief eta from the current prior eta
      * keep_lam_side_rebuild_beliefs: preserve lam-side state, zero eta-side
        messages, then rebuild beliefs through the reference update logic
    """
    if sweeps <= 0:
        return x
    rhs = b - state.level.a @ x
    _set_graph_rhs(
        state.graph,
        rhs,
        state.level.a,
        state.block_dofs,
        pair_eta_mode=state.pair_eta_mode,
    )
    if mode == "reset_all":
        _reset_graph_all_state(state.graph)
    elif mode == "keep_lam_side":
        _reset_graph_eta_only_keep_lam(state.graph)
    elif mode == "keep_lam_side_prior_eta":
        _reset_graph_eta_keep_lam_with_prior_eta(state.graph)
    elif mode == "keep_lam_side_rebuild_beliefs":
        _reset_graph_eta_keep_lam_rebuild_beliefs(state.graph)
    else:
        raise ValueError(f"Unknown reuse mode: {mode}")
    for _ in range(sweeps):
        state.graph.synchronous_iteration(fixed_lam=fixed_lam)
    return x + graph_mean_vector(state.graph)


def _set_graph_mean_keep_lam(graph, x_new: np.ndarray) -> None:
    """Inject an absolute/correction state into a persistent graph, preserving lam-side."""
    offset = 0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        sl = slice(offset, offset + var.dofs)
        var.mu = np.array(x_new[sl], copy=True)
        try:
            chol, lower = scipy.linalg.cho_factor(var.belief.lam, lower=False, check_finite=False)
            var.Sigma = scipy.linalg.cho_solve((chol, lower), np.eye(var.dofs))
        except np.linalg.LinAlgError:
            var.Sigma = np.linalg.inv(var.belief.lam)
        var.belief.eta = var.belief.lam @ var.mu
        for factor in var.adj_factors:
            belief_ix = factor.adj_var_nodes.index(var)
            factor.adj_beliefs[belief_ix].eta = np.array(var.belief.eta, copy=True)
            factor.adj_beliefs[belief_ix].lam = np.array(var.belief.lam, copy=True)
        offset += var.dofs


def smooth_reference_gbp_absolute_level0(
    state: ReferenceGBPLevelState,
    x: np.ndarray,
    b: np.ndarray,
    sweeps: int,
    fixed_lam: bool = False,
    reset_all: bool = False,
) -> np.ndarray:
    """Raylib-style absolute smoother for level 0.

    Level 0 represents the absolute variable state, not a defect/correction
    variable. We therefore set the true RHS `b`, inject the current absolute
    iterate into the persistent graph, and continue synchronous GBP updates.
    """
    if sweeps <= 0:
        return x
    if reset_all:
        _set_graph_rhs(
            state.graph,
            b,
            state.level.a,
            state.block_dofs,
            pair_eta_mode=state.pair_eta_mode,
        )
        _reset_graph_all_state(state.graph)
    # In the raylib-style path, level 0 is a persistent absolute graph whose
    # internal messages/beliefs already encode the current iterate. Do not
    # re-inject `x` every cycle; just continue iterating the persistent graph.
    for _ in range(sweeps):
        state.graph.synchronous_iteration(fixed_lam=fixed_lam)
    return graph_mean_vector(state.graph)


def _inject_graph_correction_keep_lam(graph, delta: np.ndarray) -> np.ndarray:
    """Apply a correction to a persistent graph state, preserving lam-side."""
    x_new = graph_mean_vector(graph) + delta
    _set_graph_mean_keep_lam(graph, x_new)
    return x_new


def run_reference_gbp_direct(
    eta: np.ndarray,
    lam,
    block_dofs: int,
    x_star: np.ndarray,
    tol: float,
    max_iters: int,
) -> dict[str, object]:
    """Run the reference GBP core directly on a fixed linear system."""
    graph = build_reference_gbp_graph_from_operator(eta, lam, block_dofs=block_dofs)
    x = np.zeros_like(eta)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(eta - lam @ x))]
    error_history = [float(initial_error)]
    t0 = perf_counter()

    for it in range(1, max_iters + 1):
        graph.synchronous_iteration()
        x = graph_mean_vector(graph)
        residual_history.append(float(np.linalg.norm(eta - lam @ x)))
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


def v_cycle_with_reference_gbp_raylib_style(
    levels,
    level_states: list[ReferenceGBPLevelState],
    level_idx: int,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
    mode: str = "keep_lam_side_prior_eta",
    fixed_lam: bool = False,
    correction_damping: float = 1.0,
    top_level_solver: str = "iterative",
) -> np.ndarray:
    """Reference-core V-cycle with raylib-style state semantics.

    - level 0 is an absolute variable layer
    - levels >= 1 are residual/correction layers
    - only coarse levels use the eta-reset reuse modes
    - level 0 has no post-smoothing, matching raylib's current V-cycle
    """
    if top_level_solver not in {"iterative", "direct"}:
        raise ValueError(f"Unknown top_level_solver: {top_level_solver}")

    state = level_states[level_idx]
    level = state.level
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / level_states[0].level.a.shape[0])

    is_top_level = level_idx == len(level_states) - 1

    if level_idx == 0:
        x0 = graph_mean_vector(state.graph)
        x0 = smooth_reference_gbp_absolute_level0(
            state,
            x0,
            b,
            pre_sweeps,
            fixed_lam=fixed_lam,
            reset_all=(mode == "reset_all"),
        )
        if is_top_level:
            return x0
        residual = b - level.a @ x0
        coarse_rhs = level.r @ residual
        coarse_error = v_cycle_with_reference_gbp_raylib_style(
            levels,
            level_states,
            level_idx + 1,
            coarse_rhs,
            pre_sweeps,
            post_sweeps,
            stats,
            mode=mode,
            fixed_lam=fixed_lam,
            correction_damping=correction_damping,
            top_level_solver=top_level_solver,
        )
        x0 = _inject_graph_correction_keep_lam(state.graph, correction_damping * (level.p @ coarse_error))
        return x0

    e = graph_mean_vector(state.graph)
    e = smooth_reference_gbp_reuse(
        state,
        e,
        b,
        pre_sweeps,
        mode=mode,
        fixed_lam=fixed_lam,
    )
    if is_top_level:
        if top_level_solver == "direct":
            stats["coarse_solves"] += 1
            e = spla.spsolve(level.a, b)
            _set_graph_mean_keep_lam(state.graph, e)
            return e
        e = smooth_reference_gbp_reuse(
            state,
            e,
            b,
            post_sweeps,
            mode=mode,
            fixed_lam=fixed_lam,
        )
        return e
    residual = b - level.a @ e
    coarse_rhs = level.r @ residual
    coarse_error = v_cycle_with_reference_gbp_raylib_style(
        levels,
        level_states,
        level_idx + 1,
        coarse_rhs,
        pre_sweeps,
        post_sweeps,
        stats,
        mode=mode,
        fixed_lam=fixed_lam,
        correction_damping=correction_damping,
        top_level_solver=top_level_solver,
    )
    e = _inject_graph_correction_keep_lam(state.graph, correction_damping * (level.p @ coarse_error))
    e = smooth_reference_gbp_reuse(
        state,
        e,
        b,
        post_sweeps,
        mode=mode,
        fixed_lam=fixed_lam,
    )
    return e


def v_cycle_with_reference_gbp_fresh(
    levels,
    block_dofs: int,
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
) -> np.ndarray:
    """Geometric V-cycle using the reference GBP core as fresh smoother."""
    level = levels[level_idx]
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / levels[0].a.shape[0])

    if level_idx == len(levels) - 1:
        stats["coarse_solves"] += 1
        return spla.spsolve(level.a, b)

    x = smooth_reference_gbp_fresh(level, x, b, pre_sweeps, block_dofs=block_dofs)
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_with_reference_gbp_fresh(
        levels,
        block_dofs,
        level_idx + 1,
        coarse_error,
        coarse_rhs,
        pre_sweeps,
        post_sweeps,
        stats,
    )
    x = x + level.p @ coarse_error
    x = smooth_reference_gbp_fresh(level, x, b, post_sweeps, block_dofs=block_dofs)
    return x


def v_cycle_with_reference_gbp_reuse(
    levels,
    level_states: list[ReferenceGBPLevelState],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    stats: dict[str, float],
    mode: str = "reset_all",
    fixed_lam: bool = False,
    correction_damping: float = 1.0,
) -> np.ndarray:
    if level_idx == len(level_states):
        level = levels[level_idx]
        stats["coarse_solves"] += 1
        return spla.spsolve(level.a, b)

    level = level_states[level_idx].level
    stats["smoothing_work"] += (pre_sweeps + post_sweeps) * (level.a.shape[0] / level_states[0].level.a.shape[0])

    x = smooth_reference_gbp_reuse(
        level_states[level_idx], x, b, pre_sweeps, mode=mode, fixed_lam=fixed_lam
    )
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_with_reference_gbp_reuse(
        levels,
        level_states,
        level_idx + 1,
        coarse_error,
        coarse_rhs,
        pre_sweeps,
        post_sweeps,
        stats,
        mode=mode,
        fixed_lam=fixed_lam,
        correction_damping=correction_damping,
    )
    x = x + correction_damping * (level.p @ coarse_error)
    x = smooth_reference_gbp_reuse(
        level_states[level_idx], x, b, post_sweeps, mode=mode, fixed_lam=fixed_lam
    )
    return x


def run_multigrid_with_reference_gbp_fresh(
    levels,
    block_dofs: int,
    b: np.ndarray,
    x_star: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
) -> dict[str, object]:
    """Run geometric multigrid with the exact reference GBP smoother."""
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle_with_reference_gbp_fresh(
            levels,
            block_dofs,
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


def run_multigrid_with_reference_gbp_reuse(
    levels,
    block_dofs: int,
    b: np.ndarray,
    x_star: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
    mode: str = "reset_all",
    preconverge_lam: bool = False,
    fixed_lam: bool = False,
    lam_tol: float = 1e-10,
    lam_max_iters: int = 500,
    correction_damping: float = 1.0,
    top_level_solver: str = "iterative",
    pair_eta_mode: str = "unary",
) -> dict[str, object]:
    """Run geometric MG with persistent per-level reference FactorGraphs."""
    x = np.zeros_like(b)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    level_states = build_reference_gbp_level_states(
        levels,
        block_dofs=block_dofs,
        preconverge_lam=preconverge_lam,
        lam_tol=lam_tol,
        lam_max_iters=lam_max_iters,
        include_top=False,
        pair_eta_mode=pair_eta_mode,
    )
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle_with_reference_gbp_reuse(
            levels,
            level_states,
            0,
            x,
            b,
            pre_sweeps=pre_sweeps,
            post_sweeps=post_sweeps,
            stats=stats,
            mode=mode,
            fixed_lam=fixed_lam,
            correction_damping=correction_damping,
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


def run_multigrid_with_reference_gbp_raylib_style(
    levels,
    block_dofs: int,
    b: np.ndarray,
    x_star: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    tol: float,
    max_cycles: int,
    mode: str = "keep_lam_side_prior_eta",
    preconverge_lam: bool = False,
    fixed_lam: bool = False,
    lam_tol: float = 1e-10,
    lam_max_iters: int = 500,
    correction_damping: float = 1.0,
    top_level_solver: str = "iterative",
    pair_eta_mode: str = "incidence_lstsq",
) -> dict[str, object]:
    """Reference-core multigrid with raylib-style level semantics."""
    level_states = build_reference_gbp_level_states_raylib_style(
        levels,
        block_dofs=block_dofs,
        level0_eta=b,
        preconverge_lam=preconverge_lam,
        lam_tol=lam_tol,
        lam_max_iters=lam_max_iters,
        pair_eta_mode=pair_eta_mode,
    )
    x = graph_mean_vector(level_states[0].graph)
    initial_error = np.linalg.norm(x - x_star)
    residual_history = [float(np.linalg.norm(b - levels[0].a @ x))]
    error_history = [float(initial_error)]
    stats = {"smoothing_work": 0.0, "coarse_solves": 0.0}
    t0 = perf_counter()

    for cycle in range(1, max_cycles + 1):
        x = v_cycle_with_reference_gbp_raylib_style(
            levels,
            level_states,
            0,
            b,
            pre_sweeps,
            post_sweeps,
            stats,
            mode=mode,
            fixed_lam=fixed_lam,
            correction_damping=correction_damping,
            top_level_solver=top_level_solver,
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
