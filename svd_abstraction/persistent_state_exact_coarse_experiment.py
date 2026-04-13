from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.sparse

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs, group_list
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_absolute_factors(graph) -> None:
    for factor in graph.factors[: graph.n_factor_nodes]:
        factor.compute_factor_absolute(update_self=True)


def mean_vector(graph) -> np.ndarray:
    return np.concatenate([np.asarray(v.mu).reshape(-1) for v in graph.var_nodes[: graph.n_var_nodes]])


def relative_error_vec(x: np.ndarray, x_star: np.ndarray) -> float:
    return float(np.linalg.norm(x - x_star) / max(np.linalg.norm(x_star), 1e-15))


def init_odom_with_belief_precision(graph, belief_prec: float) -> None:
    """Absolute-state persistent initialization.

    - variable means follow the odometry chain
    - belief/adjacent-belief precision is seeded with a small positive value
    - messages start from zero
    """
    chain_meas: dict[tuple[int, int], np.ndarray] = {}
    for factor in graph.factors[: graph.n_factor_nodes]:
        if getattr(factor, "type", None) != "odometry":
            continue
        i, j = [int(v) for v in factor.adj_vIDs]
        if j == i + 1:
            meas = factor.measurement[0] if isinstance(factor.measurement, list) else factor.measurement
            chain_meas[(i, j)] = np.asarray(meas, dtype=float).reshape(-1)

    mus = {0: np.asarray(graph.var_nodes[0].GT, dtype=float).copy()}
    for i in range(graph.n_var_nodes - 1):
        mus[i + 1] = mus[i] + chain_meas[(i, i + 1)]

    lam0 = belief_prec * np.eye(2, dtype=float)
    sigma0 = np.eye(2, dtype=float) / max(belief_prec, 1e-30)
    for var in graph.var_nodes[: graph.n_var_nodes]:
        mu = mus[int(var.variableID)].copy()
        var.mu = mu
        var.belief.lam = lam0.copy()
        var.belief.eta = lam0 @ mu
        var.Sigma = sigma0.copy()

    for factor in graph.factors[: graph.n_factor_nodes]:
        for adj_var, adj_belief in zip(factor.adj_var_nodes, factor.adj_beliefs):
            adj_belief.lam = np.asarray(adj_var.belief.lam, dtype=float).copy()
            adj_belief.eta = np.asarray(adj_var.belief.eta, dtype=float).copy()
        for msg in factor.messages:
            msg.eta = np.zeros_like(msg.eta)
            msg.lam = np.zeros_like(msg.lam)


def inject_correction_keep_messages(graph, delta: np.ndarray) -> None:
    """Minimal persistent-state transport after coarse correction.

    Keep factor-to-variable messages unchanged, but move the variable-side state
    to the corrected absolute iterate:
    - x <- x + delta
    - belief eta <- belief lam * x
    - adjacent beliefs mirror the variable belief
    """
    x_new = mean_vector(graph) + delta
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


def pin_state_to_x_keep_messages(graph, x_pin: np.ndarray) -> None:
    """Hard-reset variable/belief state to a prescribed geometry, keep messages unchanged."""
    offset = 0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        sl = slice(offset, offset + var.dofs)
        var.mu = np.array(x_pin[sl], copy=True)
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


def inject_correction_transport_binary_messages(graph, delta: np.ndarray, omega: float = 1.0) -> None:
    """Transport coarse correction into beliefs and binary message eta.

    For each variable, prescribe the belief-eta shift
        delta_eta_i = Lambda_i^bel delta_x_i.
    Then, for each binary factor, solve the 2x2 local coupled linear system for
    the two outgoing message-eta increments implied by the fixed-lam Schur
    complement update. Unary factor messages are left unchanged.
    Finally, update beliefs from the transported messages.
    """
    x_old = mean_vector(graph)
    x_new = x_old + delta

    desired_belief_eta: dict[int, np.ndarray] = {}
    offset = 0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        sl = slice(offset, offset + var.dofs)
        dx_i = np.asarray(delta[sl], dtype=float)
        desired_belief_eta[var.variableID] = np.asarray(var.belief.eta, dtype=float) + var.belief.lam @ dx_i
        var.mu = np.array(x_new[sl], copy=True)
        offset += var.dofs

    for factor in graph.factors[: graph.n_factor_nodes]:
        if len(factor.adj_var_nodes) == 1:
            var = factor.adj_var_nodes[0]
            belief_ix = 0
            factor.adj_beliefs[belief_ix].eta = np.array(desired_belief_eta[var.variableID], copy=True)
            factor.adj_beliefs[belief_ix].lam = np.array(var.belief.lam, copy=True)
            continue

        if len(factor.adj_var_nodes) != 2:
            raise NotImplementedError("Only unary/binary factors are supported.")

        v0, v1 = factor.adj_var_nodes
        b0_eta_new = np.array(desired_belief_eta[v0.variableID], copy=True)
        b1_eta_new = np.array(desired_belief_eta[v1.variableID], copy=True)
        db0 = b0_eta_new - factor.adj_beliefs[0].eta
        db1 = b1_eta_new - factor.adj_beliefs[1].eta

        for belief_ix, (var, eta_new) in enumerate(((v0, b0_eta_new), (v1, b1_eta_new))):
            factor.adj_beliefs[belief_ix].eta = np.array(eta_new, copy=True)
            factor.adj_beliefs[belief_ix].lam = np.array(var.belief.lam, copy=True)

        split = v0.dofs
        lam_base = factor.factor.lam

        lam0 = lam_base[:split, :split]
        lam01 = lam_base[:split, split:]
        lam10 = lam_base[split:, :split]
        lam1 = lam_base[split:, split:]

        cav1 = factor.adj_beliefs[1].lam - factor.messages[1].lam
        cav0 = factor.adj_beliefs[0].lam - factor.messages[0].lam
        tilde1 = lam1 + cav1 + 1e-10 * np.eye(v1.dofs)
        tilde0 = lam0 + cav0 + 1e-10 * np.eye(v0.dofs)

        try:
            inv_tilde1 = scipy.linalg.cho_solve(scipy.linalg.cho_factor(tilde1, lower=False, check_finite=False), np.eye(v1.dofs))
        except np.linalg.LinAlgError:
            inv_tilde1 = np.linalg.inv(tilde1)
        try:
            inv_tilde0 = scipy.linalg.cho_solve(scipy.linalg.cho_factor(tilde0, lower=False, check_finite=False), np.eye(v0.dofs))
        except np.linalg.LinAlgError:
            inv_tilde0 = np.linalg.inv(tilde0)

        a = -lam01 @ inv_tilde1
        b = -lam10 @ inv_tilde0

        mmat = np.block(
            [
                [np.eye(v0.dofs), a],
                [b, np.eye(v1.dofs)],
            ]
        )
        rhs = np.concatenate([a @ db1, b @ db0])
        try:
            delta_m = np.linalg.solve(mmat, rhs)
        except np.linalg.LinAlgError:
            delta_m, *_ = np.linalg.lstsq(mmat, rhs, rcond=None)

        factor.messages[0].eta = factor.messages[0].eta + omega * delta_m[: v0.dofs]
        factor.messages[1].eta = factor.messages[1].eta + omega * delta_m[v0.dofs :]

    graph.update_all_beliefs()


def rebalance_message_eta_to_beliefs(graph) -> None:
    """Project incoming message eta onto the current variable belief eta sums."""
    for var in graph.var_nodes[: graph.n_var_nodes]:
        desired = np.asarray(var.belief.eta, dtype=float)
        current = np.asarray(var.prior.eta, dtype=float).copy()
        incoming: list[tuple[object, int]] = []
        for factor in var.adj_factors:
            msg_ix = factor.adj_var_nodes.index(var)
            current += factor.messages[msg_ix].eta
            incoming.append((factor, msg_ix))

        gap = desired - current
        if not incoming:
            continue

        share = gap / len(incoming)
        for factor, msg_ix in incoming:
            factor.messages[msg_ix].eta = factor.messages[msg_ix].eta + share
            factor.adj_beliefs[msg_ix].eta = np.array(var.belief.eta, copy=True)
            factor.adj_beliefs[msg_ix].lam = np.array(var.belief.lam, copy=True)


def eta_balance_reequilibrate(graph, rounds: int, fixed_lam: bool = True) -> None:
    """Naive eta-only re-equilibration with belief-pinned rebalancing."""
    for _ in range(rounds):
        rebalance_message_eta_to_beliefs(graph)
        for factor in graph.factors[: graph.n_factor_nodes]:
            factor.compute_messages(0.0, fixed_lam=fixed_lam)
        rebalance_message_eta_to_beliefs(graph)


def message_only_reequilibrate(graph, sweeps: int, fixed_lam: bool = True) -> None:
    """Update messages only, keeping current beliefs pinned."""
    for _ in range(sweeps):
        graph.compute_all_messages(local_relin=False, fixed_lam=fixed_lam)


def damped_pinned_reequilibrate(
    graph,
    x_pin: np.ndarray,
    rounds: int,
    tau: float,
    alpha: float,
    fixed_lam: bool = True,
) -> None:
    """Small-step message update with belief means pinned near corrected geometry.

    Per round:
    1. damped message update, m <- (1-tau) m + tau T_m
    2. compute belief proposal from prior+messages
    3. only move means a fraction alpha away from x_pin toward the proposal
    """
    msg_damping = 1.0 - tau
    for _ in range(rounds):
        for factor in graph.factors[: graph.n_factor_nodes]:
            factor.compute_messages(msg_damping, fixed_lam=fixed_lam)

        offset = 0
        for var in graph.var_nodes[: graph.n_var_nodes]:
            eta_prop = np.asarray(var.prior.eta, dtype=float).copy()
            lam_prop = np.asarray(var.prior.lam, dtype=float).copy()
            for factor in var.adj_factors:
                msg_ix = factor.adj_var_nodes.index(var)
                eta_prop += factor.messages[msg_ix].eta
                lam_prop += factor.messages[msg_ix].lam

            try:
                chol, lower = scipy.linalg.cho_factor(lam_prop, lower=False, check_finite=False)
                mu_prop = scipy.linalg.cho_solve((chol, lower), eta_prop)
                sigma_prop = scipy.linalg.cho_solve((chol, lower), np.eye(var.dofs))
            except np.linalg.LinAlgError:
                mu_prop = np.linalg.solve(lam_prop, eta_prop)
                sigma_prop = np.linalg.inv(lam_prop)

            sl = slice(offset, offset + var.dofs)
            mu_new = (1.0 - alpha) * x_pin[sl] + alpha * mu_prop
            var.mu = mu_new
            var.Sigma = sigma_prop
            var.belief.lam = lam_prop
            var.belief.eta = lam_prop @ mu_new

            for factor in var.adj_factors:
                belief_ix = factor.adj_var_nodes.index(var)
                factor.adj_beliefs[belief_ix].eta = np.array(var.belief.eta, copy=True)
                factor.adj_beliefs[belief_ix].lam = np.array(var.belief.lam, copy=True)
            offset += var.dofs


def run_schedule(
    *,
    belief_prec: float,
    pre_sweeps: int,
    re_eq_sweeps: int,
    re_eq_fixed_lam: bool,
    cycles: int,
) -> dict[str, object]:
    nodes, edges, exact_graph, _, base_graph = build_graphs(
        n=512,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=0,
    )
    set_absolute_factors(exact_graph)
    set_absolute_factors(base_graph)

    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    eta, a = exact_graph.joint_distribution_inf_absolute()

    init_odom_with_belief_precision(base_graph, belief_prec=belief_prec)

    groups = group_list(
        nodes=nodes,
        graph=base_graph,
        method="order",
        group_size=20,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    level = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups,
        r_reduced=4,
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=True,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(mean_vector(base_graph), x_star),
            "algebraic_residual": float(np.linalg.norm(eta - a @ mean_vector(base_graph))),
        }
    ]

    for cyc in range(1, cycles + 1):
        for _ in range(pre_sweeps):
            base_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_keep_messages(base_graph, delta_x)

        for _ in range(re_eq_sweeps):
            base_graph.synchronous_iteration(fixed_lam=re_eq_fixed_lam)

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, x_star),
                "algebraic_residual": float(np.linalg.norm(eta - a @ x)),
            }
        )

    return {
        "config": {
            "belief_prec": belief_prec,
            "pre_sweeps": pre_sweeps,
            "re_eq_sweeps": re_eq_sweeps,
            "re_eq_fixed_lam": re_eq_fixed_lam,
            "cycles": cycles,
        },
        "history": history,
    }


def run_balance_schedule(
    *,
    belief_prec: float,
    pre_sweeps: int,
    balance_rounds: int,
    cycles: int,
) -> dict[str, object]:
    nodes, edges, exact_graph, _, base_graph = build_graphs(
        n=512,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=0,
    )
    set_absolute_factors(exact_graph)
    set_absolute_factors(base_graph)

    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    eta, a = exact_graph.joint_distribution_inf_absolute()
    init_odom_with_belief_precision(base_graph, belief_prec=belief_prec)

    groups = group_list(
        nodes=nodes,
        graph=base_graph,
        method="order",
        group_size=20,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    level = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups,
        r_reduced=4,
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=True,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(mean_vector(base_graph), x_star),
            "algebraic_residual": float(np.linalg.norm(eta - a @ mean_vector(base_graph))),
        }
    ]

    for cyc in range(1, cycles + 1):
        for _ in range(pre_sweeps):
            base_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_keep_messages(base_graph, delta_x)
        eta_balance_reequilibrate(base_graph, rounds=balance_rounds, fixed_lam=True)

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, x_star),
                "algebraic_residual": float(np.linalg.norm(eta - a @ x)),
            }
        )

    return {
        "config": {
            "belief_prec": belief_prec,
            "pre_sweeps": pre_sweeps,
            "balance_rounds": balance_rounds,
            "cycles": cycles,
        },
        "history": history,
    }


def run_message_only_schedule(
    *,
    belief_prec: float,
    pre_sweeps: int,
    msg_only_sweeps: int,
    msg_only_fixed_lam: bool,
    cycles: int,
) -> dict[str, object]:
    nodes, edges, exact_graph, _, base_graph = build_graphs(
        n=512,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=0,
    )
    set_absolute_factors(exact_graph)
    set_absolute_factors(base_graph)

    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    eta, a = exact_graph.joint_distribution_inf_absolute()
    init_odom_with_belief_precision(base_graph, belief_prec=belief_prec)

    groups = group_list(
        nodes=nodes,
        graph=base_graph,
        method="order",
        group_size=20,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    level = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups,
        r_reduced=4,
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=True,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(mean_vector(base_graph), x_star),
            "algebraic_residual": float(np.linalg.norm(eta - a @ mean_vector(base_graph))),
        }
    ]

    for cyc in range(1, cycles + 1):
        for _ in range(pre_sweeps):
            base_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_keep_messages(base_graph, delta_x)
        message_only_reequilibrate(base_graph, sweeps=msg_only_sweeps, fixed_lam=msg_only_fixed_lam)

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, x_star),
                "algebraic_residual": float(np.linalg.norm(eta - a @ x)),
            }
        )

    return {
        "config": {
            "belief_prec": belief_prec,
            "pre_sweeps": pre_sweeps,
            "msg_only_sweeps": msg_only_sweeps,
            "msg_only_fixed_lam": msg_only_fixed_lam,
            "cycles": cycles,
        },
        "history": history,
    }


def run_transport_schedule(
    *,
    belief_prec: float,
    pre_sweeps: int,
    transport_omega: float,
    cycles: int,
) -> dict[str, object]:
    nodes, edges, exact_graph, _, base_graph = build_graphs(
        n=512,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=0,
    )
    set_absolute_factors(exact_graph)
    set_absolute_factors(base_graph)

    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    eta, a = exact_graph.joint_distribution_inf_absolute()
    init_odom_with_belief_precision(base_graph, belief_prec=belief_prec)

    groups = group_list(
        nodes=nodes,
        graph=base_graph,
        method="order",
        group_size=20,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    level = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups,
        r_reduced=4,
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=True,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(mean_vector(base_graph), x_star),
            "algebraic_residual": float(np.linalg.norm(eta - a @ mean_vector(base_graph))),
        }
    ]

    for cyc in range(1, cycles + 1):
        for _ in range(pre_sweeps):
            base_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_transport_binary_messages(base_graph, delta_x, omega=transport_omega)

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, x_star),
                "algebraic_residual": float(np.linalg.norm(eta - a @ x)),
            }
        )

    return {
        "config": {
            "belief_prec": belief_prec,
            "pre_sweeps": pre_sweeps,
            "transport_omega": transport_omega,
            "cycles": cycles,
        },
        "history": history,
    }


def inject_correction_transport_least_squares(
    graph,
    delta: np.ndarray,
    var_weight: float = 1.0,
    reg: float = 1e-6,
) -> None:
    """Global least-squares message transport after coarse correction.

    Unknowns are binary factor-to-variable message-eta increments. We minimize
    the joint mismatch of:
    1. local fixed-lam factor transport equations
    2. variable belief-sum consistency with the desired belief-eta shift
    """
    x_old = mean_vector(graph)
    x_new = x_old + delta

    desired_belief_eta: dict[int, np.ndarray] = {}
    delta_belief_eta: dict[int, np.ndarray] = {}
    offset = 0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        sl = slice(offset, offset + var.dofs)
        dx_i = np.asarray(delta[sl], dtype=float)
        desired = np.asarray(var.belief.eta, dtype=float) + var.belief.lam @ dx_i
        desired_belief_eta[var.variableID] = desired
        delta_belief_eta[var.variableID] = desired - np.asarray(var.belief.eta, dtype=float)
        var.mu = np.array(x_new[sl], copy=True)
        offset += var.dofs

    unknown_offsets: dict[tuple[int, int], tuple[int, int]] = {}
    n_unknowns = 0
    for factor in graph.factors[: graph.n_factor_nodes]:
        if len(factor.adj_var_nodes) != 2:
            continue
        d0 = factor.adj_var_nodes[0].dofs
        d1 = factor.adj_var_nodes[1].dofs
        unknown_offsets[(factor.factorID, 0)] = (n_unknowns, d0)
        n_unknowns += d0
        unknown_offsets[(factor.factorID, 1)] = (n_unknowns, d1)
        n_unknowns += d1

    rows = []
    cols = []
    data = []
    rhs_parts = []
    row_cursor = 0

    def add_block(row0: int, col0: int, block: np.ndarray) -> None:
        rr, cc = np.nonzero(np.abs(block) > 0.0)
        for r, c in zip(rr, cc):
            rows.append(row0 + int(r))
            cols.append(col0 + int(c))
            data.append(float(block[r, c]))

    for factor in graph.factors[: graph.n_factor_nodes]:
        if len(factor.adj_var_nodes) != 2:
            continue

        v0, v1 = factor.adj_var_nodes
        d0 = v0.dofs
        d1 = v1.dofs
        off0, _ = unknown_offsets[(factor.factorID, 0)]
        off1, _ = unknown_offsets[(factor.factorID, 1)]

        split = d0
        lam_base = factor.factor.lam
        lam0 = lam_base[:split, :split]
        lam01 = lam_base[:split, split:]
        lam10 = lam_base[split:, :split]
        lam1 = lam_base[split:, split:]

        cav1 = factor.adj_beliefs[1].lam - factor.messages[1].lam
        cav0 = factor.adj_beliefs[0].lam - factor.messages[0].lam
        tilde1 = lam1 + cav1 + 1e-10 * np.eye(d1)
        tilde0 = lam0 + cav0 + 1e-10 * np.eye(d0)

        try:
            inv_tilde1 = scipy.linalg.cho_solve(scipy.linalg.cho_factor(tilde1, lower=False, check_finite=False), np.eye(d1))
        except np.linalg.LinAlgError:
            inv_tilde1 = np.linalg.inv(tilde1)
        try:
            inv_tilde0 = scipy.linalg.cho_solve(scipy.linalg.cho_factor(tilde0, lower=False, check_finite=False), np.eye(d0))
        except np.linalg.LinAlgError:
            inv_tilde0 = np.linalg.inv(tilde0)

        a = -lam01 @ inv_tilde1
        b = -lam10 @ inv_tilde0
        db0 = delta_belief_eta[v0.variableID]
        db1 = delta_belief_eta[v1.variableID]

        # dm0 + A dm1 = A db1
        add_block(row_cursor, off0, np.eye(d0))
        add_block(row_cursor, off1, a)
        rhs_parts.append(a @ db1)
        row_cursor += d0

        # B dm0 + dm1 = B db0
        add_block(row_cursor, off0, b)
        add_block(row_cursor, off1, np.eye(d1))
        rhs_parts.append(b @ db0)
        row_cursor += d1

    # Variable consistency rows: sum incoming message increments = desired belief increment.
    for var in graph.var_nodes[: graph.n_var_nodes]:
        d = var.dofs
        for factor in var.adj_factors:
            if len(factor.adj_var_nodes) != 2:
                continue
            msg_ix = factor.adj_var_nodes.index(var)
            off, _ = unknown_offsets[(factor.factorID, msg_ix)]
            add_block(row_cursor, off, np.sqrt(var_weight) * np.eye(d))
        rhs_parts.append(np.sqrt(var_weight) * delta_belief_eta[var.variableID])
        row_cursor += d

    mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(row_cursor, n_unknowns)).tocsc()
    rhs = np.concatenate(rhs_parts)

    normal = (mat.T @ mat).tocsc()
    if reg > 0:
        normal = normal + reg * scipy.sparse.eye(n_unknowns, format="csc")
    rhs_n = mat.T @ rhs

    delta_m = scipy.sparse.linalg.spsolve(normal, rhs_n)
    delta_m = np.asarray(delta_m, dtype=float).reshape(-1)

    for factor in graph.factors[: graph.n_factor_nodes]:
        if len(factor.adj_var_nodes) != 2:
            continue
        off0, d0 = unknown_offsets[(factor.factorID, 0)]
        off1, d1 = unknown_offsets[(factor.factorID, 1)]
        factor.messages[0].eta = factor.messages[0].eta + delta_m[off0 : off0 + d0]
        factor.messages[1].eta = factor.messages[1].eta + delta_m[off1 : off1 + d1]

    graph.update_all_beliefs()


def run_transport_ls_schedule(
    *,
    belief_prec: float,
    pre_sweeps: int,
    var_weight: float,
    reg: float,
    cycles: int,
) -> dict[str, object]:
    nodes, edges, exact_graph, _, base_graph = build_graphs(
        n=512,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=0,
    )
    set_absolute_factors(exact_graph)
    set_absolute_factors(base_graph)

    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    eta, a = exact_graph.joint_distribution_inf_absolute()
    init_odom_with_belief_precision(base_graph, belief_prec=belief_prec)

    groups = group_list(
        nodes=nodes,
        graph=base_graph,
        method="order",
        group_size=20,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    level = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups,
        r_reduced=4,
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=True,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(mean_vector(base_graph), x_star),
            "algebraic_residual": float(np.linalg.norm(eta - a @ mean_vector(base_graph))),
        }
    ]

    for cyc in range(1, cycles + 1):
        for _ in range(pre_sweeps):
            base_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_transport_least_squares(base_graph, delta_x, var_weight=var_weight, reg=reg)

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, x_star),
                "algebraic_residual": float(np.linalg.norm(eta - a @ x)),
            }
        )

    return {
        "config": {
            "belief_prec": belief_prec,
            "pre_sweeps": pre_sweeps,
            "var_weight": var_weight,
            "reg": reg,
            "cycles": cycles,
        },
        "history": history,
    }


def run_pinned_schedule(
    *,
    belief_prec: float,
    pre_sweeps: int,
    rounds: int,
    tau: float,
    alpha: float,
    fixed_lam: bool,
    cycles: int,
) -> dict[str, object]:
    nodes, edges, exact_graph, _, base_graph = build_graphs(
        n=512,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=0,
    )
    set_absolute_factors(exact_graph)
    set_absolute_factors(base_graph)

    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    eta, a = exact_graph.joint_distribution_inf_absolute()
    init_odom_with_belief_precision(base_graph, belief_prec=belief_prec)

    groups = group_list(
        nodes=nodes,
        graph=base_graph,
        method="order",
        group_size=20,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    level = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups,
        r_reduced=4,
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=True,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(mean_vector(base_graph), x_star),
            "algebraic_residual": float(np.linalg.norm(eta - a @ mean_vector(base_graph))),
        }
    ]

    for cyc in range(1, cycles + 1):
        for _ in range(pre_sweeps):
            base_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_keep_messages(base_graph, delta_x)
        x_pin = mean_vector(base_graph).copy()
        damped_pinned_reequilibrate(base_graph, x_pin, rounds=rounds, tau=tau, alpha=alpha, fixed_lam=fixed_lam)

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, x_star),
                "algebraic_residual": float(np.linalg.norm(eta - a @ x)),
            }
        )

    return {
        "config": {
            "belief_prec": belief_prec,
            "pre_sweeps": pre_sweeps,
            "rounds": rounds,
            "tau": tau,
            "alpha": alpha,
            "fixed_lam": fixed_lam,
            "cycles": cycles,
        },
        "history": history,
    }


def run_decoupled_reset_schedule(
    *,
    belief_prec: float,
    pre_sweeps: int,
    rounds: int,
    cycles: int,
) -> dict[str, object]:
    nodes, edges, exact_graph, _, base_graph = build_graphs(
        n=512,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=0,
    )
    set_absolute_factors(exact_graph)
    set_absolute_factors(base_graph)

    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    eta, a = exact_graph.joint_distribution_inf_absolute()
    init_odom_with_belief_precision(base_graph, belief_prec=belief_prec)

    groups = group_list(
        nodes=nodes,
        graph=base_graph,
        method="order",
        group_size=20,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    level = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups,
        r_reduced=4,
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=True,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(mean_vector(base_graph), x_star),
            "algebraic_residual": float(np.linalg.norm(eta - a @ mean_vector(base_graph))),
        }
    ]

    for cyc in range(1, cycles + 1):
        for _ in range(pre_sweeps):
            base_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_keep_messages(base_graph, delta_x)
        x_pin = mean_vector(base_graph).copy()

        for _ in range(rounds):
            base_graph.synchronous_iteration()
            pin_state_to_x_keep_messages(base_graph, x_pin)

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, x_star),
                "algebraic_residual": float(np.linalg.norm(eta - a @ x)),
            }
        )

    return {
        "config": {
            "belief_prec": belief_prec,
            "pre_sweeps": pre_sweeps,
            "rounds": rounds,
            "cycles": cycles,
        },
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--belief-prec", type=float, default=1e-6)
    parser.add_argument("--pre-sweeps", type=int, default=2)
    args = parser.parse_args()

    schedules = [
        ("reeq0", run_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, re_eq_sweeps=0, re_eq_fixed_lam=False, cycles=args.cycles)),
        ("reeq1_sync", run_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, re_eq_sweeps=1, re_eq_fixed_lam=False, cycles=args.cycles)),
        ("reeq2_sync", run_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, re_eq_sweeps=2, re_eq_fixed_lam=False, cycles=args.cycles)),
        ("reeq1_fixedlam", run_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, re_eq_sweeps=1, re_eq_fixed_lam=True, cycles=args.cycles)),
        ("reeq2_fixedlam", run_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, re_eq_sweeps=2, re_eq_fixed_lam=True, cycles=args.cycles)),
        ("balance1_fixedlam", run_balance_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, balance_rounds=1, cycles=args.cycles)),
        ("balance2_fixedlam", run_balance_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, balance_rounds=2, cycles=args.cycles)),
        ("balance5_fixedlam", run_balance_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, balance_rounds=5, cycles=args.cycles)),
        ("balance10_fixedlam", run_balance_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, balance_rounds=10, cycles=args.cycles)),
        ("msgonly2_fixedlam", run_message_only_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, msg_only_sweeps=2, msg_only_fixed_lam=True, cycles=args.cycles)),
        ("msgonly5_fixedlam", run_message_only_schedule(belief_prec=args.belief_prec, pre_sweeps=5, msg_only_sweeps=5, msg_only_fixed_lam=True, cycles=args.cycles)),
        ("msgonly2_full", run_message_only_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, msg_only_sweeps=2, msg_only_fixed_lam=False, cycles=args.cycles)),
        ("transport2_local", run_transport_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, transport_omega=1.0, cycles=args.cycles)),
        ("transport5_local", run_transport_schedule(belief_prec=args.belief_prec, pre_sweeps=5, transport_omega=1.0, cycles=args.cycles)),
        ("transport2_w01", run_transport_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, transport_omega=0.1, cycles=args.cycles)),
        ("transport2_w005", run_transport_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, transport_omega=0.05, cycles=args.cycles)),
        ("transport_ls_v1", run_transport_ls_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, var_weight=1.0, reg=1e-6, cycles=args.cycles)),
        ("transport_ls_v10", run_transport_ls_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, var_weight=10.0, reg=1e-6, cycles=args.cycles)),
        ("pinned2_t01_a01", run_pinned_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, rounds=2, tau=0.1, alpha=0.1, fixed_lam=True, cycles=args.cycles)),
        ("pinned2_t005_a01", run_pinned_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, rounds=2, tau=0.05, alpha=0.1, fixed_lam=True, cycles=args.cycles)),
        ("pinned2_t01_a005", run_pinned_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, rounds=2, tau=0.1, alpha=0.05, fixed_lam=True, cycles=args.cycles)),
        ("decoupled2_reset", run_decoupled_reset_schedule(belief_prec=args.belief_prec, pre_sweeps=args.pre_sweeps, rounds=2, cycles=args.cycles)),
        ("decoupled5_reset", run_decoupled_reset_schedule(belief_prec=args.belief_prec, pre_sweeps=5, rounds=5, cycles=args.cycles)),
    ]

    payload = {name: result for name, result in schedules}
    out_path = OUT_DIR / "persistent_state_exact_coarse_experiment.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    for name, result in schedules:
        print(name)
        for row in result["history"]:
            print(
                row["cycle"],
                row["relative_state_error"],
                row["algebraic_residual"],
            )


if __name__ == "__main__":
    main()
