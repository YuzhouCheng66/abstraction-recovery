"""
Nonlinear SE(2) pose-graph optimization with an outer relinearization loop and
an inner linear Gaussian belief propagation solver.

This module is intentionally different from `gbp.py`:

- `gbp.py` follows the Robot Web Lie-group message-passing style where
  messages are stored as SE(2) means plus precisions in the tangent space of
  those means, and messages are transported between changing local charts.
- This file freezes all factor linearizations during the inner solve and runs
  plain linear GBP on stacked perturbations `delta_i in R^3`.

At a fixed set of linearization points Xbar_i, each factor is a quadratic in
the perturbation variables. The inner GBP loop therefore solves the same
linearized normal equations as a Gauss-Newton step. Only after the inner loop
finishes do we retract the accumulated deltas back to SE(2) and relinearize.
"""

from __future__ import annotations

import manifpy as m
import numpy as np
import scipy.linalg


def _symmetrise(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _solve_linear_system(mat: np.ndarray, rhs: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Solve a dense linear system with SPD-first fallbacks."""
    mat = _symmetrise(np.asarray(mat, dtype=float))
    rhs = np.asarray(rhs, dtype=float)

    if mat.size == 0:
        return np.zeros_like(rhs, dtype=float)

    eye = np.eye(mat.shape[0], dtype=float)
    for scale in (0.0, jitter, 1e-9, 1e-6):
        trial = mat if scale == 0.0 else mat + scale * eye
        try:
            chol, lower = scipy.linalg.cho_factor(trial, lower=False, check_finite=False)
            return scipy.linalg.cho_solve((chol, lower), rhs, check_finite=False)
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            pass
        try:
            return np.linalg.solve(trial, rhs)
        except np.linalg.LinAlgError:
            pass

    sol, *_ = np.linalg.lstsq(mat + 1e-6 * eye, rhs, rcond=None)
    return sol


def _sigma_vec(s, theta_ratio: float = 0.01) -> np.ndarray:
    v = np.array(s, dtype=float).ravel()
    if v.size == 1:
        s0 = float(v.item())
        return np.array([s0, s0, s0 * theta_ratio], dtype=float)
    if v.size != 3:
        raise ValueError("sigma must be scalar or length-3 [sx, sy, sth]")
    return v.astype(float)


def _wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class LinearMessage:
    """Information-form Gaussian message in a fixed Euclidean delta space."""

    def __init__(self, dofs: int, eta=None, lam=None):
        self.eta = np.zeros(dofs, dtype=float) if eta is None else np.array(eta, dtype=float).reshape(dofs)
        if lam is None:
            self.lam = np.zeros((dofs, dofs), dtype=float)
        else:
            self.lam = _symmetrise(np.array(lam, dtype=float).reshape(dofs, dofs))

    def reset_eta(self) -> None:
        self.eta.fill(0.0)

    def reset(self) -> None:
        self.eta.fill(0.0)
        self.lam.fill(0.0)


class DeltaVariableNode:
    """
    Variable node whose nonlinear state lives on SE(2) but whose belief update
    in the inner loop is always over a Euclidean perturbation delta in R^3.
    """

    def __init__(self, variable_id: int, dofs: int = 3, tiny: float = 1e-12):
        self.variableID = variable_id
        self.dofs = dofs
        self.tiny = tiny

        self.adj_factors = []
        self.to_factor_messages: list[LinearMessage] = []

        self.status = m.SE2(0.0, 0.0, 0.0)
        self.GT = None

        self.delta = np.zeros(dofs, dtype=float)
        self.Lam = np.eye(dofs, dtype=float) * tiny

    def reset_local_state(self) -> None:
        self.delta.fill(0.0)
        self.Lam = np.eye(self.dofs, dtype=float) * self.tiny

    def update_belief(self) -> None:
        eta_all = np.zeros(self.dofs, dtype=float)
        lam_all = np.eye(self.dofs, dtype=float) * self.tiny
        incoming = []

        for factor in self.adj_factors:
            message_ix = factor.adj_var_nodes.index(self)
            msg = factor.to_variable_messages[message_ix]
            eta_all += msg.eta
            lam_all += msg.lam
            incoming.append((factor, message_ix, msg))

        lam_all = _symmetrise(lam_all)
        self.delta = _solve_linear_system(lam_all, eta_all, jitter=self.tiny)
        self.Lam = lam_all

        for factor, message_ix, msg in incoming:
            cavity = self.to_factor_messages[self.adj_factors.index(factor)]
            cavity.eta = eta_all - msg.eta
            cavity.lam = _symmetrise(lam_all - msg.lam)


class LinearisedSE2Factor:
    """Base class for SE(2) factors linearised around fixed manifold states."""

    def __init__(
        self,
        factor_id: int,
        adj_var_nodes: list[DeltaVariableNode],
        measurement,
        measurement_lambda,
        robustify: bool = False,
        tiny: float = 1e-12,
    ):
        self.factorID = factor_id
        self.adj_var_nodes = adj_var_nodes
        self.measurement = measurement
        self.measurement_lambda = np.array(measurement_lambda, dtype=float)
        self.robustify = robustify
        self.tiny = tiny

        self.threshold = 1e8
        self.linpoints = []

        total_dofs = sum(v.dofs for v in self.adj_var_nodes)
        self.factor_eta = np.zeros(total_dofs, dtype=float)
        self.factor_Lam = np.eye(total_dofs, dtype=float) * tiny

        self.last_messages_eta = [np.zeros(v.dofs, dtype=float) for v in self.adj_var_nodes]
        self.last_messages_Lam = [np.zeros((v.dofs, v.dofs), dtype=float) for v in self.adj_var_nodes]
        self.to_variable_messages = [LinearMessage(v.dofs) for v in self.adj_var_nodes]

        self._block_slices = []
        offset = 0
        for var in self.adj_var_nodes:
            self._block_slices.append(slice(offset, offset + var.dofs))
            offset += var.dofs

    def robust_kernel(self, error: float) -> float:
        if error > self.threshold:
            return self.threshold / error
        return 1.0

    def set_factor_eta_lam(self, h: np.ndarray, jac: np.ndarray) -> None:
        if self.robustify:
            error = float(h.T @ self.measurement_lambda @ h)
            scale = self.robust_kernel(error)
        else:
            scale = 1.0

        factor_eta = scale * jac.T @ self.measurement_lambda @ (-h)
        factor_lam = scale * jac.T @ self.measurement_lambda @ jac

        self.factor_eta = np.array(factor_eta, dtype=float).reshape(-1)
        self.factor_Lam = _symmetrise(np.array(factor_lam, dtype=float)) + np.eye(jac.shape[1]) * self.tiny

    def linearise(self) -> bool:
        raise NotImplementedError

    def loss(self) -> float:
        raise NotImplementedError

    def _conditioned_joint(self) -> tuple[np.ndarray, np.ndarray]:
        eta_c = self.factor_eta.copy()
        lam_c = self.factor_Lam.copy()

        for i, adj_var_node in enumerate(self.adj_var_nodes):
            message_ix = adj_var_node.adj_factors.index(self)
            msg = adj_var_node.to_factor_messages[message_ix]
            sl = self._block_slices[i]

            eta_c[sl] += msg.eta
            lam_c[sl, sl] += msg.lam

        return eta_c, _symmetrise(lam_c)

    def compute_messages(self, damping: float = 0.0) -> None:
        eta_joint, lam_joint = self._conditioned_joint()

        for i, adj_var_node in enumerate(self.adj_var_nodes):
            sl_out = self._block_slices[i]
            out_idx = np.arange(sl_out.start, sl_out.stop, dtype=int)
            other_idx = np.concatenate(
                [
                    np.arange(sl.start, sl.stop, dtype=int)
                    for j, sl in enumerate(self._block_slices)
                    if j != i
                ]
            ) if len(self._block_slices) > 1 else np.zeros(0, dtype=int)

            message_ix = adj_var_node.adj_factors.index(self)
            incoming = adj_var_node.to_factor_messages[message_ix]

            eta_d = eta_joint.copy()
            lam_d = lam_joint.copy()
            eta_d[sl_out] -= incoming.eta
            lam_d[sl_out, sl_out] -= incoming.lam
            lam_d = _symmetrise(lam_d)

            eo = eta_d[out_idx]
            loo = lam_d[np.ix_(out_idx, out_idx)]

            if other_idx.size == 0:
                msg_eta = eo
                msg_lam = loo
            else:
                eno = eta_d[other_idx]
                lono = lam_d[np.ix_(out_idx, other_idx)]
                lnoo = lam_d[np.ix_(other_idx, out_idx)]
                lnono = lam_d[np.ix_(other_idx, other_idx)]

                rhs = np.concatenate([lnoo, eno.reshape(-1, 1)], axis=1)
                solved = _solve_linear_system(lnono, rhs, jitter=self.tiny)
                solved_lam = solved[:, : out_idx.size]
                solved_eta = solved[:, -1]

                msg_lam = loo - lono @ solved_lam
                msg_eta = eo - lono @ solved_eta

            msg_lam = _symmetrise(msg_lam)
            if damping > 0.0:
                msg_eta = (1.0 - damping) * msg_eta + damping * self.last_messages_eta[i]
                msg_lam = (1.0 - damping) * msg_lam + damping * self.last_messages_Lam[i]
                msg_lam = _symmetrise(msg_lam)

            self.last_messages_eta[i] = np.array(msg_eta, copy=True)
            self.last_messages_Lam[i] = np.array(msg_lam, copy=True)
            self.to_variable_messages[i].eta = np.array(msg_eta, copy=True)
            self.to_variable_messages[i].lam = np.array(msg_lam, copy=True)


class PriorSE2Factor(LinearisedSE2Factor):
    def residual_prior(self, x, p):
        jac = np.zeros((3, 3), dtype=float)
        ret = p.minus(x, None, jac).coeffs()
        return np.array(ret, dtype=float).reshape(-1), jac

    def linearise(self) -> bool:
        h, jac = self.residual_prior(self.linpoints[0], self.measurement)
        self.set_factor_eta_lam(h, jac)
        return True

    def loss(self) -> float:
        h, _ = self.residual_prior(self.adj_var_nodes[0].status, self.measurement)
        return float(h.T @ self.measurement_lambda @ h)


class BetweenSE2Factor(LinearisedSE2Factor):
    def between(self, x1, x2):
        jac1 = np.zeros((3, 3), dtype=float)
        jac2 = np.zeros((3, 3), dtype=float)
        ret = x1.between(x2, jac1, jac2)

        jac = np.zeros((3, 6), dtype=float)
        jac[:, 0:3] = jac1
        jac[:, 3:6] = jac2
        return ret, jac

    def residual_between(self, x1, x2, meas):
        bt, jac = self.between(x1, x2)
        jac_r = np.zeros((3, 3), dtype=float)
        ret = meas.rminus(bt, None, jac_r).coeffs()
        return np.array(ret, dtype=float).reshape(-1), jac_r @ jac

    def linearise(self) -> bool:
        h, jac = self.residual_between(self.linpoints[0], self.linpoints[1], self.measurement)
        self.set_factor_eta_lam(h, jac)
        return True

    def loss(self) -> float:
        h, _ = self.residual_between(self.adj_var_nodes[0].status, self.adj_var_nodes[1].status, self.measurement)
        return float(h.T @ self.measurement_lambda @ h)


class NonlinearSE2GBPGraph:
    """
    Nonlinear outer loop around a linear inner GBP solver on fixed perturbations.

    Warm-start policy:
    - graph topology persists forever;
    - `reset_messages("full")` is the mathematically clean default after each
      relinearization because the delta coordinates changed with the new base
      points;
    - `reset_messages("eta")` is an optional heuristic that preserves the
      lam-side message state while resetting eta-side messages.
    """

    def __init__(self, message_damping: float = 0.0, tiny: float = 1e-12):
        self.message_damping = message_damping
        self.tiny = tiny

        self.var_nodes: list[DeltaVariableNode] = []
        self.factors: list[LinearisedSE2Factor] = []
        self.n_var_nodes = 0
        self.n_factor_nodes = 0

    def reset_messages(self, mode: str = "full") -> None:
        if mode not in {"full", "eta"}:
            raise ValueError("mode must be one of {'full', 'eta'}")

        for var in self.var_nodes:
            var.reset_local_state()
            for msg in var.to_factor_messages:
                if mode == "full":
                    msg.reset()
                elif mode == "eta":
                    msg.reset_eta()

        for factor in self.factors:
            for msg in factor.to_variable_messages:
                if mode == "full":
                    msg.reset()
                elif mode == "eta":
                    msg.reset_eta()

            for i in range(len(factor.last_messages_eta)):
                if mode in {"full", "eta"}:
                    factor.last_messages_eta[i].fill(0.0)
                if mode == "full":
                    factor.last_messages_Lam[i].fill(0.0)

    def relinearise_factors(self, reset_messages: str = "full") -> None:
        for factor in self.factors:
            factor.linpoints = [var.status for var in factor.adj_var_nodes]
            factor.linearise()
        self.reset_messages(mode=reset_messages)

    def compute_all_messages(self) -> None:
        for factor in self.factors:
            factor.compute_messages(damping=self.message_damping)

    def update_all_beliefs(self) -> None:
        for var in self.var_nodes:
            var.update_belief()

    def inner_iteration(self) -> None:
        self.compute_all_messages()
        self.update_all_beliefs()

    def run_inner(self, iters: int) -> None:
        for _ in range(int(iters)):
            self.inner_iteration()

    def current_delta_vector(self) -> np.ndarray:
        return np.concatenate([var.delta for var in self.var_nodes], axis=0)

    def apply_deltas(self, step_size: float = 1.0) -> None:
        for var in self.var_nodes:
            var.status = var.status + m.SE2Tangent(step_size * var.delta)

    def outer_iteration(self, inner_iters: int, step_size: float = 1.0, reset_messages: str = "full") -> None:
        self.relinearise_factors(reset_messages=reset_messages)
        self.run_inner(inner_iters)
        self.apply_deltas(step_size=step_size)

    def optimise(
        self,
        outer_iters: int,
        inner_iters: int,
        step_size: float = 1.0,
        reset_messages: str = "full",
    ) -> None:
        for _ in range(int(outer_iters)):
            self.outer_iteration(inner_iters=inner_iters, step_size=step_size, reset_messages=reset_messages)

    def joint_distribution_inf(self) -> tuple[np.ndarray, np.ndarray]:
        var_ix = np.zeros(len(self.var_nodes), dtype=int)
        total_dofs = 0
        for var in self.var_nodes:
            var_ix[var.variableID] = total_dofs
            total_dofs += var.dofs

        eta = np.zeros(total_dofs, dtype=float)
        lam = np.zeros((total_dofs, total_dofs), dtype=float)

        for factor in self.factors:
            factor_offset = 0
            for adj_var in factor.adj_var_nodes:
                v_id = adj_var.variableID
                dofs = adj_var.dofs
                start = var_ix[v_id]
                stop = start + dofs

                eta[start:stop] += factor.factor_eta[factor_offset : factor_offset + dofs]
                lam[start:stop, start:stop] += factor.factor_Lam[
                    factor_offset : factor_offset + dofs,
                    factor_offset : factor_offset + dofs,
                ]

                other_offset = 0
                for other_adj_var in factor.adj_var_nodes:
                    if other_adj_var.variableID > adj_var.variableID:
                        other_start = var_ix[other_adj_var.variableID]
                        other_stop = other_start + other_adj_var.dofs
                        lam[start:stop, other_start:other_stop] += factor.factor_Lam[
                            factor_offset : factor_offset + dofs,
                            other_offset : other_offset + other_adj_var.dofs,
                        ]
                        lam[other_start:other_stop, start:stop] += factor.factor_Lam[
                            other_offset : other_offset + other_adj_var.dofs,
                            factor_offset : factor_offset + dofs,
                        ]
                    other_offset += other_adj_var.dofs
                factor_offset += dofs

        return eta, _symmetrise(lam)

    def joint_distribution_cov(self) -> tuple[np.ndarray, np.ndarray]:
        eta, lam = self.joint_distribution_inf()
        sigma = _solve_linear_system(lam, np.eye(lam.shape[0], dtype=float), jitter=self.tiny)
        sigma = np.array(sigma, dtype=float)
        if sigma.ndim == 1:
            sigma = sigma.reshape(1, 1)
        mu = sigma @ eta
        return mu, _symmetrise(sigma)

    def direct_solve_delta(self) -> np.ndarray:
        eta, lam = self.joint_distribution_inf()
        return _solve_linear_system(lam, eta, jitter=self.tiny)

    def linear_residual(self) -> np.ndarray:
        eta, lam = self.joint_distribution_inf()
        return eta - lam @ self.current_delta_vector()

    def linear_residual_norm(self) -> float:
        return float(np.linalg.norm(self.linear_residual()))

    def delta_error_to_direct(self) -> float:
        return float(np.linalg.norm(self.current_delta_vector() - self.direct_solve_delta()))

    def loss(self) -> float:
        return 0.5 * sum(factor.loss() for factor in self.factors)

    def SE(self) -> float:
        squared_error = 0.0
        for var in self.var_nodes:
            squared_error += (var.status - var.GT).squaredWeightedNorm()
        return float(squared_error)


def build_nonlinear_se2_pose_graph(
    nodes,
    edges,
    prior_sigma: float | tuple | np.ndarray = 1.0,
    odom_sigma: float | tuple | np.ndarray = 1.0,
    loop_sigma: float | tuple | np.ndarray = 1.0,
    theta_ratio: float = 0.01,
    seed=None,
    message_damping: float = 0.0,
    init_mode: str = "chain",
) -> NonlinearSE2GBPGraph:
    """
    Build a nonlinear SE(2) graph whose inner solve operates on perturbations.

    `init_mode="chain"` mirrors the current repo:
    - anchor node 0 strongly,
    - propagate sequential odometry,
    - reset at prior nodes when they exist.
    """

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    graph = NonlinearSE2GBPGraph(message_damping=message_damping)

    var_nodes = []
    for i, node in enumerate(nodes):
        var = DeltaVariableNode(i, dofs=3, tiny=graph.tiny)
        th = float(node["data"].get("theta", 0.0))
        var.GT = m.SE2(node["position"]["x"], node["position"]["y"], th)
        var.status = m.SE2(0.0, 0.0, 0.0)
        var_nodes.append(var)

    graph.var_nodes = var_nodes
    graph.n_var_nodes = len(var_nodes)

    prior_sigma_vec = _sigma_vec(prior_sigma, theta_ratio=theta_ratio)
    odom_sigma_vec = _sigma_vec(odom_sigma, theta_ratio=theta_ratio)
    loop_sigma_vec = _sigma_vec(loop_sigma, theta_ratio=theta_ratio)

    lambda_prior = np.diag(1.0 / (prior_sigma_vec ** 2))
    lambda_odom = np.diag(1.0 / (odom_sigma_vec ** 2))
    lambda_loop = np.diag(1.0 / (loop_sigma_vec ** 2))
    lambda_anchor = np.diag(1.0 / (np.array([1e-3, 1e-3, 1e-5], dtype=float) ** 2))

    odom_meas = {}
    prior_meas = {}
    factors = []
    fid = 0

    # Strong anchor at node 0.
    v0 = var_nodes[0]
    factors.append(
        PriorSE2Factor(
            fid,
            [v0],
            measurement=v0.GT,
            measurement_lambda=lambda_anchor,
            robustify=False,
            tiny=graph.tiny,
        )
    )
    v0.adj_factors.append(factors[-1])
    fid += 1

    for edge in edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]

        if dst != "prior":
            i, j = int(src), int(dst)
            z = np.array(edge["data"]["z"], dtype=float).ravel()
            kind = edge["data"].get("kind", "between")

            if kind == "loop":
                noise_vec = rng.normal(0.0, loop_sigma_vec, size=3)
                this_lambda = lambda_loop
            else:
                noise_vec = rng.normal(0.0, odom_sigma_vec, size=3)
                this_lambda = lambda_odom

            z_noisy = z.copy()
            z_noisy[:2] += noise_vec[:2]
            z_noisy[2] = _wrap_angle(z_noisy[2] + noise_vec[2])
            z_noisy_se2 = m.SE2(z_noisy[0], z_noisy[1], z_noisy[2])

            if kind == "odom" and (j == i + 1):
                odom_meas[(i, j)] = z_noisy_se2

            vi, vj = var_nodes[i], var_nodes[j]
            factor = BetweenSE2Factor(
                fid,
                [vi, vj],
                measurement=z_noisy_se2,
                measurement_lambda=this_lambda,
                robustify=False,
                tiny=graph.tiny,
            )
            factors.append(factor)
            vi.adj_factors.append(factor)
            vj.adj_factors.append(factor)
            fid += 1
        else:
            i = int(src)
            z = [*var_nodes[i].GT.translation(), var_nodes[i].GT.angle()]
            noise = rng.normal(0.0, prior_sigma_vec, size=3)
            z[:2] += noise[:2]
            z[2] = _wrap_angle(z[2] + noise[2])
            z_se2 = m.SE2(z[0], z[1], z[2])
            prior_meas[i] = z_se2

            vi = var_nodes[i]
            factor = PriorSE2Factor(
                fid,
                [vi],
                measurement=z_se2,
                measurement_lambda=lambda_prior,
                robustify=False,
                tiny=graph.tiny,
            )
            factors.append(factor)
            vi.adj_factors.append(factor)
            fid += 1

    if init_mode == "chain":
        var_nodes[0].status = var_nodes[0].GT
        for i in range(len(var_nodes) - 1):
            if (i, i + 1) in odom_meas:
                var_nodes[i + 1].status = var_nodes[i].status * odom_meas[(i, i + 1)]
            else:
                var_nodes[i + 1].status = var_nodes[i].status
            if (i + 1) in prior_meas:
                var_nodes[i + 1].status = prior_meas[i + 1]
    elif init_mode == "gt":
        for var in var_nodes:
            var.status = var.GT
    else:
        raise ValueError("init_mode must be 'chain' or 'gt'")

    graph.factors = factors
    graph.n_factor_nodes = len(factors)

    for var in var_nodes:
        var.to_factor_messages = [LinearMessage(var.dofs) for _ in var.adj_factors]

    graph.relinearise_factors(reset_messages="full")
    return graph
