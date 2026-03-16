"""
Lean GBP core for SVD residual-abstraction experiments.

This file intentionally starts closer to `hierarchy/gbp/gbp.py` than to the
full `raylib_gbp` multigrid implementation. The goal is to keep the single-layer
GBP machinery small and understandable, while exposing the minimum hooks needed
for the upcoming residual-coarse-correction work.
"""

import heapq

import numpy as np
import scipy.linalg

from svd_abstraction.utils.gaussian import NdimGaussian


class FactorGraph:
    def __init__(
        self,
        nonlinear_factors=True,
        eta_damping=0.0,
        beta=None,
        num_undamped_iters=None,
        min_linear_iters=None,
        wild_thresh=0.0,
    ):
        self.var_nodes = []
        self.factors = []

        self.n_var_nodes = 0
        self.n_factor_nodes = 0
        self.n_msgs = 0

        self.nonlinear_factors = nonlinear_factors
        self.eta_damping = eta_damping
        self.wild_thresh = wild_thresh

        self.mus = []
        self.var_residual = {}
        self.var_heap = []

        if nonlinear_factors:
            self.beta = beta
            self.num_undamped_iters = num_undamped_iters
            self.min_linear_iters = min_linear_iters

    def compute_all_messages(self, factors=None, local_relin=True, fixed_lam=False):
        if factors is None:
            factors = self.factors[: self.n_factor_nodes]

        for factor in factors:
            if not factor.active:
                continue
            if self.nonlinear_factors and local_relin:
                if factor.iters_since_relin == self.num_undamped_iters:
                    factor.eta_damping = self.eta_damping
                factor.compute_messages(factor.eta_damping, fixed_lam=fixed_lam)
            else:
                factor.compute_messages(self.eta_damping, fixed_lam=fixed_lam)
            self.n_msgs += len(factor.adj_var_nodes)

    def update_all_beliefs(self, vars=None):
        if vars is None:
            vars = self.var_nodes[: self.n_var_nodes]
        for var in vars:
            if var.active:
                var.update_belief()

    def compute_all_factors(self, factors=None):
        if factors is None:
            factors = self.factors[: self.n_factor_nodes]
        for factor in factors:
            factor.compute_factor()

    def update_all_residuals(self, vars=None):
        if vars is None:
            vars = self.var_nodes[: self.n_var_nodes]
        for var in vars:
            if var.active:
                var.compute_residual()

    def energy(self, vars=None):
        if vars is None:
            vars = self.var_nodes[: self.n_var_nodes]
        energy = 0.0
        for var in vars:
            if var.active:
                energy += 0.5 * np.linalg.norm(var.compute_residual()) ** 2
        return energy

    def relinearise_factors(self):
        if not self.nonlinear_factors:
            return

        for factor in self.factors:
            adj_belief_means = np.concatenate([belief.mu() for belief in factor.adj_beliefs])
            if (
                np.linalg.norm(factor.linpoint - adj_belief_means) > self.beta
                and factor.iters_since_relin >= self.min_linear_iters
            ):
                factor.compute_factor(linpoint=adj_belief_means)
                factor.iters_since_relin = 0
                factor.eta_damping = 0.0
            else:
                factor.iters_since_relin += 1

    def synchronous_iteration(self, factors=None, local_relin=True, fixed_lam=False):
        if factors is None:
            factors = self.factors[: self.n_factor_nodes]

        if self.nonlinear_factors and local_relin:
            self.relinearise_factors()

        self.compute_all_messages(factors, local_relin=local_relin, fixed_lam=fixed_lam)
        self.update_all_beliefs()

    def push_var(self, var, residual=1.0):
        residual = float(residual)
        self.var_residual[var] = residual
        heapq.heappush(self.var_heap, (-residual, var.variableID, var))

    def residual_iteration_var_heap(self, max_updates=50):
        """
        Variable-priority GBP scheduler.

        This is not the same as multilevel defect correction yet, but it gives
        us a clean residual-aware single-layer update order without bringing in
        the whole AMG stack.
        """

        if len(self.var_heap) == 0:
            for var in self.var_nodes[: self.n_var_nodes]:
                self.var_residual[var] = 0.0
                heapq.heappush(self.var_heap, (-0.0, var.variableID, var))

        n_updates = 0
        while self.var_heap and n_updates < max_updates:
            neg_residual, _, var = heapq.heappop(self.var_heap)
            residual = -neg_residual
            current_residual = self.var_residual.get(var, 0.0)

            if abs(residual - current_residual) > 1e-12:
                continue

            for factor in var.adj_factors:
                factor.compute_messages(self.eta_damping)
                for other_var in factor.adj_var_nodes:
                    old_eta = np.array(other_var.belief.eta, copy=True)
                    other_var.update_belief()
                    new_eta = np.array(other_var.belief.eta, copy=True)

                    est_residual = float(np.linalg.norm(new_eta - old_eta))
                    old_residual = self.var_residual.get(other_var, 0.0)
                    if est_residual > old_residual + 1e-12:
                        self.var_residual[other_var] = est_residual
                        heapq.heappush(self.var_heap, (-est_residual, other_var.variableID, other_var))

            self.var_residual[var] = 0.0
            n_updates += 1

    def joint_distribution_inf(self):
        sizes = [var.dofs for var in self.var_nodes[: self.n_var_nodes]]
        total = sum(sizes)

        eta = np.empty(total, dtype=float)
        lam = np.zeros((total, total), dtype=float)
        var_ix = np.empty(self.n_var_nodes, dtype=int)

        offset = 0
        for var in self.var_nodes[: self.n_var_nodes]:
            dofs = var.dofs
            var_ix[var.variableID] = offset
            eta[offset : offset + dofs] = var.prior.eta
            lam[offset : offset + dofs, offset : offset + dofs] = var.prior.lam
            offset += dofs

        for factor in self.factors[: self.n_factor_nodes]:
            factor_offset = 0
            for adj_var in factor.adj_var_nodes:
                v_id = adj_var.variableID
                dofs = adj_var.dofs
                start = var_ix[v_id]
                stop = start + dofs

                eta[start:stop] += factor.factor.eta[factor_offset : factor_offset + dofs]
                lam[start:stop, start:stop] += factor.factor.lam[
                    factor_offset : factor_offset + dofs,
                    factor_offset : factor_offset + dofs,
                ]

                other_offset = 0
                for other_adj_var in factor.adj_var_nodes:
                    if other_adj_var.variableID > adj_var.variableID:
                        other_start = var_ix[other_adj_var.variableID]
                        other_stop = other_start + other_adj_var.dofs
                        lam[start:stop, other_start:other_stop] += factor.factor.lam[
                            factor_offset : factor_offset + dofs,
                            other_offset : other_offset + other_adj_var.dofs,
                        ]
                        lam[other_start:other_stop, start:stop] += factor.factor.lam[
                            other_offset : other_offset + other_adj_var.dofs,
                            factor_offset : factor_offset + dofs,
                        ]
                    other_offset += other_adj_var.dofs
                factor_offset += dofs

        return eta, lam

    def joint_distribution_cov(self):
        eta, lam = self.joint_distribution_inf()
        sigma = np.linalg.inv(lam)
        mu = sigma @ eta
        return mu, sigma


class VariableNode:
    def __init__(self, variable_id, dofs):
        self.variableID = variable_id
        self.adj_factors = []
        self.InfoMat = []
        self.EtaVec = []
        self.type = "variable"
        self.active = True

        self.mu = np.zeros(dofs)
        self.Sigma = np.eye(dofs) * 1e12
        self.residual = np.zeros(dofs)
        self.belief = NdimGaussian(dofs)
        self.prior = NdimGaussian(dofs)
        self.dofs = dofs

    def update_belief(self):
        eta = self.prior.eta.copy()
        lam = self.prior.lam.copy()

        for factor in self.adj_factors:
            message_ix = factor.adj_var_nodes.index(self)
            eta += factor.messages[message_ix].eta
            lam += factor.messages[message_ix].lam

        self.belief.eta = eta
        self.belief.lam = lam

        try:
            chol, lower = scipy.linalg.cho_factor(lam, lower=False, check_finite=False)
            self.mu = scipy.linalg.cho_solve((chol, lower), eta)
            self.Sigma = scipy.linalg.cho_solve((chol, lower), np.eye(self.dofs))
        except np.linalg.LinAlgError:
            self.mu = np.linalg.solve(lam, eta)
            self.Sigma = np.linalg.inv(lam)

        for factor in self.adj_factors:
            belief_ix = factor.adj_var_nodes.index(self)
            factor.adj_beliefs[belief_ix].eta = self.belief.eta
            factor.adj_beliefs[belief_ix].lam = self.belief.lam

    def compute_residual(self):
        res = self.prior.eta - self.prior.lam @ self.mu
        for factor in self.adj_factors:
            x = np.concatenate([np.asarray(var.mu).reshape(-1) for var in factor.adj_var_nodes])
            start = 0
            stop = self.dofs
            for var in factor.adj_var_nodes:
                if var is self:
                    stop = start + var.dofs
                    break
                start += var.dofs
            res += factor.factor.eta[start:stop] - factor.factor.lam[start:stop, :] @ x

        self.residual = res
        return res

    def __repr__(self):
        return f"VariableNode({self.variableID})"


class Factor:
    def __init__(
        self,
        factor_id,
        adj_var_nodes,
        measurement,
        measurement_lambda,
        meas_fn,
        jac_fn,
        loss=None,
        mahalanobis_threshold=2,
        wildfire=False,
        *args,
    ):
        self.factorID = factor_id
        self.adj_var_nodes = adj_var_nodes
        self.adj_vIDs = []
        self.adj_beliefs = []
        self.messages = []
        self.messages_prior = []
        self.messages_dist = []
        self.dofs_conditional_vars = 0

        self.active = True
        self.type = "factor"
        self.level = 0
        self.in_queue = False

        for var in self.adj_var_nodes:
            self.dofs_conditional_vars += var.dofs
            self.adj_vIDs.append(var.variableID)
            self.adj_beliefs.append(NdimGaussian(var.dofs))
            self.messages.append(NdimGaussian(var.dofs))
            self.messages_prior.append(NdimGaussian(var.dofs))
            self.messages_dist.append(np.zeros(var.dofs))

        self.factor = NdimGaussian(self.dofs_conditional_vars)
        self.linpoint = np.zeros(self.dofs_conditional_vars)
        self.residual = None
        self.b_calc_mess_dist = wildfire

        self.measurement = measurement
        self.measurement_lambda = measurement_lambda
        self.meas_fn = meas_fn
        self.jac_fn = jac_fn
        self.args = args
        self.loss = loss
        self.mahalanobis_threshold = mahalanobis_threshold
        self.robust_flag = False

        self.eta_damping = 0.0
        self.iters_since_relin = 1

    def compute_residual(self):
        x = np.concatenate([belief.mu() for belief in self.adj_beliefs])
        self.residual = self.factor.eta - self.factor.lam @ x
        return self.residual

    def energy(self):
        return 0.5 * np.linalg.norm(self.compute_residual()) ** 2

    def compute_factor(self, linpoint=None, update_self=True):
        if linpoint is None:
            self.linpoint = np.concatenate([np.asarray(var.mu).reshape(-1) for var in self.adj_var_nodes])
        else:
            self.linpoint = np.asarray(linpoint).reshape(-1)

        J = self.jac_fn(self.linpoint, *self.args)
        pred_measurement = self.meas_fn(self.linpoint, *self.args)

        lambda_factor = np.zeros_like(self.factor.lam)
        eta_factor = np.zeros_like(self.factor.eta)

        for jac_block, lam_z, meas, pred in zip(J, self.measurement_lambda, self.measurement, pred_measurement):
            lambda_factor += jac_block.T @ lam_z @ jac_block
            eta_factor += jac_block.T @ (lam_z @ (jac_block @ self.linpoint + meas - pred))

        if update_self:
            self.factor.eta = eta_factor
            self.factor.lam = lambda_factor

        return eta_factor, lambda_factor

    def compute_messages(self, eta_damping, fixed_lam=False):
        if len(self.adj_vIDs) == 1:
            self.messages[0].eta = self.factor.eta.copy()
            if not fixed_lam:
                self.messages[0].lam = self.factor.lam.copy()
            return

        if len(self.adj_vIDs) != 2:
            raise NotImplementedError("The current svd_abstraction GBP core supports unary and binary factors only.")

        messages_eta = []
        messages_lam = []
        split = self.adj_var_nodes[0].dofs

        for target_idx in range(2):
            eta_factor = self.factor.eta.copy()
            lam_factor = self.factor.lam.copy()

            offset = 0
            for other_idx, other_var in enumerate(self.adj_var_nodes):
                dofs = other_var.dofs
                if other_idx != target_idx:
                    eta_factor[offset : offset + dofs] += self.adj_beliefs[other_idx].eta - self.messages[other_idx].eta
                    lam_factor[offset : offset + dofs, offset : offset + dofs] += (
                        self.adj_beliefs[other_idx].lam - self.messages[other_idx].lam
                    )
                offset += dofs

            if target_idx == 0:
                eta_o = eta_factor[:split]
                eta_no = eta_factor[split:]
                lam_oo = lam_factor[:split, :split]
                lam_ono = lam_factor[:split, split:]
                lam_noo = lam_factor[split:, :split]
                lam_nono = lam_factor[split:, split:]
            else:
                eta_o = eta_factor[split:]
                eta_no = eta_factor[:split]
                lam_oo = lam_factor[split:, split:]
                lam_ono = lam_factor[split:, :split]
                lam_noo = lam_factor[:split, split:]
                lam_nono = lam_factor[:split, :split]

            lam_nono = lam_nono + 1e-10 * np.eye(lam_nono.shape[0])
            rhs = np.concatenate([lam_noo, eta_no.reshape(-1, 1)], axis=1)
            try:
                chol, lower = scipy.linalg.cho_factor(lam_nono, lower=False, check_finite=False)
                solved = scipy.linalg.cho_solve((chol, lower), rhs)
            except np.linalg.LinAlgError:
                solved = np.linalg.solve(lam_nono, rhs)

            solved_lam = solved[:, : lam_noo.shape[1]]
            solved_eta = solved[:, -1]

            new_lam = lam_oo - lam_ono @ solved_lam
            new_eta = eta_o - lam_ono @ solved_eta

            if fixed_lam:
                messages_lam.append(self.messages[target_idx].lam.copy())
            else:
                messages_lam.append((1.0 - eta_damping) * new_lam + eta_damping * self.messages[target_idx].lam)
            messages_eta.append((1.0 - eta_damping) * new_eta + eta_damping * self.messages[target_idx].eta)

        for idx in range(2):
            self.messages[idx].eta = messages_eta[idx]
            self.messages[idx].lam = messages_lam[idx]

    def __repr__(self):
        return f"Factor({self.factorID})"
