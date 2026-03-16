"""Residual abstraction with frozen SVD bases and coarse GBP sweeps."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from svd_abstraction.gbp.gbp import Factor
from svd_abstraction.gbp.gbp import FactorGraph
from svd_abstraction.gbp.gbp import VariableNode


@dataclass
class CycleStats:
    residual_norm_before: float
    residual_norm_after: float
    coarse_residual_norm: float
    correction_norm: float


class SVDResidualAbstraction:
    """
    Base -> abs residual correction without a separate iterative super layer.

    Each group owns a local basis `B_g`, and the global prolongation matrix is
    the block diagonal composition `P = blkdiag(B_1, ..., B_G)`.
    """

    def __init__(
        self,
        base_graph,
        groups,
        r_reduced=2,
        basis_source="belief_covariance",
        freeze_basis=True,
        ridge=1e-10,
    ):
        self.base_graph = base_graph
        self.groups = [list(group) for group in groups]
        self.r_reduced = r_reduced
        self.basis_source = basis_source
        self.freeze_basis = freeze_basis
        self.ridge = ridge

        self.var_slices = self._build_var_slices()
        self.groups = self._normalize_groups(self.groups)

        self.group_dims = []
        self.group_full_dofs = []
        self.group_reduced_slices = []
        self.var_to_group = {}
        self.var_local_rows = {}
        self.total_reduced_dim = 0
        self.Bs = []
        self.P = None
        self.bases_initialized = False
        self.coarse_graph = None
        self.coarse_var_nodes = []

    def _build_var_slices(self):
        var_slices = {}
        offset = 0
        for var in self.base_graph.var_nodes[: self.base_graph.n_var_nodes]:
            var_slices[var.variableID] = slice(offset, offset + var.dofs)
            offset += var.dofs
        return var_slices

    def _normalize_groups(self, groups):
        known_ids = {var.variableID for var in self.base_graph.var_nodes[: self.base_graph.n_var_nodes]}
        normalized = []
        covered = set()

        for group in groups:
            clean_group = [int(var_id) for var_id in group if int(var_id) in known_ids]
            if clean_group:
                normalized.append(clean_group)
                covered.update(clean_group)

        for var_id in sorted(known_ids - covered):
            normalized.append([var_id])

        return normalized

    def current_mean_vector(self):
        parts = [np.asarray(var.mu).reshape(-1) for var in self.base_graph.var_nodes[: self.base_graph.n_var_nodes]]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    def average_error(self):
        errs = []
        for var in self.base_graph.var_nodes[: self.base_graph.n_var_nodes]:
            gt = getattr(var, "GT", None)
            if gt is None:
                continue
            errs.append(np.linalg.norm(np.asarray(var.mu) - np.asarray(gt)))
        return float(np.mean(errs)) if errs else 0.0

    def warmup(self, iterations=5, scheduler="sync", fixed_lam=False):
        for _ in range(iterations):
            if scheduler == "sync":
                self.base_graph.synchronous_iteration(fixed_lam=fixed_lam)
            elif scheduler == "residual":
                self.base_graph.residual_iteration_var_heap(max_updates=self.base_graph.n_var_nodes)
            else:
                raise ValueError(f"Unknown scheduler: {scheduler}")

    def _source_matrices(self):
        if self.basis_source == "joint_information":
            _, lam = self.base_graph.joint_distribution_inf()
            return lam
        if self.basis_source == "joint_covariance":
            _, sigma = self.base_graph.joint_distribution_cov()
            return sigma
        return None

    def _group_block(self, group_var_ids, source_matrix=None):
        if self.basis_source == "belief_covariance":
            blocks = [self.base_graph.var_nodes[var_id].Sigma for var_id in group_var_ids]
            return scipy.linalg.block_diag(*blocks)
        if self.basis_source == "belief_information":
            blocks = [self.base_graph.var_nodes[var_id].belief.lam for var_id in group_var_ids]
            return scipy.linalg.block_diag(*blocks)

        dof_indices = []
        for var_id in group_var_ids:
            sl = self.var_slices[var_id]
            dof_indices.extend(range(sl.start, sl.stop))
        dof_indices = np.array(dof_indices, dtype=int)
        return source_matrix[np.ix_(dof_indices, dof_indices)]

    def _basis_from_block(self, block):
        dim = block.shape[0]
        r = min(self.r_reduced, dim)
        if r == dim:
            return np.eye(dim)

        eigvals, eigvecs = np.linalg.eigh(block)
        if self.basis_source.endswith("covariance"):
            order = np.argsort(eigvals)[::-1]
        else:
            order = np.argsort(eigvals)
        return eigvecs[:, order[:r]]

    def _refresh_group_maps(self):
        self.var_to_group = {}
        self.var_local_rows = {}
        for group_idx, group_var_ids in enumerate(self.groups):
            local_offset = 0
            for var_id in group_var_ids:
                dofs = self.base_graph.var_nodes[var_id].dofs
                self.var_to_group[var_id] = group_idx
                self.var_local_rows[var_id] = slice(local_offset, local_offset + dofs)
                local_offset += dofs

    def _projection_rows(self, var_id):
        group_idx = self.var_to_group[var_id]
        local_rows = self.var_local_rows[var_id]
        return group_idx, self.Bs[group_idx][local_rows, :]

    def _new_static_factor(self, factor_id, adj_var_nodes, lam):
        factor = Factor(
            factor_id,
            adj_var_nodes,
            [np.zeros(1, dtype=float)],
            [np.eye(1, dtype=float)],
            lambda x, *args: [np.zeros(1, dtype=float)],
            lambda x, *args: [np.zeros((1, x.shape[0]), dtype=float)],
        )
        factor.factor.eta = np.zeros(lam.shape[0], dtype=float)
        factor.factor.lam = 0.5 * (lam + lam.T)
        return factor

    def initialize_bases(self, force=False):
        if self.bases_initialized and self.freeze_basis and not force:
            return

        source_matrix = self._source_matrices()
        self.Bs = []
        self.group_dims = []
        self.group_full_dofs = []
        self.group_reduced_slices = []

        reduced_offset = 0
        total_full_dim = sum(var.dofs for var in self.base_graph.var_nodes[: self.base_graph.n_var_nodes])

        for group_var_ids in self.groups:
            block = self._group_block(group_var_ids, source_matrix=source_matrix)
            basis = self._basis_from_block(block)
            self.Bs.append(basis)
            self.group_full_dofs.append(group_var_ids)
            self.group_dims.append(block.shape[0])
            self.group_reduced_slices.append(slice(reduced_offset, reduced_offset + basis.shape[1]))
            reduced_offset += basis.shape[1]

        self.total_reduced_dim = reduced_offset
        self.P = np.zeros((total_full_dim, self.total_reduced_dim), dtype=float)

        for group_var_ids, basis, reduced_slice in zip(self.groups, self.Bs, self.group_reduced_slices):
            full_indices = []
            for var_id in group_var_ids:
                sl = self.var_slices[var_id]
                full_indices.extend(range(sl.start, sl.stop))
            self.P[np.array(full_indices, dtype=int), reduced_slice] = basis

        self.bases_initialized = True
        self._refresh_group_maps()
        self.coarse_graph = None
        self.coarse_var_nodes = []

    def joint_system(self):
        eta, lam = self.base_graph.joint_distribution_inf()
        x = self.current_mean_vector()
        residual = eta - lam @ x
        return x, eta, lam, residual

    def coarse_system(self):
        if not self.bases_initialized:
            self.initialize_bases()

        _, _, lam, residual = self.joint_system()
        coarse_lam = self.P.T @ lam @ self.P
        coarse_lam = 0.5 * (coarse_lam + coarse_lam.T)
        coarse_residual = self.P.T @ residual
        return coarse_lam, coarse_residual

    def build_coarse_graph(self, force=False):
        if not self.bases_initialized:
            self.initialize_bases()

        if self.coarse_graph is not None and self.freeze_basis and not force:
            return

        coarse_graph = FactorGraph(nonlinear_factors=False, eta_damping=self.base_graph.eta_damping)
        coarse_var_nodes = []

        for group_idx, reduced_slice in enumerate(self.group_reduced_slices):
            dim = reduced_slice.stop - reduced_slice.start
            coarse_var = VariableNode(group_idx, dofs=dim)
            coarse_var.type = "coarse"
            coarse_var.prior.lam = np.eye(dim, dtype=float) * 1e-12
            coarse_var.prior.eta = np.zeros(dim, dtype=float)
            coarse_var_nodes.append(coarse_var)

        coarse_graph.var_nodes = coarse_var_nodes
        coarse_graph.n_var_nodes = len(coarse_var_nodes)

        factor_id = 0

        for var in self.base_graph.var_nodes[: self.base_graph.n_var_nodes]:
            group_idx, proj_rows = self._projection_rows(var.variableID)
            lam = proj_rows.T @ var.prior.lam @ proj_rows
            if np.linalg.norm(lam) <= 1e-14:
                continue
            factor = self._new_static_factor(factor_id, [coarse_var_nodes[group_idx]], lam)
            coarse_var_nodes[group_idx].adj_factors.append(factor)
            coarse_graph.factors.append(factor)
            factor_id += 1

        for factor in self.base_graph.factors[: self.base_graph.n_factor_nodes]:
            adj_vars = factor.adj_var_nodes
            if len(adj_vars) == 1:
                base_var = adj_vars[0]
                group_idx, proj_rows = self._projection_rows(base_var.variableID)
                lam = proj_rows.T @ factor.factor.lam @ proj_rows
                if np.linalg.norm(lam) <= 1e-14:
                    continue
                coarse_factor = self._new_static_factor(factor_id, [coarse_var_nodes[group_idx]], lam)
                coarse_var_nodes[group_idx].adj_factors.append(coarse_factor)
                coarse_graph.factors.append(coarse_factor)
                factor_id += 1
                continue

            if len(adj_vars) != 2:
                raise NotImplementedError("Only unary and binary base factors are supported in the coarse projection.")

            var_i, var_j = adj_vars
            group_i, proj_i = self._projection_rows(var_i.variableID)
            group_j, proj_j = self._projection_rows(var_j.variableID)

            if group_i == group_j:
                proj = np.vstack((proj_i, proj_j))
                lam = proj.T @ factor.factor.lam @ proj
                if np.linalg.norm(lam) <= 1e-14:
                    continue
                coarse_factor = self._new_static_factor(factor_id, [coarse_var_nodes[group_i]], lam)
                coarse_var_nodes[group_i].adj_factors.append(coarse_factor)
                coarse_graph.factors.append(coarse_factor)
                factor_id += 1
                continue

            dim_i = proj_i.shape[1]
            dim_j = proj_j.shape[1]
            proj = np.zeros((var_i.dofs + var_j.dofs, dim_i + dim_j), dtype=float)
            proj[: var_i.dofs, :dim_i] = proj_i
            proj[var_i.dofs :, dim_i:] = proj_j
            lam = proj.T @ factor.factor.lam @ proj
            if np.linalg.norm(lam) <= 1e-14:
                continue

            coarse_factor = self._new_static_factor(
                factor_id,
                [coarse_var_nodes[group_i], coarse_var_nodes[group_j]],
                lam,
            )
            coarse_var_nodes[group_i].adj_factors.append(coarse_factor)
            coarse_var_nodes[group_j].adj_factors.append(coarse_factor)
            coarse_graph.factors.append(coarse_factor)
            factor_id += 1

        coarse_graph.n_factor_nodes = len(coarse_graph.factors)
        for coarse_var in coarse_graph.var_nodes[: coarse_graph.n_var_nodes]:
            coarse_var.update_belief()

        self.coarse_graph = coarse_graph
        self.coarse_var_nodes = coarse_var_nodes

    def update_coarse_residual_eta(self):
        if self.coarse_graph is None:
            self.build_coarse_graph()

        _, _, _, residual = self.joint_system()
        coarse_residual = self.P.T @ residual

        for group_idx, coarse_var in enumerate(self.coarse_var_nodes):
            reduced_slice = self.group_reduced_slices[group_idx]
            coarse_var.prior.eta = coarse_residual[reduced_slice].copy()

        return coarse_residual

    def coarse_mean_vector(self):
        if self.coarse_graph is None:
            self.build_coarse_graph()
        parts = [np.asarray(var.mu).reshape(-1) for var in self.coarse_var_nodes]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

    def solve_coarse_correction(self):
        coarse_lam, coarse_residual = self.coarse_system()
        if coarse_lam.size == 0:
            return np.zeros(0, dtype=float), coarse_lam, coarse_residual

        stabilized = coarse_lam + self.ridge * np.eye(coarse_lam.shape[0], dtype=float)
        try:
            chol, lower = scipy.linalg.cho_factor(stabilized, lower=False, check_finite=False)
            delta_z = scipy.linalg.cho_solve((chol, lower), coarse_residual)
        except np.linalg.LinAlgError:
            delta_z = np.linalg.solve(stabilized, coarse_residual)
        return delta_z, coarse_lam, coarse_residual

    def prolongate(self, delta_z):
        if not self.bases_initialized:
            self.initialize_bases()
        return self.P @ delta_z

    def apply_correction(self, delta_x, step_size=1.0):
        scaled = step_size * delta_x
        for var in self.base_graph.var_nodes[: self.base_graph.n_var_nodes]:
            sl = self.var_slices[var.variableID]
            local_delta = scaled[sl]
            var.mu = var.mu + local_delta
            var.belief.eta = var.belief.lam @ var.mu

            for factor in var.adj_factors:
                belief_ix = factor.adj_var_nodes.index(var)
                factor.adj_beliefs[belief_ix].eta = var.belief.eta
                factor.adj_beliefs[belief_ix].lam = var.belief.lam

    def v_cycle(
        self,
        pre_smooth=1,
        post_smooth=1,
        upward_coarse_sweeps=1,
        downward_coarse_sweeps=1,
        scheduler="sync",
        fixed_lam=False,
        step_size=1.0,
        recompute_basis=False,
    ):
        if pre_smooth > 0:
            self.warmup(iterations=pre_smooth, scheduler=scheduler, fixed_lam=fixed_lam)

        if recompute_basis or not self.bases_initialized or not self.freeze_basis:
            self.initialize_bases(force=True)
        else:
            self.initialize_bases()

        if recompute_basis or self.coarse_graph is None or not self.freeze_basis:
            self.build_coarse_graph(force=True)
        else:
            self.build_coarse_graph()

        _, _, _, residual_before = self.joint_system()
        coarse_residual = self.update_coarse_residual_eta()
        self.coarse_graph.update_all_beliefs()

        for _ in range(upward_coarse_sweeps):
            if scheduler == "sync":
                self.coarse_graph.synchronous_iteration()
            elif scheduler == "residual":
                self.coarse_graph.residual_iteration_var_heap(max_updates=self.coarse_graph.n_var_nodes)
            else:
                raise ValueError(f"Unknown scheduler: {scheduler}")

        for _ in range(downward_coarse_sweeps):
            if scheduler == "sync":
                self.coarse_graph.synchronous_iteration()
            elif scheduler == "residual":
                self.coarse_graph.residual_iteration_var_heap(max_updates=self.coarse_graph.n_var_nodes)
            else:
                raise ValueError(f"Unknown scheduler: {scheduler}")

        delta_z = self.coarse_mean_vector()
        delta_x = self.prolongate(delta_z)
        self.apply_correction(delta_x, step_size=step_size)

        if post_smooth > 0:
            self.warmup(iterations=post_smooth, scheduler=scheduler, fixed_lam=fixed_lam)

        _, _, _, residual_after = self.joint_system()

        return CycleStats(
            residual_norm_before=float(np.linalg.norm(residual_before)),
            residual_norm_after=float(np.linalg.norm(residual_after)),
            coarse_residual_norm=float(np.linalg.norm(coarse_residual)),
            correction_norm=float(np.linalg.norm(delta_x)),
        )


def ordered_groups_from_ids(ids, group_size, tail_heavy=True):
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not ids:
        return []

    if not tail_heavy:
        return [ids[start : start + group_size] for start in range(0, len(ids), group_size)]

    groups = []
    start = 0
    while start + 2 * group_size <= len(ids):
        groups.append(ids[start : start + group_size])
        start += group_size
    groups.append(ids[start:])
    return [group for group in groups if group]


class SVDResidualHierarchy:
    def __init__(
        self,
        base_graph,
        groups,
        group_size,
        num_levels=2,
        r_reduced=2,
        basis_source="belief_covariance",
        freeze_basis=True,
        ridge=1e-10,
    ):
        self.group_size = group_size
        self.num_levels = max(2, int(num_levels))
        self.r_reduced = r_reduced
        self.basis_source = basis_source
        self.freeze_basis = freeze_basis
        self.ridge = ridge

        self.levels = [
            SVDResidualAbstraction(
                base_graph=base_graph,
                groups=groups,
                r_reduced=r_reduced,
                basis_source=basis_source,
                freeze_basis=freeze_basis,
                ridge=ridge,
            )
        ]

    def warmup(self, iterations=5, scheduler="sync", fixed_lam=False):
        self.levels[0].warmup(iterations=iterations, scheduler=scheduler, fixed_lam=fixed_lam)

    def build_hierarchy(self, force=False):
        if force:
            self.levels = self.levels[:1]

        base_level = self.levels[0]
        base_level.initialize_bases(force=force)
        base_level.build_coarse_graph(force=force)

        while len(self.levels) < self.num_levels - 1:
            lower_to_upper = self.levels[-1]
            lower_to_upper.initialize_bases(force=force)
            lower_to_upper.build_coarse_graph(force=force)

            coarse_graph = lower_to_upper.coarse_graph
            coarse_ids = [var.variableID for var in coarse_graph.var_nodes[: coarse_graph.n_var_nodes]]
            if len(coarse_ids) <= 1:
                break

            groups = ordered_groups_from_ids(coarse_ids, self.group_size, tail_heavy=True)
            if len(groups) >= len(coarse_ids):
                break

            self.levels.append(
                SVDResidualAbstraction(
                    base_graph=coarse_graph,
                    groups=groups,
                    r_reduced=self.r_reduced,
                    basis_source=self.basis_source,
                    freeze_basis=self.freeze_basis,
                    ridge=self.ridge,
                )
            )

    def total_levels(self):
        return 1 + len(self.levels)

    def v_cycle(
        self,
        pre_smooth=1,
        post_smooth=0,
        upward_coarse_sweeps=1,
        downward_coarse_sweeps=1,
        scheduler="sync",
        fixed_lam=False,
        step_size=1.0,
        recompute_basis=False,
    ):
        self.build_hierarchy(force=recompute_basis)

        finest = self.levels[0]
        if pre_smooth > 0:
            finest.warmup(iterations=pre_smooth, scheduler=scheduler, fixed_lam=fixed_lam)

        _, _, _, residual_before = finest.joint_system()

        coarse_residual_norm = 0.0
        for level in self.levels:
            level.initialize_bases(force=recompute_basis)
            level.build_coarse_graph(force=recompute_basis)
            coarse_residual = level.update_coarse_residual_eta()
            coarse_residual_norm = float(np.linalg.norm(coarse_residual))
            level.coarse_graph.update_all_beliefs()

            for _ in range(upward_coarse_sweeps):
                if scheduler == "sync":
                    level.coarse_graph.synchronous_iteration()
                elif scheduler == "residual":
                    level.coarse_graph.residual_iteration_var_heap(max_updates=level.coarse_graph.n_var_nodes)
                else:
                    raise ValueError(f"Unknown scheduler: {scheduler}")

        correction_norm = 0.0
        for level in reversed(self.levels):
            for _ in range(downward_coarse_sweeps):
                if scheduler == "sync":
                    level.coarse_graph.synchronous_iteration()
                elif scheduler == "residual":
                    level.coarse_graph.residual_iteration_var_heap(max_updates=level.coarse_graph.n_var_nodes)
                else:
                    raise ValueError(f"Unknown scheduler: {scheduler}")

            delta_z = level.coarse_mean_vector()
            delta_x = level.prolongate(delta_z)
            correction_norm += float(np.linalg.norm(delta_x))
            level.apply_correction(delta_x, step_size=step_size)

        if post_smooth > 0:
            finest.warmup(iterations=post_smooth, scheduler=scheduler, fixed_lam=fixed_lam)

        _, _, _, residual_after = finest.joint_system()

        return CycleStats(
            residual_norm_before=float(np.linalg.norm(residual_before)),
            residual_norm_after=float(np.linalg.norm(residual_after)),
            coarse_residual_norm=coarse_residual_norm,
            correction_norm=correction_norm,
        )
