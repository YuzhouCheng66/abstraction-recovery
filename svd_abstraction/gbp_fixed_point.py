from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass(frozen=True)
class DirectedBinaryMessage:
    factor_id: int
    target_idx: int
    target_var_id: int
    other_idx: int
    other_var_id: int
    target_slice: slice
    other_slice: slice
    global_slice: slice


@dataclass
class EtaFixedPointSystem:
    matrix: sparse.csc_matrix
    rhs: np.ndarray
    directed_messages: list[DirectedBinaryMessage]
    incoming_binary_to_var: dict[int, list[DirectedBinaryMessage]]


def _local_slices(factor) -> list[slice]:
    out = []
    offset = 0
    for var in factor.adj_var_nodes:
        out.append(slice(offset, offset + var.dofs))
        offset += var.dofs
    return out


def _incoming_binary_messages(graph) -> tuple[list[DirectedBinaryMessage], dict[int, list[DirectedBinaryMessage]]]:
    directed: list[DirectedBinaryMessage] = []
    incoming: dict[int, list[DirectedBinaryMessage]] = {
        int(var.variableID): [] for var in graph.var_nodes[: graph.n_var_nodes]
    }

    offset = 0
    for factor in graph.factors[: graph.n_factor_nodes]:
        if len(factor.adj_var_nodes) != 2:
            continue
        local = _local_slices(factor)
        for target_idx, target_var in enumerate(factor.adj_var_nodes):
            other_idx = 1 - target_idx
            target_slice = local[target_idx]
            other_slice = local[other_idx]
            dofs = target_var.dofs
            desc = DirectedBinaryMessage(
                factor_id=int(factor.factorID),
                target_idx=target_idx,
                target_var_id=int(target_var.variableID),
                other_idx=other_idx,
                other_var_id=int(factor.adj_var_nodes[other_idx].variableID),
                target_slice=target_slice,
                other_slice=other_slice,
                global_slice=slice(offset, offset + dofs),
            )
            directed.append(desc)
            incoming[desc.target_var_id].append(desc)
            offset += dofs
    return directed, incoming


def build_eta_fixed_point_system(graph) -> EtaFixedPointSystem:
    """
    Build the frozen-lam eta/message fixed-point system

        (I - G_lambda) m = c_lambda

    for the current graph state.

    Assumptions:
    - Unary factor messages are fixed and equal to their factor canonical eta.
    - Binary message precisions and adj_belief precisions are already frozen.
    - The graph only uses unary and binary factors.
    """

    directed, incoming_binary_to_var = _incoming_binary_messages(graph)
    total_dim = sum(msg.global_slice.stop - msg.global_slice.start for msg in directed)

    const_eta: dict[int, np.ndarray] = {}
    factor_by_id = {int(f.factorID): f for f in graph.factors[: graph.n_factor_nodes]}
    for var in graph.var_nodes[: graph.n_var_nodes]:
        const_eta[int(var.variableID)] = np.asarray(var.prior.eta, dtype=float).reshape(-1).copy()

    for factor in graph.factors[: graph.n_factor_nodes]:
        if len(factor.adj_var_nodes) == 1:
            var = factor.adj_var_nodes[0]
            const_eta[int(var.variableID)] += np.asarray(factor.factor.eta, dtype=float).reshape(-1)
        elif len(factor.adj_var_nodes) != 2:
            raise NotImplementedError("Only unary and binary factors are supported.")

    mat = sparse.lil_matrix((total_dim, total_dim), dtype=float)
    rhs = np.zeros(total_dim, dtype=float)

    directed_by_key = {(msg.factor_id, msg.target_idx): msg for msg in directed}

    for msg in directed:
        factor = factor_by_id[msg.factor_id]
        row = msg.global_slice
        target_block = np.asarray(factor.factor.eta[msg.target_slice], dtype=float).reshape(-1)
        other_block = np.asarray(factor.factor.eta[msg.other_slice], dtype=float).reshape(-1)

        cavity_lam = (
            np.asarray(factor.adj_beliefs[msg.other_idx].lam, dtype=float)
            - np.asarray(factor.messages[msg.other_idx].lam, dtype=float)
        )
        lam_oo = np.asarray(factor.factor.lam[msg.target_slice, msg.other_slice], dtype=float)
        lam_no = np.asarray(factor.factor.lam[msg.other_slice, msg.other_slice], dtype=float) + cavity_lam
        transfer = lam_oo @ np.linalg.inv(lam_no)

        mat[row, row] = np.eye(row.stop - row.start)
        rhs[row] = target_block - transfer @ (other_block + const_eta[msg.other_var_id])

        opposite = directed_by_key[(msg.factor_id, msg.other_idx)]
        for incoming in incoming_binary_to_var[msg.other_var_id]:
            if incoming == opposite:
                continue
            mat[row, incoming.global_slice] += transfer

    return EtaFixedPointSystem(
        matrix=mat.tocsc(),
        rhs=rhs,
        directed_messages=directed,
        incoming_binary_to_var=incoming_binary_to_var,
    )


def solve_eta_fixed_point(system: EtaFixedPointSystem) -> np.ndarray:
    """
    Solve the frozen-lam eta/message fixed-point system with a sparse direct solver.

    The matrix is generally non-symmetric, so we use sparse LU rather than Cholesky.
    """

    lu = spla.splu(system.matrix)
    return lu.solve(system.rhs)


def apply_eta_fixed_point_solution(graph, system: EtaFixedPointSystem, solution: np.ndarray) -> None:
    """
    Write the eta fixed-point solution back into graph messages, then rebuild beliefs.
    """

    factor_by_id = {int(f.factorID): f for f in graph.factors[: graph.n_factor_nodes]}

    for factor in graph.factors[: graph.n_factor_nodes]:
        if len(factor.adj_var_nodes) == 1:
            factor.messages[0].eta = np.asarray(factor.factor.eta, dtype=float).reshape(-1).copy()

    for msg in system.directed_messages:
        factor = factor_by_id[msg.factor_id]
        factor.messages[msg.target_idx].eta = np.asarray(solution[msg.global_slice], dtype=float).reshape(-1).copy()

    graph.update_all_beliefs()


def max_message_eta_residual(graph, system: EtaFixedPointSystem) -> float:
    """
    Compute the max fixed-point residual in eta-message space using the current graph state.
    """

    factor_by_id = {int(f.factorID): f for f in graph.factors[: graph.n_factor_nodes]}
    worst = 0.0
    for msg in system.directed_messages:
        factor = factor_by_id[msg.factor_id]
        target_idx = msg.target_idx
        target_slice = msg.target_slice
        other_slice = msg.other_slice
        target_eta = np.asarray(factor.factor.eta[target_slice], dtype=float).reshape(-1)
        other_eta = np.asarray(factor.factor.eta[other_slice], dtype=float).reshape(-1)

        cavity_lam = (
            np.asarray(factor.adj_beliefs[msg.other_idx].lam, dtype=float)
            - np.asarray(factor.messages[msg.other_idx].lam, dtype=float)
        )
        cavity_eta = (
            np.asarray(factor.adj_beliefs[msg.other_idx].eta, dtype=float)
            - np.asarray(factor.messages[msg.other_idx].eta, dtype=float)
        )
        lam_oo = np.asarray(factor.factor.lam[target_slice, other_slice], dtype=float)
        lam_no = np.asarray(factor.factor.lam[other_slice, other_slice], dtype=float) + cavity_lam
        rhs_eta = other_eta + cavity_eta
        new_eta = target_eta - lam_oo @ np.linalg.solve(lam_no, rhs_eta)
        old_eta = np.asarray(factor.messages[target_idx].eta, dtype=float).reshape(-1)
        worst = max(worst, float(np.max(np.abs(new_eta - old_eta))))
    return worst
