"""Build reference GBP factor graphs from a block canonical linear system."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from svd_abstraction.gbp.gbp import Factor
from svd_abstraction.gbp.gbp import FactorGraph
from svd_abstraction.gbp.gbp import VariableNode


def decompose_block_operator(
    eta: np.ndarray,
    lam: sparse.spmatrix | np.ndarray,
    block_dofs: int,
    tol: float = 1e-12,
):
    """Decompose a block M-matrix into unary terms and pairwise attractive factors."""
    lam_csr = sparse.csr_matrix(lam)
    if lam_csr.shape[0] != lam_csr.shape[1]:
        raise ValueError("Expected square information matrix.")
    if lam_csr.shape[0] % block_dofs != 0:
        raise ValueError("Matrix dimension must be divisible by block_dofs.")

    n_nodes = lam_csr.shape[0] // block_dofs
    bsr = lam_csr.tobsr(blocksize=(block_dofs, block_dofs))

    diag_blocks = [np.zeros((block_dofs, block_dofs), dtype=float) for _ in range(n_nodes)]
    pair_weights: list[tuple[int, int, np.ndarray]] = []
    w_sums = [np.zeros((block_dofs, block_dofs), dtype=float) for _ in range(n_nodes)]

    for i in range(n_nodes):
        for pos in range(bsr.indptr[i], bsr.indptr[i + 1]):
            j = int(bsr.indices[pos])
            block = np.asarray(bsr.data[pos], dtype=float)
            if i == j:
                diag_blocks[i] = block
                continue
            if j < i:
                continue
            if np.any(block > tol):
                raise ValueError(f"Expected non-positive off-diagonal blocks, found positive block at {(i, j)}.")
            if np.linalg.norm(block) <= tol:
                continue
            weight = -block
            pair_weights.append((i, j, weight))
            w_sums[i] += weight
            w_sums[j] += weight

    unary_lam = []
    unary_eta = []
    for i in range(n_nodes):
        block_slice = slice(block_dofs * i, block_dofs * (i + 1))
        unary_lam.append(diag_blocks[i] - w_sums[i])
        unary_eta.append(np.asarray(eta[block_slice], dtype=float).copy())

    return unary_eta, unary_lam, pair_weights


def distribute_eta_to_pairwise_lstsq(
    unary_eta: list[np.ndarray],
    pair_weights: list[tuple[int, int, np.ndarray]],
    block_dofs: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Distribute node eta onto pairwise factors via an incidence least-squares fit."""
    n_nodes = len(unary_eta)
    n_edges = len(pair_weights)
    if n_edges == 0:
        return [np.array(v, copy=True) for v in unary_eta], []

    b = np.zeros((n_nodes * block_dofs, n_edges * block_dofs), dtype=float)
    for edge_idx, (i, j, _weight) in enumerate(pair_weights):
        node_i = slice(i * block_dofs, (i + 1) * block_dofs)
        node_j = slice(j * block_dofs, (j + 1) * block_dofs)
        edge = slice(edge_idx * block_dofs, (edge_idx + 1) * block_dofs)
        b[node_i, edge] = -np.eye(block_dofs)
        b[node_j, edge] = np.eye(block_dofs)

    eta_nodes = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in unary_eta])
    q, *_ = np.linalg.lstsq(b, eta_nodes, rcond=None)
    residual = eta_nodes - b @ q

    unary_residual = [
        residual[i * block_dofs : (i + 1) * block_dofs].copy()
        for i in range(n_nodes)
    ]
    pair_eta = [
        q[edge_idx * block_dofs : (edge_idx + 1) * block_dofs].copy()
        for edge_idx in range(n_edges)
    ]
    return unary_residual, pair_eta


def build_reference_gbp_graph_from_operator(
    eta: np.ndarray,
    lam: sparse.spmatrix | np.ndarray,
    block_dofs: int,
    eta_damping: float = 0.0,
    pair_eta_mode: str = "unary",
) -> FactorGraph:
    """Build a GBP factor graph whose canonical product matches (eta, lam)."""
    unary_eta, unary_lam, pair_weights = decompose_block_operator(eta, lam, block_dofs)
    if pair_eta_mode == "unary":
        pair_eta = [np.zeros(block_dofs, dtype=float) for _ in pair_weights]
    elif pair_eta_mode == "incidence_lstsq":
        unary_eta, pair_eta = distribute_eta_to_pairwise_lstsq(unary_eta, pair_weights, block_dofs)
    else:
        raise ValueError(f"Unknown pair_eta_mode: {pair_eta_mode}")

    graph = FactorGraph(nonlinear_factors=False, eta_damping=eta_damping)
    var_nodes = []
    for idx, (local_eta, local_lam) in enumerate(zip(unary_eta, unary_lam)):
        var = VariableNode(idx, block_dofs)
        var.prior.eta = local_eta.copy()
        var.prior.lam = local_lam.copy()
        var_nodes.append(var)

    graph.var_nodes = var_nodes
    graph.n_var_nodes = len(var_nodes)

    def meas_fn(x, *args):
        return [np.zeros(block_dofs)]

    def jac_fn(x, *args):
        return [np.zeros((block_dofs, 2 * block_dofs))]

    factors = []
    for factor_id, ((i, j, weight), factor_eta_edge) in enumerate(zip(pair_weights, pair_eta)):
        factor = Factor(
            factor_id,
            [var_nodes[i], var_nodes[j]],
            [np.zeros(block_dofs)],
            [np.eye(block_dofs)],
            meas_fn,
            jac_fn,
        )
        factor.factor.lam = np.block([[weight, -weight], [-weight, weight]])
        factor.factor.eta = np.concatenate([-factor_eta_edge, factor_eta_edge])
        factors.append(factor)
        var_nodes[i].adj_factors.append(factor)
        var_nodes[j].adj_factors.append(factor)

    graph.factors = factors
    graph.n_factor_nodes = len(factors)

    for var in graph.var_nodes:
        var.update_belief()

    return graph


def graph_mean_vector(graph: FactorGraph) -> np.ndarray:
    return np.concatenate([var.mu for var in graph.var_nodes[: graph.n_var_nodes]])
