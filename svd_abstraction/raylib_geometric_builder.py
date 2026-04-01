"""Independent geometric hierarchy builder around raylib GBP.

This module does not modify `raylib_gbp` sources. Instead it:

1. builds a standard raylib base graph
2. attaches a geometric multigrid hierarchy (chain/grid)
3. reuses raylib's persistent multigrid variables, factors, and `vcycle_step()`

The purpose is to compare:
* raylib AMG-style splitting/interpolation
* raylib-style persistent GBP on a geometric hierarchy
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys

import numpy as np
from scipy import sparse

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOCAL_RAYLIB_ROOT = SCRIPT_DIR / "raylib_gbp_local"
EXTERNAL_RAYLIB_ROOT = pathlib.Path("/home/yuzhou/Desktop/raylib_gbp")
RAYLIB_ROOT = EXTERNAL_RAYLIB_ROOT if EXTERNAL_RAYLIB_ROOT.exists() else LOCAL_RAYLIB_ROOT

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(RAYLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(RAYLIB_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(1, str(WORKSPACE_ROOT))

from gbp.factors import linear_displacement
from gbp.gbp import Factor
from gbp.gbp import VariableNode

from svd_abstraction.gbp_from_operator import decompose_block_operator
from svd_abstraction.raylib_recursive_experiment import build_raylib_graph


@dataclass
class GeometricLevelSpec:
    kind: str
    nx: int
    ny: int
    coarse_x: np.ndarray
    coarse_y: np.ndarray
    p_scalar: sparse.csr_matrix
    r_scalar: sparse.csr_matrix

    @property
    def n_fine(self) -> int:
        return self.nx * self.ny

    @property
    def n_coarse(self) -> int:
        return self.coarse_x.size * self.coarse_y.size

    @property
    def coarse_nx(self) -> int:
        return int(self.coarse_x.size)

    @property
    def coarse_ny(self) -> int:
        return int(self.coarse_y.size)


def _coarse_positions_1d(n_fine: int) -> np.ndarray:
    coarse = np.arange(0, n_fine, 2, dtype=int)
    if coarse[-1] != n_fine - 1:
        coarse = np.r_[coarse, n_fine - 1]
    return coarse


def _prolongation_1d_from_positions(n_fine: int, coarse_pos: np.ndarray) -> sparse.csr_matrix:
    n_coarse = coarse_pos.size
    coarse_lookup = {int(pos): idx for idx, pos in enumerate(coarse_pos.tolist())}
    rows = []
    cols = []
    data = []
    for i in range(n_fine):
        if i in coarse_lookup:
            rows.append(i)
            cols.append(coarse_lookup[i])
            data.append(1.0)
            continue

        right_idx = int(np.searchsorted(coarse_pos, i))
        left_idx = max(right_idx - 1, 0)
        right_idx = min(right_idx, n_coarse - 1)
        left_pos = int(coarse_pos[left_idx])
        right_pos = int(coarse_pos[right_idx])
        if right_pos == left_pos:
            rows.append(i)
            cols.append(left_idx)
            data.append(1.0)
            continue
        t = (i - left_pos) / float(right_pos - left_pos)
        rows.extend([i, i])
        cols.extend([left_idx, right_idx])
        data.extend([1.0 - t, t])
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_fine, n_coarse))


def build_chain_level_spec(n_fine: int) -> GeometricLevelSpec:
    coarse_x = _coarse_positions_1d(n_fine)
    p_scalar = _prolongation_1d_from_positions(n_fine, coarse_x)
    r_scalar = (0.5 * p_scalar.transpose()).tocsr()
    return GeometricLevelSpec(
        kind="chain",
        nx=n_fine,
        ny=1,
        coarse_x=coarse_x,
        coarse_y=np.array([0], dtype=int),
        p_scalar=p_scalar.tocsr(),
        r_scalar=r_scalar,
    )


def build_grid_level_spec(nx: int, ny: int) -> GeometricLevelSpec:
    coarse_x = _coarse_positions_1d(nx)
    coarse_y = _coarse_positions_1d(ny)
    px = _prolongation_1d_from_positions(nx, coarse_x)
    py = _prolongation_1d_from_positions(ny, coarse_y)
    p_scalar = sparse.kron(py, px, format="csr")
    r_scalar = (0.25 * p_scalar.transpose()).tocsr()
    return GeometricLevelSpec(
        kind="grid",
        nx=nx,
        ny=ny,
        coarse_x=coarse_x,
        coarse_y=coarse_y,
        p_scalar=p_scalar,
        r_scalar=r_scalar,
    )


def build_geometric_specs(kind: str, nx: int, ny: int, max_total_levels: int | None = None, min_coarse_n: int = 2):
    specs = []
    cur_nx, cur_ny = nx, ny
    while True:
        if kind == "chain":
            spec = build_chain_level_spec(cur_nx)
        elif kind == "grid":
            spec = build_grid_level_spec(cur_nx, cur_ny)
        else:
            raise ValueError(f"Unknown geometric kind: {kind}")
        if kind == "chain" and spec.n_coarse >= spec.n_fine:
            break
        if kind == "grid" and spec.coarse_nx >= cur_nx and spec.coarse_ny >= cur_ny:
            break
        specs.append(spec)
        if max_total_levels is not None and len(specs) >= max_total_levels - 1:
            break
        if spec.n_coarse <= min_coarse_n:
            break
        if kind == "chain":
            cur_nx = spec.n_coarse
            cur_ny = 1
        else:
            cur_nx = spec.coarse_nx
            cur_ny = spec.coarse_ny
    return specs


def _append_coarse_var(graph, level_idx: int, dofs: int = 2):
    var = VariableNode(graph.n_var_nodes, dofs)
    var.type = "multigrid"
    var.multigrid.level = level_idx
    var.GT = np.zeros(dofs)
    graph.var_nodes.append(var)
    while len(graph.multigrid_vars) <= level_idx:
        graph.multigrid_vars.append([])
        graph.multigrid_factors.append([])
    graph.multigrid_vars[level_idx].append(var)
    graph.n_var_nodes += 1
    return var


def _build_level_links(fine_vars, coarse_vars, spec: GeometricLevelSpec):
    for fine in fine_vars:
        fine.multigrid.restriction_vars = []
        fine.multigrid.restriction_coefficients = []
        fine.multigrid.classification = "fine"

    for coarse in coarse_vars:
        coarse.multigrid.interpolation_vars = []
        coarse.multigrid.interpolation_coefficients = []
        coarse.multigrid.res_incoming = []
        coarse.multigrid.corrections_outgoing = []
        coarse.multigrid.classification = "coarse"

    p = spec.p_scalar.tocsr()
    r_csc = spec.r_scalar.tocsc()
    eye2 = np.eye(2, dtype=float)

    for fine_idx, fine in enumerate(fine_vars):
        col_start = r_csc.indptr[fine_idx]
        col_end = r_csc.indptr[fine_idx + 1]
        for coarse_idx, coeff in zip(r_csc.indices[col_start:col_end], r_csc.data[col_start:col_end]):
            coeff = float(coeff)
            if abs(coeff) <= 1e-15:
                continue
            fine.multigrid.restriction_vars.append(coarse_vars[int(coarse_idx)])
            fine.multigrid.restriction_coefficients.append(coeff * eye2)

    for fine_idx, fine in enumerate(fine_vars):
        row_start = p.indptr[fine_idx]
        row_end = p.indptr[fine_idx + 1]
        for coarse_idx, coeff in zip(p.indices[row_start:row_end], p.data[row_start:row_end]):
            coeff = float(coeff)
            if abs(coeff) <= 1e-15:
                continue
            coarse = coarse_vars[int(coarse_idx)]
            coarse.multigrid.interpolation_vars.append(fine)
            coarse.multigrid.interpolation_coefficients.append(coeff * eye2)
            coarse.multigrid.res_incoming.append(np.zeros(2, dtype=float))
            coarse.multigrid.corrections_outgoing.append(np.zeros(2, dtype=float))
            if abs(coeff - 1.0) <= 1e-15 and fine.multigrid.parent is None:
                fine.multigrid.parent = coarse
                coarse.multigrid.child = fine


def _build_level_operator_and_factors(graph, fine_level_idx: int, coarse_level_idx: int, spec: GeometricLevelSpec):
    _, a_fine = graph.joint_distribution_inf_level(fine_level_idx)
    eye2 = sparse.eye(2, format="csr")
    p = sparse.kron(spec.p_scalar, eye2, format="csr")
    r = sparse.kron(spec.r_scalar, eye2, format="csr")
    a_coarse = (r @ sparse.csr_matrix(a_fine) @ p).tocsr()
    zero_eta = np.zeros(a_coarse.shape[0], dtype=float)
    unary_eta, unary_lam, pair_weights = decompose_block_operator(zero_eta, a_coarse, block_dofs=2)

    coarse_vars = graph.multigrid_vars[coarse_level_idx]
    for coarse_var, local_lam in zip(coarse_vars, unary_lam):
        coarse_var.prior.lam = np.array(local_lam, copy=True)
        coarse_var.prior.eta = np.zeros(2, dtype=float)

    factor_id = graph.n_factor_nodes
    for i, j, weight in pair_weights:
        vi = coarse_vars[i]
        vj = coarse_vars[j]
        factor = Factor(
            factor_id,
            [vi, vj],
            np.zeros(2, dtype=float),
            np.ones(2, dtype=float),
            linear_displacement.meas_fn,
            linear_displacement.jac_fn,
            loss=None,
            mahalanobis_threshold=2,
        )
        lam_ij = -weight
        lam_ii = weight
        factor.factor.lam = np.block([[lam_ii, lam_ij], [lam_ij, lam_ii]])
        factor.factor.eta = np.zeros(4, dtype=float)
        factor.type = f"multigrid-geom lvl {coarse_level_idx}"
        factor.level = coarse_level_idx
        graph.factors.append(factor)
        graph.multigrid_factors[coarse_level_idx].append(factor)
        graph.n_factor_nodes += 1
        vi.adj_factors.append(factor)
        vj.adj_factors.append(factor)
        factor_id += 1

    for coarse_var in coarse_vars:
        coarse_var.update_belief()


def attach_geometric_hierarchy(graph, kind: str, nx: int, ny: int = 1, max_total_levels: int | None = None):
    specs = build_geometric_specs(kind=kind, nx=nx, ny=ny, max_total_levels=max_total_levels)
    fine_vars = list(graph.multigrid_vars[0])

    for level_offset, spec in enumerate(specs, start=1):
        coarse_vars = [_append_coarse_var(graph, level_offset) for _ in range(spec.n_coarse)]
        _build_level_links(fine_vars, coarse_vars, spec)
        _build_level_operator_and_factors(graph, level_offset - 1, level_offset, spec)
        fine_vars = coarse_vars

    return graph


def build_geometric_multigrid_graph(
    nodes,
    edges,
    kind: str,
    nx: int,
    ny: int = 1,
    max_total_levels: int | None = None,
    prior_sigma: float = 6.0,
    odom_sigma: float = 3.0,
    seed: int = 0,
):
    graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )
    attach_geometric_hierarchy(graph, kind=kind, nx=nx, ny=ny, max_total_levels=max_total_levels)
    return graph
