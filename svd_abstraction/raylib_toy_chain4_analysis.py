"""Toy 4-node chain analysis for raylib AMG.

This script keeps raylib's own graph / AMG / message semantics and manually
unrolls one two-level V-cycle on a tiny deterministic chain:

    (0,0) -- (1,0) -- (2,0) -- (3,0)

The goal is to inspect why the level-1 -> level-0 prolongation may only give a
small residual improvement (or even become harmful) depending on the hierarchy
and coarse solve quality.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

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

from amg import functions as amg_fnc
from gbp.factors import linear_displacement
from gbp.gbp import Factor
from gbp.gbp import FactorGraph
from gbp.gbp import VariableNode


@dataclass
class AMGArgs:
    theta: float = 0.25
    split_mode: str = "pmis2"
    interp_mode: str = "extended_if_needed"
    disable_second_pass_coarse_match: bool = True


def make_toy_nodes_edges() -> tuple[list[dict], list[dict]]:
    nodes = []
    for i in range(4):
        nodes.append(
            {
                "id": i,
                "position": {"x": float(i), "y": 0.0},
            }
        )

    edges = []
    for i in range(3):
        edges.append({"data": {"source": i, "target": i + 1}})
    return nodes, edges


def build_deterministic_raylib_graph(
    nodes: list[dict],
    edges: list[dict],
    odom_sigma: float = 1.0,
    tiny_prior: float = 1e-12,
) -> FactorGraph:
    graph = FactorGraph(nonlinear_factors=False, eta_damping=0.0)
    eye2 = np.eye(2, dtype=float)

    var_nodes: list[VariableNode] = []
    for i, node in enumerate(nodes):
        var = VariableNode(i, 2)
        var.GT = np.array([node["position"]["x"], node["position"]["y"]], dtype=float)
        var.type = "base"
        var.prior.lam = tiny_prior * eye2
        var.prior.eta = np.zeros(2, dtype=float)
        var_nodes.append(var)

    factors: list[Factor] = []
    factor_id = 0
    for edge in edges:
        i = int(edge["data"]["source"])
        j = int(edge["data"]["target"])
        var_i = var_nodes[i]
        var_j = var_nodes[j]
        measurement = var_j.GT - var_i.GT
        factor = Factor(
            factor_id,
            [var_i, var_j],
            measurement,
            odom_sigma,
            linear_displacement.meas_fn,
            linear_displacement.jac_fn,
            loss=None,
            mahalanobis_threshold=2,
        )
        factor.type = "odometry"
        factor.compute_factor(linpoint=np.r_[var_i.GT, var_j.GT], update_self=True)
        factors.append(factor)
        var_i.adj_factors.append(factor)
        var_j.adj_factors.append(factor)
        factor_id += 1

    anchor_var = var_nodes[0]
    anchor_lam = eye2 / ((1e-4) ** 2)
    anchor_var.prior.lam = anchor_var.prior.lam + anchor_lam
    anchor_var.prior.eta = anchor_var.prior.eta + anchor_lam @ anchor_var.GT

    graph.var_nodes = var_nodes.copy()
    graph.factors = factors.copy()
    graph.n_var_nodes = len(var_nodes)
    graph.n_factor_nodes = len(factors)
    graph.multigrid_vars[0].extend(var_nodes)
    graph.multigrid_factors[0].extend(factors)

    for var in graph.var_nodes[: graph.n_var_nodes]:
        var.update_belief()

    return graph


def build_toy_multigrid_graph(args: AMGArgs) -> FactorGraph:
    nodes, edges = make_toy_nodes_edges()
    graph = build_deterministic_raylib_graph(nodes, edges)
    graph.enable_second_pass_coarse_match = not args.disable_second_pass_coarse_match
    graph.multigrid_split_mode = args.split_mode
    for var in graph.multigrid_vars[0]:
        var.multigrid.theta = args.theta
        var.multigrid.interp_mode = args.interp_mode
    amg_fnc.coarsen_graph(graph, graph.multigrid_vars[0].copy())
    return graph


def level_vars(graph: FactorGraph, level: int):
    return [var for var in graph.multigrid_vars[level] if var.type != "dead"]


def base_vars(graph: FactorGraph):
    return [var for var in graph.multigrid_vars[0] if var.type != "dead"]


def residual_table(vars_list) -> list[dict]:
    rows = []
    for var in vars_list:
        residual = np.array(var.compute_residual(), copy=True)
        rows.append(
            {
                "id": int(var.variableID),
                "mu": np.array(var.mu, copy=True),
                "residual": residual,
                "res_norm": float(np.linalg.norm(residual)),
            }
        )
    return rows


def print_residual_table(title: str, rows: list[dict]) -> None:
    print(title)
    print("  id | mu                 | residual           | ||r||")
    for row in rows:
        mu = np.array2string(row["mu"], precision=6, suppress_small=False)
        residual = np.array2string(row["residual"], precision=6, suppress_small=False)
        print(f"  {row['id']:>2} | {mu:<18} | {residual:<18} | {row['res_norm']:.6e}")
    total = float(np.linalg.norm(np.concatenate([row["residual"] for row in rows])))
    print(f"  total residual norm = {total:.6e}")
    print()


def print_base_classification(graph: FactorGraph) -> None:
    print("Base-level AMG classification")
    print("  id | class  | parent | restriction coeffs")
    for var in base_vars(graph):
        parent_id = None if var.multigrid.parent is None else int(var.multigrid.parent.variableID)
        coeffs = [
            np.array2string(np.array(coeff), precision=3, suppress_small=False)
            for coeff in var.multigrid.restriction_coefficients
        ]
        print(
            f"  {var.variableID:>2} | {var.multigrid.classification:<6} | "
            f"{str(parent_id):<6} | {coeffs}"
        )
    print()


def print_coarse_state(graph: FactorGraph) -> None:
    print("Level-1 coarse variables")
    print("  id | child | mu                 | prior.eta           | interpolation targets")
    for var in level_vars(graph, 1):
        child_id = None if var.multigrid.child is None else int(var.multigrid.child.variableID)
        targets = []
        for coeff, child in zip(var.multigrid.interpolation_coefficients, var.multigrid.interpolation_vars):
            coeff_str = np.array2string(np.array(coeff), precision=3, suppress_small=False)
            targets.append((int(child.variableID), coeff_str))
        print(
            f"  {var.variableID:>2} | {str(child_id):<5} | "
            f"{np.array2string(var.mu, precision=6):<18} | "
            f"{np.array2string(var.prior.eta, precision=6):<18} | {targets}"
        )
    print()


def print_level_operator(graph: FactorGraph, level: int) -> None:
    eta, lam = graph.joint_distribution_inf_level(level)
    print(f"Level {level} joint operator")
    print("eta =", np.array2string(eta, precision=6, suppress_small=False))
    print("lam =")
    print(np.array2string(lam, precision=6, suppress_small=False))
    print()


def prolongation_delta(graph: FactorGraph) -> np.ndarray:
    child_vars = base_vars(graph)
    child_slices = {}
    offset = 0
    for var in child_vars:
        child_slices[var.variableID] = slice(offset, offset + var.dofs)
        offset += var.dofs

    delta = np.zeros(offset, dtype=float)
    for coarse_var in level_vars(graph, 1):
        for coeff, child_var in zip(
            coarse_var.multigrid.interpolation_coefficients,
            coarse_var.multigrid.interpolation_vars,
        ):
            sl = child_slices.get(child_var.variableID)
            if sl is None:
                continue
            delta[sl] += coeff @ coarse_var.mu
    return delta


def run_one_manual_vcycle(graph: FactorGraph) -> None:
    print_residual_table("Initial base-node state", residual_table(base_vars(graph)))

    graph.synchronous_iteration(level=0)
    print_residual_table("After level-0 pre-smooth", residual_table(base_vars(graph)))

    graph.update_all_residual_etas(level=1)
    print_coarse_state(graph)

    graph.update_all_beliefs(level=1)
    print_coarse_state(graph)

    graph.synchronous_iteration(level=1)
    print_coarse_state(graph)

    graph.update_all_residuals(level=1)
    graph.synchronous_iteration(level=1)
    print_coarse_state(graph)

    before_rows = residual_table(base_vars(graph))
    print_residual_table("Base residuals just before level-1 -> level-0 prolongation", before_rows)

    delta = prolongation_delta(graph)
    print("Prolongation delta to level 0")
    for idx, var in enumerate(base_vars(graph)):
        sl = slice(2 * idx, 2 * idx + 2)
        print(f"  base node {var.variableID}: delta = {np.array2string(delta[sl], precision=6)}")
    print()

    graph.prolongate_corrections(level=1)

    after_rows = residual_table(base_vars(graph))
    print_residual_table("Base residuals immediately after level-1 -> level-0 prolongation", after_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--split-mode", choices=["rs", "pmis", "pmis2"], default="pmis2")
    parser.add_argument(
        "--interp-mode",
        choices=["direct", "extended_if_needed", "extended_all"],
        default="extended_if_needed",
    )
    parser.add_argument("--enable-second-pass", action="store_true")
    args = parser.parse_args()

    graph = build_toy_multigrid_graph(
        AMGArgs(
            theta=args.theta,
            split_mode=args.split_mode,
            interp_mode=args.interp_mode,
            disable_second_pass_coarse_match=not args.enable_second_pass,
        )
    )

    sizes = [len(level_vars(graph, level)) for level in range(len(graph.multigrid_vars))]
    print("Toy chain-4 raylib AMG")
    print(f"levels = {sizes}")
    print(f"split_mode = {args.split_mode}, interp_mode = {args.interp_mode}, second_pass = {args.enable_second_pass}")
    print()

    print_base_classification(graph)
    print_level_operator(graph, 1)
    run_one_manual_vcycle(graph)


if __name__ == "__main__":
    main()
