from __future__ import annotations

import argparse
import pathlib
import sys
from contextlib import contextmanager

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOCAL_RAYLIB_ROOT = SCRIPT_DIR / "raylib_gbp_local"

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(LOCAL_RAYLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_RAYLIB_ROOT))

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(1, str(WORKSPACE_ROOT))


def _module_comes_from_local_raylib(module) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        try:
            return pathlib.Path(module_file).resolve().is_relative_to(LOCAL_RAYLIB_ROOT.resolve())
        except ValueError:
            return False

    module_paths = getattr(module, "__path__", None)
    if module_paths is None:
        return False

    for path_entry in module_paths:
        try:
            if pathlib.Path(path_entry).resolve().is_relative_to(LOCAL_RAYLIB_ROOT.resolve()):
                return True
        except ValueError:
            continue
    return False


def _purge_conflicting_local_raylib_modules():
    for prefix in ("gbp", "amg"):
        for name, module in list(sys.modules.items()):
            if name != prefix and not name.startswith(prefix + "."):
                continue
            if module is None:
                sys.modules.pop(name, None)
                continue
            if not _module_comes_from_local_raylib(module):
                sys.modules.pop(name, None)


_purge_conflicting_local_raylib_modules()

from amg import functions as amg_fnc
from gbp.factors import linear_displacement
from gbp.gbp import Factor
from gbp.gbp import FactorGraph
from gbp.gbp import VariableNode

from svd_abstraction.pose_graph import make_slam_like_graph


def build_local_raylib_graph(
    nodes,
    edges,
    prior_sigma=1.0,
    odom_sigma=1.0,
    tiny_prior=1e-12,
    seed=0,
):
    rng = np.random.default_rng(seed)
    graph = FactorGraph(nonlinear_factors=False, eta_damping=0.0)
    eye2 = np.eye(2, dtype=float)

    prior_noises = {}
    odom_noises = {}
    for edge in edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]
        if dst not in {"prior", "anchor"}:
            odom_noises[(int(src), int(dst))] = rng.normal(0.0, odom_sigma, size=2)
        elif dst == "prior":
            prior_noises[int(src)] = rng.normal(0.0, prior_sigma, size=2)

    var_nodes = []
    for i, node in enumerate(nodes):
        var = VariableNode(i, 2)
        var.GT = np.array([node["position"]["x"], node["position"]["y"]], dtype=float)
        var.type = "base"
        var.prior.lam = tiny_prior * eye2
        var.prior.eta = np.zeros(2, dtype=float)
        var_nodes.append(var)

    factors = []
    factor_id = 0
    for edge in edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]

        if dst not in {"prior", "anchor"}:
            i = int(src)
            j = int(dst)
            var_i = var_nodes[i]
            var_j = var_nodes[j]
            measurement = (var_j.GT - var_i.GT) + odom_noises[(i, j)]
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
            continue

        if dst == "prior":
            i = int(src)
            var = var_nodes[i]
            measurement = var.GT + prior_noises[i]
            lam = eye2 / (prior_sigma**2)
            var.prior.lam = var.prior.lam + lam
            var.prior.eta = var.prior.eta + lam @ measurement

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


def build_slam_graph(
    n=128,
    step_size=25,
    loop_prob=0.05,
    loop_radius=50,
    prior_prop=0.02,
    prior_sigma=1.0,
    odom_sigma=1.0,
    seed=0,
):
    nodes, edges = make_slam_like_graph(
        N=n,
        step_size=step_size,
        loop_prob=loop_prob,
        loop_radius=loop_radius,
        prior_prop=prior_prop,
        seed=seed,
    )
    return build_local_raylib_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )


def build_chain_graph(n=128, prior_sigma=1.0, odom_sigma=1.0, seed=0):
    nodes = [
        {
            "data": {"id": str(i), "layer": 0, "dim": 2, "num_base": 1},
            "position": {"x": float(i), "y": 0.0},
        }
        for i in range(int(n))
    ]
    edges = [{"data": {"source": str(i), "target": str(i + 1)}} for i in range(int(n) - 1)]
    edges.append({"data": {"source": "0", "target": "anchor"}})
    return build_local_raylib_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )


def build_hierarchy(graph, split_mode="pmis2", interp_mode="extended_if_needed", theta=0.25):
    graph.enable_second_pass_coarse_match = False
    graph.multigrid_split_mode = split_mode
    for var in graph.multigrid_vars[0]:
        var.multigrid.theta = theta
        var.multigrid.interp_mode = interp_mode
    amg_fnc.coarsen_graph(graph, graph.multigrid_vars[0].copy())
    return graph


def eta_consistency_stats(graph, level=0):
    vars_level = [var for var in graph.multigrid_vars[level] if var.type != "dead"]
    max_norm = 0.0
    mean_norm = 0.0
    count = 0
    for var in vars_level:
        if not var.active:
            continue
        incoming = np.zeros_like(var.belief.eta)
        for factor in var.adj_factors:
            msg_ix = factor.adj_vIDs.index(var.variableID)
            incoming += factor.messages[msg_ix].eta
        mismatch = var.belief.eta - (var.prior.eta + incoming)
        mismatch_norm = float(np.linalg.norm(mismatch))
        max_norm = max(max_norm, mismatch_norm)
        mean_norm += mismatch_norm
        count += 1

        for factor in var.adj_factors:
            belief_ix = factor.adj_vIDs.index(var.variableID)
            adj_mismatch = factor.adj_beliefs[belief_ix].eta - var.belief.eta
            max_norm = max(max_norm, float(np.linalg.norm(adj_mismatch)))

    if count == 0:
        return {"max": 0.0, "mean": 0.0}
    return {"max": max_norm, "mean": mean_norm / count}


def run_first_two_level_prolongation(graph):
    if len(graph.multigrid_vars) < 2:
        raise RuntimeError("Need at least two levels for prolongation validation")

    graph.synchronous_iteration(level=0)
    graph.update_all_residual_etas(level=1)
    graph.update_all_beliefs(level=1)
    graph.synchronous_iteration(level=1)
    graph.update_all_residuals(level=1)
    graph.synchronous_iteration(level=1)
    graph.prolongate_corrections(level=1)


def mean_vector(graph):
    base_vars = [var for var in graph.var_nodes[: graph.n_var_nodes] if var.type != "multigrid"]
    return np.concatenate([var.mu for var in base_vars])


def exact_mean(graph):
    eta, lam = graph.joint_distribution_inf()
    return np.linalg.solve(lam, eta)


def relative_error(graph, mu_star):
    err = mean_vector(graph) - mu_star
    return float(np.linalg.norm(err) / np.linalg.norm(mu_star))


def base_residual_norm(graph):
    graph.update_all_residuals(level=0)
    base_vars = [var for var in graph.multigrid_vars[0] if var.type != "multigrid"]
    res = np.concatenate([var.residual for var in base_vars])
    return float(np.linalg.norm(res))


def run_behavior(mode, cycles=100):
    graph = build_slam_graph()
    mu_star = exact_mean(graph)
    build_hierarchy(graph)
    graph.set_prolongation_eta_mode(mode)

    relerrs = []
    for _ in range(cycles):
        graph.vcycle_step()
        relerrs.append(relative_error(graph, mu_star))
    return relerrs


def run_consistency(mode):
    graph = build_slam_graph()
    build_hierarchy(graph)
    graph.set_prolongation_eta_mode(mode)
    run_first_two_level_prolongation(graph)
    return eta_consistency_stats(graph, level=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none", "uniform", "message_lam_trace", "factor_diag_trace", "local_direct", "local_iter5"],
    )
    parser.add_argument("--cycles", type=int, default=50)
    args = parser.parse_args()

    print("Consistency after first prolongation:")
    for mode in args.modes:
        stats = run_consistency(mode)
        print(
            f"  {mode:>18s} | max mismatch {stats['max']:.6e} | mean mismatch {stats['mean']:.6e}"
        )

    print("\nBehavior on slam128:")
    for mode in args.modes:
        relerrs = run_behavior(mode, cycles=args.cycles)
        print(
            f"  {mode:>18s} | relerr@1 {relerrs[0]:.6e} | relerr@10 {relerrs[min(9, len(relerrs)-1)]:.6e} | relerr@{args.cycles} {relerrs[-1]:.6e}"
        )


if __name__ == "__main__":
    main()
