from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import scipy.linalg

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs, group_list
from svd_abstraction.pose_graph import build_noisy_pose_graph
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [1, 2, 5, 10, 20, 50]
POINTS = [0, 1, 2, 5, 10, 20, 50, 100]


def set_absolute_factors(graph) -> None:
    for factor in graph.factors[: graph.n_factor_nodes]:
        factor.compute_factor_absolute(update_self=True)


def mean_vector(graph) -> np.ndarray:
    return np.concatenate([np.asarray(v.mu).reshape(-1) for v in graph.var_nodes[: graph.n_var_nodes]])


def relative_error_vec(x: np.ndarray, x_star: np.ndarray) -> float:
    denom = max(np.linalg.norm(x_star), 1e-15)
    return float(np.linalg.norm(x - x_star) / denom)


def var_slices(graph):
    mapping = {}
    offset = 0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        mapping[int(var.variableID)] = slice(offset, offset + var.dofs)
        offset += var.dofs
    return mapping


def flatten_message_eta(graph) -> np.ndarray:
    pieces = []
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg in factor.messages:
            pieces.append(np.asarray(msg.eta, dtype=float).reshape(-1))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=float)


def flatten_message_lam(graph) -> np.ndarray:
    pieces = []
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg in factor.messages:
            pieces.append(np.asarray(msg.lam, dtype=float).reshape(-1))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=float)


def max_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def build_joint_cov_basis(graph, a0: np.ndarray, groups: list[list[int]], r_reduced: int) -> np.ndarray:
    slices = var_slices(graph)
    cov = np.linalg.inv(a0)
    total_dim = a0.shape[0]

    total_reduced = 0
    full_indices_per_group = []
    local_bases = []
    for group in groups:
        full_indices = []
        for var_id in group:
            sl = slices[int(var_id)]
            full_indices.extend(range(sl.start, sl.stop))
        full_indices = np.array(full_indices, dtype=int)
        block = cov[np.ix_(full_indices, full_indices)]
        eigvals, eigvecs = np.linalg.eigh(block)
        order = np.argsort(eigvals)[::-1]
        r_local = min(int(r_reduced), block.shape[0])
        basis_local = eigvecs[:, order[:r_local]]
        full_indices_per_group.append(full_indices)
        local_bases.append(basis_local)
        total_reduced += r_local

    p = np.zeros((total_dim, total_reduced), dtype=float)
    offset = 0
    for full_indices, basis_local in zip(full_indices_per_group, local_bases):
        r_local = basis_local.shape[1]
        p[np.ix_(full_indices, np.arange(offset, offset + r_local))] = basis_local
        offset += r_local
    return p


def odom_tiny_init_graph(graph, n: int, tiny: float = 1e-12) -> None:
    chain_meas = {}
    for factor in graph.factors[: graph.n_factor_nodes]:
        if getattr(factor, "type", None) != "odometry":
            continue
        i, j = [int(v) for v in factor.adj_vIDs]
        if j == i + 1:
            meas = factor.measurement[0] if isinstance(factor.measurement, list) else factor.measurement
            chain_meas[(i, j)] = np.asarray(meas, dtype=float).reshape(-1)

    mus = {0: np.asarray(graph.var_nodes[0].GT, dtype=float).copy()}
    for i in range(n - 1):
        mus[i + 1] = mus[i] + chain_meas[(i, i + 1)]

    for var in graph.var_nodes[: graph.n_var_nodes]:
        mu = mus[int(var.variableID)].copy()
        var.mu = mu
        var.prior.eta = np.asarray(var.prior.eta, dtype=float).reshape(-1) + tiny * mu
        var.belief.eta = np.asarray(var.belief.lam, dtype=float) @ mu

    for factor in graph.factors[: graph.n_factor_nodes]:
        for adj_var, adj_belief in zip(factor.adj_var_nodes, factor.adj_beliefs):
            adj_belief.eta = np.asarray(adj_var.belief.eta, dtype=float).copy()
            adj_belief.lam = np.asarray(adj_var.belief.lam, dtype=float).copy()
        for msg in factor.messages:
            msg.eta = np.zeros_like(msg.eta)


def reset_residual_graph(residual_graph, template_graph, x: np.ndarray, slices: dict[int, slice]) -> None:
    residual_graph.var_heap.clear()
    residual_graph.var_residual.clear()
    for orig_var, var in zip(
        template_graph.var_nodes[: template_graph.n_var_nodes],
        residual_graph.var_nodes[: residual_graph.n_var_nodes],
    ):
        xi = np.asarray(x[slices[int(orig_var.variableID)]]).reshape(-1)
        var.prior.lam = np.asarray(orig_var.prior.lam, dtype=float).copy()
        var.prior.eta = np.asarray(orig_var.prior.eta, dtype=float).reshape(-1) - np.asarray(orig_var.prior.lam, dtype=float) @ xi
        var.mu = np.zeros(var.dofs, dtype=float)
        var.belief.lam = var.prior.lam.copy()
        var.belief.eta = np.zeros(var.dofs, dtype=float)

    for orig_factor, factor in zip(
        template_graph.factors[: template_graph.n_factor_nodes],
        residual_graph.factors[: residual_graph.n_factor_nodes],
    ):
        local_x = np.concatenate(
            [np.asarray(x[slices[int(orig_var.variableID)]]).reshape(-1) for orig_var in orig_factor.adj_var_nodes]
        )
        abs_eta, abs_lam = orig_factor.compute_factor_absolute(update_self=False)
        factor.factor.lam = np.asarray(abs_lam, dtype=float).copy()
        factor.factor.eta = np.asarray(abs_eta, dtype=float).reshape(-1) - np.asarray(abs_lam, dtype=float) @ local_x
        for msg in factor.messages:
            msg.eta = np.zeros_like(msg.eta)
            msg.lam = np.zeros_like(msg.lam)
        for adj_var, adj_belief in zip(factor.adj_var_nodes, factor.adj_beliefs):
            adj_belief.eta = np.zeros_like(adj_belief.eta)
            adj_belief.lam = np.asarray(adj_var.prior.lam, dtype=float).copy()


def init_odom_with_belief_precision(graph, belief_prec: float) -> None:
    chain_meas = {}
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


@dataclass
class CommonData:
    nodes: list
    edges: list
    x_star: np.ndarray
    b: np.ndarray
    a: np.ndarray
    p: np.ndarray
    ac: np.ndarray


def build_common_data() -> CommonData:
    nodes, edges, exact_graph, _, mg_graph = build_graphs(
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
    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    b, a = exact_graph.joint_distribution_inf_absolute()
    groups = group_list(
        nodes=nodes,
        graph=mg_graph,
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
    p = build_joint_cov_basis(mg_graph, a, groups, r_reduced=4)
    ac = p.T @ a @ p
    return CommonData(nodes=nodes, edges=edges, x_star=x_star, b=b, a=a, p=p, ac=ac)


def selected_points(history: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for p in POINTS:
        if p < len(history):
            out[str(p)] = history[p]
    return out


def run_residual_threshold(common: CommonData, k: int, cycles: int = 100) -> dict[str, object]:
    _, _, exact_graph, base_graph, _ = build_graphs(
        n=512,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=0,
    )
    odom_tiny_init_graph(base_graph, n=512)
    x = mean_vector(base_graph)
    slices = var_slices(base_graph)
    residual_graph = build_noisy_pose_graph(
        common.nodes,
        common.edges,
        prior_sigma=1.0,
        odom_sigma=1.0,
        tiny_prior=1e-12,
        seed=0,
    )

    a_chol = scipy.linalg.cho_factor(common.a, lower=False, check_finite=False)
    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(x, common.x_star),
            "algebraic_residual": float(np.linalg.norm(common.b - common.a @ x)),
        }
    ]
    probes: dict[str, dict[str, float]] = {}

    for cyc in range(1, cycles + 1):
        cycle_start_x = x.copy()
        cycle_start_res = common.b - common.a @ cycle_start_x
        cycle_start_res_norm = float(np.linalg.norm(cycle_start_res))

        if cyc <= 2:
            e_star = scipy.linalg.cho_solve(a_chol, cycle_start_res)

        reset_residual_graph(residual_graph, base_graph, cycle_start_x, slices)
        prev_eta = flatten_message_eta(residual_graph)
        prev_lam = flatten_message_lam(residual_graph)
        prev_e = mean_vector(residual_graph)

        last_eta_delta = 0.0
        last_lam_delta = 0.0
        last_e_step = 0.0
        e_curr = prev_e
        for _ in range(k):
            residual_graph.synchronous_iteration()
            e_curr = mean_vector(residual_graph)
            curr_eta = flatten_message_eta(residual_graph)
            curr_lam = flatten_message_lam(residual_graph)
            last_eta_delta = max_abs_delta(curr_eta, prev_eta)
            last_lam_delta = max_abs_delta(curr_lam, prev_lam)
            last_e_step = float(np.linalg.norm(e_curr - prev_e))
            prev_eta = curr_eta
            prev_lam = curr_lam
            prev_e = e_curr

        if cyc <= 2:
            lin_res_ratio = float(np.linalg.norm(cycle_start_res - common.a @ e_curr) / max(cycle_start_res_norm, 1e-15))
            e_rel_to_exact = float(np.linalg.norm(e_curr - e_star) / max(np.linalg.norm(e_star), 1e-15))
            probes[f"cycle{cyc}"] = {
                "cycle_start_relerr": relative_error_vec(cycle_start_x, common.x_star),
                "cycle_start_residual": cycle_start_res_norm,
                "inner_linear_residual_ratio": lin_res_ratio,
                "inner_e_rel_to_exact": e_rel_to_exact,
                "last_message_eta_delta_max": last_eta_delta,
                "last_message_lam_delta_max": last_lam_delta,
                "last_e_step_norm": last_e_step,
                "pre_end_relerr": relative_error_vec(cycle_start_x + e_curr, common.x_star),
                "pre_end_residual": float(np.linalg.norm(common.b - common.a @ (cycle_start_x + e_curr))),
            }

        x = cycle_start_x + e_curr
        residual = common.b - common.a @ x
        yc = np.linalg.solve(common.ac, common.p.T @ residual)
        x = x + common.p @ yc
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, common.x_star),
                "algebraic_residual": float(np.linalg.norm(common.b - common.a @ x)),
            }
        )
        if not np.isfinite(history[-1]["relative_state_error"]) or history[-1]["relative_state_error"] > 1e12:
            break

    rel_hist = [row["relative_state_error"] for row in history]
    best_cycle = int(np.argmin(rel_hist))
    return {
        "method": "residual_base_preK_post0",
        "k": k,
        "probes": probes,
        "summary": {
            "final_cycle": history[-1]["cycle"],
            "final_relerr": rel_hist[-1],
            "final_residual": history[-1]["algebraic_residual"],
            "best_relerr": rel_hist[best_cycle],
            "best_cycle": best_cycle,
            "diverged": bool(not np.isfinite(rel_hist[-1]) or rel_hist[-1] > 1e6),
        },
        "points": selected_points(history),
    }


def run_persistent_threshold(common: CommonData, k: int, cycles: int = 100) -> dict[str, object]:
    _, _, exact_graph, _, base_graph = build_graphs(
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
    init_odom_with_belief_precision(base_graph, belief_prec=1e-6)

    groups = group_list(
        nodes=common.nodes,
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

    x = mean_vector(base_graph)
    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(x, common.x_star),
            "algebraic_residual": float(np.linalg.norm(common.b - common.a @ x)),
        }
    ]
    probes: dict[str, dict[str, float]] = {}

    for cyc in range(1, cycles + 1):
        cycle_start_x = mean_vector(base_graph)
        cycle_start_res = common.b - common.a @ cycle_start_x
        cycle_start_res_norm = float(np.linalg.norm(cycle_start_res))
        prev_eta = flatten_message_eta(base_graph)
        prev_lam = flatten_message_lam(base_graph)
        prev_x = cycle_start_x.copy()

        last_eta_delta = 0.0
        last_lam_delta = 0.0
        last_x_step = 0.0
        for _ in range(k):
            base_graph.synchronous_iteration()
            curr_x = mean_vector(base_graph)
            curr_eta = flatten_message_eta(base_graph)
            curr_lam = flatten_message_lam(base_graph)
            last_eta_delta = max_abs_delta(curr_eta, prev_eta)
            last_lam_delta = max_abs_delta(curr_lam, prev_lam)
            last_x_step = float(np.linalg.norm(curr_x - prev_x))
            prev_eta = curr_eta
            prev_lam = curr_lam
            prev_x = curr_x

        pre_end_x = mean_vector(base_graph)
        if cyc <= 2:
            probes[f"cycle{cyc}"] = {
                "cycle_start_relerr": relative_error_vec(cycle_start_x, common.x_star),
                "cycle_start_residual": cycle_start_res_norm,
                "pre_residual_ratio": float(np.linalg.norm(common.b - common.a @ pre_end_x) / max(cycle_start_res_norm, 1e-15)),
                "last_message_eta_delta_max": last_eta_delta,
                "last_message_lam_delta_max": last_lam_delta,
                "last_x_step_norm": last_x_step,
                "pre_end_relerr": relative_error_vec(pre_end_x, common.x_star),
                "pre_end_residual": float(np.linalg.norm(common.b - common.a @ pre_end_x)),
            }

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_keep_messages(base_graph, delta_x)

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, common.x_star),
                "algebraic_residual": float(np.linalg.norm(common.b - common.a @ x)),
            }
        )
        if not np.isfinite(history[-1]["relative_state_error"]) or history[-1]["relative_state_error"] > 1e12:
            break

    rel_hist = [row["relative_state_error"] for row in history]
    best_cycle = int(np.argmin(rel_hist))
    return {
        "method": "persistent_state_preK_post0",
        "k": k,
        "probes": probes,
        "summary": {
            "final_cycle": history[-1]["cycle"],
            "final_relerr": rel_hist[-1],
            "final_residual": history[-1]["algebraic_residual"],
            "best_relerr": rel_hist[best_cycle],
            "best_cycle": best_cycle,
            "diverged": bool(not np.isfinite(rel_hist[-1]) or rel_hist[-1] > 1e6),
        },
        "points": selected_points(history),
    }


def write_summary_csv(path: pathlib.Path, results: dict[str, dict[str, object]]) -> None:
    lines = [
        "family,k,final_cycle,final_relerr,final_residual,best_relerr,best_cycle,diverged,"
        "cycle1_metric_a,cycle1_metric_b,cycle2_metric_a,cycle2_metric_b"
    ]
    for family, fam_results in results.items():
        for k in K_LIST:
            row = fam_results[str(k)]
            summary = row["summary"]
            probes = row["probes"]
            if family == "residual_base":
                c1a = probes["cycle1"]["inner_linear_residual_ratio"]
                c1b = probes["cycle1"]["last_message_lam_delta_max"]
                c2a = probes["cycle2"]["inner_linear_residual_ratio"]
                c2b = probes["cycle2"]["last_message_lam_delta_max"]
            else:
                c1a = probes["cycle1"]["pre_residual_ratio"]
                c1b = probes["cycle1"]["last_message_lam_delta_max"]
                c2a = probes["cycle2"]["pre_residual_ratio"]
                c2b = probes["cycle2"]["last_message_lam_delta_max"]
            lines.append(
                f"{family},{k},{summary['final_cycle']},{summary['final_relerr']},{summary['final_residual']},"
                f"{summary['best_relerr']},{summary['best_cycle']},{summary['diverged']},"
                f"{c1a},{c1b},{c2a},{c2b}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    common = build_common_data()
    results = {"residual_base": {}, "persistent_state": {}}
    for k in K_LIST:
        results["residual_base"][str(k)] = run_residual_threshold(common, k=k, cycles=100)
        results["persistent_state"][str(k)] = run_persistent_threshold(common, k=k, cycles=100)

    json_path = OUT_DIR / "inner_sync_threshold_analysis.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = OUT_DIR / "inner_sync_threshold_analysis_summary.csv"
    write_summary_csv(csv_path, results)

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
