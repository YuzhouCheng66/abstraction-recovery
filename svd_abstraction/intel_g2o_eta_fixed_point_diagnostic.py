from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
from scipy import sparse

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import build_linearized_local_graph_g2o
from svd_abstraction.g2o_se2 import linearize_g2o_problem
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.gbp_fixed_point import apply_eta_fixed_point_solution
from svd_abstraction.gbp_fixed_point import build_eta_fixed_point_system
from svd_abstraction.gbp_fixed_point import max_message_eta_residual
from svd_abstraction.gbp_fixed_point import solve_eta_fixed_point
from svd_abstraction.intel_g2o_adaptive_policy import RESULT_DIR
from svd_abstraction.intel_g2o_adaptive_policy import direct_optimum_objective
from svd_abstraction.intel_g2o_adaptive_policy import direct_reference_poses
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import lam_state_messages
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import mean_vector
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph_eta_only
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices


def belief_lam_state(graph) -> np.ndarray:
    parts = [
        np.asarray(var.belief.lam, dtype=float).reshape(-1)
        for var in graph.var_nodes[: graph.n_var_nodes]
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)


def resolve_g2o_path() -> pathlib.Path:
    candidates = [
        pathlib.Path("/home/yuzhou/Desktop/input_INTEL_g2o.g2o"),
        pathlib.Path("/home/yuzhou/Desktop/g2o/input_INTEL_g2o.g2o"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find input_INTEL_g2o.g2o in any known location: "
        + ", ".join(str(path) for path in candidates)
    )


def message_eta_state(graph) -> np.ndarray:
    parts = [
        np.asarray(msg.eta, dtype=float).reshape(-1)
        for factor in graph.factors[: graph.n_factor_nodes]
        for msg in factor.messages
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)


def collect_state_metrics(graph) -> dict[str, float]:
    mu = mean_vector(graph)
    msg_eta = message_eta_state(graph)

    belief_eta_max = 0.0
    belief_eta_norm_sq = 0.0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        eta = np.asarray(var.belief.eta, dtype=float).reshape(-1)
        belief_eta_max = max(belief_eta_max, float(np.max(np.abs(eta))))
        belief_eta_norm_sq += float(eta @ eta)

    return {
        "mu_norm": float(np.linalg.norm(mu)),
        "mu_max_abs": float(np.max(np.abs(mu))) if mu.size else 0.0,
        "message_eta_norm": float(np.linalg.norm(msg_eta)),
        "message_eta_max_abs": float(np.max(np.abs(msg_eta))) if msg_eta.size else 0.0,
        "belief_eta_norm": float(np.sqrt(belief_eta_norm_sq)),
        "belief_eta_max_abs": belief_eta_max,
    }


def warmup_lam_history(
    residual_graph,
    template_graph,
    zero: np.ndarray,
    slices: dict[int, slice],
    max_sweeps: int,
    tol: float,
    print_every: int,
) -> tuple[list[dict[str, float]], int | None]:
    reset_residual_graph(residual_graph, template_graph, zero, slices)

    prev_msg = lam_state_messages(residual_graph)
    prev_bel = belief_lam_state(residual_graph)
    history: list[dict[str, float]] = []
    converged_sweep: int | None = None

    for sweep in range(1, int(max_sweeps) + 1):
        residual_graph.synchronous_iteration()
        curr_msg = lam_state_messages(residual_graph)
        curr_bel = belief_lam_state(residual_graph)

        msg_delta = float(np.max(np.abs(curr_msg - prev_msg))) if curr_msg.size else 0.0
        bel_delta = float(np.max(np.abs(curr_bel - prev_bel))) if curr_bel.size else 0.0
        row = {
            "sweep": int(sweep),
            "message_lam_delta_max_abs": msg_delta,
            "belief_lam_delta_max_abs": bel_delta,
            "message_lam_norm": float(np.linalg.norm(curr_msg)),
            "belief_lam_norm": float(np.linalg.norm(curr_bel)),
        }
        history.append(row)

        if sweep <= 10 or sweep in {20, 50, 100, 200, 500, 1000, 2000, 5000, 10000} or (
            print_every > 0 and sweep % print_every == 0
        ):
            print(
                f"[lam] sweep={sweep} msg_delta={msg_delta:.3e} belief_delta={bel_delta:.3e}",
                flush=True,
            )

        if msg_delta < tol and bel_delta < tol:
            converged_sweep = sweep
            break

        prev_msg = curr_msg
        prev_bel = curr_bel

    return history, converged_sweep


def compute_variable_balance_residuals(graph) -> tuple[float, float]:
    worst = 0.0
    accum = 0.0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        eta = np.asarray(var.prior.eta, dtype=float).reshape(-1).copy()
        for factor in var.adj_factors:
            msg_ix = factor.var_index[var.variableID]
            eta += np.asarray(factor.messages[msg_ix].eta, dtype=float).reshape(-1)
        eta -= np.asarray(var.belief.eta, dtype=float).reshape(-1)
        worst = max(worst, float(np.max(np.abs(eta))))
        accum += float(eta @ eta)
    return worst, float(np.sqrt(accum))


def save_csv(rows: list[dict], path: pathlib.Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_graph_state_npz(graph, path: pathlib.Path) -> None:
    vars_ = graph.var_nodes[: graph.n_var_nodes]
    factors = graph.factors[: graph.n_factor_nodes]

    var_ids = np.array([int(var.variableID) for var in vars_], dtype=int)
    var_mu = np.stack([np.asarray(var.mu, dtype=float).reshape(-1) for var in vars_], axis=0)
    var_prior_eta = np.stack([np.asarray(var.prior.eta, dtype=float).reshape(-1) for var in vars_], axis=0)
    var_prior_lam = np.stack([np.asarray(var.prior.lam, dtype=float) for var in vars_], axis=0)
    var_belief_eta = np.stack([np.asarray(var.belief.eta, dtype=float).reshape(-1) for var in vars_], axis=0)
    var_belief_lam = np.stack([np.asarray(var.belief.lam, dtype=float) for var in vars_], axis=0)

    factor_ids = np.array([int(factor.factorID) for factor in factors], dtype=int)
    factor_types = np.array([str(getattr(factor, "type", "factor")) for factor in factors], dtype="U32")
    factor_arities = np.array([len(factor.adj_var_nodes) for factor in factors], dtype=int)
    factor_total_dofs = np.array([int(factor.factor.eta.size) for factor in factors], dtype=int)
    factor_adj_padded = np.full((len(factors), 2), -1, dtype=int)
    factor_eta_padded = np.zeros((len(factors), 6), dtype=float)
    factor_lam_padded = np.zeros((len(factors), 6, 6), dtype=float)
    for idx, factor in enumerate(factors):
        adj = [int(var.variableID) for var in factor.adj_var_nodes]
        factor_adj_padded[idx, : len(adj)] = adj
        dofs = factor_total_dofs[idx]
        factor_eta = np.asarray(factor.factor.eta, dtype=float).reshape(-1)
        factor_lam = np.asarray(factor.factor.lam, dtype=float)
        factor_eta_padded[idx, :dofs] = factor_eta
        factor_lam_padded[idx, :dofs, :dofs] = factor_lam

    message_rows = []
    message_eta = []
    message_lam = []
    adj_belief_eta = []
    adj_belief_lam = []
    for factor in factors:
        for local_idx, (var, msg, adj) in enumerate(zip(factor.adj_var_nodes, factor.messages, factor.adj_beliefs)):
            other_idx = 1 - local_idx if len(factor.adj_var_nodes) == 2 else -1
            other_var_id = int(factor.adj_var_nodes[other_idx].variableID) if other_idx >= 0 else -1
            message_rows.append(
                {
                    "factor_id": int(factor.factorID),
                    "factor_type": str(getattr(factor, "type", "factor")),
                    "target_idx": int(local_idx),
                    "target_var_id": int(var.variableID),
                    "other_idx": int(other_idx),
                    "other_var_id": int(other_var_id),
                }
            )
            message_eta.append(np.asarray(msg.eta, dtype=float).reshape(-1))
            message_lam.append(np.asarray(msg.lam, dtype=float))
            adj_belief_eta.append(np.asarray(adj.eta, dtype=float).reshape(-1))
            adj_belief_lam.append(np.asarray(adj.lam, dtype=float))

    np.savez_compressed(
        path,
        var_ids=var_ids,
        var_mu=var_mu,
        var_prior_eta=var_prior_eta,
        var_prior_lam=var_prior_lam,
        var_belief_eta=var_belief_eta,
        var_belief_lam=var_belief_lam,
        factor_ids=factor_ids,
        factor_types=factor_types,
        factor_arities=factor_arities,
        factor_total_dofs=factor_total_dofs,
        factor_adj_padded=factor_adj_padded,
        factor_eta_padded=factor_eta_padded,
        factor_lam_padded=factor_lam_padded,
        message_eta=np.stack(message_eta, axis=0),
        message_lam=np.stack(message_lam, axis=0),
        adj_belief_eta=np.stack(adj_belief_eta, axis=0),
        adj_belief_lam=np.stack(adj_belief_lam, axis=0),
    )

    save_csv(message_rows, path.with_name(path.stem + "_message_descriptors.csv"))


def run_diagnostic(
    lam_tol: float,
    lam_max_sweeps: int,
    print_every: int,
    stem: str,
) -> dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    g2o_path = resolve_g2o_path()
    problem = parse_g2o_se2(g2o_path)
    base_poses = direct_reference_poses()
    direct_obj = direct_optimum_objective()

    A, b = linearize_g2o_problem(problem, base_poses)
    A = A + 1e-10 * sparse.eye(A.shape[0], format="csc")
    e_star = sparse.linalg.spsolve(A, b)

    template_graph = build_linearized_local_graph_g2o(problem, base_poses)
    residual_graph = build_linearized_local_graph_g2o(problem, base_poses)
    slices = var_slices(template_graph)
    zero = np.zeros(template_graph.n_var_nodes * 3, dtype=float)

    lam_history, converged_sweep = warmup_lam_history(
        residual_graph=residual_graph,
        template_graph=template_graph,
        zero=zero,
        slices=slices,
        max_sweeps=lam_max_sweeps,
        tol=lam_tol,
        print_every=print_every,
    )
    lam_last = lam_history[-1] if lam_history else {
        "sweep": 0,
        "message_lam_delta_max_abs": 0.0,
        "belief_lam_delta_max_abs": 0.0,
        "message_lam_norm": 0.0,
        "belief_lam_norm": 0.0,
    }

    reset_residual_graph_eta_only(
        residual_graph=residual_graph,
        template_graph=template_graph,
        x=zero,
        slices=slices,
    )

    system = build_eta_fixed_point_system(residual_graph)
    solution = solve_eta_fixed_point(system)
    apply_eta_fixed_point_solution(residual_graph, system, solution)

    fixed_point_metrics = collect_state_metrics(residual_graph)
    variable_balance_max, variable_balance_norm = compute_variable_balance_residuals(residual_graph)
    fixed_point_message_residual = max_message_eta_residual(residual_graph, system)
    pre_mu = mean_vector(residual_graph).copy()
    pre_messages = message_eta_state(residual_graph).copy()

    fixed_point_state_path = RESULT_DIR / f"{stem}_state_fixed_point.npz"
    save_graph_state_npz(residual_graph, fixed_point_state_path)

    residual_graph.synchronous_iteration(fixed_lam=True)

    post_mu = mean_vector(residual_graph).copy()
    post_messages = message_eta_state(residual_graph).copy()
    post_metrics = collect_state_metrics(residual_graph)
    post_balance_max, post_balance_norm = compute_variable_balance_residuals(residual_graph)

    after_state_path = RESULT_DIR / f"{stem}_state_after_one_more_fixedlam.npz"
    save_graph_state_npz(residual_graph, after_state_path)

    message_descriptor_rows = []
    factor_type_by_id = {
        int(factor.factorID): str(getattr(factor, "type", "factor"))
        for factor in residual_graph.factors[: residual_graph.n_factor_nodes]
    }
    for unknown_idx, msg in enumerate(system.directed_messages):
        message_descriptor_rows.append(
            {
                "unknown_index": int(unknown_idx),
                "factor_id": int(msg.factor_id),
                "factor_type": factor_type_by_id[msg.factor_id],
                "target_idx": int(msg.target_idx),
                "target_var_id": int(msg.target_var_id),
                "other_idx": int(msg.other_idx),
                "other_var_id": int(msg.other_var_id),
                "global_start": int(msg.global_slice.start),
                "global_stop": int(msg.global_slice.stop),
            }
        )

    save_csv(lam_history, RESULT_DIR / f"{stem}_lam_history.csv")
    save_csv(message_descriptor_rows, RESULT_DIR / f"{stem}_eta_system_unknowns.csv")
    sparse.save_npz(RESULT_DIR / f"{stem}_eta_system_matrix.npz", system.matrix)
    np.savez_compressed(
        RESULT_DIR / f"{stem}_eta_system_vectors.npz",
        rhs=system.rhs,
        solution=solution,
        b=b,
        e_star=e_star,
    )

    summary = {
        "config": {
            "g2o_path": str(g2o_path),
            "lam_tol": float(lam_tol),
            "lam_max_sweeps": int(lam_max_sweeps),
            "print_every": int(print_every),
            "stem": stem,
        },
        "direct_reference": {
            "objective": float(direct_obj),
            "linear_rhs_norm": float(np.linalg.norm(b)),
            "linear_rhs_max_abs": float(np.max(np.abs(b))) if b.size else 0.0,
            "exact_local_step_norm": float(np.linalg.norm(e_star)),
        },
        "lam_warmup": {
            "converged": converged_sweep is not None,
            "converged_sweep": None if converged_sweep is None else int(converged_sweep),
            "last_sweep": int(lam_last["sweep"]),
            "last_message_lam_delta_max_abs": float(lam_last["message_lam_delta_max_abs"]),
            "last_belief_lam_delta_max_abs": float(lam_last["belief_lam_delta_max_abs"]),
            "last_message_lam_norm": float(lam_last["message_lam_norm"]),
            "last_belief_lam_norm": float(lam_last["belief_lam_norm"]),
        },
        "eta_system": {
            "num_unknowns": int(system.matrix.shape[0]),
            "nnz": int(system.matrix.nnz),
            "rhs_norm": float(np.linalg.norm(system.rhs)),
            "solution_norm": float(np.linalg.norm(solution)),
        },
        "fixed_point_state": {
            **fixed_point_metrics,
            "variable_balance_max_abs": float(variable_balance_max),
            "variable_balance_norm": float(variable_balance_norm),
            "message_eta_fixed_point_residual_max_abs": float(fixed_point_message_residual),
        },
        "one_more_fixed_lam_iteration": {
            **post_metrics,
            "variable_balance_max_abs": float(post_balance_max),
            "variable_balance_norm": float(post_balance_norm),
            "message_eta_delta_max_abs": float(np.max(np.abs(post_messages - pre_messages))) if pre_messages.size else 0.0,
            "message_eta_delta_norm": float(np.linalg.norm(post_messages - pre_messages)),
            "mu_delta_norm": float(np.linalg.norm(post_mu - pre_mu)),
        },
        "artifacts": {
            "lam_history_csv": str(RESULT_DIR / f"{stem}_lam_history.csv"),
            "eta_system_unknowns_csv": str(RESULT_DIR / f"{stem}_eta_system_unknowns.csv"),
            "eta_system_matrix_npz": str(RESULT_DIR / f"{stem}_eta_system_matrix.npz"),
            "eta_system_vectors_npz": str(RESULT_DIR / f"{stem}_eta_system_vectors.npz"),
            "state_fixed_point_npz": str(fixed_point_state_path),
            "state_fixed_point_message_descriptors_csv": str(
                fixed_point_state_path.with_name(fixed_point_state_path.stem + "_message_descriptors.csv")
            ),
            "state_after_one_more_fixedlam_npz": str(after_state_path),
            "state_after_one_more_fixedlam_message_descriptors_csv": str(
                after_state_path.with_name(after_state_path.stem + "_message_descriptors.csv")
            ),
        },
    }

    summary_path = RESULT_DIR / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converge Intel residual-graph lam state, then solve eta fixed point exactly.",
    )
    parser.add_argument("--lam-tol", type=float, default=1e-12)
    parser.add_argument("--lam-max-sweeps", type=int, default=5000)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument(
        "--stem",
        type=str,
        default="intel_g2o_optimal_residual_eta_fixed_point",
    )
    args = parser.parse_args()

    run_diagnostic(
        lam_tol=args.lam_tol,
        lam_max_sweeps=args.lam_max_sweeps,
        print_every=args.print_every,
        stem=args.stem,
    )


if __name__ == "__main__":
    main()
