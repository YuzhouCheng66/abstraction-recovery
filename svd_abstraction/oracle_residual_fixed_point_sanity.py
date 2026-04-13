from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.gbp_fixed_point import (
    apply_eta_fixed_point_solution,
    build_eta_fixed_point_system,
    max_message_eta_residual,
    solve_eta_fixed_point,
)
from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs
from svd_abstraction.pose_graph import build_noisy_pose_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import (
    mean_vector,
    reset_residual_graph_eta_only,
    var_slices,
    warmup_residual_lam,
)


def relative_state_error(x: np.ndarray, x_star: np.ndarray) -> float:
    return float(np.linalg.norm(x - x_star) / max(np.linalg.norm(x_star), 1e-15))


def metrics(x: np.ndarray, x_star: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    err = x - x_star
    residual = b - a @ x
    return {
        "relative_state_error": relative_state_error(x, x_star),
        "algebraic_residual": float(np.linalg.norm(residual)),
        "energy_gap": float(0.5 * err @ (a @ err)),
    }


def run_oracle_residual_fixed_point_sanity(
    n: int = 512,
    seed: int = 0,
    prior_sigma: float = 1.0,
    odom_sigma: float = 1.0,
    lam_tol: float = 1e-10,
    lam_max_sweeps: int = 5000,
) -> dict:
    nodes, edges, exact_graph, base_graph, _ = build_graphs(
        n=n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        seed=seed,
    )
    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    b0, a0 = exact_graph.joint_distribution_inf_absolute()
    slices = var_slices(base_graph)

    residual_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )

    lam_sweeps, lam_delta = warmup_residual_lam(
        residual_graph=residual_graph,
        template_graph=base_graph,
        x=x_star,
        slices=slices,
        max_sweeps=lam_max_sweeps,
        tol=lam_tol,
    )

    reset_residual_graph_eta_only(
        residual_graph=residual_graph,
        template_graph=base_graph,
        x=x_star,
        slices=slices,
    )

    system = build_eta_fixed_point_system(residual_graph)
    solution = solve_eta_fixed_point(system)
    apply_eta_fixed_point_solution(residual_graph, system, solution)

    e = mean_vector(residual_graph)
    x = x_star + e
    before = metrics(x, x_star, a0, b0)

    belief_eta_max = 0.0
    belief_mu_max = 0.0
    for var in residual_graph.var_nodes[: residual_graph.n_var_nodes]:
        belief_eta_max = max(
            belief_eta_max,
            float(np.max(np.abs(np.asarray(var.belief.eta, dtype=float).reshape(-1)))),
        )
        belief_mu_max = max(
            belief_mu_max,
            float(np.max(np.abs(np.asarray(var.mu, dtype=float).reshape(-1)))),
        )

    fixed_point_message_residual = max_message_eta_residual(residual_graph, system)

    pre_messages = []
    for factor in residual_graph.factors[: residual_graph.n_factor_nodes]:
        for msg in factor.messages:
            pre_messages.append(np.asarray(msg.eta, dtype=float).reshape(-1).copy())
    pre_messages_vec = np.concatenate(pre_messages) if pre_messages else np.zeros(0, dtype=float)
    pre_e = mean_vector(residual_graph).copy()

    residual_graph.synchronous_iteration(fixed_lam=True)

    post_messages = []
    for factor in residual_graph.factors[: residual_graph.n_factor_nodes]:
        for msg in factor.messages:
            post_messages.append(np.asarray(msg.eta, dtype=float).reshape(-1).copy())
    post_messages_vec = np.concatenate(post_messages) if post_messages else np.zeros(0, dtype=float)
    post_e = mean_vector(residual_graph).copy()
    x_after = x_star + post_e
    after = metrics(x_after, x_star, a0, b0)

    return {
        "config": {
            "n": n,
            "seed": seed,
            "prior_sigma": prior_sigma,
            "odom_sigma": odom_sigma,
            "lam_tol": lam_tol,
            "lam_max_sweeps": lam_max_sweeps,
        },
        "lam_warmup": {
            "sweeps": lam_sweeps,
            "delta": lam_delta,
        },
        "eta_system": {
            "num_unknowns": int(system.matrix.shape[0]),
            "nnz": int(system.matrix.nnz),
            "solution_norm": float(np.linalg.norm(solution)),
        },
        "fixed_point_state": {
            **before,
            "e_norm": float(np.linalg.norm(e)),
            "belief_eta_max_abs": belief_eta_max,
            "belief_mu_max_abs": belief_mu_max,
            "message_eta_fixed_point_residual_max_abs": fixed_point_message_residual,
        },
        "one_more_fixed_lam_iteration": {
            **after,
            "message_eta_delta_max_abs": float(np.max(np.abs(post_messages_vec - pre_messages_vec)))
            if pre_messages_vec.size
            else 0.0,
            "message_eta_delta_norm": float(np.linalg.norm(post_messages_vec - pre_messages_vec)),
            "e_delta_norm": float(np.linalg.norm(post_e - pre_e)),
            "post_e_norm": float(np.linalg.norm(post_e)),
        },
    }


def main() -> None:
    result = run_oracle_residual_fixed_point_sanity()
    out_dir = pathlib.Path("svd_abstraction/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oracle_residual_fixed_point_sanity.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
