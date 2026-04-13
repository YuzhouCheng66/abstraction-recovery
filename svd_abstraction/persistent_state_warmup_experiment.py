from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs, group_list
from svd_abstraction.persistent_state_exact_coarse_experiment import (
    OUT_DIR,
    init_odom_with_belief_precision,
    inject_correction_keep_messages,
    mean_vector,
    pin_state_to_x_keep_messages,
    relative_error_vec,
    set_absolute_factors,
)
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


def snapshot_message_lams(graph) -> list[np.ndarray]:
    return [
        np.array(msg.lam, copy=True)
        for factor in graph.factors[: graph.n_factor_nodes]
        for msg in factor.messages
    ]


def max_message_lam_delta(prev: list[np.ndarray], graph) -> float:
    deltas = []
    idx = 0
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg in factor.messages:
            deltas.append(float(np.max(np.abs(msg.lam - prev[idx]))))
            idx += 1
    return max(deltas, default=0.0)


def warmup_variance_at_pinned_geometry(graph, x_pin: np.ndarray, tol: float = 1e-8, max_sweeps: int = 500) -> dict[str, float]:
    sweeps = 0
    delta = np.inf
    while sweeps < max_sweeps and delta > tol:
        prev = snapshot_message_lams(graph)
        graph.synchronous_iteration()
        pin_state_to_x_keep_messages(graph, x_pin)
        delta = max_message_lam_delta(prev, graph)
        sweeps += 1
    return {"warmup_sweeps": sweeps, "warmup_lam_delta": float(delta)}


def build_level(base_graph, nodes):
    groups = group_list(
        nodes=nodes,
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
    return level


def run_schedule(pre_sweeps: int, post_sweeps: int, cycles: int, do_warmup: bool) -> dict[str, object]:
    nodes, _, exact_graph, _, base_graph = build_graphs(
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
    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    eta, a = exact_graph.joint_distribution_inf_absolute()

    init_odom_with_belief_precision(base_graph, belief_prec=1e-6)
    x_pin0 = mean_vector(base_graph).copy()

    warmup_info = {}
    if do_warmup:
        warmup_info = warmup_variance_at_pinned_geometry(base_graph, x_pin0)

    level = build_level(base_graph, nodes)

    history = [
        {
            "cycle": 0,
            "relative_state_error": relative_error_vec(mean_vector(base_graph), x_star),
            "algebraic_residual": float(np.linalg.norm(eta - a @ mean_vector(base_graph))),
        }
    ]

    for cyc in range(1, cycles + 1):
        for _ in range(pre_sweeps):
            base_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_keep_messages(base_graph, delta_x)

        for _ in range(post_sweeps):
            base_graph.synchronous_iteration()

        x = mean_vector(base_graph)
        history.append(
            {
                "cycle": cyc,
                "relative_state_error": relative_error_vec(x, x_star),
                "algebraic_residual": float(np.linalg.norm(eta - a @ x)),
            }
        )

    return {
        "config": {
            "pre_sweeps": pre_sweeps,
            "post_sweeps": post_sweeps,
            "cycles": cycles,
            "warmup": do_warmup,
        }
        | warmup_info,
        "history": history,
    }


def main() -> None:
    results = {
        "pre2_post0_no_warmup": run_schedule(pre_sweeps=2, post_sweeps=0, cycles=20, do_warmup=False),
        "pre2_post0_warmup": run_schedule(pre_sweeps=2, post_sweeps=0, cycles=20, do_warmup=True),
        "pre10_post0_no_warmup": run_schedule(pre_sweeps=10, post_sweeps=0, cycles=20, do_warmup=False),
        "pre10_post0_warmup": run_schedule(pre_sweeps=10, post_sweeps=0, cycles=20, do_warmup=True),
        "pre10_post10_no_warmup": run_schedule(pre_sweeps=10, post_sweeps=10, cycles=20, do_warmup=False),
        "pre10_post10_warmup": run_schedule(pre_sweeps=10, post_sweeps=10, cycles=20, do_warmup=True),
    }

    out_path = OUT_DIR / "persistent_state_warmup_experiment.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for name, result in results.items():
        print(name)
        cfg = result["config"]
        if cfg.get("warmup"):
            print("warmup", cfg["warmup_sweeps"], cfg["warmup_lam_delta"])
        for row in result["history"]:
            if row["cycle"] in (0, 1, 2, 3, 4, 5, 10, 15, 20):
                print(row["cycle"], row["relative_state_error"], row["algebraic_residual"])


if __name__ == "__main__":
    main()
