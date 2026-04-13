from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.inner_sync_threshold_analysis import (
    build_common_data,
    flatten_message_eta,
    flatten_message_lam,
    inject_correction_keep_messages,
    init_odom_with_belief_precision,
    max_abs_delta,
    mean_vector,
    relative_error_vec,
    set_absolute_factors,
)
from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs, group_list
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_adaptive_target(target_ratio: float, cycles: int = 100, max_pre_sweeps: int = 200) -> dict[str, object]:
    common = build_common_data()
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

    history: list[dict[str, float]] = []
    x0 = mean_vector(base_graph)
    history.append(
        {
            "cycle": 0,
            "k_used": 0,
            "target_ratio": float(target_ratio),
            "pre_residual_ratio": 1.0,
            "last_message_lam_delta_max": 0.0,
            "last_message_eta_delta_max": 0.0,
            "last_x_step_norm": 0.0,
            "pre_end_relative_state_error": relative_error_vec(x0, common.x_star),
            "pre_end_algebraic_residual": float(np.linalg.norm(common.b - common.a @ x0)),
            "relative_state_error": relative_error_vec(x0, common.x_star),
            "algebraic_residual": float(np.linalg.norm(common.b - common.a @ x0)),
        }
    )

    for cyc in range(1, cycles + 1):
        cycle_start_x = mean_vector(base_graph)
        cycle_start_residual = common.b - common.a @ cycle_start_x
        cycle_start_res_norm = float(np.linalg.norm(cycle_start_residual))

        prev_eta = flatten_message_eta(base_graph)
        prev_lam = flatten_message_lam(base_graph)
        prev_x = cycle_start_x.copy()

        current_ratio = 1.0
        current_x = cycle_start_x.copy()
        current_residual = cycle_start_res_norm
        last_eta_delta = 0.0
        last_lam_delta = 0.0
        last_x_step = 0.0
        k_used = 0

        while k_used < max_pre_sweeps and current_ratio > target_ratio:
            base_graph.synchronous_iteration()
            k_used += 1

            current_x = mean_vector(base_graph)
            current_residual = float(np.linalg.norm(common.b - common.a @ current_x))
            current_ratio = current_residual / max(cycle_start_res_norm, 1e-15)

            curr_eta = flatten_message_eta(base_graph)
            curr_lam = flatten_message_lam(base_graph)
            last_eta_delta = max_abs_delta(curr_eta, prev_eta)
            last_lam_delta = max_abs_delta(curr_lam, prev_lam)
            last_x_step = float(np.linalg.norm(current_x - prev_x))
            prev_eta = curr_eta
            prev_lam = curr_lam
            prev_x = current_x

        pre_end_x = current_x.copy()
        pre_end_rel = relative_error_vec(pre_end_x, common.x_star)
        pre_end_res = current_residual

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_x = level.prolongate(delta_z)
        inject_correction_keep_messages(base_graph, delta_x)

        x = mean_vector(base_graph)
        residual = float(np.linalg.norm(common.b - common.a @ x))
        rel = relative_error_vec(x, common.x_star)
        history.append(
            {
                "cycle": cyc,
                "k_used": int(k_used),
                "target_ratio": float(target_ratio),
                "pre_residual_ratio": float(current_ratio),
                "last_message_lam_delta_max": float(last_lam_delta),
                "last_message_eta_delta_max": float(last_eta_delta),
                "last_x_step_norm": float(last_x_step),
                "pre_end_relative_state_error": float(pre_end_rel),
                "pre_end_algebraic_residual": float(pre_end_res),
                "relative_state_error": float(rel),
                "algebraic_residual": float(residual),
            }
        )
        if not np.isfinite(rel) or rel > 1e12:
            break

    rel_hist = [row["relative_state_error"] for row in history]
    best_cycle = int(np.argmin(rel_hist))
    return {
        "config": {
            "target_ratio": float(target_ratio),
            "cycles": int(cycles),
            "max_pre_sweeps": int(max_pre_sweeps),
        },
        "summary": {
            "final_cycle": int(history[-1]["cycle"]),
            "final_relerr": float(history[-1]["relative_state_error"]),
            "final_residual": float(history[-1]["algebraic_residual"]),
            "best_relerr": float(rel_hist[best_cycle]),
            "best_cycle": int(best_cycle),
            "mean_k_used": float(np.mean([row["k_used"] for row in history[1:]])) if len(history) > 1 else 0.0,
            "max_k_used": int(max([row["k_used"] for row in history[1:]], default=0)),
            "min_k_used": int(min([row["k_used"] for row in history[1:]], default=0)),
        },
        "history": history,
    }


def write_csv(path: pathlib.Path, results: dict[str, dict[str, object]]) -> None:
    lines = [
        "target,cycle,k_used,pre_residual_ratio,last_message_lam_delta_max,last_message_eta_delta_max,last_x_step_norm,"
        "pre_end_relative_state_error,pre_end_algebraic_residual,relative_state_error,algebraic_residual"
    ]
    for target, payload in results.items():
        for row in payload["history"]:
            lines.append(
                f"{target},{row['cycle']},{row['k_used']},{row['pre_residual_ratio']},"
                f"{row['last_message_lam_delta_max']},{row['last_message_eta_delta_max']},{row['last_x_step_norm']},"
                f"{row['pre_end_relative_state_error']},{row['pre_end_algebraic_residual']},"
                f"{row['relative_state_error']},{row['algebraic_residual']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = {
        "0.5": run_adaptive_target(0.5, cycles=100, max_pre_sweeps=200),
        "0.1": run_adaptive_target(0.1, cycles=100, max_pre_sweeps=200),
    }
    json_path = OUT_DIR / "persistent_state_adaptive_ratio_experiment.json"
    csv_path = OUT_DIR / "persistent_state_adaptive_ratio_experiment.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(csv_path, results)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
