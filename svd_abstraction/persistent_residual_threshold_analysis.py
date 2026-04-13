from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.persistent_residual_fixed_problem_experiment import (
    build_setup,
    current_metrics,
    reset_fixed_residual_state,
)
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LIST = [1, 2, 5, 10, 20, 50]
POINTS = [0, 1, 2, 5, 10, 20, 50, 100]


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


def selected_points(history: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for p in POINTS:
        if p < len(history):
            out[str(p)] = history[p]
    return out


def run_threshold(k: int, cycles: int = 100) -> dict[str, object]:
    setup = build_setup()
    reset_fixed_residual_state(setup)

    history = [{"cycle": 0, **current_metrics(setup)}]
    probes: dict[str, dict[str, float]] = {}

    for cyc in range(1, cycles + 1):
        e_start = np.concatenate(
            [np.asarray(v.mu, dtype=float).reshape(-1) for v in setup.residual_graph.var_nodes[: setup.residual_graph.n_var_nodes]]
        )
        fixed_res_start = float(np.linalg.norm(setup.r - setup.a @ e_start))

        prev_eta = flatten_message_eta(setup.residual_graph)
        prev_lam = flatten_message_lam(setup.residual_graph)
        prev_e = e_start.copy()

        last_eta_delta = 0.0
        last_lam_delta = 0.0
        last_e_step = 0.0
        for _ in range(k):
            setup.residual_graph.synchronous_iteration()
            curr_e = np.concatenate(
                [np.asarray(v.mu, dtype=float).reshape(-1) for v in setup.residual_graph.var_nodes[: setup.residual_graph.n_var_nodes]]
            )
            curr_eta = flatten_message_eta(setup.residual_graph)
            curr_lam = flatten_message_lam(setup.residual_graph)
            last_eta_delta = max_abs_delta(curr_eta, prev_eta)
            last_lam_delta = max_abs_delta(curr_lam, prev_lam)
            last_e_step = float(np.linalg.norm(curr_e - prev_e))
            prev_eta = curr_eta
            prev_lam = curr_lam
            prev_e = curr_e

        pre_metrics = current_metrics(setup)
        if cyc <= 2:
            probes[f"cycle{cyc}"] = {
                "cycle_start_relative_state_error": float(history[-1]["relative_state_error"]),
                "cycle_start_algebraic_residual": float(history[-1]["algebraic_residual"]),
                "cycle_start_fixed_residual_norm": float(fixed_res_start),
                "pre_residual_ratio": float(pre_metrics["fixed_residual_norm"] / max(fixed_res_start, 1e-15)),
                "last_message_eta_delta_max": float(last_eta_delta),
                "last_message_lam_delta_max": float(last_lam_delta),
                "last_e_step_norm": float(last_e_step),
                "pre_end_relative_state_error": float(pre_metrics["relative_state_error"]),
                "pre_end_algebraic_residual": float(pre_metrics["algebraic_residual"]),
                "pre_end_fixed_residual_norm": float(pre_metrics["fixed_residual_norm"]),
                "pre_end_e_rel_to_exact": float(pre_metrics["e_rel_to_exact"]),
            }

        setup.level.update_coarse_residual_eta()
        delta_z = setup.level.direct_solve_coarse_graph()
        delta_e = setup.level.prolongate(delta_z)
        inject_correction_keep_messages(setup.residual_graph, delta_e)
        history.append({"cycle": cyc, **current_metrics(setup)})
        if not np.isfinite(history[-1]["relative_state_error"]) or history[-1]["relative_state_error"] > 1e12:
            break

    rel_hist = [row["relative_state_error"] for row in history]
    best_cycle = int(np.argmin(rel_hist))
    return {
        "method": "persistent_residual_preK_post0",
        "k": int(k),
        "probes": probes,
        "summary": {
            "final_cycle": int(history[-1]["cycle"]),
            "final_relerr": float(history[-1]["relative_state_error"]),
            "final_residual": float(history[-1]["algebraic_residual"]),
            "final_fixed_residual_norm": float(history[-1]["fixed_residual_norm"]),
            "best_relerr": float(rel_hist[best_cycle]),
            "best_cycle": int(best_cycle),
            "diverged": bool(not np.isfinite(rel_hist[-1]) or rel_hist[-1] > 1e6),
        },
        "points": selected_points(history),
    }


def write_summary_csv(path: pathlib.Path, results: dict[str, dict[str, object]]) -> None:
    lines = [
        "k,final_cycle,final_relerr,final_residual,final_fixed_residual_norm,best_relerr,best_cycle,diverged,"
        "cycle1_pre_ratio,cycle1_lam_delta,cycle1_e_step,cycle2_pre_ratio,cycle2_lam_delta,cycle2_e_step"
    ]
    for k in K_LIST:
        row = results[str(k)]
        summary = row["summary"]
        c1 = row["probes"]["cycle1"]
        c2 = row["probes"]["cycle2"]
        lines.append(
            f"{k},{summary['final_cycle']},{summary['final_relerr']},{summary['final_residual']},"
            f"{summary['final_fixed_residual_norm']},{summary['best_relerr']},{summary['best_cycle']},"
            f"{summary['diverged']},{c1['pre_residual_ratio']},{c1['last_message_lam_delta_max']},"
            f"{c1['last_e_step_norm']},{c2['pre_residual_ratio']},{c2['last_message_lam_delta_max']},"
            f"{c2['last_e_step_norm']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = {str(k): run_threshold(k=k, cycles=100) for k in K_LIST}
    json_path = OUT_DIR / "persistent_residual_threshold_analysis.json"
    csv_path = OUT_DIR / "persistent_residual_threshold_analysis_summary.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary_csv(csv_path, results)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
