from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs, group_list
from svd_abstraction.pose_graph import build_noisy_pose_graph
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import (
    mean_vector,
    odom_tiny_init_graph,
    reset_residual_graph,
    var_slices,
)
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_absolute_factors(graph) -> None:
    for factor in graph.factors[: graph.n_factor_nodes]:
        factor.compute_factor_absolute(update_self=True)


def relative_error_vec(x: np.ndarray, x_star: np.ndarray) -> float:
    return float(np.linalg.norm(x - x_star) / max(np.linalg.norm(x_star), 1e-15))


@dataclass
class FixedResidualSetup:
    nodes: list
    edges: list
    x0: np.ndarray
    x_star: np.ndarray
    a: np.ndarray
    b: np.ndarray
    r: np.ndarray
    e_star: np.ndarray
    residual_graph: object
    level: SVDResidualAbstraction
    base_graph_template: object
    base_slices: dict[int, slice]


def build_setup() -> FixedResidualSetup:
    nodes, edges, exact_graph, base_graph, _ = build_graphs(
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

    b, a = exact_graph.joint_distribution_inf_absolute()
    x_star, _ = exact_graph.joint_distribution_cov_absolute()

    odom_tiny_init_graph(base_graph, n=512)
    x0 = mean_vector(base_graph)

    r = b - a @ x0
    e_star = np.linalg.solve(a, r)

    residual_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=1.0,
        odom_sigma=1.0,
        tiny_prior=1e-12,
        seed=0,
    )
    slices = var_slices(base_graph)
    reset_residual_graph(residual_graph, base_graph, x0, slices)

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
        base_graph=residual_graph,
        groups=groups,
        r_reduced=4,
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=False,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    return FixedResidualSetup(
        nodes=nodes,
        edges=edges,
        x0=x0,
        x_star=x_star,
        a=a,
        b=b,
        r=r,
        e_star=e_star,
        residual_graph=residual_graph,
        level=level,
        base_graph_template=base_graph,
        base_slices=slices,
    )


def reset_fixed_residual_state(setup: FixedResidualSetup) -> None:
    reset_residual_graph(setup.residual_graph, setup.base_graph_template, setup.x0, setup.base_slices)
    setup.level.base_graph = setup.residual_graph


def current_metrics(setup: FixedResidualSetup) -> dict[str, float]:
    e = mean_vector(setup.residual_graph)
    x = setup.x0 + e
    return {
        "relative_state_error": relative_error_vec(x, setup.x_star),
        "algebraic_residual": float(np.linalg.norm(setup.b - setup.a @ x)),
        "fixed_residual_norm": float(np.linalg.norm(setup.r - setup.a @ e)),
        "e_rel_to_exact": float(np.linalg.norm(e - setup.e_star) / max(np.linalg.norm(setup.e_star), 1e-15)),
    }


def run_fixed_k(setup: FixedResidualSetup, k: int, cycles: int = 100) -> dict[str, object]:
    reset_fixed_residual_state(setup)
    history = [{"cycle": 0, "k_used": 0, **current_metrics(setup)}]

    for cyc in range(1, cycles + 1):
        for _ in range(k):
            setup.residual_graph.synchronous_iteration()

        setup.level.update_coarse_residual_eta()
        delta_z = setup.level.direct_solve_coarse_graph()
        delta_e = setup.level.prolongate(delta_z)
        inject_correction_keep_messages(setup.residual_graph, delta_e)

        history.append({"cycle": cyc, "k_used": int(k), **current_metrics(setup)})
        if not np.isfinite(history[-1]["relative_state_error"]) or history[-1]["relative_state_error"] > 1e12:
            break

    rel_hist = [row["relative_state_error"] for row in history]
    best_cycle = int(np.argmin(rel_hist))
    return {
        "config": {"k": int(k), "cycles": int(cycles)},
        "summary": {
            "final_cycle": int(history[-1]["cycle"]),
            "final_relerr": float(history[-1]["relative_state_error"]),
            "final_residual": float(history[-1]["algebraic_residual"]),
            "final_fixed_residual_norm": float(history[-1]["fixed_residual_norm"]),
            "best_relerr": float(rel_hist[best_cycle]),
            "best_cycle": int(best_cycle),
        },
        "history": history,
    }


def run_single_layer_sync(setup: FixedResidualSetup, sweeps: int, points: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000)) -> dict[str, object]:
    reset_fixed_residual_state(setup)
    wanted = set(int(p) for p in points if p <= sweeps)
    history = []
    history.append({"iter": 0, **current_metrics(setup)})
    for it in range(1, sweeps + 1):
        setup.residual_graph.synchronous_iteration()
        if it in wanted:
            history.append({"iter": it, **current_metrics(setup)})
    if sweeps not in wanted:
        history.append({"iter": sweeps, **current_metrics(setup)})
    best_idx = int(np.argmin([row["relative_state_error"] for row in history]))
    return {
        "config": {"sweeps": int(sweeps)},
        "summary": {
            "final_iter": int(history[-1]["iter"]),
            "final_relerr": float(history[-1]["relative_state_error"]),
            "final_residual": float(history[-1]["algebraic_residual"]),
            "final_fixed_residual_norm": float(history[-1]["fixed_residual_norm"]),
            "best_relerr": float(history[best_idx]["relative_state_error"]),
            "best_iter": int(history[best_idx]["iter"]),
        },
        "history": history,
    }


def run_adaptive_ratio(setup: FixedResidualSetup, target_ratio: float, cycles: int = 100, max_pre_sweeps: int = 200) -> dict[str, object]:
    reset_fixed_residual_state(setup)
    history = [
        {
            "cycle": 0,
            "k_used": 0,
            "target_ratio": float(target_ratio),
            "pre_residual_ratio": 1.0,
            **current_metrics(setup),
        }
    ]

    for cyc in range(1, cycles + 1):
        e_start = mean_vector(setup.residual_graph)
        res_start = float(np.linalg.norm(setup.r - setup.a @ e_start))

        k_used = 0
        current_ratio = 1.0
        while k_used < max_pre_sweeps and current_ratio > target_ratio:
            setup.residual_graph.synchronous_iteration()
            k_used += 1
            e_now = mean_vector(setup.residual_graph)
            current_ratio = float(np.linalg.norm(setup.r - setup.a @ e_now) / max(res_start, 1e-15))

        pre_metrics = current_metrics(setup)

        setup.level.update_coarse_residual_eta()
        delta_z = setup.level.direct_solve_coarse_graph()
        delta_e = setup.level.prolongate(delta_z)
        inject_correction_keep_messages(setup.residual_graph, delta_e)

        history.append(
            {
                "cycle": cyc,
                "k_used": int(k_used),
                "target_ratio": float(target_ratio),
                "pre_residual_ratio": float(current_ratio),
                "pre_end_relative_state_error": float(pre_metrics["relative_state_error"]),
                "pre_end_algebraic_residual": float(pre_metrics["algebraic_residual"]),
                "pre_end_fixed_residual_norm": float(pre_metrics["fixed_residual_norm"]),
                **current_metrics(setup),
            }
        )
        if not np.isfinite(history[-1]["relative_state_error"]) or history[-1]["relative_state_error"] > 1e12:
            break

    rel_hist = [row["relative_state_error"] for row in history]
    best_cycle = int(np.argmin(rel_hist))
    ks = [row["k_used"] for row in history[1:]]
    return {
        "config": {"target_ratio": float(target_ratio), "cycles": int(cycles), "max_pre_sweeps": int(max_pre_sweeps)},
        "summary": {
            "final_cycle": int(history[-1]["cycle"]),
            "final_relerr": float(history[-1]["relative_state_error"]),
            "final_residual": float(history[-1]["algebraic_residual"]),
            "final_fixed_residual_norm": float(history[-1]["fixed_residual_norm"]),
            "best_relerr": float(rel_hist[best_cycle]),
            "best_cycle": int(best_cycle),
            "mean_k_used": float(np.mean(ks)) if ks else 0.0,
            "max_k_used": int(max(ks, default=0)),
            "min_k_used": int(min(ks, default=0)),
        },
        "history": history,
    }


def write_csv(path: pathlib.Path, results: dict[str, dict[str, object]]) -> None:
    lines = [
        "experiment,step_kind,step,k_used,relative_state_error,algebraic_residual,fixed_residual_norm,e_rel_to_exact,pre_residual_ratio"
    ]
    for name, payload in results.items():
        for row in payload["history"]:
            step_kind = "cycle" if "cycle" in row else "iter"
            step = row.get("cycle", row.get("iter", 0))
            lines.append(
                f"{name},{step_kind},{step},{row.get('k_used', '')},{row['relative_state_error']},"
                f"{row['algebraic_residual']},{row['fixed_residual_norm']},{row['e_rel_to_exact']},"
                f"{row.get('pre_residual_ratio', 1.0)}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup = build_setup()
    results = {
        "single_sync_1000": run_single_layer_sync(setup, 1000),
        "single_sync_5000": run_single_layer_sync(setup, 5000),
        "fixed_k_2": run_fixed_k(setup, 2, cycles=100),
        "fixed_k_10": run_fixed_k(setup, 10, cycles=100),
        "fixed_k_50": run_fixed_k(setup, 50, cycles=100),
        "adaptive_ratio_0.5": run_adaptive_ratio(setup, 0.5, cycles=100, max_pre_sweeps=200),
        "adaptive_ratio_0.1": run_adaptive_ratio(setup, 0.1, cycles=100, max_pre_sweeps=200),
    }

    json_path = OUT_DIR / "persistent_residual_fixed_problem_experiment.json"
    csv_path = OUT_DIR / "persistent_residual_fixed_problem_experiment.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(csv_path, results)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
