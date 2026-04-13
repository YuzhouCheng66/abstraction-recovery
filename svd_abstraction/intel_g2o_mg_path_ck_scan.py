from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import build_linearized_local_graph_g2o
from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import poses_to_nodes_g2o
from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.intel_g2o_persistent_residual_mg import G2O_PATH
from svd_abstraction.intel_g2o_persistent_residual_mg import RESULT_DIR
from svd_abstraction.intel_g2o_persistent_residual_mg import exact_local_solve_g2o
from svd_abstraction.intel_g2o_persistent_residual_mg import save_csv
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import mean_vector
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices
from svd_abstraction.se2_utils import apply_pose_deltas


BASELINE_TRAJ = (
    RESULT_DIR / "intel_g2o_direct_vs_persistent_mg_outer40_c3_k100_message_conditioned_information_trajectories.npz"
)
DIRECT_META = RESULT_DIR / "intel_g2o_direct_newton_damping0p1_outer300.json"


def _build_level(problem, base_poses, group_size: int, r_reduced: int, basis_source: str):
    template_graph = build_linearized_local_graph_g2o(problem, base_poses)
    residual_graph = build_linearized_local_graph_g2o(problem, base_poses)
    zero = np.zeros(template_graph.n_var_nodes * 3, dtype=float)
    reset_residual_graph(residual_graph, template_graph, zero, var_slices(template_graph))

    groups = group_list(
        nodes=poses_to_nodes_g2o(base_poses),
        graph=template_graph,
        method="order",
        group_size=group_size,
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
        r_reduced=r_reduced,
        basis_source=basis_source,
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=False,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)
    return residual_graph, level


def run_outer_ck_scan(
    problem,
    base_poses: np.ndarray,
    outer_sample: int,
    k: int,
    max_cycles: int,
    basis_source: str,
    group_size: int,
    r_reduced: int,
    direct_optimum: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    exact = exact_local_solve_g2o(problem, base_poses)
    A = exact["A"]
    b = np.asarray(exact["b"], dtype=float)
    e_star = np.asarray(exact["e_star"], dtype=float)
    outer_start_obj = float(nonlinear_objective_g2o(problem, base_poses))

    residual_graph, level = _build_level(
        problem=problem,
        base_poses=base_poses,
        group_size=group_size,
        r_reduced=r_reduced,
        basis_source=basis_source,
    )

    cycle_rows: list[dict[str, object]] = []
    first_cycle_below_start = None

    for cyc in range(1, int(max_cycles) + 1):
        e_cycle_start = mean_vector(residual_graph)
        cycle_start_res = float(np.linalg.norm(A @ e_cycle_start - b))
        cycle_start_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_cycle_start)))

        for _ in range(int(k)):
            residual_graph.synchronous_iteration()
        e_pre = mean_vector(residual_graph)
        pre_res = float(np.linalg.norm(A @ e_pre - b))
        pre_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_pre)))

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_e = level.prolongate(delta_z)
        inject_correction_keep_messages(residual_graph, delta_e)

        e_post = mean_vector(residual_graph)
        post_res = float(np.linalg.norm(A @ e_post - b))
        post_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_post)))

        if first_cycle_below_start is None and post_obj < outer_start_obj:
            first_cycle_below_start = int(cyc)

        cycle_rows.append(
            {
                "outer_sample": int(outer_sample),
                "k": int(k),
                "cycle": int(cyc),
                "cycle_start_linear_residual": cycle_start_res,
                "pre_linear_residual": pre_res,
                "post_linear_residual": post_res,
                "pre_residual_ratio_vs_cycle_start": float(pre_res / max(cycle_start_res, 1e-300)),
                "post_residual_ratio_vs_cycle_start": float(post_res / max(cycle_start_res, 1e-300)),
                "outer_start_objective": outer_start_obj,
                "cycle_start_objective": cycle_start_obj,
                "pre_objective": pre_obj,
                "post_objective": post_obj,
                "post_objective_delta_vs_outer_start": float(post_obj - outer_start_obj),
                "post_objective_gap_to_direct_optimum": float(post_obj - direct_optimum),
                "delta_z_norm": float(np.linalg.norm(delta_z)),
                "delta_e_norm": float(np.linalg.norm(delta_e)),
                "e_post_norm": float(np.linalg.norm(e_post)),
                "e_rel_to_exact": float(np.linalg.norm(e_post - e_star) / max(np.linalg.norm(e_star), 1e-15)),
            }
        )

    best_row = min(cycle_rows, key=lambda row: row["post_objective"])
    summary = {
        "outer_sample": int(outer_sample),
        "k": int(k),
        "max_cycles": int(max_cycles),
        "outer_start_objective": outer_start_obj,
        "direct_optimum_objective": float(direct_optimum),
        "exact_after_objective": float(exact["after_objective"]),
        "exact_gap_to_direct_optimum": float(exact["after_objective"] - direct_optimum),
        "exact_step_norm": float(np.linalg.norm(e_star)),
        "first_cycle_below_outer_start": int(first_cycle_below_start) if first_cycle_below_start else None,
        "best_cycle": int(best_row["cycle"]),
        "best_cycle_objective": float(best_row["post_objective"]),
        "best_gap_to_direct_optimum": float(best_row["post_objective"] - direct_optimum),
        "best_pre_residual_ratio": float(best_row["pre_residual_ratio_vs_cycle_start"]),
        "best_post_residual_ratio": float(best_row["post_residual_ratio_vs_cycle_start"]),
    }
    return summary, cycle_rows


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-samples", type=str, default="1,2,3,4,5,6,9,12,13,19,20")
    parser.add_argument("--k-values", type=str, default="50,100,150,200,300")
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--basis-source", type=str, default="message_conditioned_information")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--tag", type=str, default="deep_diagnosis")
    args = parser.parse_args()

    outer_samples = _parse_int_list(args.outer_samples)
    k_values = _parse_int_list(args.k_values)
    problem = parse_g2o_se2(G2O_PATH)
    traj = np.load(BASELINE_TRAJ)
    pose_history = np.asarray(traj["mg_pose_history"], dtype=float)
    direct_opt = float(json.loads(DIRECT_META.read_text(encoding="utf-8"))["best_objective"])

    summary_rows: list[dict[str, object]] = []
    cycle_rows_all: list[dict[str, object]] = []

    for outer in outer_samples:
        if outer < 0 or outer >= pose_history.shape[0]:
            continue
        base_poses = pose_history[outer]
        for k in k_values:
            print(f"scan outer={outer} k={k} max_cycles={args.max_cycles}", flush=True)
            summary, cycle_rows = run_outer_ck_scan(
                problem=problem,
                base_poses=base_poses,
                outer_sample=outer,
                k=k,
                max_cycles=args.max_cycles,
                basis_source=args.basis_source,
                group_size=args.group_size,
                r_reduced=args.r_reduced,
                direct_optimum=direct_opt,
            )
            summary_rows.append(summary)
            cycle_rows_all.extend(cycle_rows)

    stem = (
        "intel_g2o_mg_path_ck_scan"
        f"_{args.basis_source}"
        f"_g{args.group_size}_r{args.r_reduced}"
        f"_cmax{args.max_cycles}_{args.tag}"
    )
    summary_csv = RESULT_DIR / f"{stem}_summary.csv"
    cycle_csv = RESULT_DIR / f"{stem}_cycle.csv"
    json_path = RESULT_DIR / f"{stem}.json"
    save_csv(summary_rows, summary_csv)
    save_csv(cycle_rows_all, cycle_csv)
    json_path.write_text(
        json.dumps(
            {
                "trajectory_source": str(BASELINE_TRAJ),
                "direct_meta": str(DIRECT_META),
                "basis_source": args.basis_source,
                "group_size": int(args.group_size),
                "r_reduced": int(args.r_reduced),
                "outer_samples": outer_samples,
                "k_values": k_values,
                "max_cycles": int(args.max_cycles),
                "direct_optimum_objective": direct_opt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "cycle_csv": str(cycle_csv),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
