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


DIRECT_DAMPED_PATH = RESULT_DIR / "intel_g2o_direct_newton_damping0p1_outer300_trajectories.npz"
DIRECT_DAMPED_META = RESULT_DIR / "intel_g2o_direct_newton_damping0p1_outer300.json"


def _result_stem(
    basis_source: str,
    group_size: int,
    r_reduced: int,
    max_cycles: int,
    max_sweeps: int,
    tag: str,
) -> str:
    stem = (
        "intel_g2o_relinearization_ck_requirement_analysis"
        f"_{basis_source}"
        f"_g{group_size}"
        f"_r{r_reduced}"
        f"_cmax{max_cycles}"
        f"_kmax{max_sweeps}"
    )
    if tag:
        stem += f"_{tag}"
    return stem


def _write_checkpoint(
    stem: str,
    args: argparse.Namespace,
    ratio_targets: list[float],
    outer_samples: list[int],
    direct_opt: float,
    summary_rows: list[dict[str, object]],
    cycle_rows_all: list[dict[str, object]],
    sweep_rows_all: list[dict[str, object]],
) -> dict[str, pathlib.Path]:
    summary_csv = RESULT_DIR / f"{stem}_summary.csv"
    cycle_csv = RESULT_DIR / f"{stem}_cycle.csv"
    sweep_csv = RESULT_DIR / f"{stem}_sweep.csv"
    json_path = RESULT_DIR / f"{stem}.json"
    save_csv(summary_rows, summary_csv)
    save_csv(cycle_rows_all, cycle_csv)
    save_csv(sweep_rows_all, sweep_csv)
    json_path.write_text(
        json.dumps(
            {
                "basis_source": args.basis_source,
                "group_size": int(args.group_size),
                "r_reduced": int(args.r_reduced),
                "max_cycles": int(args.max_cycles),
                "max_sweeps": int(args.max_sweeps),
                "ratio_targets": ratio_targets,
                "outer_samples": outer_samples,
                "direct_optimum_objective": direct_opt,
                "num_summary_rows": int(len(summary_rows)),
                "num_cycle_rows": int(len(cycle_rows_all)),
                "num_sweep_rows": int(len(sweep_rows_all)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_csv,
        "cycle_csv": cycle_csv,
        "sweep_csv": sweep_csv,
        "json": json_path,
    }


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
    return residual_graph, level, groups


def _target_reached(cur: float, start: float, ratio_target: float) -> bool:
    if start <= 0.0:
        return True
    if ratio_target >= 1.0:
        return cur < start - 1e-12
    return cur <= ratio_target * start


def analyze_relinearization_point(
    problem,
    base_poses: np.ndarray,
    ratio_target: float,
    basis_source: str,
    group_size: int,
    r_reduced: int,
    max_cycles: int,
    max_sweeps: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    exact = exact_local_solve_g2o(problem, base_poses)
    A = exact["A"]
    b = exact["b"]
    e_star = np.asarray(exact["e_star"], dtype=float)
    start_obj = float(nonlinear_objective_g2o(problem, base_poses))
    residual_graph, level, groups = _build_level(problem, base_poses, group_size, r_reduced, basis_source)

    cycle_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    success = False
    success_cycle = None
    success_obj = None

    for cyc in range(1, int(max_cycles) + 1):
        e_cycle_start = mean_vector(residual_graph)
        cycle_start_res = float(np.linalg.norm(A @ e_cycle_start - b))
        sweeps = 0
        e_pre = e_cycle_start
        pre_res = cycle_start_res
        while sweeps < int(max_sweeps):
            residual_graph.synchronous_iteration()
            sweeps += 1
            e_pre = mean_vector(residual_graph)
            pre_res = float(np.linalg.norm(A @ e_pre - b))
            sweep_rows.append(
                {
                    "cycle": int(cyc),
                    "sweep": int(sweeps),
                    "cycle_start_residual": cycle_start_res,
                    "current_residual": pre_res,
                    "residual_ratio": float(pre_res / max(cycle_start_res, 1e-300)),
                }
            )
            if _target_reached(pre_res, cycle_start_res, ratio_target):
                break

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_e = level.prolongate(delta_z)
        inject_correction_keep_messages(residual_graph, delta_e)
        e_post = mean_vector(residual_graph)
        post_res = float(np.linalg.norm(A @ e_post - b))
        post_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_post)))

        row = {
            "cycle": int(cyc),
            "k_used": int(sweeps),
            "cycle_start_residual": cycle_start_res,
            "pre_coarse_residual": pre_res,
            "post_coarse_residual": post_res,
            "pre_residual_ratio": float(pre_res / max(cycle_start_res, 1e-300)),
            "post_residual_ratio": float(post_res / max(cycle_start_res, 1e-300)),
            "post_cycle_objective": post_obj,
            "objective_ratio_to_start": float(post_obj / max(start_obj, 1e-300)),
            "objective_delta_to_start": float(post_obj - start_obj),
            "delta_e_norm": float(np.linalg.norm(delta_e)),
            "e_post_norm": float(np.linalg.norm(e_post)),
            "e_rel_to_exact": float(np.linalg.norm(e_post - e_star) / max(np.linalg.norm(e_star), 1e-15)),
            "num_groups": int(len(groups)),
            "coarse_dim": int(level.total_reduced_dim),
        }
        cycle_rows.append(row)

        if post_obj < start_obj:
            success = True
            success_cycle = cyc
            success_obj = post_obj
            break

    summary = {
        "base_objective": start_obj,
        "exact_after_objective": float(exact["after_objective"]),
        "exact_step_norm": float(np.linalg.norm(e_star)),
        "success": bool(success),
        "success_cycle": int(success_cycle) if success_cycle is not None else None,
        "success_objective": float(success_obj) if success_obj is not None else None,
        "best_cycle_objective": float(min(row["post_cycle_objective"] for row in cycle_rows)) if cycle_rows else None,
        "best_cycle_index": int(min(cycle_rows, key=lambda row: row["post_cycle_objective"])["cycle"]) if cycle_rows else None,
        "cycles_ran": int(len(cycle_rows)),
        "max_sweeps_cap": int(max_sweeps),
        "max_cycles_cap": int(max_cycles),
    }
    return cycle_rows, sweep_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-source", type=str, default="message_conditioned_information")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--max-sweeps", type=int, default=1000)
    parser.add_argument("--ratio-targets", type=str, default="1.0,0.8,0.5")
    parser.add_argument("--outer-samples", type=str, default="0,10,20,50,100,150,200,300")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    ratio_targets = [float(x) for x in args.ratio_targets.split(",") if x.strip()]
    outer_samples = [int(x) for x in args.outer_samples.split(",") if x.strip()]

    problem = parse_g2o_se2(G2O_PATH)
    npz = np.load(DIRECT_DAMPED_PATH)
    pose_history = np.asarray(npz["pose_history"], dtype=float)
    direct_meta = json.loads(DIRECT_DAMPED_META.read_text(encoding="utf-8"))
    direct_opt = float(direct_meta["best_objective"])

    summary_rows: list[dict[str, object]] = []
    cycle_rows_all: list[dict[str, object]] = []
    sweep_rows_all: list[dict[str, object]] = []
    stem = _result_stem(
        basis_source=args.basis_source,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        max_cycles=args.max_cycles,
        max_sweeps=args.max_sweeps,
        tag=args.tag,
    )

    for outer_idx in outer_samples:
        if outer_idx < 0 or outer_idx >= pose_history.shape[0]:
            continue
        base_poses = pose_history[outer_idx]
        base_obj = float(nonlinear_objective_g2o(problem, base_poses))
        for ratio_target in ratio_targets:
            print(
                f"analyze outer={outer_idx} ratio_target={ratio_target:g} base_obj={base_obj:.6f}",
                flush=True,
            )
            cycle_rows, sweep_rows, summary = analyze_relinearization_point(
                problem=problem,
                base_poses=base_poses,
                ratio_target=ratio_target,
                basis_source=args.basis_source,
                group_size=args.group_size,
                r_reduced=args.r_reduced,
                max_cycles=args.max_cycles,
                max_sweeps=args.max_sweeps,
            )
            summary_rows.append(
                {
                    "outer_sample": int(outer_idx),
                    "ratio_target": float(ratio_target),
                    "base_objective": base_obj,
                    "gap_to_direct_opt": float(base_obj - direct_opt),
                    **summary,
                }
            )
            for row in cycle_rows:
                row["outer_sample"] = int(outer_idx)
                row["ratio_target"] = float(ratio_target)
                cycle_rows_all.append(row)
            for row in sweep_rows:
                row["outer_sample"] = int(outer_idx)
                row["ratio_target"] = float(ratio_target)
                sweep_rows_all.append(row)
            paths = _write_checkpoint(
                stem=stem,
                args=args,
                ratio_targets=ratio_targets,
                outer_samples=outer_samples,
                direct_opt=direct_opt,
                summary_rows=summary_rows,
                cycle_rows_all=cycle_rows_all,
                sweep_rows_all=sweep_rows_all,
            )
            print(
                f"checkpoint outer={outer_idx} ratio_target={ratio_target:g} "
                f"summary_rows={len(summary_rows)} cycle_rows={len(cycle_rows_all)} "
                f"sweep_rows={len(sweep_rows_all)}",
                flush=True,
            )

    paths = _write_checkpoint(
        stem=stem,
        args=args,
        ratio_targets=ratio_targets,
        outer_samples=outer_samples,
        direct_opt=direct_opt,
        summary_rows=summary_rows,
        cycle_rows_all=cycle_rows_all,
        sweep_rows_all=sweep_rows_all,
    )
    print(
        json.dumps(
            {
                "summary_csv": str(paths["summary_csv"]),
                "cycle_csv": str(paths["cycle_csv"]),
                "sweep_csv": str(paths["sweep_csv"]),
                "json": str(paths["json"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
