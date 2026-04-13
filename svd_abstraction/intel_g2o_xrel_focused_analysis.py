from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.intel_g2o_ck_beta_policy_experiment import BASELINE_TRAJ
from svd_abstraction.intel_g2o_ck_beta_policy_experiment import parse_ck_schedule
from svd_abstraction.intel_g2o_ck_beta_policy_experiment import parse_float_list
from svd_abstraction.intel_g2o_ck_beta_policy_experiment import run_candidate
from svd_abstraction.intel_g2o_persistent_residual_mg import G2O_PATH
from svd_abstraction.intel_g2o_persistent_residual_mg import RESULT_DIR
from svd_abstraction.intel_g2o_persistent_residual_mg import exact_local_solve_g2o
from svd_abstraction.intel_g2o_persistent_residual_mg import save_csv
from svd_abstraction.se2_utils import apply_pose_deltas


DIRECT_TRAJ = RESULT_DIR / "intel_g2o_direct_newton_damping0p1_outer300_trajectories.npz"


def xy_stack(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    return poses[:, :2].reshape(-1)


def xy_rel_to_ref(poses: np.ndarray, ref_poses: np.ndarray) -> float:
    x = xy_stack(poses)
    x_ref = xy_stack(ref_poses)
    denom = max(float(np.linalg.norm(x_ref)), 1e-15)
    return float(np.linalg.norm(x - x_ref) / denom)


def reference_direct_poses() -> np.ndarray:
    data = np.load(DIRECT_TRAJ)
    return np.asarray(data["final_poses"], dtype=float)


def base_pose_from_mode(
    problem,
    mode: str,
    completed_outer: int,
    trajectory_file: str | None = None,
    trajectory_key: str = "pose_history",
) -> tuple[np.ndarray, str]:
    if mode == "initial":
        return np.asarray(problem.init_poses, dtype=float).copy(), "initial"
    if mode == "fixed_path":
        traj = np.load(BASELINE_TRAJ)
        pose_history = np.asarray(traj["mg_pose_history"], dtype=float)
        if completed_outer < 0 or completed_outer >= pose_history.shape[0]:
            raise ValueError(
                f"completed_outer={completed_outer} out of range for pose history length {pose_history.shape[0]}"
            )
        return np.asarray(pose_history[completed_outer], dtype=float).copy(), f"fixed_path_completed_outer{completed_outer}"
    if mode == "trajectory_npz":
        if not trajectory_file:
            raise ValueError("--trajectory-file is required when --base-mode=trajectory_npz")
        traj_path = pathlib.Path(trajectory_file)
        traj = np.load(traj_path)
        pose_history = np.asarray(traj[trajectory_key], dtype=float)
        if completed_outer < 0 or completed_outer >= pose_history.shape[0]:
            raise ValueError(
                f"completed_outer={completed_outer} out of range for trajectory length {pose_history.shape[0]}"
            )
        return (
            np.asarray(pose_history[completed_outer], dtype=float).copy(),
            f"{traj_path.stem}_{trajectory_key}_completed_outer{completed_outer}",
        )
    raise ValueError(f"Unsupported mode={mode!r}")


def scan_base_state(
    problem,
    base_poses: np.ndarray,
    base_label: str,
    ref_poses: np.ndarray,
    ck_schedule: list[tuple[int, int]],
    beta_candidates: list[float],
    basis_source: str,
    group_size: int,
    r_reduced: int,
    tag: str,
) -> dict[str, pathlib.Path]:
    base_obj = float(nonlinear_objective_g2o(problem, base_poses))
    base_x_rel = float(xy_rel_to_ref(base_poses, ref_poses))
    exact = exact_local_solve_g2o(problem, base_poses)
    exact_after_obj = float(exact["after_objective"])
    exact_after_poses = np.asarray(exact["next_poses"], dtype=float)
    exact_after_x_rel = float(xy_rel_to_ref(exact_after_poses, ref_poses))

    summary_rows: list[dict[str, object]] = []
    beta_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []

    for inner_cycles, pre_sweeps in ck_schedule:
        candidate = run_candidate(
            problem=problem,
            base_poses=base_poses,
            exact=exact,
            inner_cycles=inner_cycles,
            pre_sweeps=pre_sweeps,
            group_size=group_size,
            r_reduced=r_reduced,
            basis_source=basis_source,
        )
        e_hat = np.asarray(candidate["e_hat"], dtype=float)

        candidate_beta_rows: list[dict[str, object]] = []
        for beta in beta_candidates:
            beta = float(beta)
            next_poses = apply_pose_deltas(base_poses, beta * e_hat)
            post_obj = float(nonlinear_objective_g2o(problem, next_poses))
            post_x_rel = float(xy_rel_to_ref(next_poses, ref_poses))
            row = {
                "base_label": base_label,
                "start_objective": base_obj,
                "start_x_rel_to_direct": base_x_rel,
                "exact_after_objective": exact_after_obj,
                "exact_after_x_rel_to_direct": exact_after_x_rel,
                "c": int(inner_cycles),
                "k": int(pre_sweeps),
                "ck_cost": int(inner_cycles * pre_sweeps),
                "beta": beta,
                "post_objective": post_obj,
                "objective_improvement": float(base_obj - post_obj),
                "x_rel_to_direct": post_x_rel,
                "x_rel_improvement": float(base_x_rel - post_x_rel),
                "linear_residual_approx": float(candidate["linear_residual_approx"]),
                "e_rel_to_exact": float(candidate["e_rel_to_exact"]),
                "raw_full_step_objective": float(candidate["raw_full_step_objective"]),
                "last_cycle_pre_residual_ratio": float(candidate["cycle_rows"][-1]["pre_residual_ratio_vs_cycle_start"]),
                "last_cycle_post_residual_ratio": float(candidate["cycle_rows"][-1]["post_residual_ratio_vs_cycle_start"]),
            }
            beta_rows.append(row)
            candidate_beta_rows.append(row)

        beta1_row = next((row for row in candidate_beta_rows if np.isclose(row["beta"], 1.0)), None)
        best_obj_row = min(candidate_beta_rows, key=lambda row: row["post_objective"])
        best_xrel_row = min(candidate_beta_rows, key=lambda row: row["x_rel_to_direct"])
        summary_rows.append(
            {
                "base_label": base_label,
                "start_objective": base_obj,
                "start_x_rel_to_direct": base_x_rel,
                "exact_after_objective": exact_after_obj,
                "exact_after_x_rel_to_direct": exact_after_x_rel,
                "c": int(inner_cycles),
                "k": int(pre_sweeps),
                "ck_cost": int(inner_cycles * pre_sweeps),
                "beta1_objective": float(beta1_row["post_objective"]) if beta1_row is not None else "",
                "beta1_x_rel_to_direct": float(beta1_row["x_rel_to_direct"]) if beta1_row is not None else "",
                "beta1_objective_improvement": float(beta1_row["objective_improvement"]) if beta1_row is not None else "",
                "beta1_x_rel_improvement": float(beta1_row["x_rel_improvement"]) if beta1_row is not None else "",
                "best_objective_beta": float(best_obj_row["beta"]),
                "best_objective": float(best_obj_row["post_objective"]),
                "best_objective_x_rel_to_direct": float(best_obj_row["x_rel_to_direct"]),
                "best_x_rel_beta": float(best_xrel_row["beta"]),
                "best_x_rel_to_direct": float(best_xrel_row["x_rel_to_direct"]),
                "best_x_rel_objective": float(best_xrel_row["post_objective"]),
                "linear_residual_approx": float(candidate["linear_residual_approx"]),
                "e_rel_to_exact": float(candidate["e_rel_to_exact"]),
                "raw_full_step_objective": float(candidate["raw_full_step_objective"]),
                "last_cycle_pre_residual_ratio": float(candidate["cycle_rows"][-1]["pre_residual_ratio_vs_cycle_start"]),
                "last_cycle_post_residual_ratio": float(candidate["cycle_rows"][-1]["post_residual_ratio_vs_cycle_start"]),
                "exact_step_norm": float(exact["e_norm"]),
                "approx_step_norm": float(candidate["e_hat_norm"]),
            }
        )

        for cycle_row in candidate["cycle_rows"]:
            cycle_rows.append(
                {
                    "base_label": base_label,
                    "c": int(inner_cycles),
                    "k": int(pre_sweeps),
                    **cycle_row,
                }
            )

    stem = f"intel_g2o_xrel_focus_{base_label}_{basis_source}_{tag}"
    summary_csv = RESULT_DIR / f"{stem}_summary.csv"
    beta_csv = RESULT_DIR / f"{stem}_beta.csv"
    cycle_csv = RESULT_DIR / f"{stem}_cycle.csv"
    json_path = RESULT_DIR / f"{stem}.json"
    save_csv(summary_rows, summary_csv)
    save_csv(beta_rows, beta_csv)
    save_csv(cycle_rows, cycle_csv)
    json_path.write_text(
        json.dumps(
            {
                "base_label": base_label,
                "basis_source": basis_source,
                "group_size": int(group_size),
                "r_reduced": int(r_reduced),
                "ck_schedule": [[int(c), int(k)] for c, k in ck_schedule],
                "beta_candidates": [float(beta) for beta in beta_candidates],
                "start_objective": base_obj,
                "start_x_rel_to_direct": base_x_rel,
                "exact_after_objective": exact_after_obj,
                "exact_after_x_rel_to_direct": exact_after_x_rel,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_csv,
        "beta_csv": beta_csv,
        "cycle_csv": cycle_csv,
        "json": json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mode", choices=["initial", "fixed_path", "trajectory_npz"], required=True)
    parser.add_argument("--completed-outer", type=int, default=19)
    parser.add_argument("--trajectory-file", type=str, default=None)
    parser.add_argument("--trajectory-key", type=str, default="pose_history")
    parser.add_argument(
        "--ck-schedule",
        type=str,
        default="1x20,1x50,1x100,1x200,2x20,2x50,2x100,2x200,3x50,3x100,3x200,5x50,5x100,5x200,10x50,10x100",
    )
    parser.add_argument(
        "--beta-candidates",
        type=str,
        default="1,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001",
    )
    parser.add_argument("--basis-source", type=str, default="message_conditioned_information")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    problem = parse_g2o_se2(G2O_PATH)
    ref_poses = reference_direct_poses()
    base_poses, base_label = base_pose_from_mode(
        problem,
        args.base_mode,
        args.completed_outer,
        trajectory_file=args.trajectory_file,
        trajectory_key=args.trajectory_key,
    )
    paths = scan_base_state(
        problem=problem,
        base_poses=base_poses,
        base_label=base_label,
        ref_poses=ref_poses,
        ck_schedule=parse_ck_schedule(args.ck_schedule),
        beta_candidates=parse_float_list(args.beta_candidates),
        basis_source=args.basis_source,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        tag=args.tag,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
