from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.se2_newton_vs_persistent_mg_experiment import approx_local_persistent_mg_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import exact_local_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import pose_metrics
from svd_abstraction.se2_utils import build_se2_problem


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def xy_stack(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    return poses[:, :2].reshape(-1)


def xy_rel_to_ref(poses: np.ndarray, ref_poses: np.ndarray) -> float:
    x = xy_stack(poses)
    x_ref = xy_stack(ref_poses)
    denom = max(float(np.linalg.norm(x_ref)), 1e-15)
    return float(np.linalg.norm(x - x_ref) / denom)


def run_direct(problem, num_outer: int, ref_poses: np.ndarray, optimum_objective: float) -> list[dict[str, float]]:
    poses = problem.init_poses.copy()
    rows = []
    rows.append(
        {
            "outer": 0,
            "nonlinear_objective": float(pose_metrics(problem, poses)["nonlinear_objective"]),
            "objective_minus_optimum": float(pose_metrics(problem, poses)["nonlinear_objective"] - optimum_objective),
            "xy_rel_to_direct50": float(xy_rel_to_ref(poses, ref_poses)),
        }
    )
    for outer in range(1, num_outer + 1):
        step = exact_local_solve(problem, poses)
        poses = step["next_poses"]
        metrics = pose_metrics(problem, poses)
        rows.append(
            {
                "outer": int(outer),
                "nonlinear_objective": float(metrics["nonlinear_objective"]),
                "objective_minus_optimum": float(metrics["nonlinear_objective"] - optimum_objective),
                "xy_rel_to_direct50": float(xy_rel_to_ref(poses, ref_poses)),
            }
        )
    return rows


def run_mg(
    problem,
    num_outer: int,
    inner_cycles: int,
    pre_sweeps: int,
    ref_poses: np.ndarray,
    optimum_objective: float,
) -> list[dict[str, float]]:
    poses = problem.init_poses.copy()
    rows = []
    rows.append(
        {
            "outer": 0,
            "inner_cycles": int(inner_cycles),
            "nonlinear_objective": float(pose_metrics(problem, poses)["nonlinear_objective"]),
            "objective_minus_optimum": float(pose_metrics(problem, poses)["nonlinear_objective"] - optimum_objective),
            "xy_rel_to_direct50": float(xy_rel_to_ref(poses, ref_poses)),
        }
    )
    for outer in range(1, num_outer + 1):
        step = approx_local_persistent_mg_solve(
            problem=problem,
            base_poses=poses,
            inner_cycles=inner_cycles,
            pre_sweeps=pre_sweeps,
            group_size=20,
            r_reduced=4,
        )
        poses = step["next_poses"]
        metrics = pose_metrics(problem, poses)
        rows.append(
            {
                "outer": int(outer),
                "inner_cycles": int(inner_cycles),
                "nonlinear_objective": float(metrics["nonlinear_objective"]),
                "objective_minus_optimum": float(metrics["nonlinear_objective"] - optimum_objective),
                "xy_rel_to_direct50": float(xy_rel_to_ref(poses, ref_poses)),
                "linear_residual_approx": float(step["linear_residual_approx"]),
                "next_objective_gap_to_exact": float(step["next_objective_gap_to_exact"]),
                "next_pose_gap_to_exact": float(step["next_pose_gap_to_exact"]),
            }
        )
    return rows


def main() -> None:
    problem = build_se2_problem(
        n=64,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        seed=0,
    )

    poses_opt = problem.init_poses.copy()
    for _ in range(50):
        step = exact_local_solve(problem, poses_opt)
        poses_opt = step["next_poses"]
    optimum_objective = float(pose_metrics(problem, poses_opt)["nonlinear_objective"])

    direct20 = run_direct(problem, num_outer=20, ref_poses=poses_opt, optimum_objective=optimum_objective)
    mg10 = run_mg(problem, num_outer=20, inner_cycles=10, pre_sweeps=50, ref_poses=poses_opt, optimum_objective=optimum_objective)
    mg5 = run_mg(problem, num_outer=20, inner_cycles=5, pre_sweeps=50, ref_poses=poses_opt, optimum_objective=optimum_objective)
    mg2 = run_mg(problem, num_outer=20, inner_cycles=2, pre_sweeps=50, ref_poses=poses_opt, optimum_objective=optimum_objective)
    mg1 = run_mg(problem, num_outer=20, inner_cycles=1, pre_sweeps=50, ref_poses=poses_opt, optimum_objective=optimum_objective)

    out_rows = []
    for outer in range(21):
        out_rows.append(
            {
                "outer": outer,
                "optimum_objective_direct50": optimum_objective,
                "direct20_objective_minus_optimum": direct20[outer]["objective_minus_optimum"],
                "direct20_xy_rel_to_direct50": direct20[outer]["xy_rel_to_direct50"],
                "mg_c10_objective_minus_optimum": mg10[outer]["objective_minus_optimum"],
                "mg_c10_xy_rel_to_direct50": mg10[outer]["xy_rel_to_direct50"],
                "mg_c5_objective_minus_optimum": mg5[outer]["objective_minus_optimum"],
                "mg_c5_xy_rel_to_direct50": mg5[outer]["xy_rel_to_direct50"],
                "mg_c2_objective_minus_optimum": mg2[outer]["objective_minus_optimum"],
                "mg_c2_xy_rel_to_direct50": mg2[outer]["xy_rel_to_direct50"],
                "mg_c1_objective_minus_optimum": mg1[outer]["objective_minus_optimum"],
                "mg_c1_xy_rel_to_direct50": mg1[outer]["xy_rel_to_direct50"],
            }
        )

    csv_path = OUT_DIR / "se2_outer20_inner_cycles_objective_and_xy_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    json_path = OUT_DIR / "se2_outer20_inner_cycles_objective_and_xy_table.json"
    json_path.write_text(
        json.dumps(
            {
                "optimum_objective_direct50": optimum_objective,
                "table": out_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "optimum_objective_direct50": optimum_objective}, indent=2))


if __name__ == "__main__":
    main()
