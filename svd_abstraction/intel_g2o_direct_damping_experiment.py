from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import direct_newton_step_g2o
from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.se2_utils import apply_pose_deltas


G2O_PATH = pathlib.Path("/home/yuzhou/Desktop/input_INTEL_g2o.g2o")
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_damped_direct(problem, damping: float, num_outer: int) -> dict[str, object]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    history: list[dict[str, object]] = [
        {
            "outer": 0,
            "damping": float(damping),
            "nonlinear_objective": float(nonlinear_objective_g2o(problem, poses)),
        }
    ]

    for outer in range(1, int(num_outer) + 1):
        step = direct_newton_step_g2o(problem, poses)
        delta = float(damping) * np.asarray(step["delta"], dtype=float)
        poses = apply_pose_deltas(poses, delta)
        pose_history.append(poses.copy())
        obj = float(nonlinear_objective_g2o(problem, poses))
        history.append(
            {
                "outer": int(outer),
                "damping": float(damping),
                "nonlinear_objective": obj,
                "undamped_step_norm": float(step["linear_step_norm"]),
                "damped_step_norm": float(np.linalg.norm(delta)),
                "exact_linear_residual_norm": float(step["linear_residual_norm"]),
            }
        )

    return {
        "damping": float(damping),
        "num_outer": int(num_outer),
        "history": history,
        "pose_history": np.asarray(pose_history, dtype=float),
        "final_poses": poses.copy(),
    }


def main() -> None:
    problem = parse_g2o_se2(G2O_PATH)
    dampings = [1.0, 0.8, 0.5, 0.1]
    num_outer = 100

    all_rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    pose_histories: dict[str, np.ndarray] = {
        "initial_poses": np.asarray(problem.init_poses, dtype=float),
    }

    for damping in dampings:
        result = run_damped_direct(problem, damping=damping, num_outer=num_outer)
        key = f"damping_{damping:g}".replace(".", "p")
        pose_histories[f"{key}_pose_history"] = result["pose_history"]
        pose_histories[f"{key}_final_poses"] = result["final_poses"]
        all_rows.extend(result["history"])
        best = min(result["history"], key=lambda row: float(row["nonlinear_objective"]))
        final = result["history"][-1]
        summary.append(
            {
                "damping": float(damping),
                "num_outer": int(num_outer),
                "best_outer": int(best["outer"]),
                "best_objective": float(best["nonlinear_objective"]),
                "final_objective": float(final["nonlinear_objective"]),
                "final_step_norm": final.get("damped_step_norm", ""),
            }
        )
        print(
            f"damping={damping:g} best_outer={best['outer']} "
            f"best_obj={float(best['nonlinear_objective']):.12g} "
            f"final_obj={float(final['nonlinear_objective']):.12g}",
            flush=True,
        )

    stem = "intel_g2o_direct_newton_damping_outer100"
    history_csv = RESULT_DIR / f"{stem}_history.csv"
    summary_csv = RESULT_DIR / f"{stem}_summary.csv"
    json_path = RESULT_DIR / f"{stem}.json"
    trajectories_path = RESULT_DIR / f"{stem}_trajectories.npz"

    save_csv(all_rows, history_csv)
    save_csv(summary, summary_csv)
    json_path.write_text(
        json.dumps(
            {
                "path": str(G2O_PATH),
                "num_outer": num_outer,
                "dampings": dampings,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(trajectories_path, **pose_histories)
    print(
        json.dumps(
            {
                "history_csv": str(history_csv),
                "summary_csv": str(summary_csv),
                "json": str(json_path),
                "trajectories": str(trajectories_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
