from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import gtsam

from svd_abstraction.g2o_se2 import parse_g2o_se2


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

X = gtsam.symbol_shorthand.X


def save_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(key, "")) for key in keys) + "\n")


def stem_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return slug.strip("_") or "gtsam"


def pose2_from_array(arr: np.ndarray) -> gtsam.Pose2:
    arr = np.asarray(arr, dtype=float).reshape(3)
    return gtsam.Pose2(float(arr[0]), float(arr[1]), float(arr[2]))


def values_to_pose_array(values: gtsam.Values, num_poses: int) -> np.ndarray:
    poses = np.zeros((int(num_poses), 3), dtype=float)
    for i in range(int(num_poses)):
        pose = values.atPose2(X(i))
        poses[i, 0] = float(pose.x())
        poses[i, 1] = float(pose.y())
        poses[i, 2] = float(pose.theta())
    return poses


def build_gtsam_pose2_graph(problem) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    for i, pose in enumerate(np.asarray(problem.init_poses, dtype=float)):
        initial.insert(X(i), pose2_from_array(pose))

    prior_noise = gtsam.noiseModel.Gaussian.Information(np.asarray(problem.anchor_information, dtype=float))
    graph.add(gtsam.PriorFactorPose2(X(0), pose2_from_array(problem.anchor_pose), prior_noise))

    for edge in problem.edges:
        noise = gtsam.noiseModel.Gaussian.Information(np.asarray(edge.information, dtype=float))
        measurement = pose2_from_array(edge.measurement)
        graph.add(gtsam.BetweenFactorPose2(X(int(edge.i)), X(int(edge.j)), measurement, noise))

    return graph, initial


def plot_initial_vs_result(problem, result_poses: np.ndarray, out_path: pathlib.Path, title: str) -> pathlib.Path:
    init = np.asarray(problem.init_poses, dtype=float)
    result = np.asarray(result_poses, dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)
    for edge in problem.edges:
        if edge.kind != "loop":
            continue
        pi = init[edge.i, :2]
        pj = init[edge.j, :2]
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#8a8a8a", alpha=0.25, linewidth=0.30, zorder=1)

    ax.plot(
        init[:, 0],
        init[:, 1],
        color="#8f8f8f",
        linewidth=1.0,
        linestyle=(0, (10, 3, 2, 3)),
        label="Initial",
        zorder=2,
    )
    ax.plot(
        result[:, 0],
        result[:, 1],
        color="#d62728",
        linewidth=1.25,
        label="GTSAM LM",
        zorder=3,
    )
    ax.scatter([init[0, 0]], [init[0, 1]], color="black", s=18, label="Anchor/start", zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def params_summary(params: gtsam.LevenbergMarquardtParams) -> dict[str, object]:
    return {
        "relative_error_tol": float(params.getRelativeErrorTol()),
        "absolute_error_tol": float(params.getAbsoluteErrorTol()),
        "error_tol": float(params.getErrorTol()),
        "max_iterations": int(params.getMaxIterations()),
        "lambda_initial": float(params.getlambdaInitial()),
        "lambda_factor": float(params.getlambdaFactor()),
        "lambda_upper_bound": float(params.getlambdaUpperBound()),
        "lambda_lower_bound": float(params.getlambdaLowerBound()),
        "diagonal_damping": bool(params.getDiagonalDamping()),
        "use_fixed_lambda_factor": bool(params.getUseFixedLambdaFactor()),
    }


def run_gtsam_lm(problem, params: gtsam.LevenbergMarquardtParams) -> dict[str, object]:
    graph, initial = build_gtsam_pose2_graph(problem)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    history: list[dict[str, object]] = []
    pose_history = [values_to_pose_array(initial, problem.init_poses.shape[0])]

    prev_error = float(graph.error(initial))
    history.append(
        {
            "iteration": 0,
            "graph_error": prev_error,
            "lambda": float(optimizer.lambda_()),
        }
    )

    termination_reason = "max_iterations"
    for _ in range(int(params.getMaxIterations())):
        old_error = float(optimizer.error())
        optimizer.iterate()
        current_values = optimizer.values()
        new_error = float(optimizer.error())
        lambda_now = float(optimizer.lambda_())
        pose_history.append(values_to_pose_array(current_values, problem.init_poses.shape[0]))
        abs_decrease = float(abs(old_error - new_error))
        rel_decrease = float(abs_decrease / max(abs(old_error), 1e-15))

        history.append(
            {
                "iteration": int(optimizer.iterations()),
                "graph_error": new_error,
                "lambda": lambda_now,
                "absolute_decrease": abs_decrease,
                "relative_decrease": rel_decrease,
            }
        )

        if lambda_now >= float(params.getlambdaUpperBound()) and new_error >= old_error:
            termination_reason = "lambda_upper_bound"
            break
        if new_error <= float(params.getErrorTol()):
            termination_reason = "error_tolerance"
            break
        if abs_decrease < float(params.getAbsoluteErrorTol()):
            termination_reason = "absolute_error_tolerance"
            break
        if rel_decrease < float(params.getRelativeErrorTol()):
            termination_reason = "relative_error_tolerance"
            break

        prev_error = new_error

    final_values = optimizer.values()
    final_poses = values_to_pose_array(final_values, problem.init_poses.shape[0])
    return {
        "history": history,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": final_poses.tolist(),
        "termination_reason": termination_reason,
        "final_graph_error": float(graph.error(final_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2o-path", type=pathlib.Path, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--relative-error-tol", type=float, default=None)
    parser.add_argument("--absolute-error-tol", type=float, default=None)
    parser.add_argument("--error-tol", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--lambda-initial", type=float, default=None)
    parser.add_argument("--lambda-factor", type=float, default=None)
    parser.add_argument("--lambda-upper-bound", type=float, default=None)
    parser.add_argument("--lambda-lower-bound", type=float, default=None)
    parser.add_argument("--diagonal-damping", action="store_true")
    parser.add_argument("--use-fixed-lambda-factor", action="store_true", default=True)
    parser.add_argument("--no-fixed-lambda-factor", dest="use_fixed_lambda_factor", action="store_false")
    args = parser.parse_args()

    label = args.label or args.g2o_path.stem
    label_stem = f"{stem_slug(label)}_g2o"
    stem = f"{label_stem}_true_gtsam_lm_baseline"

    params = gtsam.LevenbergMarquardtParams()
    if args.relative_error_tol is not None:
        params.setRelativeErrorTol(float(args.relative_error_tol))
    if args.absolute_error_tol is not None:
        params.setAbsoluteErrorTol(float(args.absolute_error_tol))
    if args.error_tol is not None:
        params.setErrorTol(float(args.error_tol))
    if args.max_iterations is not None:
        params.setMaxIterations(int(args.max_iterations))
    if args.lambda_initial is not None:
        params.setlambdaInitial(float(args.lambda_initial))
    if args.lambda_factor is not None:
        params.setlambdaFactor(float(args.lambda_factor))
    if args.lambda_upper_bound is not None:
        params.setlambdaUpperBound(float(args.lambda_upper_bound))
    if args.lambda_lower_bound is not None:
        params.setlambdaLowerBound(float(args.lambda_lower_bound))
    params.setDiagonalDamping(bool(args.diagonal_damping))
    params.setUseFixedLambdaFactor(bool(args.use_fixed_lambda_factor))

    problem = parse_g2o_se2(args.g2o_path)
    result = run_gtsam_lm(problem, params)
    final_poses = np.asarray(result["final_poses"], dtype=float)

    plot_path = plot_initial_vs_result(
        problem,
        final_poses,
        OUT_DIR / f"{label_stem}_true_gtsam_initial_vs_lm.png",
        f"{label} g2o: Initial vs True GTSAM LM",
    )
    json_path = RESULT_DIR / f"{stem}.json"
    history_csv = RESULT_DIR / f"{stem}_history.csv"
    traj_path = RESULT_DIR / f"{stem}_trajectories.npz"

    out = {
        "config": {
            "path": str(args.g2o_path),
            "label": label,
            "gtsam_params": params_summary(params),
        },
        "result": result,
    }
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_csv(result["history"], history_csv)
    np.savez_compressed(
        traj_path,
        initial_poses=np.asarray(problem.init_poses, dtype=float),
        final_poses=final_poses,
        pose_history=np.asarray(result["pose_history"], dtype=float),
    )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "history_csv": str(history_csv),
                "trajectories": str(traj_path),
                "plot": str(plot_path),
                "termination_reason": result["termination_reason"],
                "final_row": result["history"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
