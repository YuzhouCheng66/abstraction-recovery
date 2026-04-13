from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys


REPO_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SLOW_GTSAM_ROOT = REPO_ROOT / "third_party" / "gtsam_build" / "build-slow"
SLOW_GTSAM_PYTHON = SLOW_GTSAM_ROOT / "python"
SLOW_GTSAM_LIB = SLOW_GTSAM_ROOT / "gtsam"
OUT_DIR = REPO_ROOT / "svd_abstraction" / "out"
RESULT_DIR = REPO_ROOT / "svd_abstraction" / "output_results"


def _prepend_env_path(existing: str | None, values: list[str]) -> str:
    parts: list[str] = []
    for value in values:
        if value and value not in parts:
            parts.append(value)
    for value in (existing or "").split(":"):
        if value and value not in parts:
            parts.append(value)
    return ":".join(parts)


def _early_variant_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gtsam-variant",
        choices=("auto", "slow_correct", "system"),
        default="auto",
    )
    return parser


def ensure_gtsam_variant(argv: list[str]) -> str:
    args, _ = _early_variant_parser().parse_known_args(argv[1:])
    requested = args.gtsam_variant
    if requested == "system":
        return "system"

    have_local_build = SLOW_GTSAM_PYTHON.exists() and SLOW_GTSAM_LIB.exists()
    if requested == "slow_correct" and not have_local_build:
        raise RuntimeError(
            f"Requested slow-correct GTSAM, but local build was not found at {SLOW_GTSAM_ROOT}"
        )
    if requested == "auto" and not have_local_build:
        return "system"

    desired = "slow_correct"
    active = os.environ.get("ABSTRACTION_GTSAM_VARIANT_ACTIVE")
    python_ok = str(SLOW_GTSAM_PYTHON) in os.environ.get("PYTHONPATH", "").split(":")
    lib_ok = str(SLOW_GTSAM_LIB) in os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if active == desired and python_ok and lib_ok:
        return desired

    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_env_path(env.get("PYTHONPATH"), [str(SLOW_GTSAM_PYTHON)])
    env["LD_LIBRARY_PATH"] = _prepend_env_path(
        env.get("LD_LIBRARY_PATH"),
        ["/usr/lib/x86_64-linux-gnu", str(SLOW_GTSAM_LIB)],
    )
    env["ABSTRACTION_GTSAM_VARIANT_ACTIVE"] = desired
    os.execvpe(sys.executable, [sys.executable, *argv], env)
    raise AssertionError("unreachable")


ACTIVE_GTSAM_VARIANT = ensure_gtsam_variant(sys.argv)

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import gtsam

from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2


OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def stem_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return slug.strip("_") or "gtsam"


def values_to_pose_array(values: gtsam.Values, original_ids: list[int]) -> np.ndarray:
    poses = np.zeros((len(original_ids), 3), dtype=float)
    for index, original_id in enumerate(original_ids):
        pose = values.atPose2(int(original_id))
        poses[index, 0] = float(pose.x())
        poses[index, 1] = float(pose.y())
        poses[index, 2] = float(pose.theta())
    return poses


def build_lago_graph_with_diagonal_noise(
    graph: gtsam.NonlinearFactorGraph,
) -> gtsam.NonlinearFactorGraph:
    diagonal_graph = gtsam.NonlinearFactorGraph()
    for factor_index in range(graph.size()):
        factor = graph.at(factor_index)
        if isinstance(factor, gtsam.BetweenFactorPose2):
            covariance = factor.noiseModel().covariance()
            diagonal_graph.add(
                gtsam.BetweenFactorPose2(
                    int(factor.keys()[0]),
                    int(factor.keys()[1]),
                    factor.measured(),
                    gtsam.noiseModel.Diagonal.Variances(np.diag(covariance)),
                )
            )
        elif isinstance(factor, gtsam.PriorFactorPose2):
            covariance = factor.noiseModel().covariance()
            diagonal_graph.add(
                gtsam.PriorFactorPose2(
                    int(factor.keys()[0]),
                    factor.prior(),
                    gtsam.noiseModel.Diagonal.Variances(np.diag(covariance)),
                )
            )
        else:
            raise TypeError(f"Unsupported factor type for Pose2 solver: {type(factor)}")
    return diagonal_graph


def plot_initial_vs_result(
    problem,
    initial_poses: np.ndarray,
    result_poses: np.ndarray,
    out_path: pathlib.Path,
    title: str,
    initial_label: str,
    result_label: str,
) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)
    for edge in problem.edges:
        if edge.kind != "loop":
            continue
        pi = initial_poses[edge.i, :2]
        pj = initial_poses[edge.j, :2]
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#8a8a8a", alpha=0.25, linewidth=0.30, zorder=1)

    ax.plot(
        initial_poses[:, 0],
        initial_poses[:, 1],
        color="#8f8f8f",
        linewidth=1.0,
        linestyle=(0, (10, 3, 2, 3)),
        label=initial_label,
        zorder=2,
    )
    ax.plot(
        result_poses[:, 0],
        result_poses[:, 1],
        color="#d62728",
        linewidth=1.25,
        label=result_label,
        zorder=3,
    )
    ax.scatter([initial_poses[0, 0]], [initial_poses[0, 1]], color="black", s=18, label="Anchor/start", zorder=4)
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


def params_summary(params) -> dict[str, object]:
    summary = {
        "relative_error_tol": float(params.getRelativeErrorTol()),
        "absolute_error_tol": float(params.getAbsoluteErrorTol()),
        "error_tol": float(params.getErrorTol()),
        "max_iterations": int(params.getMaxIterations()),
    }
    if hasattr(params, "getlambdaInitial"):
        summary.update(
            {
                "lambda_initial": float(params.getlambdaInitial()),
                "lambda_factor": float(params.getlambdaFactor()),
                "lambda_upper_bound": float(params.getlambdaUpperBound()),
                "lambda_lower_bound": float(params.getlambdaLowerBound()),
                "diagonal_damping": bool(params.getDiagonalDamping()),
                "use_fixed_lambda_factor": bool(params.getUseFixedLambdaFactor()),
            }
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Official-style GTSAM Pose2 pose-graph solver for g2o files: "
            "readG2o + PriorFactorPose2 + optional LAGO initialization + GN/LM refine."
        )
    )
    parser.add_argument("--g2o-path", type=pathlib.Path, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--gtsam-variant",
        choices=("auto", "slow_correct", "system"),
        default="auto",
        help="Select which GTSAM build to run. 'auto' prefers the local slow-correct build if present.",
    )
    parser.add_argument(
        "--initialization",
        choices=("readg2o", "lago"),
        default="lago",
        help="Initialization used before nonlinear refinement.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("lm", "gn"),
        default="lm",
        help="Official GTSAM nonlinear optimizer used for refinement.",
    )
    parser.add_argument("--relative-error-tol", type=float, default=1e-10)
    parser.add_argument("--absolute-error-tol", type=float, default=1e-8)
    parser.add_argument("--error-tol", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--lambda-initial", type=float, default=1e-5)
    parser.add_argument("--lambda-factor", type=float, default=None)
    parser.add_argument("--lambda-upper-bound", type=float, default=1e12)
    parser.add_argument("--lambda-lower-bound", type=float, default=None)
    parser.add_argument("--diagonal-damping", action="store_true")
    parser.add_argument("--use-fixed-lambda-factor", action="store_true", default=True)
    parser.add_argument("--no-fixed-lambda-factor", dest="use_fixed_lambda_factor", action="store_false")
    parser.add_argument("--output-g2o", type=pathlib.Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    problem = parse_g2o_se2(args.g2o_path)
    graph, readg2o_initial = gtsam.readG2o(str(args.g2o_path), False)
    prior_model = gtsam.noiseModel.Diagonal.Variances(gtsam.Point3(1e-6, 1e-6, 1e-8))
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(), prior_model))

    file_initial_poses = np.asarray(problem.init_poses, dtype=float)
    readg2o_initial_poses = values_to_pose_array(readg2o_initial, problem.original_ids)

    initialization_error = float(graph.error(readg2o_initial))
    initialization_custom_objective = float(nonlinear_objective_g2o(problem, readg2o_initial_poses))
    initialization_source = "readg2o"
    optimizer_initial_values = readg2o_initial
    optimizer_initial_poses = readg2o_initial_poses
    lago_error = None
    lago_custom_objective = None

    if args.initialization == "lago":
        diagonal_graph = build_lago_graph_with_diagonal_noise(graph)
        optimizer_initial_values = gtsam.lago.initialize(diagonal_graph)
        optimizer_initial_poses = values_to_pose_array(optimizer_initial_values, problem.original_ids)
        initialization_source = "lago_on_diagonalized_graph"
        lago_error = float(graph.error(optimizer_initial_values))
        lago_custom_objective = float(nonlinear_objective_g2o(problem, optimizer_initial_poses))

    if args.optimizer == "lm":
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(int(args.max_iterations))
        params.setRelativeErrorTol(float(args.relative_error_tol))
        params.setAbsoluteErrorTol(float(args.absolute_error_tol))
        params.setlambdaInitial(float(args.lambda_initial))
        params.setlambdaUpperBound(float(args.lambda_upper_bound))
        if hasattr(params, "setVerbosityLM"):
            params.setVerbosityLM("SILENT")
        if args.error_tol is not None:
            params.setErrorTol(float(args.error_tol))
        if args.lambda_factor is not None:
            params.setlambdaFactor(float(args.lambda_factor))
        if args.lambda_lower_bound is not None:
            params.setlambdaLowerBound(float(args.lambda_lower_bound))
        params.setDiagonalDamping(bool(args.diagonal_damping))
        params.setUseFixedLambdaFactor(bool(args.use_fixed_lambda_factor))
        result_values = gtsam.LevenbergMarquardtOptimizer(graph, optimizer_initial_values, params).optimize()
    else:
        params = gtsam.GaussNewtonParams()
        params.setMaxIterations(int(args.max_iterations))
        params.setRelativeErrorTol(float(args.relative_error_tol))
        params.setAbsoluteErrorTol(float(args.absolute_error_tol))
        params.setVerbosity("SILENT")
        if args.error_tol is not None:
            params.setErrorTol(float(args.error_tol))
        result_values = gtsam.GaussNewtonOptimizer(graph, optimizer_initial_values, params).optimize()

    result_poses = values_to_pose_array(result_values, problem.original_ids)
    final_error = float(graph.error(result_values))
    final_custom_objective = float(nonlinear_objective_g2o(problem, result_poses))

    label = args.label or args.g2o_path.stem
    stem = f"{stem_slug(label)}_g2o_true_gtsam_official_{args.initialization}_{args.optimizer}_solve"
    plot_path = plot_initial_vs_result(
        problem,
        file_initial_poses,
        result_poses,
        OUT_DIR / f"{stem}_initial_vs_result.png",
        f"{label} g2o: official GTSAM {args.initialization.upper()} + {args.optimizer.upper()}",
        initial_label="g2o initial",
        result_label=f"GTSAM {args.initialization.upper()} + {args.optimizer.upper()}",
    )
    json_path = RESULT_DIR / f"{stem}.json"
    traj_path = RESULT_DIR / f"{stem}_trajectories.npz"

    if args.output_g2o is not None:
        gtsam.writeG2o(graph, result_values, str(args.output_g2o))

    out = {
        "config": {
            "path": str(args.g2o_path),
            "label": label,
            "gtsam_variant_requested": args.gtsam_variant,
            "gtsam_variant_active": ACTIVE_GTSAM_VARIANT,
            "gtsam_module_path": str(pathlib.Path(gtsam.__file__).resolve()),
            "initialization": initialization_source,
            "optimizer": args.optimizer,
            "optimizer_params": params_summary(params),
            "prior_variances": [1e-6, 1e-6, 1e-8],
            "output_g2o": None if args.output_g2o is None else str(args.output_g2o),
        },
        "result": {
            "graph_size": int(graph.size()),
            "num_poses": int(file_initial_poses.shape[0]),
            "readg2o_initial_error": initialization_error,
            "readg2o_initial_custom_objective": initialization_custom_objective,
            "optimizer_initial_error": initialization_error if lago_error is None else lago_error,
            "optimizer_initial_custom_objective": initialization_custom_objective
            if lago_custom_objective is None
            else lago_custom_objective,
            "final_graph_error": final_error,
            "final_custom_objective": final_custom_objective,
        },
    }
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(
        traj_path,
        g2o_vertex_poses=file_initial_poses,
        readg2o_initial_poses=readg2o_initial_poses,
        optimizer_initial_poses=optimizer_initial_poses,
        final_poses=result_poses,
    )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "trajectories": str(traj_path),
                "plot": str(plot_path),
                "initialization": initialization_source,
                "optimizer": args.optimizer,
                "gtsam_variant_active": ACTIVE_GTSAM_VARIANT,
                "readg2o_initial_error": initialization_error,
                "optimizer_initial_error": initialization_error if lago_error is None else lago_error,
                "final_graph_error": final_error,
                "final_custom_objective": final_custom_objective,
                "output_g2o": None if args.output_g2o is None else str(args.output_g2o),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
