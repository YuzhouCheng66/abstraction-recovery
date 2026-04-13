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
from gtsam import symbol

from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.se2_utils import se2_compose


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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


def symbol_key(i: int) -> int:
    return symbol("x", int(i))


def pose2_from_array(arr: np.ndarray) -> gtsam.Pose2:
    arr = np.asarray(arr, dtype=float).reshape(3)
    return gtsam.Pose2(float(arr[0]), float(arr[1]), float(arr[2]))


def _lookup_key(index: int, original_id: int, key_mode: str) -> int:
    if key_mode == "symbol":
        return symbol_key(index)
    if key_mode == "index":
        return int(original_id)
    raise ValueError(f"Unsupported key_mode: {key_mode}")


def values_to_pose_array(values: gtsam.Values, original_ids: list[int], key_mode: str) -> np.ndarray:
    poses = np.zeros((len(original_ids), 3), dtype=float)
    for index, original_id in enumerate(original_ids):
        pose = values.atPose2(_lookup_key(index, original_id, key_mode))
        poses[index, 0] = float(pose.x())
        poses[index, 1] = float(pose.y())
        poses[index, 2] = float(pose.theta())
    return poses


def build_odometry_chain_initial(problem) -> np.ndarray:
    num_poses = int(problem.init_poses.shape[0])
    initial = np.zeros((num_poses, 3), dtype=float)
    initial[0] = np.asarray(problem.anchor_pose, dtype=float).reshape(3)

    odom_edges = {
        (int(edge.i), int(edge.j)): np.asarray(edge.measurement, dtype=float).reshape(3)
        for edge in problem.edges
        if edge.kind == "odometry" and int(edge.j) == int(edge.i) + 1
    }

    for i in range(num_poses - 1):
        meas = odom_edges.get((i, i + 1))
        if meas is None:
            raise ValueError(f"Missing sequential odometry edge {(i, i + 1)}")
        initial[i + 1] = se2_compose(initial[i], meas)

    return initial


def marginal_sigmas_from_information(info: np.ndarray) -> np.ndarray:
    cov = np.linalg.inv(np.asarray(info, dtype=float).reshape(3, 3))
    return np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))


def diagonal_sigmas_from_information_diag(info: np.ndarray) -> np.ndarray:
    diag = np.diag(np.asarray(info, dtype=float).reshape(3, 3))
    return np.sqrt(1.0 / diag)


def build_anchor_noise_model(problem, anchor_noise_mode: str, anchor_sigmas: tuple[float, float, float] | None):
    if anchor_noise_mode == "information":
        return gtsam.noiseModel.Gaussian.Information(
            np.asarray(problem.anchor_information, dtype=float).reshape(3, 3)
        )
    if anchor_noise_mode == "sigmas":
        if anchor_sigmas is None:
            anchor_sigmas = (1e-3, 1e-3, 1e-5)
        sigmas = np.asarray(anchor_sigmas, dtype=float).reshape(3)
        return gtsam.noiseModel.Diagonal.Sigmas(
            gtsam.Point3(float(sigmas[0]), float(sigmas[1]), float(sigmas[2]))
        )
    raise ValueError(f"Unsupported anchor_noise_mode: {anchor_noise_mode}")


def build_manual_pose2_graph(
    problem,
    edge_noise_mode: str = "information",
    anchor_noise_mode: str = "information",
    anchor_sigmas: tuple[float, float, float] | None = None,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, np.ndarray, str]:
    graph = gtsam.NonlinearFactorGraph()
    initial_poses = build_odometry_chain_initial(problem)
    initial = gtsam.Values()

    for i, pose in enumerate(initial_poses):
        initial.insert(symbol_key(i), pose2_from_array(pose))

    graph.add(
        gtsam.PriorFactorPose2(
            symbol_key(0),
            pose2_from_array(problem.anchor_pose),
            build_anchor_noise_model(problem, anchor_noise_mode, anchor_sigmas),
        )
    )

    for edge in problem.edges:
        info = np.asarray(edge.information, dtype=float).reshape(3, 3)
        if edge_noise_mode == "information":
            noise = gtsam.noiseModel.Gaussian.Information(info)
        elif edge_noise_mode == "covariance":
            noise = gtsam.noiseModel.Gaussian.Covariance(np.linalg.inv(info))
        elif edge_noise_mode == "diag_info":
            sigmas = diagonal_sigmas_from_information_diag(info)
            noise = gtsam.noiseModel.Diagonal.Sigmas(
                gtsam.Point3(float(sigmas[0]), float(sigmas[1]), float(sigmas[2]))
            )
        elif edge_noise_mode == "diag_cov":
            sigmas = marginal_sigmas_from_information(info)
            noise = gtsam.noiseModel.Diagonal.Sigmas(
                gtsam.Point3(float(sigmas[0]), float(sigmas[1]), float(sigmas[2]))
            )
        else:
            raise ValueError(f"Unsupported edge_noise_mode: {edge_noise_mode}")

        graph.add(
            gtsam.BetweenFactorPose2(
                symbol_key(int(edge.i)),
                symbol_key(int(edge.j)),
                pose2_from_array(edge.measurement),
                noise,
            )
        )

    return graph, initial, initial_poses, "symbol"


def build_official_readg2o_graph(
    problem,
    anchor_noise_mode: str = "information",
    anchor_sigmas: tuple[float, float, float] | None = None,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, np.ndarray, str]:
    graph, initial = gtsam.readG2o(problem.source_path, False)
    root_key = int(problem.original_ids[0])
    graph.add(
        gtsam.PriorFactorPose2(
            root_key,
            pose2_from_array(problem.anchor_pose),
            build_anchor_noise_model(problem, anchor_noise_mode, anchor_sigmas),
        )
    )
    initial_poses = values_to_pose_array(initial, problem.original_ids, key_mode="index")
    return graph, initial, initial_poses, "index"


def plot_initial_vs_result(
    problem,
    initial_poses: np.ndarray,
    result_poses: np.ndarray,
    out_path: pathlib.Path,
    title: str,
    initial_label: str,
    result_label: str,
) -> pathlib.Path:
    initial = np.asarray(initial_poses, dtype=float)
    result = np.asarray(result_poses, dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)
    for edge in problem.edges:
        if edge.kind != "loop":
            continue
        pi = initial[edge.i, :2]
        pj = initial[edge.j, :2]
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#8a8a8a", alpha=0.25, linewidth=0.30, zorder=1)

    ax.plot(
        initial[:, 0],
        initial[:, 1],
        color="#8f8f8f",
        linewidth=1.0,
        linestyle=(0, (10, 3, 2, 3)),
        label=initial_label,
        zorder=2,
    )
    ax.plot(
        result[:, 0],
        result[:, 1],
        color="#d62728",
        linewidth=1.25,
        label=result_label,
        zorder=3,
    )
    ax.scatter([initial[0, 0]], [initial[0, 1]], color="black", s=18, label="Anchor/start", zorder=4)
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


def run_gtsam_lm_with_history(
    problem,
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    original_ids: list[int],
    key_mode: str,
    params: gtsam.LevenbergMarquardtParams,
) -> dict[str, object]:
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    initial_poses = values_to_pose_array(initial, original_ids, key_mode)

    history: list[dict[str, object]] = []
    pose_history = [initial_poses]
    initial_error = float(graph.error(initial))
    initial_custom_objective = float(nonlinear_objective_g2o(problem, initial_poses))
    history.append(
        {
            "iteration": 0,
            "graph_error": initial_error,
            "custom_objective": initial_custom_objective,
            "lambda": float(optimizer.lambda_()),
        }
    )

    termination_reason = "max_iterations"
    exception_message = None
    for _ in range(int(params.getMaxIterations())):
        old_error = float(optimizer.error())
        try:
            optimizer.iterate()
        except Exception as ex:  # pragma: no cover - defensive around wrapped C++ exceptions
            termination_reason = "exception"
            exception_message = str(ex)
            break

        current_values = optimizer.values()
        current_poses = values_to_pose_array(current_values, original_ids, key_mode)
        new_error = float(optimizer.error())
        lambda_now = float(optimizer.lambda_())
        abs_decrease = float(abs(old_error - new_error))
        rel_decrease = float(abs_decrease / max(abs(old_error), 1e-15))

        history.append(
            {
                "iteration": int(optimizer.iterations()),
                "graph_error": new_error,
                "custom_objective": float(nonlinear_objective_g2o(problem, current_poses)),
                "lambda": lambda_now,
                "absolute_decrease": abs_decrease,
                "relative_decrease": rel_decrease,
            }
        )
        pose_history.append(current_poses)

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

    final_values = optimizer.values()
    final_poses = values_to_pose_array(final_values, original_ids, key_mode)
    return {
        "history": history,
        "pose_history": [poses.tolist() for poses in pose_history],
        "final_poses": final_poses.tolist(),
        "termination_reason": termination_reason,
        "exception_message": exception_message,
        "initial_graph_error": initial_error,
        "initial_custom_objective": initial_custom_objective,
        "final_graph_error": float(graph.error(final_values)),
        "final_custom_objective": float(nonlinear_objective_g2o(problem, final_poses)),
        "accepted_iterations": int(optimizer.iterations()),
    }


def run_gtsam_gn_with_history(
    problem,
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    original_ids: list[int],
    key_mode: str,
    params: gtsam.GaussNewtonParams,
) -> dict[str, object]:
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
    initial_poses = values_to_pose_array(initial, original_ids, key_mode)

    history: list[dict[str, object]] = []
    pose_history = [initial_poses]
    initial_error = float(graph.error(initial))
    initial_custom_objective = float(nonlinear_objective_g2o(problem, initial_poses))
    history.append(
        {
            "iteration": 0,
            "graph_error": initial_error,
            "custom_objective": initial_custom_objective,
        }
    )

    termination_reason = "max_iterations"
    exception_message = None
    for _ in range(int(params.getMaxIterations())):
        old_error = float(optimizer.error())
        try:
            optimizer.iterate()
        except Exception as ex:  # pragma: no cover - defensive around wrapped C++ exceptions
            termination_reason = "exception"
            exception_message = str(ex)
            break

        current_values = optimizer.values()
        current_poses = values_to_pose_array(current_values, original_ids, key_mode)
        new_error = float(optimizer.error())
        abs_decrease = float(abs(old_error - new_error))
        rel_decrease = float(abs_decrease / max(abs(old_error), 1e-15))

        history.append(
            {
                "iteration": int(optimizer.iterations()),
                "graph_error": new_error,
                "custom_objective": float(nonlinear_objective_g2o(problem, current_poses)),
                "absolute_decrease": abs_decrease,
                "relative_decrease": rel_decrease,
            }
        )
        pose_history.append(current_poses)

        if new_error > old_error:
            termination_reason = "error_increased"
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

    final_values = optimizer.values()
    final_poses = values_to_pose_array(final_values, original_ids, key_mode)
    return {
        "history": history,
        "pose_history": [poses.tolist() for poses in pose_history],
        "final_poses": final_poses.tolist(),
        "termination_reason": termination_reason,
        "exception_message": exception_message,
        "initial_graph_error": initial_error,
        "initial_custom_objective": initial_custom_objective,
        "final_graph_error": float(graph.error(final_values)),
        "final_custom_objective": float(nonlinear_objective_g2o(problem, final_poses)),
        "accepted_iterations": int(optimizer.iterations()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2o-path", type=pathlib.Path, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--construction",
        type=str,
        choices=("official_readg2o", "manual_odom_chain"),
        default="official_readg2o",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("lm", "gn"),
        default="lm",
    )
    parser.add_argument(
        "--edge-noise-mode",
        type=str,
        choices=("information", "covariance", "diag_cov", "diag_info"),
        default="information",
    )
    parser.add_argument(
        "--anchor-noise-mode",
        type=str,
        choices=("information", "sigmas"),
        default="information",
    )
    parser.add_argument("--anchor-sigmas", nargs=3, type=float, default=None)
    parser.add_argument("--relative-error-tol", type=float, default=1e-10)
    parser.add_argument("--absolute-error-tol", type=float, default=1e-8)
    parser.add_argument("--error-tol", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--lambda-initial", type=float, default=1e-5)
    parser.add_argument("--lambda-factor", type=float, default=None)
    parser.add_argument("--lambda-upper-bound", type=float, default=None)
    parser.add_argument("--lambda-lower-bound", type=float, default=None)
    parser.add_argument("--diagonal-damping", action="store_true")
    parser.add_argument("--use-fixed-lambda-factor", action="store_true", default=True)
    parser.add_argument("--no-fixed-lambda-factor", dest="use_fixed_lambda_factor", action="store_false")
    args = parser.parse_args(argv)

    problem = parse_g2o_se2(args.g2o_path)
    if args.construction == "official_readg2o":
        if args.edge_noise_mode != "information":
            raise ValueError("official_readg2o always uses the file-provided per-factor precision")
        graph, initial, initial_poses, key_mode = build_official_readg2o_graph(
            problem,
            anchor_noise_mode=args.anchor_noise_mode,
            anchor_sigmas=None if args.anchor_sigmas is None else tuple(float(x) for x in args.anchor_sigmas),
        )
        effective_edge_noise_mode = "file_via_readg2o"
        initial_label = "g2o initial"
    else:
        graph, initial, initial_poses, key_mode = build_manual_pose2_graph(
            problem,
            edge_noise_mode=args.edge_noise_mode,
            anchor_noise_mode=args.anchor_noise_mode,
            anchor_sigmas=None if args.anchor_sigmas is None else tuple(float(x) for x in args.anchor_sigmas),
        )
        effective_edge_noise_mode = args.edge_noise_mode
        initial_label = "odom-chain initial"

    label = args.label or args.g2o_path.stem
    label_stem = f"{stem_slug(label)}_g2o"
    construction_stem = "official" if args.construction == "official_readg2o" else "manual"
    edge_stem = "file" if args.construction == "official_readg2o" else effective_edge_noise_mode.lower()
    optimizer_stem = args.optimizer.lower()
    stem = f"{label_stem}_true_gtsam_{construction_stem}_{edge_stem}_{optimizer_stem}_baseline"

    if args.optimizer == "lm":
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(int(args.max_iterations))
        params.setRelativeErrorTol(float(args.relative_error_tol))
        params.setAbsoluteErrorTol(float(args.absolute_error_tol))
        params.setlambdaInitial(float(args.lambda_initial))
        if hasattr(params, "setVerbosityLM"):
            params.setVerbosityLM("SILENT")
        if args.error_tol is not None:
            params.setErrorTol(float(args.error_tol))
        if args.lambda_factor is not None:
            params.setlambdaFactor(float(args.lambda_factor))
        if args.lambda_upper_bound is not None:
            params.setlambdaUpperBound(float(args.lambda_upper_bound))
        if args.lambda_lower_bound is not None:
            params.setlambdaLowerBound(float(args.lambda_lower_bound))
        params.setDiagonalDamping(bool(args.diagonal_damping))
        params.setUseFixedLambdaFactor(bool(args.use_fixed_lambda_factor))
        result = run_gtsam_lm_with_history(problem, graph, initial, problem.original_ids, key_mode, params)
    else:
        params = gtsam.GaussNewtonParams()
        params.setMaxIterations(int(args.max_iterations))
        params.setRelativeErrorTol(float(args.relative_error_tol))
        params.setAbsoluteErrorTol(float(args.absolute_error_tol))
        params.setVerbosity("SILENT")
        if args.error_tol is not None:
            params.setErrorTol(float(args.error_tol))
        result = run_gtsam_gn_with_history(problem, graph, initial, problem.original_ids, key_mode, params)

    final_poses = np.asarray(result["final_poses"], dtype=float)
    plot_path = plot_initial_vs_result(
        problem,
        initial_poses,
        final_poses,
        OUT_DIR / f"{stem}_initial_vs_result.png",
        f"{label} g2o: True GTSAM {args.optimizer.upper()}",
        initial_label=initial_label,
        result_label=f"True GTSAM {args.optimizer.upper()}",
    )
    json_path = RESULT_DIR / f"{stem}.json"
    history_csv = RESULT_DIR / f"{stem}_history.csv"
    traj_path = RESULT_DIR / f"{stem}_trajectories.npz"

    out = {
        "config": {
            "path": str(args.g2o_path),
            "label": label,
            "construction": args.construction,
            "optimizer": args.optimizer,
            "edge_noise_mode": effective_edge_noise_mode,
            "anchor_noise_mode": args.anchor_noise_mode,
            "anchor_sigmas": None if args.anchor_sigmas is None else [float(x) for x in args.anchor_sigmas],
            "anchor_information": np.asarray(problem.anchor_information, dtype=float).tolist(),
            "initialization": initial_label,
            "optimizer_params": params_summary(params),
        },
        "result": result,
    }
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_csv(result["history"], history_csv)
    np.savez_compressed(
        traj_path,
        g2o_vertex_poses=np.asarray(problem.init_poses, dtype=float),
        initial_poses=np.asarray(initial_poses, dtype=float),
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
                "initial_graph_error": result["initial_graph_error"],
                "initial_custom_objective": result["initial_custom_objective"],
                "final_graph_error": result["final_graph_error"],
                "final_custom_objective": result["final_custom_objective"],
                "final_row": result["history"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
