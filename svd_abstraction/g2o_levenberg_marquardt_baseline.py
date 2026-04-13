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

from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import run_gtsam_levenberg_marquardt_g2o
from svd_abstraction.g2o_se2 import run_levenberg_marquardt_g2o


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
    return slug.strip("_") or "g2o"


def plot_initial_vs_lm(problem, lm_poses: np.ndarray, out_path: pathlib.Path, title: str) -> pathlib.Path:
    init = np.asarray(problem.init_poses, dtype=float)
    lm = np.asarray(lm_poses, dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)

    for edge in problem.edges:
        if edge.kind != "loop":
            continue
        pi = init[edge.i, :2]
        pj = init[edge.j, :2]
        ax.plot(
            [pi[0], pj[0]],
            [pi[1], pj[1]],
            color="#8a8a8a",
            alpha=0.25,
            linewidth=0.30,
            zorder=1,
        )

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
        lm[:, 0],
        lm[:, 1],
        color="#d62728",
        linewidth=1.25,
        label="Levenberg-Marquardt",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2o-path", type=pathlib.Path, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--style", choices=("gtsam", "g2o"), default="gtsam")
    parser.add_argument("--num-outer", type=int, default=100)
    parser.add_argument("--rel-obj-tol", type=float, default=None)
    parser.add_argument("--step-tol", type=float, default=None)
    parser.add_argument("--initial-lambda", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=1e-5)
    parser.add_argument("--good-step-lower-scale", type=float, default=1.0 / 3.0)
    parser.add_argument("--good-step-upper-scale", type=float, default=2.0 / 3.0)
    parser.add_argument("--max-trials-after-failure", type=int, default=10)
    parser.add_argument("--relative-error-tol", type=float, default=1e-5)
    parser.add_argument("--absolute-error-tol", type=float, default=1e-5)
    parser.add_argument("--error-tol", type=float, default=0.0)
    parser.add_argument("--lambda-initial", type=float, default=1e-5)
    parser.add_argument("--lambda-factor", type=float, default=10.0)
    parser.add_argument("--lambda-upper-bound", type=float, default=1e5)
    parser.add_argument("--lambda-lower-bound", type=float, default=0.0)
    parser.add_argument("--min-model-fidelity", type=float, default=1e-3)
    parser.add_argument("--diagonal-damping", action="store_true")
    parser.add_argument("--use-fixed-lambda-factor", action="store_true", default=True)
    parser.add_argument("--no-fixed-lambda-factor", dest="use_fixed_lambda_factor", action="store_false")
    parser.add_argument("--min-diagonal", type=float, default=1e-6)
    parser.add_argument("--max-diagonal", type=float, default=1e32)
    parser.add_argument("--ridge", type=float, default=0.0)
    args = parser.parse_args()

    label = args.label or args.g2o_path.stem
    label_stem = f"{stem_slug(label)}_g2o"
    stem = f"{label_stem}_{args.style}_levenberg_marquardt_baseline"

    problem = parse_g2o_se2(args.g2o_path)
    if args.style == "gtsam":
        lm = run_gtsam_levenberg_marquardt_g2o(
            problem,
            num_outer=args.num_outer,
            relative_error_tol=args.relative_error_tol,
            absolute_error_tol=args.absolute_error_tol,
            error_tol=args.error_tol,
            lambda_initial=args.lambda_initial,
            lambda_factor=args.lambda_factor,
            lambda_upper_bound=args.lambda_upper_bound,
            lambda_lower_bound=args.lambda_lower_bound,
            min_model_fidelity=args.min_model_fidelity,
            diagonal_damping=args.diagonal_damping,
            use_fixed_lambda_factor=args.use_fixed_lambda_factor,
            min_diagonal=args.min_diagonal,
            max_diagonal=args.max_diagonal,
            ridge=args.ridge,
        )
    else:
        lm = run_levenberg_marquardt_g2o(
            problem,
            num_outer=args.num_outer,
            rel_obj_tol=args.rel_obj_tol,
            step_tol=args.step_tol,
            initial_lambda=args.initial_lambda,
            tau=args.tau,
            good_step_lower_scale=args.good_step_lower_scale,
            good_step_upper_scale=args.good_step_upper_scale,
            max_trials_after_failure=args.max_trials_after_failure,
            ridge=args.ridge,
        )
    final_poses = np.asarray(lm["final_poses"], dtype=float)

    plot_path = plot_initial_vs_lm(
        problem,
        final_poses,
        OUT_DIR / f"{label_stem}_{args.style}_initial_vs_levenberg_marquardt.png",
        f"{label} g2o: Initial vs Levenberg-Marquardt ({args.style.upper()} style)",
    )
    json_path = RESULT_DIR / f"{stem}.json"
    history_csv = RESULT_DIR / f"{stem}_history.csv"
    attempts_csv = RESULT_DIR / f"{stem}_attempts.csv"
    traj_path = RESULT_DIR / f"{stem}_trajectories.npz"

    out = {
        "config": {
            "path": str(args.g2o_path),
            "label": label,
            "style": args.style,
            "anchor_precision": 1e8,
            "num_outer": int(args.num_outer),
            "rel_obj_tol": args.rel_obj_tol,
            "step_tol": args.step_tol,
            "initial_lambda": float(args.initial_lambda),
            "tau": float(args.tau),
            "good_step_lower_scale": float(args.good_step_lower_scale),
            "good_step_upper_scale": float(args.good_step_upper_scale),
            "max_trials_after_failure": int(args.max_trials_after_failure),
            "relative_error_tol": float(args.relative_error_tol),
            "absolute_error_tol": float(args.absolute_error_tol),
            "error_tol": float(args.error_tol),
            "lambda_initial": float(args.lambda_initial),
            "lambda_factor": float(args.lambda_factor),
            "lambda_upper_bound": float(args.lambda_upper_bound),
            "lambda_lower_bound": float(args.lambda_lower_bound),
            "min_model_fidelity": float(args.min_model_fidelity),
            "diagonal_damping": bool(args.diagonal_damping),
            "use_fixed_lambda_factor": bool(args.use_fixed_lambda_factor),
            "min_diagonal": float(args.min_diagonal),
            "max_diagonal": float(args.max_diagonal),
            "ridge": float(args.ridge),
        },
        "initial_objective": float(nonlinear_objective_g2o(problem, problem.init_poses)),
        "levenberg_marquardt": lm,
    }
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_csv(lm["history"], history_csv)
    save_csv(lm["attempt_rows"], attempts_csv)
    np.savez_compressed(
        traj_path,
        initial_poses=np.asarray(problem.init_poses, dtype=float),
        final_poses=final_poses,
        pose_history=np.asarray(lm["pose_history"], dtype=float),
    )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "history_csv": str(history_csv),
                "attempts_csv": str(attempts_csv),
                "trajectories": str(traj_path),
                "plot": str(plot_path),
                "initial_objective": out["initial_objective"],
                "termination_reason": lm["termination_reason"],
                "final_row": lm["history"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
