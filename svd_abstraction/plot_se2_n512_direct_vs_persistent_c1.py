from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.se2_newton_vs_persistent_mg_experiment import approx_local_persistent_mg_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import basis_suffix
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import exact_local_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import pose_metrics
from svd_abstraction.se2_utils import build_se2_problem


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
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


def run_direct_outer(problem, num_outer: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    poses = np.array(problem.init_poses, copy=True)
    history: list[dict[str, float]] = [{"outer": 0, **pose_metrics(problem, poses)}]
    for _ in range(num_outer):
        step = exact_local_solve(problem, poses)
        poses = step["next_poses"]
        history.append({"outer": len(history), **pose_metrics(problem, poses)})
    return poses, history


def run_mg_outer(
    problem,
    num_outer: int,
    inner_cycles: int,
    pre_sweeps: int,
    basis_source: str,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    poses = np.array(problem.init_poses, copy=True)
    history: list[dict[str, float]] = [{"outer": 0, **pose_metrics(problem, poses)}]
    for _ in range(num_outer):
        step = approx_local_persistent_mg_solve(
            problem=problem,
            base_poses=poses,
            inner_cycles=inner_cycles,
            pre_sweeps=pre_sweeps,
            group_size=20,
            r_reduced=4,
            basis_source=basis_source,
        )
        poses = step["next_poses"]
        history.append(
            {
                "outer": len(history),
                "inner_cycles": int(inner_cycles),
                "pre_sweeps": int(pre_sweeps),
                "e_hat_norm": float(step["e_hat_norm"]),
                "e_star_norm": float(step["e_star_norm"]),
                "e_rel_to_exact": float(step["e_rel_to_exact"]),
                "linear_residual_exact": float(step["linear_residual_exact"]),
                "linear_residual_approx": float(step["linear_residual_approx"]),
                "next_pose_gap_to_exact": float(step["next_pose_gap_to_exact"]),
                "next_objective_gap_to_exact": float(step["next_objective_gap_to_exact"]),
                "num_groups": int(step["num_groups"]),
                "coarse_dim": int(step["coarse_dim"]),
                **pose_metrics(problem, poses),
            }
        )
    return poses, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--inner-cycles", type=int, default=1)
    parser.add_argument("--pre-sweeps", type=int, default=50)
    parser.add_argument("--outer-cycles", type=int, default=20)
    parser.add_argument("--basis-source", type=str, default="joint_covariance")
    args = parser.parse_args()

    problem = build_se2_problem(
        n=args.n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        seed=0,
    )

    gt = np.asarray(problem.gt_poses, dtype=float)
    odom = np.asarray(problem.init_poses, dtype=float)
    direct_outer, direct_history = run_direct_outer(problem, num_outer=args.outer_cycles)
    mg_outer, mg_history = run_mg_outer(
        problem,
        num_outer=args.outer_cycles,
        inner_cycles=args.inner_cycles,
        pre_sweeps=args.pre_sweeps,
        basis_source=args.basis_source,
    )

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)

    ax.plot(
        gt[:, 0],
        gt[:, 1],
        color="black",
        linewidth=1.6,
        label="GT",
        zorder=1,
    )
    ax.plot(
        odom[:, 0],
        odom[:, 1],
        color="#8f8f8f",
        linewidth=1.0,
        linestyle=(0, (10, 3, 2, 3)),
        label="Initial odometry",
        zorder=2,
    )
    ax.plot(
        direct_outer[:, 0],
        direct_outer[:, 1],
        color="#1f77b4",
        linewidth=1.25,
        label=f"Direct Newton, outer={args.outer_cycles}",
        zorder=3,
    )
    ax.plot(
        mg_outer[:, 0],
        mg_outer[:, 1],
        color="#d62728",
        linewidth=1.05,
        linestyle=(0, (4, 2)),
        label=(
            f"Persistent residual MG, c={args.inner_cycles}, "
            f"outer={args.outer_cycles}, {args.basis_source}"
        ),
        zorder=4,
    )

    ax.scatter([gt[0, 0]], [gt[0, 1]], color="black", s=20, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"SE(2) Trajectories, N={args.n}, c={args.inner_cycles}, k={args.pre_sweeps}")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, loc="best")

    stem = (
        f"se2_n{args.n}_gt_odom_direct{args.outer_cycles}"
        f"_persistent_c{args.inner_cycles}_outer{args.outer_cycles}"
    )
    if args.pre_sweeps != 50:
        stem += f"_k{args.pre_sweeps}"
    stem += basis_suffix(args.basis_source)

    metrics_json = RESULT_DIR / f"{stem}.json"
    direct_csv = RESULT_DIR / f"{stem}_direct.csv"
    mg_csv = RESULT_DIR / f"{stem}_mg.csv"
    traj_npz = RESULT_DIR / f"{stem}_trajectories.npz"

    metrics_json.write_text(
        json.dumps(
            {
                "config": {
                    "n": int(args.n),
                    "outer_cycles": int(args.outer_cycles),
                    "inner_cycles": int(args.inner_cycles),
                    "pre_sweeps": int(args.pre_sweeps),
                    "basis_source": args.basis_source,
                },
                "initial_metrics": pose_metrics(problem, odom),
                "gt_metrics": pose_metrics(problem, gt),
                "direct_final_metrics": pose_metrics(problem, direct_outer),
                "mg_final_metrics": pose_metrics(problem, mg_outer),
                "direct_history": direct_history,
                "mg_history": mg_history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_csv(direct_history, direct_csv)
    save_csv(mg_history, mg_csv)
    np.savez_compressed(
        traj_npz,
        gt=gt,
        odom=odom,
        direct=direct_outer,
        mg=mg_outer,
    )

    out_path = OUT_DIR / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
