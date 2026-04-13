from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.se2_newton_vs_persistent_mg_experiment import approx_local_persistent_mg_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import basis_suffix
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import exact_local_solve
from svd_abstraction.se2_utils import build_se2_problem


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_direct_outer(problem, num_outer: int) -> np.ndarray:
    poses = np.array(problem.init_poses, copy=True)
    for _ in range(num_outer):
        step = exact_local_solve(problem, poses)
        poses = step["next_poses"]
    return poses


def run_mg_outer(
    problem,
    num_outer: int,
    inner_cycles: int,
    pre_sweeps: int,
    basis_source: str,
) -> np.ndarray:
    poses = np.array(problem.init_poses, copy=True)
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
    return poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-source", type=str, default="joint_covariance")
    parser.add_argument("--inner-cycles", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="")
    args = parser.parse_args()

    problem = build_se2_problem(
        n=64,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        seed=0,
    )

    odom = np.asarray(problem.init_poses, dtype=float)
    direct3 = run_direct_outer(problem, num_outer=3)
    mg_c1_outer3 = run_mg_outer(
        problem,
        num_outer=3,
        inner_cycles=args.inner_cycles,
        pre_sweeps=50,
        basis_source=args.basis_source,
    )

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=180)

    ax.plot(
        odom[:, 0],
        odom[:, 1],
        color="#8f8f8f",
        linewidth=1.2,
        linestyle=(0, (10, 3, 2, 3)),
        label="Initial odometry",
        zorder=1,
    )
    ax.plot(
        direct3[:, 0],
        direct3[:, 1],
        color="#1f77b4",
        linewidth=1.35,
        label="Direct Newton, outer=3",
        zorder=2,
    )
    ax.plot(
        mg_c1_outer3[:, 0],
        mg_c1_outer3[:, 1],
        color="#d62728",
        linewidth=1.1,
        linestyle=(0, (4, 2)),
        label=f"Persistent residual MG, c={args.inner_cycles}, outer=3, {args.basis_source}",
        zorder=3,
    )

    ax.scatter([odom[0, 0]], [odom[0, 1]], color="black", s=28, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SE(2) Geometry After 3 Outer Iterations")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")

    if args.output_path:
        out_path = pathlib.Path(args.output_path)
    else:
        stem = f"se2_outer3_direct_vs_mg_c{args.inner_cycles}_geometry"
        stem += basis_suffix(args.basis_source)
        out_path = OUT_DIR / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()
