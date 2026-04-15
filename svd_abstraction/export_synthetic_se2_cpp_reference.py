from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.se2_newton_vs_persistent_mg_experiment import approx_local_persistent_mg_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import exact_local_solve
from svd_abstraction.se2_utils import build_se2_problem
from svd_abstraction.se2_utils import nonlinear_objective


def write_problem(problem, path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("SYNTHETIC_SE2_PROBLEM\n")
        fh.write(f"N {problem.gt_poses.shape[0]}\n")
        fh.write("POSES\n")
        for idx, (gt_pose, init_pose) in enumerate(zip(problem.gt_poses, problem.init_poses)):
            fh.write(
                f"{idx} "
                f"{gt_pose[0]:.17g} {gt_pose[1]:.17g} {gt_pose[2]:.17g} "
                f"{init_pose[0]:.17g} {init_pose[1]:.17g} {init_pose[2]:.17g}\n"
            )
        anchor = np.asarray(problem.anchor_pose, dtype=float).reshape(3)
        anchor_info = np.asarray(problem.anchor_information, dtype=float).reshape(3, 3)
        fh.write(
            "ANCHOR "
            f"{anchor[0]:.17g} {anchor[1]:.17g} {anchor[2]:.17g} "
            + " ".join(f"{anchor_info[r, c]:.17g}" for r in range(3) for c in range(3))
            + "\n"
        )
        fh.write(f"EDGES {len(problem.edges)}\n")
        for edge in problem.edges:
            meas = np.asarray(edge.measurement, dtype=float).reshape(3)
            info = np.asarray(edge.information, dtype=float).reshape(3, 3)
            fh.write(
                f"{int(edge.i)} {int(edge.j)} {edge.kind} "
                f"{meas[0]:.17g} {meas[1]:.17g} {meas[2]:.17g} "
                + " ".join(f"{info[r, c]:.17g}" for r in range(3) for c in range(3))
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-path", type=pathlib.Path, required=True)
    parser.add_argument("--reference-path", type=pathlib.Path, required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-outer", type=int, default=3)
    parser.add_argument("--inner-cycles", type=int, default=3)
    parser.add_argument("--pre-sweeps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    args = parser.parse_args()

    problem = build_se2_problem(
        n=args.n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        seed=args.seed,
    )

    args.problem_path.parent.mkdir(parents=True, exist_ok=True)
    args.reference_path.parent.mkdir(parents=True, exist_ok=True)
    write_problem(problem, args.problem_path)

    direct_history = [
        {
            "outer": 0,
            "nonlinear_objective": float(nonlinear_objective(problem, problem.init_poses)),
            "linear_step_norm": 0.0,
            "linear_residual_norm": 0.0,
        }
    ]
    direct_poses = problem.init_poses.copy()
    for outer in range(1, args.num_outer + 1):
        step = exact_local_solve(problem, direct_poses)
        direct_poses = step["next_poses"]
        direct_history.append(
            {
                "outer": int(outer),
                "nonlinear_objective": float(nonlinear_objective(problem, direct_poses)),
                "linear_step_norm": float(step["e_norm"]),
                "linear_residual_norm": float(step["linear_residual_norm"]),
            }
        )

    mg_history = [
        {
            "outer": 0,
            "nonlinear_objective": float(nonlinear_objective(problem, problem.init_poses)),
            "e_hat_norm": 0.0,
            "e_star_norm": 0.0,
            "e_rel_to_exact": 0.0,
            "linear_residual_exact": 0.0,
            "linear_residual_approx": 0.0,
            "num_groups": 0,
            "coarse_dim": 0,
        }
    ]
    mg_poses = problem.init_poses.copy()
    for outer in range(1, args.num_outer + 1):
        step = approx_local_persistent_mg_solve(
            problem=problem,
            base_poses=mg_poses,
            inner_cycles=args.inner_cycles,
            pre_sweeps=args.pre_sweeps,
            group_size=args.group_size,
            r_reduced=args.r_reduced,
            basis_source="message_conditioned_information",
        )
        mg_poses = step["next_poses"]
        mg_history.append(
            {
                "outer": int(outer),
                "nonlinear_objective": float(nonlinear_objective(problem, mg_poses)),
                "e_hat_norm": float(step["e_hat_norm"]),
                "e_star_norm": float(step["e_star_norm"]),
                "e_rel_to_exact": float(step["e_rel_to_exact"]),
                "linear_residual_exact": float(step["linear_residual_exact"]),
                "linear_residual_approx": float(step["linear_residual_approx"]),
                "num_groups": int(step["num_groups"]),
                "coarse_dim": int(step["coarse_dim"]),
            }
        )

    payload = {
        "config": {
            "num_outer": int(args.num_outer),
            "inner_cycles": int(args.inner_cycles),
            "pre_sweeps": int(args.pre_sweeps),
            "group_size": int(args.group_size),
            "r_reduced": int(args.r_reduced),
            "basis_source": "message_conditioned_information",
        },
        "problem": {
            "num_poses": int(problem.gt_poses.shape[0]),
            "num_edges": int(len(problem.edges)),
        },
        "initial_objective": float(nonlinear_objective(problem, problem.init_poses)),
        "direct_history": direct_history,
        "mg_history": mg_history,
    }
    args.reference_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
