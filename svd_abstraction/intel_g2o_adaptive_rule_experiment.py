from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.intel_g2o_adaptive_policy import build_default_problem
from svd_abstraction.intel_g2o_adaptive_policy import parse_beta_candidates
from svd_abstraction.intel_g2o_adaptive_policy import run_adaptive_policy_outer_g2o
from svd_abstraction.intel_g2o_adaptive_policy import supported_policy_presets
from svd_abstraction.intel_g2o_persistent_residual_mg import RESULT_DIR
from svd_abstraction.intel_g2o_persistent_residual_mg import save_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--basis-source", type=str, default="message_conditioned_information")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--preset", choices=supported_policy_presets(), default="formal_v1")
    parser.add_argument(
        "--beta-candidates",
        type=str,
        default="1,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001",
    )
    args = parser.parse_args()

    problem = build_default_problem()
    result = run_adaptive_policy_outer_g2o(
        problem=problem,
        num_outer=args.num_outer,
        basis_source=args.basis_source,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        preset=args.preset,
        beta_candidates=parse_beta_candidates(args.beta_candidates),
    )
    stem = f"intel_g2o_adaptive_rule_{args.preset}_{args.basis_source}_outer{args.num_outer}"
    json_path = RESULT_DIR / f"{stem}.json"
    outer_csv = RESULT_DIR / f"{stem}_outer.csv"
    attempt_csv = RESULT_DIR / f"{stem}_attempts.csv"
    traj_path = RESULT_DIR / f"{stem}_trajectories.npz"
    json_path.write_text(json.dumps(result["config"], indent=2), encoding="utf-8")
    save_csv(result["history"], outer_csv)
    save_csv(result["attempt_rows"], attempt_csv)
    np.savez(
        traj_path,
        pose_history=np.asarray(result["pose_history"], dtype=float),
        final_poses=np.asarray(result["final_poses"], dtype=float),
    )
    print(json.dumps({"json": str(json_path), "outer_csv": str(outer_csv), "attempt_csv": str(attempt_csv), "traj": str(traj_path)}, indent=2))


if __name__ == "__main__":
    main()
