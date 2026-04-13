from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.residual_base_gbp_direct_coarse_experiment import mean_vector
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import OUT_DIR
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import exact_local_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import pairwise_pose_gap
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import pose_metrics
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import run_direct_baseline
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import save_csv
from svd_abstraction.se2_utils import apply_pose_deltas
from svd_abstraction.se2_utils import build_linearized_local_graph
from svd_abstraction.se2_utils import build_se2_problem
from svd_abstraction.se2_utils import nonlinear_objective


def approx_local_base_sync_solve(problem, base_poses: np.ndarray, sweeps: int) -> dict[str, object]:
    template_graph = build_linearized_local_graph(problem, base_poses)
    eta, lam = template_graph.joint_distribution_inf_absolute()
    e_star = np.linalg.solve(0.5 * (lam + lam.T), eta)

    residual_graph = build_linearized_local_graph(problem, base_poses)
    zero = np.zeros(template_graph.n_var_nodes * 3, dtype=float)
    reset_residual_graph(residual_graph, template_graph, zero, var_slices(template_graph))

    for _ in range(int(sweeps)):
        residual_graph.synchronous_iteration()

    e_hat = mean_vector(residual_graph)
    lin_res = float(np.linalg.norm(eta - lam @ e_hat))
    next_poses = apply_pose_deltas(base_poses, e_hat)
    exact_next_poses = apply_pose_deltas(base_poses, e_star)
    exact_after = pose_metrics(problem, exact_next_poses)
    approx_after = pose_metrics(problem, next_poses)
    out = {
        "eta": eta,
        "lam": lam,
        "e_star": e_star,
        "e_hat": e_hat,
        "e_star_norm": float(np.linalg.norm(e_star)),
        "e_hat_norm": float(np.linalg.norm(e_hat)),
        "e_rel_to_exact": float(np.linalg.norm(e_hat - e_star) / max(np.linalg.norm(e_star), 1e-15)),
        "linear_residual_exact": float(np.linalg.norm(eta - lam @ e_star)),
        "linear_residual_approx": lin_res,
        "next_poses": next_poses,
        "exact_next_poses": exact_next_poses,
        "next_pose_gap_to_exact": float(pairwise_pose_gap(exact_next_poses, next_poses)),
        "next_objective_gap_to_exact": float(approx_after["nonlinear_objective"] - exact_after["nonlinear_objective"]),
    }
    out.update({f"after_{k}": v for k, v in approx_after.items()})
    return out


def run_base_sync_outer(problem, num_outer: int, sweeps: int) -> dict[str, object]:
    poses = problem.init_poses.copy()
    history: list[dict[str, float]] = []
    history.append({"outer": 0, **pose_metrics(problem, poses)})

    for outer in range(1, num_outer + 1):
        step = approx_local_base_sync_solve(problem=problem, base_poses=poses, sweeps=sweeps)
        poses = step["next_poses"]
        history.append(
            {
                "outer": int(outer),
                "inner_sweeps": int(sweeps),
                "e_hat_norm": float(step["e_hat_norm"]),
                "e_star_norm": float(step["e_star_norm"]),
                "e_rel_to_exact": float(step["e_rel_to_exact"]),
                "linear_residual_exact": float(step["linear_residual_exact"]),
                "linear_residual_approx": float(step["linear_residual_approx"]),
                "next_pose_gap_to_exact": float(step["next_pose_gap_to_exact"]),
                "next_objective_gap_to_exact": float(step["next_objective_gap_to_exact"]),
                **pose_metrics(problem, poses),
            }
        )

    return {
        "config": {"num_outer": int(num_outer), "inner_sweeps": int(sweeps)},
        "history": history,
        "final_poses": poses.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--inner-sweeps", type=int, default=1000)
    args = parser.parse_args()

    problem = build_se2_problem(
        n=args.n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        seed=args.seed,
    )

    direct = run_direct_baseline(problem, num_outer=args.num_outer)
    base_sync = run_base_sync_outer(problem, num_outer=args.num_outer, sweeps=args.inner_sweeps)

    direct_final_poses = problem.init_poses.copy()
    for _ in range(args.num_outer):
        step = exact_local_solve(problem, direct_final_poses)
        direct_final_poses = step["next_poses"]

    out = {
        "config": {
            "n": int(args.n),
            "seed": int(args.seed),
            "num_outer": int(args.num_outer),
            "inner_sweeps": int(args.inner_sweeps),
        },
        "initial_metrics": pose_metrics(problem, problem.init_poses),
        "gt_objective": float(nonlinear_objective(problem, problem.gt_poses)),
        "gt_poses": np.asarray(problem.gt_poses, dtype=float).tolist(),
        "init_poses": np.asarray(problem.init_poses, dtype=float).tolist(),
        "direct_newton": {
            **direct,
            "final_poses": np.asarray(direct_final_poses, dtype=float).tolist(),
        },
        "base_sync_only": base_sync,
    }

    stem = f"se2_newton_vs_base_sync_n{args.n}_seed{args.seed}_outer{args.num_outer}_s{args.inner_sweeps}"
    json_path = OUT_DIR / f"{stem}.json"
    direct_csv = OUT_DIR / f"{stem}_direct.csv"
    base_csv = OUT_DIR / f"{stem}_base_sync.csv"

    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_csv(direct["history"], direct_csv)
    save_csv(base_sync["history"], base_csv)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "direct_csv": str(direct_csv),
                "base_sync_csv": str(base_csv),
                "direct_final": direct["history"][-1],
                "base_sync_final": base_sync["history"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
