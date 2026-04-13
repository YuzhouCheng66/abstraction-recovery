from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import build_linearized_local_graph_g2o
from svd_abstraction.g2o_se2 import direct_newton_step_g2o
from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import poses_to_nodes_g2o
from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.intel_g2o_persistent_residual_mg import G2O_PATH
from svd_abstraction.intel_g2o_persistent_residual_mg import RESULT_DIR
from svd_abstraction.intel_g2o_persistent_residual_mg import exact_local_solve_g2o
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.se2_utils import apply_pose_deltas


def save_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        path.write_text("")
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


def direct_newton_pose_history(problem, num_outer: int = 20) -> list[dict[str, object]]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    obj_prev = float(nonlinear_objective_g2o(problem, poses))
    history: list[dict[str, object]] = [
        {
            "outer": 0,
            "poses": poses.copy(),
            "nonlinear_objective": obj_prev,
        }
    ]

    for outer in range(1, int(num_outer) + 1):
        step = direct_newton_step_g2o(problem, poses)
        poses = np.asarray(step["next_poses"], dtype=float)
        obj = float(nonlinear_objective_g2o(problem, poses))
        rel_improve = abs(obj_prev - obj) / max(abs(obj_prev), 1e-15)
        history.append(
            {
                "outer": int(outer),
                "poses": poses.copy(),
                "nonlinear_objective": obj,
                "linear_step_norm": float(step["linear_step_norm"]),
                "linear_residual_norm": float(step["linear_residual_norm"]),
                "relative_objective_improvement": float(rel_improve),
            }
        )
        if rel_improve < 1e-12 or float(step["linear_step_norm"]) < 1e-10:
            break
        obj_prev = obj

    return history


def a_norm(A, v: np.ndarray) -> float:
    q = float(v @ (A @ v))
    return float(np.sqrt(max(q, 0.0)))


def evaluate_r_ability(problem, base_poses: np.ndarray, group_size: int = 20, r_reduced: int = 4) -> dict[str, object]:
    exact = exact_local_solve_g2o(problem, base_poses)
    graph = build_linearized_local_graph_g2o(problem, base_poses)
    groups = group_list(
        nodes=poses_to_nodes_g2o(base_poses),
        graph=graph,
        method="order",
        group_size=group_size,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    level = SVDResidualAbstraction(
        base_graph=graph,
        groups=groups,
        r_reduced=r_reduced,
        basis_source="joint_information",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=False,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)

    e_star = np.asarray(exact["e_star"], dtype=float).reshape(-1)
    A = exact["A"]
    e_norm = max(float(np.linalg.norm(e_star)), 1e-15)
    e_a_norm = max(a_norm(A, e_star), 1e-15)

    z_proj, *_ = np.linalg.lstsq(level.P, e_star, rcond=None)
    e_proj = level.P @ z_proj
    proj_err = e_star - e_proj

    delta_z, coarse_lam, coarse_residual = level.solve_coarse_correction()
    e_coarse = level.prolongate(delta_z)
    coarse_err = e_star - e_coarse

    next_exact = np.asarray(exact["next_poses"], dtype=float)
    next_proj = apply_pose_deltas(base_poses, e_proj)
    next_coarse = apply_pose_deltas(base_poses, e_coarse)

    return {
        "base_objective": float(nonlinear_objective_g2o(problem, base_poses)),
        "exact_next_objective": float(exact["after_objective"]),
        "projection_next_objective": float(nonlinear_objective_g2o(problem, next_proj)),
        "ideal_coarse_next_objective": float(nonlinear_objective_g2o(problem, next_coarse)),
        "projection_next_gap_to_exact": float(nonlinear_objective_g2o(problem, next_proj) - exact["after_objective"]),
        "ideal_coarse_next_gap_to_exact": float(nonlinear_objective_g2o(problem, next_coarse) - exact["after_objective"]),
        "e_star_norm": float(np.linalg.norm(e_star)),
        "projection_rel_error": float(np.linalg.norm(proj_err) / e_norm),
        "ideal_coarse_rel_error": float(np.linalg.norm(coarse_err) / e_norm),
        "projection_a_rel_error": float(a_norm(A, proj_err) / e_a_norm),
        "ideal_coarse_a_rel_error": float(a_norm(A, coarse_err) / e_a_norm),
        "projection_linear_residual": float(np.linalg.norm(A @ e_proj - exact["b"])),
        "ideal_coarse_linear_residual": float(np.linalg.norm(A @ e_coarse - exact["b"])),
        "exact_linear_residual": float(exact["linear_residual_norm"]),
        "coarse_residual_norm": float(np.linalg.norm(coarse_residual)),
        "coarse_dim": int(level.total_reduced_dim),
        "num_groups": int(len(groups)),
        "full_dim": int(level.P.shape[0]),
        "compression_ratio": float(level.total_reduced_dim / max(level.P.shape[0], 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    args = parser.parse_args()

    problem = parse_g2o_se2(G2O_PATH)
    pose_history = direct_newton_pose_history(problem, num_outer=20)
    rows: list[dict[str, object]] = []

    for item in pose_history:
        outer = int(item["outer"])
        poses = np.asarray(item["poses"], dtype=float)
        metrics = evaluate_r_ability(
            problem,
            poses,
            group_size=args.group_size,
            r_reduced=args.r_reduced,
        )
        rows.append(
            {
                "outer": outer,
                "direct_outer_objective": float(item["nonlinear_objective"]),
                **metrics,
            }
        )
        print(
            f"outer={outer:02d} "
            f"proj_rel={metrics['projection_rel_error']:.6f} "
            f"coarse_rel={metrics['ideal_coarse_rel_error']:.6f} "
            f"proj_obj_gap={metrics['projection_next_gap_to_exact']:.6f} "
            f"coarse_obj_gap={metrics['ideal_coarse_next_gap_to_exact']:.6f}",
            flush=True,
        )

    stem = f"intel_g2o_r{int(args.r_reduced)}_optimal_error_ability"
    out_json = RESULT_DIR / f"{stem}.json"
    out_csv = RESULT_DIR / f"{stem}.csv"
    payload = {
        "config": {
            "path": str(G2O_PATH),
            "group_size": int(args.group_size),
            "r_reduced": int(args.r_reduced),
            "basis_source": "joint_information",
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(rows, out_csv)
    print(json.dumps({"json": str(out_json), "csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
