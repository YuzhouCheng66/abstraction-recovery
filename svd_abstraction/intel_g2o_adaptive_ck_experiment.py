from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import build_linearized_local_graph_g2o
from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import poses_to_nodes_g2o
from svd_abstraction.g2o_se2 import run_direct_newton_g2o
from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.intel_g2o_persistent_residual_mg import G2O_PATH
from svd_abstraction.intel_g2o_persistent_residual_mg import RESULT_DIR
from svd_abstraction.intel_g2o_persistent_residual_mg import OUT_DIR
from svd_abstraction.intel_g2o_persistent_residual_mg import exact_local_solve_g2o
from svd_abstraction.intel_g2o_persistent_residual_mg import plot_initial_direct_mg
from svd_abstraction.intel_g2o_persistent_residual_mg import save_csv
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import mean_vector
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices
from svd_abstraction.se2_utils import apply_pose_deltas


def _build_level(problem, base_poses, group_size: int, r_reduced: int, basis_source: str):
    template_graph = build_linearized_local_graph_g2o(problem, base_poses)
    residual_graph = build_linearized_local_graph_g2o(problem, base_poses)
    zero = np.zeros(template_graph.n_var_nodes * 3, dtype=float)
    reset_residual_graph(residual_graph, template_graph, zero, var_slices(template_graph))

    groups = group_list(
        nodes=poses_to_nodes_g2o(base_poses),
        graph=template_graph,
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
        base_graph=residual_graph,
        groups=groups,
        r_reduced=r_reduced,
        basis_source=basis_source,
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=False,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)
    return residual_graph, level, groups


def _strict_residual_reduction(cur: float, start: float, target_ratio: float) -> bool:
    if start <= 0.0:
        return True
    if target_ratio >= 1.0:
        return cur < start - 1e-12
    return cur <= target_ratio * start


def run_adaptive_experiment(
    num_outer: int,
    residual_ratio_target: float,
    basis_source: str,
    group_size: int,
    r_reduced: int,
    max_cycles_per_outer: int,
    max_sweeps_per_cycle: int,
) -> dict[str, object]:
    problem = parse_g2o_se2(G2O_PATH)
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    outer_history: list[dict[str, object]] = []
    cycle_history: list[dict[str, object]] = []
    sweep_history: list[dict[str, object]] = []
    pose_history = [poses.copy()]

    initial_obj = float(nonlinear_objective_g2o(problem, poses))
    outer_history.append(
        {
            "outer": 0,
            "objective_start": initial_obj,
            "objective_end": initial_obj,
            "objective_improved": False,
            "accepted": True,
            "c_used": 0,
            "total_sweeps": 0,
            "status": "initial",
        }
    )

    for outer in range(1, int(num_outer) + 1):
        outer_start_obj = float(nonlinear_objective_g2o(problem, poses))
        exact = exact_local_solve_g2o(problem, poses)
        A = exact["A"]
        b = exact["b"]
        e_star = np.asarray(exact["e_star"], dtype=float)
        residual_graph, level, groups = _build_level(
            problem=problem,
            base_poses=poses,
            group_size=group_size,
            r_reduced=r_reduced,
            basis_source=basis_source,
        )

        accepted = False
        accepted_obj = outer_start_obj
        accepted_pose = poses.copy()
        accepted_e = np.zeros_like(e_star)
        total_sweeps = 0
        status = "no_improvement"

        for cyc in range(1, int(max_cycles_per_outer) + 1):
            e_cycle_start = mean_vector(residual_graph)
            cycle_start_res = float(np.linalg.norm(A @ e_cycle_start - b))
            cycle_start_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(poses, e_cycle_start)))

            sweeps = 0
            e_pre = e_cycle_start
            pre_res = cycle_start_res
            while sweeps < int(max_sweeps_per_cycle):
                residual_graph.synchronous_iteration()
                sweeps += 1
                total_sweeps += 1
                e_pre = mean_vector(residual_graph)
                pre_res = float(np.linalg.norm(A @ e_pre - b))
                sweep_history.append(
                    {
                        "outer": int(outer),
                        "cycle": int(cyc),
                        "sweep": int(sweeps),
                        "cycle_start_residual": cycle_start_res,
                        "current_residual": pre_res,
                        "residual_ratio": float(pre_res / max(cycle_start_res, 1e-300)),
                    }
                )
                if _strict_residual_reduction(pre_res, cycle_start_res, residual_ratio_target):
                    break

            level.update_coarse_residual_eta()
            delta_z = level.direct_solve_coarse_graph()
            delta_e = level.prolongate(delta_z)
            inject_correction_keep_messages(residual_graph, delta_e)

            e_post = mean_vector(residual_graph)
            post_res = float(np.linalg.norm(A @ e_post - b))
            post_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(poses, e_post)))

            cycle_history.append(
                {
                    "outer": int(outer),
                    "cycle": int(cyc),
                    "k_used": int(sweeps),
                    "cycle_start_residual": cycle_start_res,
                    "pre_coarse_residual": pre_res,
                    "post_coarse_residual": post_res,
                    "pre_residual_ratio": float(pre_res / max(cycle_start_res, 1e-300)),
                    "post_residual_ratio": float(post_res / max(cycle_start_res, 1e-300)),
                    "cycle_start_objective": cycle_start_obj,
                    "post_cycle_objective": post_obj,
                    "objective_ratio_to_outer_start": float(post_obj / max(outer_start_obj, 1e-300)),
                    "objective_delta_to_outer_start": float(post_obj - outer_start_obj),
                    "delta_z_norm": float(np.linalg.norm(delta_z)),
                    "delta_e_norm": float(np.linalg.norm(delta_e)),
                    "e_post_norm": float(np.linalg.norm(e_post)),
                    "e_rel_to_exact": float(np.linalg.norm(e_post - e_star) / max(np.linalg.norm(e_star), 1e-15)),
                    "num_groups": int(len(groups)),
                    "coarse_dim": int(level.total_reduced_dim),
                }
            )
            print(
                f"outer={outer:02d} cycle={cyc:02d} "
                f"k_used={sweeps} pre_ratio={pre_res / max(cycle_start_res, 1e-300):.6f} "
                f"post_obj={post_obj:.6f} outer_start_obj={outer_start_obj:.6f}",
                flush=True,
            )

            if post_obj < outer_start_obj:
                accepted = True
                accepted_obj = post_obj
                accepted_pose = apply_pose_deltas(poses, e_post)
                accepted_e = e_post.copy()
                status = "objective_decreased"
                break

        poses = accepted_pose.copy()
        pose_history.append(poses.copy())
        outer_history.append(
            {
                "outer": int(outer),
                "objective_start": outer_start_obj,
                "objective_end": accepted_obj,
                "objective_improved": bool(accepted and accepted_obj < outer_start_obj),
                "accepted": bool(accepted),
                "c_used": int(cyc),
                "total_sweeps": int(total_sweeps),
                "accepted_e_norm": float(np.linalg.norm(accepted_e)),
                "status": status,
            }
        )
        print(
            f"outer={outer:02d} obj_start={outer_start_obj:.6f} "
            f"obj_end={accepted_obj:.6f} accepted={accepted} "
            f"c_used={cyc} total_sweeps={total_sweeps}",
            flush=True,
        )
        if not accepted:
            break

    return {
        "config": {
            "num_outer": int(num_outer),
            "residual_ratio_target": float(residual_ratio_target),
            "basis_source": basis_source,
            "group_size": int(group_size),
            "r_reduced": int(r_reduced),
            "max_cycles_per_outer": int(max_cycles_per_outer),
            "max_sweeps_per_cycle": int(max_sweeps_per_cycle),
        },
        "outer_history": outer_history,
        "cycle_history": cycle_history,
        "sweep_history": sweep_history,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--residual-ratio-target", type=float, default=1.0)
    parser.add_argument("--basis-source", type=str, default="message_conditioned_information")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--max-cycles-per-outer", type=int, default=50)
    parser.add_argument("--max-sweeps-per-cycle", type=int, default=1000)
    args = parser.parse_args()

    result = run_adaptive_experiment(
        num_outer=args.num_outer,
        residual_ratio_target=args.residual_ratio_target,
        basis_source=args.basis_source,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        max_cycles_per_outer=args.max_cycles_per_outer,
        max_sweeps_per_cycle=args.max_sweeps_per_cycle,
    )

    problem = parse_g2o_se2(G2O_PATH)
    direct = run_direct_newton_g2o(problem, num_outer=args.num_outer, rel_obj_tol=1e-12, step_tol=1e-10)
    stem = (
        f"intel_g2o_adaptive_ck_outer{args.num_outer}"
        f"_ratio{args.residual_ratio_target:g}"
        f"_{args.basis_source}"
    )
    json_path = RESULT_DIR / f"{stem}.json"
    outer_csv = RESULT_DIR / f"{stem}_outer.csv"
    cycle_csv = RESULT_DIR / f"{stem}_cycle.csv"
    sweep_csv = RESULT_DIR / f"{stem}_sweep.csv"
    trajectories_path = RESULT_DIR / f"{stem}_trajectories.npz"
    plot_path = OUT_DIR / f"{stem}.png"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_csv(result["outer_history"], outer_csv)
    save_csv(result["cycle_history"], cycle_csv)
    save_csv(result["sweep_history"], sweep_csv)
    np.savez_compressed(
        trajectories_path,
        initial_poses=np.asarray(problem.init_poses, dtype=float),
        adaptive_pose_history=np.asarray(result["pose_history"], dtype=float),
        adaptive_final_poses=np.asarray(result["final_poses"], dtype=float),
    )
    plot_initial_direct_mg(
        problem,
        np.asarray(direct["final_poses"], dtype=float),
        np.asarray(result["final_poses"], dtype=float),
        plot_path,
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "outer_csv": str(outer_csv),
                "cycle_csv": str(cycle_csv),
                "sweep_csv": str(sweep_csv),
                "trajectories": str(trajectories_path),
                "plot": str(plot_path),
                "final_outer": result["outer_history"][-1] if result["outer_history"] else {},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
