from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import build_linearized_local_graph_g2o
from svd_abstraction.g2o_se2 import linearize_g2o_problem
from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import poses_to_nodes_g2o
from svd_abstraction.g2o_se2 import run_direct_newton_g2o
from svd_abstraction.intel_g2o_adaptive_policy import direct_optimum_objective
from svd_abstraction.intel_g2o_adaptive_policy import direct_reference_poses
from svd_abstraction.intel_g2o_adaptive_policy import parse_beta_candidates
from svd_abstraction.intel_g2o_adaptive_policy import run_adaptive_policy_outer_g2o
from svd_abstraction.intel_g2o_adaptive_policy import supported_policy_presets
from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import mean_vector
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices
from svd_abstraction.se2_utils import apply_pose_deltas
from svd_abstraction.se2_utils import stack_pose_errors


G2O_PATH = pathlib.Path("/home/yuzhou/Desktop/input_INTEL_g2o.g2o")
OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def pairwise_pose_gap(poses_a: np.ndarray, poses_b: np.ndarray) -> float:
    errs = stack_pose_errors(np.asarray(poses_a, dtype=float), np.asarray(poses_b, dtype=float))
    if errs.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(errs.reshape(-1, 3), axis=1)))


def exact_local_solve_g2o(problem, base_poses: np.ndarray) -> dict[str, object]:
    A, b = linearize_g2o_problem(problem, base_poses)
    A = A + 1e-10 * sp.eye(A.shape[0], format="csc")
    e_star = spla.spsolve(A, b)
    lin_res = float(np.linalg.norm(A @ e_star - b))
    next_poses = apply_pose_deltas(base_poses, e_star)
    return {
        "A": A,
        "b": b,
        "e_star": e_star,
        "e_norm": float(np.linalg.norm(e_star)),
        "linear_residual_norm": lin_res,
        "next_poses": next_poses,
        "after_objective": float(nonlinear_objective_g2o(problem, next_poses)),
    }


def approx_local_persistent_mg_solve_g2o(
    problem,
    base_poses: np.ndarray,
    inner_cycles: int,
    pre_sweeps: int,
    group_size: int,
    r_reduced: int,
    basis_source: str,
    coarse_damping: float = 1.0,
    coarse_damping_mode: str = "fixed",
    coarse_damping_candidates: tuple[float, ...] = (1.0,),
) -> dict[str, object]:
    exact = exact_local_solve_g2o(problem, base_poses)
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

    inner_history: list[dict[str, float]] = []
    A = exact["A"]
    b = exact["b"]
    e_star = exact["e_star"]

    for cyc in range(1, inner_cycles + 1):
        for _ in range(pre_sweeps):
            residual_graph.synchronous_iteration()

        e_before = mean_vector(residual_graph)
        pre_lin_res = float(np.linalg.norm(A @ e_before - b))
        pre_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_before)))

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_e_raw = level.prolongate(delta_z)
        if coarse_damping_mode == "line_search":
            best_alpha = 0.0
            best_obj = pre_obj
            for alpha in coarse_damping_candidates:
                alpha = float(alpha)
                e_candidate = e_before + alpha * delta_e_raw
                obj_candidate = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_candidate)))
                if obj_candidate < best_obj:
                    best_obj = obj_candidate
                    best_alpha = alpha
            selected_damping = best_alpha
        elif coarse_damping_mode == "fixed":
            selected_damping = float(coarse_damping)
        else:
            raise ValueError(f"Unsupported coarse_damping_mode={coarse_damping_mode!r}")
        delta_e = float(selected_damping) * delta_e_raw
        inject_correction_keep_messages(residual_graph, delta_e)

        e_now = mean_vector(residual_graph)
        post_lin_res = float(np.linalg.norm(A @ e_now - b))
        post_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_now)))
        inner_history.append(
            {
                "inner_cycle": int(cyc),
                "e_rel_to_exact": float(np.linalg.norm(e_now - e_star) / max(np.linalg.norm(e_star), 1e-15)),
                "pre_linear_residual_norm": pre_lin_res,
                "post_linear_residual_norm": post_lin_res,
                "pre_objective": pre_obj,
                "post_objective": post_obj,
                "objective_gap_to_exact_after_cycle": float(post_obj - exact["after_objective"]),
                "e_norm": float(np.linalg.norm(e_now)),
                "pre_e_norm": float(np.linalg.norm(e_before)),
                "delta_z_norm": float(np.linalg.norm(delta_z)),
                "delta_e_raw_norm": float(np.linalg.norm(delta_e_raw)),
                "delta_e_norm": float(np.linalg.norm(delta_e)),
                "coarse_damping": float(selected_damping),
                "coarse_damping_mode": coarse_damping_mode,
            }
        )

    e_hat = mean_vector(residual_graph)
    lin_res = float(np.linalg.norm(A @ e_hat - b))
    next_poses = apply_pose_deltas(base_poses, e_hat)
    exact_next_poses = apply_pose_deltas(base_poses, e_star)
    after_obj = float(nonlinear_objective_g2o(problem, next_poses))
    exact_after_obj = float(nonlinear_objective_g2o(problem, exact_next_poses))
    return {
        "e_star": e_star,
        "e_hat": e_hat,
        "e_star_norm": float(np.linalg.norm(e_star)),
        "e_hat_norm": float(np.linalg.norm(e_hat)),
        "e_rel_to_exact": float(np.linalg.norm(e_hat - e_star) / max(np.linalg.norm(e_star), 1e-15)),
        "linear_residual_exact": float(exact["linear_residual_norm"]),
        "linear_residual_approx": lin_res,
        "next_poses": next_poses,
        "exact_next_poses": exact_next_poses,
        "inner_history": inner_history,
        "num_groups": int(len(groups)),
        "coarse_dim": int(level.total_reduced_dim),
        "next_pose_gap_to_exact": float(pairwise_pose_gap(exact_next_poses, next_poses)),
        "next_objective_gap_to_exact": float(after_obj - exact_after_obj),
        "after_objective": after_obj,
        "exact_after_objective": exact_after_obj,
    }


def run_mg_outer_g2o(
    problem,
    num_outer: int,
    inner_cycles: int,
    pre_sweeps: int,
    group_size: int,
    r_reduced: int,
    basis_source: str,
    coarse_damping: float = 1.0,
    coarse_damping_mode: str = "fixed",
    coarse_damping_candidates: tuple[float, ...] = (1.0,),
    outer_damping: float = 1.0,
    outer_damping_mode: str = "fixed",
    outer_damping_candidates: tuple[float, ...] = (1.0,),
) -> dict[str, object]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    history: list[dict[str, float]] = []
    inner_history: list[dict[str, float]] = []
    obj0 = float(nonlinear_objective_g2o(problem, poses))
    history.append({"outer": 0, "nonlinear_objective": obj0})

    for outer in range(1, num_outer + 1):
        step = approx_local_persistent_mg_solve_g2o(
            problem=problem,
            base_poses=poses,
            inner_cycles=inner_cycles,
            pre_sweeps=pre_sweeps,
            group_size=group_size,
            r_reduced=r_reduced,
            basis_source=basis_source,
            coarse_damping=coarse_damping,
            coarse_damping_mode=coarse_damping_mode,
            coarse_damping_candidates=coarse_damping_candidates,
        )
        if outer_damping_mode == "line_search":
            selected_outer_damping = 0.0
            selected_poses = poses.copy()
            selected_obj = float(nonlinear_objective_g2o(problem, poses))
            for beta in outer_damping_candidates:
                beta = float(beta)
                candidate_poses = apply_pose_deltas(poses, beta * np.asarray(step["e_hat"], dtype=float))
                candidate_obj = float(nonlinear_objective_g2o(problem, candidate_poses))
                if candidate_obj < selected_obj:
                    selected_outer_damping = beta
                    selected_poses = candidate_poses
                    selected_obj = candidate_obj
        elif outer_damping_mode == "fixed":
            selected_outer_damping = float(outer_damping)
            selected_poses = apply_pose_deltas(poses, selected_outer_damping * np.asarray(step["e_hat"], dtype=float))
            selected_obj = float(nonlinear_objective_g2o(problem, selected_poses))
        else:
            raise ValueError(f"Unsupported outer_damping_mode={outer_damping_mode!r}")
        poses = selected_poses
        pose_history.append(np.asarray(poses, dtype=float).copy())
        row = {
            "outer": int(outer),
            "inner_cycles": int(inner_cycles),
            "pre_sweeps": int(pre_sweeps),
            "basis_source": basis_source,
            "coarse_damping": float(coarse_damping),
            "coarse_damping_mode": coarse_damping_mode,
            "outer_damping": float(selected_outer_damping),
            "outer_damping_mode": outer_damping_mode,
            "e_hat_norm": float(step["e_hat_norm"]),
            "e_star_norm": float(step["e_star_norm"]),
            "e_rel_to_exact": float(step["e_rel_to_exact"]),
            "linear_residual_exact": float(step["linear_residual_exact"]),
            "linear_residual_approx": float(step["linear_residual_approx"]),
            "next_pose_gap_to_exact": float(step["next_pose_gap_to_exact"]),
            "next_objective_gap_to_exact": float(step["next_objective_gap_to_exact"]),
            "num_groups": int(step["num_groups"]),
            "coarse_dim": int(step["coarse_dim"]),
            "raw_full_step_objective": float(step["after_objective"]),
            "nonlinear_objective": float(selected_obj),
            "exact_next_objective": float(step["exact_after_objective"]),
        }
        history.append(row)
        for inner_row in step["inner_history"]:
            saved_inner_row = dict(inner_row)
            saved_inner_row["outer"] = int(outer)
            saved_inner_row["inner_cycles"] = int(inner_cycles)
            saved_inner_row["pre_sweeps"] = int(pre_sweeps)
            saved_inner_row["basis_source"] = basis_source
            inner_history.append(saved_inner_row)
        print(
            f"outer={outer:02d} obj={row['nonlinear_objective']:.6f} "
            f"local_gap={row['next_objective_gap_to_exact']:.6f} "
            f"lin_res={row['linear_residual_approx']:.6f} e_rel={row['e_rel_to_exact']:.6f}",
            flush=True,
        )

    return {
        "config": {
            "num_outer": int(num_outer),
            "inner_cycles": int(inner_cycles),
            "pre_sweeps": int(pre_sweeps),
            "group_size": int(group_size),
            "r_reduced": int(r_reduced),
            "basis_source": basis_source,
            "coarse_damping": float(coarse_damping),
            "coarse_damping_mode": coarse_damping_mode,
            "coarse_damping_candidates": [float(x) for x in coarse_damping_candidates],
            "outer_damping": float(outer_damping),
            "outer_damping_mode": outer_damping_mode,
            "outer_damping_candidates": [float(x) for x in outer_damping_candidates],
        },
        "history": history,
        "inner_history": inner_history,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
    }


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


def plot_initial_direct_mg(problem, direct_poses: np.ndarray, mg_poses: np.ndarray, out_path: pathlib.Path) -> pathlib.Path:
    init = np.asarray(problem.init_poses, dtype=float)
    direct = np.asarray(direct_poses, dtype=float)
    mg = np.asarray(mg_poses, dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)
    for edge in problem.edges:
        if edge.kind != "loop":
            continue
        pi = init[edge.i, :2]
        pj = init[edge.j, :2]
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="#8a8a8a", alpha=0.25, linewidth=0.30, zorder=1)
    ax.plot(init[:, 0], init[:, 1], color="#8f8f8f", linewidth=1.0, linestyle=(0, (10, 3, 2, 3)), label="Initial", zorder=2)
    ax.plot(direct[:, 0], direct[:, 1], color="#1f77b4", linewidth=1.25, label="Direct Newton", zorder=3)
    ax.plot(mg[:, 0], mg[:, 1], color="#d62728", linewidth=1.05, linestyle=(0, (4, 2)), label="Persistent residual MG", zorder=4)
    ax.scatter([init[0, 0]], [init[0, 1]], color="black", s=18, label="Anchor/start", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("INTEL g2o: Initial vs Direct vs Persistent Residual MG")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--schedule-mode", choices=("fixed", "adaptive_policy"), default="fixed")
    parser.add_argument("--inner-cycles", type=int, default=2)
    parser.add_argument("--pre-sweeps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--basis-source", type=str, default="joint_information")
    parser.add_argument("--coarse-damping", type=float, default=1.0)
    parser.add_argument("--coarse-damping-mode", type=str, default="fixed", choices=("fixed", "line_search"))
    parser.add_argument("--coarse-damping-candidates", type=str, default="1,0.8,0.5,0.25,0.1,0")
    parser.add_argument("--outer-damping", type=float, default=1.0)
    parser.add_argument("--outer-damping-mode", type=str, default="fixed", choices=("fixed", "line_search"))
    parser.add_argument("--outer-damping-candidates", type=str, default="1,0.8,0.5,0.25,0.1,0")
    parser.add_argument("--adaptive-policy", choices=supported_policy_presets(), default="formal_v1")
    parser.add_argument(
        "--adaptive-beta-candidates",
        type=str,
        default="1,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001",
    )
    args = parser.parse_args()
    coarse_damping_candidates = tuple(float(x) for x in args.coarse_damping_candidates.split(",") if x.strip())
    outer_damping_candidates = tuple(float(x) for x in args.outer_damping_candidates.split(",") if x.strip())

    problem = parse_g2o_se2(G2O_PATH)
    direct = run_direct_newton_g2o(problem, num_outer=args.num_outer, rel_obj_tol=1e-12, step_tol=1e-10)
    if args.schedule_mode == "fixed":
        mg = run_mg_outer_g2o(
            problem=problem,
            num_outer=args.num_outer,
            inner_cycles=args.inner_cycles,
            pre_sweeps=args.pre_sweeps,
            group_size=args.group_size,
            r_reduced=args.r_reduced,
            basis_source=args.basis_source,
            coarse_damping=args.coarse_damping,
            coarse_damping_mode=args.coarse_damping_mode,
            coarse_damping_candidates=coarse_damping_candidates,
            outer_damping=args.outer_damping,
            outer_damping_mode=args.outer_damping_mode,
            outer_damping_candidates=outer_damping_candidates,
        )
    else:
        mg = run_adaptive_policy_outer_g2o(
            problem=problem,
            num_outer=args.num_outer,
            basis_source=args.basis_source,
            group_size=args.group_size,
            r_reduced=args.r_reduced,
            preset=args.adaptive_policy,
            beta_candidates=parse_beta_candidates(args.adaptive_beta_candidates),
            ref_poses=direct_reference_poses(),
            direct_optimum=direct_optimum_objective(),
        )

    if args.schedule_mode == "fixed":
        stem = (
            f"intel_g2o_direct_vs_persistent_mg_outer{args.num_outer}"
            f"_c{args.inner_cycles}_k{args.pre_sweeps}_{args.basis_source}"
        )
        if float(args.coarse_damping) != 1.0:
            stem += f"_damp{args.coarse_damping:g}"
        if args.coarse_damping_mode != "fixed":
            stem += f"_{args.coarse_damping_mode}"
        if float(args.outer_damping) != 1.0:
            stem += f"_outerdamp{args.outer_damping:g}"
        if args.outer_damping_mode != "fixed":
            stem += f"_outer_{args.outer_damping_mode}"
    else:
        stem = (
            f"intel_g2o_direct_vs_persistent_mg_outer{args.num_outer}"
            f"_adaptive_{args.adaptive_policy}_{args.basis_source}"
        )
    json_path = RESULT_DIR / f"{stem}.json"
    direct_csv = RESULT_DIR / f"{stem}_direct.csv"
    mg_csv = RESULT_DIR / f"{stem}_mg.csv"
    detail_csv = RESULT_DIR / f"{stem}_{'inner' if args.schedule_mode == 'fixed' else 'attempts'}.csv"
    trajectories_path = RESULT_DIR / f"{stem}_trajectories.npz"
    plot_path = OUT_DIR / f"{stem}.png"

    out = {
        "config": {
            "path": str(G2O_PATH),
            "num_outer": int(args.num_outer),
            "schedule_mode": args.schedule_mode,
            "group_size": int(args.group_size),
            "r_reduced": int(args.r_reduced),
            "basis_source": args.basis_source,
        },
        "initial_objective": float(nonlinear_objective_g2o(problem, problem.init_poses)),
        "direct_newton": direct,
        "persistent_residual_mg": mg,
    }
    if args.schedule_mode == "fixed":
        out["config"].update(
            {
                "inner_cycles": int(args.inner_cycles),
                "pre_sweeps": int(args.pre_sweeps),
                "coarse_damping": float(args.coarse_damping),
                "coarse_damping_mode": args.coarse_damping_mode,
                "coarse_damping_candidates": [float(x) for x in coarse_damping_candidates],
                "outer_damping": float(args.outer_damping),
                "outer_damping_mode": args.outer_damping_mode,
                "outer_damping_candidates": [float(x) for x in outer_damping_candidates],
            }
        )
    else:
        out["config"].update(
            {
                "adaptive_policy": args.adaptive_policy,
                "adaptive_beta_candidates": parse_beta_candidates(args.adaptive_beta_candidates),
            }
        )
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_csv(direct["history"], direct_csv)
    save_csv(mg["history"], mg_csv)
    save_csv(mg["inner_history"] if args.schedule_mode == "fixed" else mg["attempt_rows"], detail_csv)
    np.savez_compressed(
        trajectories_path,
        initial_poses=np.asarray(problem.init_poses, dtype=float),
        direct_pose_history=np.asarray(direct["pose_history"], dtype=float),
        mg_pose_history=np.asarray(mg["pose_history"], dtype=float),
        direct_final_poses=np.asarray(direct["final_poses"], dtype=float),
        mg_final_poses=np.asarray(mg["final_poses"], dtype=float),
    )
    plot_initial_direct_mg(problem, np.asarray(direct["final_poses"], dtype=float), np.asarray(mg["final_poses"], dtype=float), plot_path)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "direct_csv": str(direct_csv),
                "mg_csv": str(mg_csv),
                "detail_csv": str(detail_csv),
                "trajectories": str(trajectories_path),
                "plot": str(plot_path),
                "direct_final": direct["history"][-1],
                "mg_final": mg["history"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
