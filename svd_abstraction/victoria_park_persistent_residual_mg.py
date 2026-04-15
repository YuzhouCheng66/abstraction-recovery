from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from svd_abstraction.g2o_se2_landmark import apply_pose_landmark_deltas
from svd_abstraction.g2o_se2_landmark import build_linearized_local_graph_g2o_se2_landmark
from svd_abstraction.g2o_se2_landmark import nonlinear_objective_g2o_se2_landmark
from svd_abstraction.g2o_se2_landmark import parse_g2o_se2_landmark
from svd_abstraction.g2o_se2_landmark import run_direct_newton_g2o_se2_landmark
from svd_abstraction.g2o_se2_landmark import summarize_g2o_se2_landmark
from svd_abstraction.g2o_se2_landmark import linearize_g2o_se2_landmark_problem
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import mean_vector
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_G2O_PATH = Path.home() / "Desktop" / "datasets" / "victoria-park-slampp.g2o"
RESULT_DIR = REPO_ROOT / "svd_abstraction" / "output_results" / "victoria_park_persistent_residual_mg"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def pairwise_pose_gap(poses_a: np.ndarray, poses_b: np.ndarray) -> float:
    poses_a = np.asarray(poses_a, dtype=float).reshape(-1, 3)
    poses_b = np.asarray(poses_b, dtype=float).reshape(-1, 3)
    if poses_a.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(poses_a - poses_b, axis=1)))


def pairwise_landmark_gap(landmarks_a: np.ndarray, landmarks_b: np.ndarray) -> float:
    landmarks_a = np.asarray(landmarks_a, dtype=float).reshape(-1, 2)
    landmarks_b = np.asarray(landmarks_b, dtype=float).reshape(-1, 2)
    if landmarks_a.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(landmarks_a - landmarks_b, axis=1)))


def ordered_groups(ids: list[int], group_size: int) -> list[list[int]]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return [ids[start : start + group_size] for start in range(0, len(ids), group_size) if ids[start : start + group_size]]


def spatial_landmark_groups(problem, landmarks: np.ndarray, group_size: int) -> list[list[int]]:
    if group_size <= 0:
        raise ValueError("landmark group_size must be positive")
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    n_landmarks = int(landmarks.shape[0])
    if n_landmarks == 0:
        return []

    pose_offset = len(problem.pose_ids)
    unassigned = set(range(n_landmarks))
    groups: list[list[int]] = []

    while unassigned:
        seed = min(unassigned)
        members = [seed]
        unassigned.remove(seed)
        while len(members) < group_size and unassigned:
            centroid = landmarks[np.array(members, dtype=int)].mean(axis=0)
            candidates = np.array(sorted(unassigned), dtype=int)
            dists = np.linalg.norm(landmarks[candidates] - centroid[None, :], axis=1)
            chosen = int(candidates[int(np.argmin(dists))])
            members.append(chosen)
            unassigned.remove(chosen)
        groups.append([pose_offset + idx for idx in members])
    return groups


def build_pose_landmark_groups(
    problem,
    poses: np.ndarray,
    landmarks: np.ndarray,
    pose_group_size: int,
    landmark_group_size: int,
) -> tuple[list[list[int]], int, int]:
    pose_groups = ordered_groups(list(range(len(problem.pose_ids))), pose_group_size)
    landmark_groups = spatial_landmark_groups(problem, landmarks, landmark_group_size)
    return pose_groups + landmark_groups, len(pose_groups), len(landmark_groups)


def exact_local_solve_g2o_landmark(problem, base_poses: np.ndarray, base_landmarks: np.ndarray) -> dict[str, object]:
    A, b = linearize_g2o_se2_landmark_problem(problem, base_poses, base_landmarks)
    A = A + 1e-10 * sp.eye(A.shape[0], format="csc")
    e_star = spla.spsolve(A, b)
    e_star = np.asarray(e_star, dtype=float).reshape(-1)
    lin_res = float(np.linalg.norm(A @ e_star - b))
    next_poses, next_landmarks = apply_pose_landmark_deltas(problem, base_poses, base_landmarks, e_star)
    return {
        "A": A,
        "b": b,
        "e_star": e_star,
        "e_norm": float(np.linalg.norm(e_star)),
        "linear_residual_norm": lin_res,
        "next_poses": next_poses,
        "next_landmarks": next_landmarks,
        "after_objective": float(nonlinear_objective_g2o_se2_landmark(problem, next_poses, next_landmarks)),
    }


def approx_local_persistent_mg_solve_g2o_landmark(
    problem,
    base_poses: np.ndarray,
    base_landmarks: np.ndarray,
    inner_cycles: int,
    pre_sweeps: int,
    pose_group_size: int,
    landmark_group_size: int,
    r_reduced: int,
    basis_source: str,
    coarse_damping: float = 1.0,
    coarse_damping_mode: str = "fixed",
    coarse_damping_candidates: tuple[float, ...] = (1.0,),
) -> dict[str, object]:
    exact = exact_local_solve_g2o_landmark(problem, base_poses, base_landmarks)
    template_graph = build_linearized_local_graph_g2o_se2_landmark(problem, base_poses, base_landmarks)
    residual_graph = build_linearized_local_graph_g2o_se2_landmark(problem, base_poses, base_landmarks)
    zero = np.zeros(sum(var.dofs for var in template_graph.var_nodes[: template_graph.n_var_nodes]), dtype=float)
    reset_residual_graph(residual_graph, template_graph, zero, var_slices(template_graph))

    groups, num_pose_groups, num_landmark_groups = build_pose_landmark_groups(
        problem=problem,
        poses=base_poses,
        landmarks=base_landmarks,
        pose_group_size=pose_group_size,
        landmark_group_size=landmark_group_size,
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
        pre_poses, pre_landmarks = apply_pose_landmark_deltas(problem, base_poses, base_landmarks, e_before)
        pre_obj = float(nonlinear_objective_g2o_se2_landmark(problem, pre_poses, pre_landmarks))

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_e_raw = level.prolongate(delta_z)
        if coarse_damping_mode == "line_search":
            best_alpha = 0.0
            best_obj = pre_obj
            for alpha in coarse_damping_candidates:
                alpha = float(alpha)
                e_candidate = e_before + alpha * delta_e_raw
                poses_candidate, landmarks_candidate = apply_pose_landmark_deltas(
                    problem, base_poses, base_landmarks, e_candidate
                )
                obj_candidate = float(
                    nonlinear_objective_g2o_se2_landmark(problem, poses_candidate, landmarks_candidate)
                )
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
        post_poses, post_landmarks = apply_pose_landmark_deltas(problem, base_poses, base_landmarks, e_now)
        post_obj = float(nonlinear_objective_g2o_se2_landmark(problem, post_poses, post_landmarks))
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
    next_poses, next_landmarks = apply_pose_landmark_deltas(problem, base_poses, base_landmarks, e_hat)
    exact_next_poses = np.asarray(exact["next_poses"], dtype=float)
    exact_next_landmarks = np.asarray(exact["next_landmarks"], dtype=float)
    after_obj = float(nonlinear_objective_g2o_se2_landmark(problem, next_poses, next_landmarks))
    exact_after_obj = float(nonlinear_objective_g2o_se2_landmark(problem, exact_next_poses, exact_next_landmarks))
    return {
        "e_star": e_star,
        "e_hat": e_hat,
        "e_star_norm": float(np.linalg.norm(e_star)),
        "e_hat_norm": float(np.linalg.norm(e_hat)),
        "e_rel_to_exact": float(np.linalg.norm(e_hat - e_star) / max(np.linalg.norm(e_star), 1e-15)),
        "linear_residual_exact": float(exact["linear_residual_norm"]),
        "linear_residual_approx": lin_res,
        "next_poses": next_poses,
        "next_landmarks": next_landmarks,
        "exact_next_poses": exact_next_poses,
        "exact_next_landmarks": exact_next_landmarks,
        "inner_history": inner_history,
        "num_groups": int(len(groups)),
        "num_pose_groups": int(num_pose_groups),
        "num_landmark_groups": int(num_landmark_groups),
        "coarse_dim": int(level.total_reduced_dim),
        "next_pose_gap_to_exact": float(pairwise_pose_gap(exact_next_poses, next_poses)),
        "next_landmark_gap_to_exact": float(pairwise_landmark_gap(exact_next_landmarks, next_landmarks)),
        "next_objective_gap_to_exact": float(after_obj - exact_after_obj),
        "after_objective": after_obj,
        "exact_after_objective": exact_after_obj,
    }


def run_mg_outer_g2o_landmark(
    problem,
    num_outer: int,
    inner_cycles: int,
    pre_sweeps: int,
    pose_group_size: int,
    landmark_group_size: int,
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
    landmarks = np.asarray(problem.init_landmarks, dtype=float).copy()
    pose_history = [poses.copy()]
    landmark_history = [landmarks.copy()]
    history: list[dict[str, float]] = []
    inner_history: list[dict[str, float]] = []
    obj0 = float(nonlinear_objective_g2o_se2_landmark(problem, poses, landmarks))
    history.append({"outer": 0, "nonlinear_objective": obj0})

    for outer in range(1, num_outer + 1):
        step = approx_local_persistent_mg_solve_g2o_landmark(
            problem=problem,
            base_poses=poses,
            base_landmarks=landmarks,
            inner_cycles=inner_cycles,
            pre_sweeps=pre_sweeps,
            pose_group_size=pose_group_size,
            landmark_group_size=landmark_group_size,
            r_reduced=r_reduced,
            basis_source=basis_source,
            coarse_damping=coarse_damping,
            coarse_damping_mode=coarse_damping_mode,
            coarse_damping_candidates=coarse_damping_candidates,
        )
        if outer_damping_mode == "line_search":
            selected_outer_damping = 0.0
            selected_poses = poses.copy()
            selected_landmarks = landmarks.copy()
            selected_obj = float(nonlinear_objective_g2o_se2_landmark(problem, poses, landmarks))
            for beta in outer_damping_candidates:
                beta = float(beta)
                candidate_poses, candidate_landmarks = apply_pose_landmark_deltas(
                    problem, poses, landmarks, beta * np.asarray(step["e_hat"], dtype=float)
                )
                candidate_obj = float(
                    nonlinear_objective_g2o_se2_landmark(problem, candidate_poses, candidate_landmarks)
                )
                if candidate_obj < selected_obj:
                    selected_outer_damping = beta
                    selected_poses = candidate_poses
                    selected_landmarks = candidate_landmarks
                    selected_obj = candidate_obj
        elif outer_damping_mode == "fixed":
            selected_outer_damping = float(outer_damping)
            selected_poses, selected_landmarks = apply_pose_landmark_deltas(
                problem, poses, landmarks, selected_outer_damping * np.asarray(step["e_hat"], dtype=float)
            )
            selected_obj = float(nonlinear_objective_g2o_se2_landmark(problem, selected_poses, selected_landmarks))
        else:
            raise ValueError(f"Unsupported outer_damping_mode={outer_damping_mode!r}")

        poses = np.asarray(selected_poses, dtype=float)
        landmarks = np.asarray(selected_landmarks, dtype=float)
        pose_history.append(poses.copy())
        landmark_history.append(landmarks.copy())

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
            "next_landmark_gap_to_exact": float(step["next_landmark_gap_to_exact"]),
            "next_objective_gap_to_exact": float(step["next_objective_gap_to_exact"]),
            "num_groups": int(step["num_groups"]),
            "num_pose_groups": int(step["num_pose_groups"]),
            "num_landmark_groups": int(step["num_landmark_groups"]),
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
            "outer"
            f" {outer:02d}"
            f" obj={selected_obj:.6f}"
            f" e_rel={row['e_rel_to_exact']:.6f}"
            f" pose_gap={row['next_pose_gap_to_exact']:.6f}"
            f" landmark_gap={row['next_landmark_gap_to_exact']:.6f}"
        )

    return {
        "config": {
            "num_outer": int(num_outer),
            "inner_cycles": int(inner_cycles),
            "pre_sweeps": int(pre_sweeps),
            "pose_group_size": int(pose_group_size),
            "landmark_group_size": int(landmark_group_size),
            "r_reduced": int(r_reduced),
            "basis_source": basis_source,
            "coarse_damping": float(coarse_damping),
            "coarse_damping_mode": coarse_damping_mode,
            "outer_damping": float(outer_damping),
            "outer_damping_mode": outer_damping_mode,
            "coarse_damping_candidates": [float(x) for x in coarse_damping_candidates],
            "outer_damping_candidates": [float(x) for x in outer_damping_candidates],
        },
        "history": history,
        "inner_history": inner_history,
        "pose_history": [arr.tolist() for arr in pose_history],
        "landmark_history": [arr.tolist() for arr in landmark_history],
        "final_poses": poses.tolist(),
        "final_landmarks": landmarks.tolist(),
    }


def plot_initial_direct_mg_pose_landmark(
    problem,
    direct_poses: np.ndarray,
    direct_landmarks: np.ndarray,
    mg_poses: np.ndarray,
    mg_landmarks: np.ndarray,
    out_path: Path,
    title: str,
) -> Path:
    init_poses = np.asarray(problem.init_poses, dtype=float)
    init_landmarks = np.asarray(problem.init_landmarks, dtype=float)
    direct_poses = np.asarray(direct_poses, dtype=float)
    direct_landmarks = np.asarray(direct_landmarks, dtype=float)
    mg_poses = np.asarray(mg_poses, dtype=float)
    mg_landmarks = np.asarray(mg_landmarks, dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 8.8), dpi=180)
    ax.plot(init_poses[:, 0], init_poses[:, 1], color="#b0b0b0", linewidth=0.9, linestyle="--", label="Initial poses")
    ax.plot(direct_poses[:, 0], direct_poses[:, 1], color="#1f77b4", linewidth=1.0, label="Direct Newton poses")
    ax.plot(mg_poses[:, 0], mg_poses[:, 1], color="#d95f02", linewidth=1.0, label="MG-SVD poses")
    ax.scatter(init_landmarks[:, 0], init_landmarks[:, 1], color="#d0d0d0", s=8, alpha=0.45, label="Initial landmarks")
    ax.scatter(direct_landmarks[:, 0], direct_landmarks[:, 1], color="#4daf4a", s=10, alpha=0.75, label="Direct Newton landmarks")
    ax.scatter(mg_landmarks[:, 0], mg_landmarks[:, 1], color="#e41a1c", s=10, alpha=0.75, label="MG-SVD landmarks")
    ax.scatter([init_poses[0, 0]], [init_poses[0, 1]], color="black", s=20, label="Anchor/start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_objective_curves(
    mg_history: list[dict[str, float]],
    direct_history: list[dict[str, float]] | None,
    out_path: Path,
    title: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=180)

    mg_outer = [int(row["outer"]) for row in mg_history]
    mg_obj = [float(row["nonlinear_objective"]) for row in mg_history]
    ax.plot(mg_outer, mg_obj, marker="o", markersize=3.0, linewidth=1.2, color="#d95f02", label="MG-SVD")

    if direct_history is not None:
        direct_outer = [int(row["outer"]) for row in direct_history]
        direct_obj = [float(row["nonlinear_objective"]) for row in direct_history]
        ax.plot(
            direct_outer,
            direct_obj,
            marker="s",
            markersize=2.8,
            linewidth=1.1,
            color="#1f77b4",
            label="Direct Newton",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Nonlinear objective")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent residual MG-SVD on Victoria Park pose+landmark g2o.")
    parser.add_argument("--g2o-path", type=str, default=str(DEFAULT_G2O_PATH))
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--inner-cycles", type=int, default=3)
    parser.add_argument("--pre-sweeps", type=int, default=100)
    parser.add_argument("--pose-group-size", type=int, default=20)
    parser.add_argument("--landmark-group-size", type=int, default=4)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--basis-source", type=str, default="message_conditioned_information")
    parser.add_argument("--coarse-damping", type=float, default=1.0)
    parser.add_argument("--outer-damping", type=float, default=1.0)
    parser.add_argument("--tag", type=str, default="victoria_slampp_c3_k100_r4_conditioned_outer20")
    parser.add_argument("--skip-direct", action="store_true")
    args = parser.parse_args()

    g2o_path = Path(args.g2o_path).resolve()
    problem = parse_g2o_se2_landmark(g2o_path)
    mg = run_mg_outer_g2o_landmark(
        problem=problem,
        num_outer=args.num_outer,
        inner_cycles=args.inner_cycles,
        pre_sweeps=args.pre_sweeps,
        pose_group_size=args.pose_group_size,
        landmark_group_size=args.landmark_group_size,
        r_reduced=args.r_reduced,
        basis_source=args.basis_source,
        coarse_damping=args.coarse_damping,
        outer_damping=args.outer_damping,
    )

    direct = None
    if not args.skip_direct:
        direct = run_direct_newton_g2o_se2_landmark(problem, num_outer=args.num_outer, ridge=1e-9)

    final_mg_poses = np.asarray(mg["final_poses"], dtype=float)
    final_mg_landmarks = np.asarray(mg["final_landmarks"], dtype=float)
    if direct is not None:
        final_direct_poses = np.asarray(direct["final_poses"], dtype=float)
        final_direct_landmarks = np.asarray(direct["final_landmarks"], dtype=float)
    else:
        final_direct_poses = final_mg_poses.copy()
        final_direct_landmarks = final_mg_landmarks.copy()

    plot_path = plot_initial_direct_mg_pose_landmark(
        problem=problem,
        direct_poses=final_direct_poses,
        direct_landmarks=final_direct_landmarks,
        mg_poses=final_mg_poses,
        mg_landmarks=final_mg_landmarks,
        out_path=RESULT_DIR / f"{args.tag}.png",
        title=f"Victoria Park Persistent MG-SVD ({args.tag})",
    )
    objective_plot_path = plot_objective_curves(
        mg_history=mg["history"],
        direct_history=None if direct is None else direct["history"],
        out_path=RESULT_DIR / f"{args.tag}_objective_curve.png",
        title=f"Victoria Park Outer Objective Curve ({args.tag})",
    )

    payload = {
        "problem_summary": summarize_g2o_se2_landmark(problem),
        "g2o_path": str(g2o_path),
        "persistent_residual_mg": mg,
        "direct_newton": direct,
        "plot_path": str(plot_path),
        "objective_plot_path": str(objective_plot_path),
    }
    json_path = RESULT_DIR / f"{args.tag}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"json_path={json_path}")
    print(f"plot_path={plot_path}")
    print(f"objective_plot_path={objective_plot_path}")
    print(f"problem_summary={payload['problem_summary']}")
    print(f"mg_final={mg['history'][-1]}")
    if direct is not None:
        print(f"direct_final={direct['history'][-1]}")


if __name__ == "__main__":
    main()
