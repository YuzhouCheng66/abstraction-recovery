from __future__ import annotations

import argparse
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
from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.intel_g2o_persistent_residual_mg import G2O_PATH
from svd_abstraction.intel_g2o_persistent_residual_mg import RESULT_DIR
from svd_abstraction.intel_g2o_persistent_residual_mg import exact_local_solve_g2o
from svd_abstraction.intel_g2o_persistent_residual_mg import save_csv
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import mean_vector
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices
from svd_abstraction.se2_utils import apply_pose_deltas


BASELINE_TRAJ = (
    RESULT_DIR / "intel_g2o_direct_vs_persistent_mg_outer40_c3_k100_message_conditioned_information_trajectories.npz"
)
DIRECT_META = RESULT_DIR / "intel_g2o_direct_newton_damping0p1_outer300.json"


def parse_ck_schedule(text: str) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        cyc_text, sweep_text = token.lower().split("x", maxsplit=1)
        schedule.append((int(cyc_text), int(sweep_text)))
    if not schedule:
        raise ValueError("Empty ck schedule.")
    return schedule


def parse_float_list(text: str) -> list[float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Empty float list.")
    return values


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
    return residual_graph, level


def _direct_optimum_objective() -> float:
    return float(json.loads(DIRECT_META.read_text(encoding="utf-8"))["best_objective"])


def baseline_pose_for_actual_outer(actual_outer: int) -> np.ndarray:
    traj = np.load(BASELINE_TRAJ)
    pose_history = np.asarray(traj["mg_pose_history"], dtype=float)
    if actual_outer < 1 or actual_outer > pose_history.shape[0]:
        raise ValueError(f"actual_outer={actual_outer} out of range for pose history length {pose_history.shape[0]}")
    # pose_history[0] is the initial pose, so the start of actual outer t is pose_history[t - 1].
    return np.asarray(pose_history[actual_outer - 1], dtype=float).copy()


def run_candidate(
    problem,
    base_poses: np.ndarray,
    exact: dict[str, object],
    inner_cycles: int,
    pre_sweeps: int,
    group_size: int,
    r_reduced: int,
    basis_source: str,
) -> dict[str, object]:
    residual_graph, level = _build_level(
        problem=problem,
        base_poses=base_poses,
        group_size=group_size,
        r_reduced=r_reduced,
        basis_source=basis_source,
    )
    A = exact["A"]
    b = np.asarray(exact["b"], dtype=float)
    e_star = np.asarray(exact["e_star"], dtype=float)
    cycle_rows: list[dict[str, object]] = []

    for cycle in range(1, int(inner_cycles) + 1):
        e_cycle_start = mean_vector(residual_graph)
        cycle_start_res = float(np.linalg.norm(A @ e_cycle_start - b))
        cycle_start_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_cycle_start)))

        for _ in range(int(pre_sweeps)):
            residual_graph.synchronous_iteration()

        e_pre = mean_vector(residual_graph)
        pre_res = float(np.linalg.norm(A @ e_pre - b))
        pre_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_pre)))

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_e = level.prolongate(delta_z)
        inject_correction_keep_messages(residual_graph, delta_e)

        e_post = mean_vector(residual_graph)
        post_res = float(np.linalg.norm(A @ e_post - b))
        post_obj = float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_post)))

        cycle_rows.append(
            {
                "cycle": int(cycle),
                "cycle_start_linear_residual": cycle_start_res,
                "pre_linear_residual": pre_res,
                "post_linear_residual": post_res,
                "pre_residual_ratio_vs_cycle_start": float(pre_res / max(cycle_start_res, 1e-300)),
                "post_residual_ratio_vs_cycle_start": float(post_res / max(cycle_start_res, 1e-300)),
                "cycle_start_objective": cycle_start_obj,
                "pre_objective": pre_obj,
                "post_objective": post_obj,
                "delta_z_norm": float(np.linalg.norm(delta_z)),
                "delta_e_norm": float(np.linalg.norm(delta_e)),
                "e_post_norm": float(np.linalg.norm(e_post)),
                "e_rel_to_exact": float(np.linalg.norm(e_post - e_star) / max(np.linalg.norm(e_star), 1e-15)),
            }
        )

    e_hat = mean_vector(residual_graph)
    return {
        "e_hat": e_hat,
        "e_hat_norm": float(np.linalg.norm(e_hat)),
        "linear_residual_approx": float(np.linalg.norm(A @ e_hat - b)),
        "e_rel_to_exact": float(np.linalg.norm(e_hat - e_star) / max(np.linalg.norm(e_star), 1e-15)),
        "raw_full_step_objective": float(nonlinear_objective_g2o(problem, apply_pose_deltas(base_poses, e_hat))),
        "exact_after_objective": float(exact["after_objective"]),
        "exact_step_norm": float(np.linalg.norm(e_star)),
        "cycle_rows": cycle_rows,
    }


def best_beta_for_e(problem, base_poses: np.ndarray, e_hat: np.ndarray, beta_candidates: list[float]) -> dict[str, object]:
    best_beta = 0.0
    best_obj = float(nonlinear_objective_g2o(problem, base_poses))
    beta_rows: list[dict[str, object]] = []
    for beta in beta_candidates:
        beta = float(beta)
        candidate_poses = apply_pose_deltas(base_poses, beta * e_hat)
        candidate_obj = float(nonlinear_objective_g2o(problem, candidate_poses))
        beta_rows.append(
            {
                "beta": beta,
                "post_objective": candidate_obj,
            }
        )
        if candidate_obj < best_obj:
            best_beta = beta
            best_obj = candidate_obj
    return {
        "best_beta": float(best_beta),
        "best_objective": float(best_obj),
        "beta_rows": beta_rows,
        "best_poses": apply_pose_deltas(base_poses, best_beta * e_hat) if best_beta > 0.0 else np.asarray(base_poses, dtype=float),
    }


def evaluate_candidates_for_base_pose(
    problem,
    base_poses: np.ndarray,
    actual_outer: int,
    ck_schedule: list[tuple[int, int]],
    beta_candidates: list[float],
    basis_source: str,
    group_size: int,
    r_reduced: int,
    stop_on_first_success: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[int, int], np.ndarray]]:
    exact = exact_local_solve_g2o(problem, base_poses)
    direct_optimum = _direct_optimum_objective()
    start_obj = float(nonlinear_objective_g2o(problem, base_poses))

    all_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    best_pose_map: dict[tuple[int, int], np.ndarray] = {}

    for inner_cycles, pre_sweeps in ck_schedule:
        candidate = run_candidate(
            problem=problem,
            base_poses=base_poses,
            exact=exact,
            inner_cycles=inner_cycles,
            pre_sweeps=pre_sweeps,
            group_size=group_size,
            r_reduced=r_reduced,
            basis_source=basis_source,
        )
        beta_result = best_beta_for_e(
            problem=problem,
            base_poses=base_poses,
            e_hat=np.asarray(candidate["e_hat"], dtype=float),
            beta_candidates=beta_candidates,
        )
        best_pose_map[(int(inner_cycles), int(pre_sweeps))] = np.asarray(beta_result["best_poses"], dtype=float).copy()
        for beta_row in beta_result["beta_rows"]:
            row = {
                "actual_outer": int(actual_outer),
                "start_objective": start_obj,
                "c": int(inner_cycles),
                "k": int(pre_sweeps),
                "ck_cost": int(inner_cycles * pre_sweeps),
                "beta": float(beta_row["beta"]),
                "post_objective": float(beta_row["post_objective"]),
                "objective_delta": float(beta_row["post_objective"] - start_obj),
                "gap_to_direct_optimum": float(beta_row["post_objective"] - direct_optimum),
                "improves": bool(beta_row["post_objective"] < start_obj),
                "raw_full_step_objective": float(candidate["raw_full_step_objective"]),
                "linear_residual_approx": float(candidate["linear_residual_approx"]),
                "e_rel_to_exact": float(candidate["e_rel_to_exact"]),
                "exact_after_objective": float(candidate["exact_after_objective"]),
                "exact_step_norm": float(candidate["exact_step_norm"]),
                "last_cycle_pre_residual_ratio": float(candidate["cycle_rows"][-1]["pre_residual_ratio_vs_cycle_start"]),
                "last_cycle_post_residual_ratio": float(candidate["cycle_rows"][-1]["post_residual_ratio_vs_cycle_start"]),
            }
            all_rows.append(row)

        pair_rows.append(
            {
                "actual_outer": int(actual_outer),
                "start_objective": start_obj,
                "c": int(inner_cycles),
                "k": int(pre_sweeps),
                "ck_cost": int(inner_cycles * pre_sweeps),
                "best_beta": float(beta_result["best_beta"]),
                "best_objective": float(beta_result["best_objective"]),
                "objective_delta": float(beta_result["best_objective"] - start_obj),
                "gap_to_direct_optimum": float(beta_result["best_objective"] - direct_optimum),
                "improves": bool(beta_result["best_objective"] < start_obj),
                "raw_full_step_objective": float(candidate["raw_full_step_objective"]),
                "linear_residual_approx": float(candidate["linear_residual_approx"]),
                "e_rel_to_exact": float(candidate["e_rel_to_exact"]),
                "exact_after_objective": float(candidate["exact_after_objective"]),
                "exact_step_norm": float(candidate["exact_step_norm"]),
                "last_cycle_pre_residual_ratio": float(candidate["cycle_rows"][-1]["pre_residual_ratio_vs_cycle_start"]),
                "last_cycle_post_residual_ratio": float(candidate["cycle_rows"][-1]["post_residual_ratio_vs_cycle_start"]),
            }
        )
        if stop_on_first_success and pair_rows[-1]["improves"]:
            return all_rows, pair_rows, best_pose_map

    return all_rows, pair_rows, best_pose_map


def select_candidate(
    pair_rows: list[dict[str, object]],
    policy: str,
    min_abs_improvement: float,
    beta_accept_threshold: float,
) -> dict[str, object] | None:
    threshold = float(min_abs_improvement)
    improving = [
        row
        for row in pair_rows
        if row["best_beta"] > 0.0 and row["best_objective"] <= row["start_objective"] - threshold
    ]
    if not improving:
        return None
    if policy == "first_success":
        return improving[0]
    if policy == "beta_guarded":
        best_row = None
        for row in pair_rows:
            if row["best_beta"] <= 0.0 or row["best_objective"] > row["start_objective"] - threshold:
                continue
            if best_row is None or row["best_objective"] < best_row["best_objective"]:
                best_row = row
            if row["best_beta"] >= float(beta_accept_threshold):
                return best_row
        return best_row
    if policy == "best_success":
        return min(improving, key=lambda row: row["best_objective"])
    raise ValueError(f"Unsupported policy={policy!r}")


def run_policy(
    num_outer: int,
    ck_schedule: list[tuple[int, int]],
    beta_candidates: list[float],
    policy: str,
    min_abs_improvement: float,
    stop_when_improvement_below: float,
    beta_accept_threshold: float,
    basis_source: str,
    group_size: int,
    r_reduced: int,
) -> dict[str, object]:
    problem = parse_g2o_se2(G2O_PATH)
    direct_optimum = _direct_optimum_objective()
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    outer_rows: list[dict[str, object]] = []
    candidate_rows_all: list[dict[str, object]] = []

    initial_obj = float(nonlinear_objective_g2o(problem, poses))
    outer_rows.append(
        {
            "actual_outer": 0,
            "start_objective": initial_obj,
            "selected_objective": initial_obj,
            "gap_to_direct_optimum": float(initial_obj - direct_optimum),
            "accepted": True,
            "status": "initial",
        }
    )

    for actual_outer in range(1, int(num_outer) + 1):
        start_obj = float(nonlinear_objective_g2o(problem, poses))
        all_rows, pair_rows, best_pose_map = evaluate_candidates_for_base_pose(
            problem=problem,
            base_poses=poses,
            actual_outer=actual_outer,
            ck_schedule=ck_schedule,
            beta_candidates=beta_candidates,
            basis_source=basis_source,
            group_size=group_size,
            r_reduced=r_reduced,
            stop_on_first_success=(policy == "first_success"),
        )
        candidate_rows_all.extend(all_rows)
        selected = select_candidate(
            pair_rows=pair_rows,
            policy=policy,
            min_abs_improvement=min_abs_improvement,
            beta_accept_threshold=beta_accept_threshold,
        )

        if selected is None:
            best_attempt = min(pair_rows, key=lambda row: row["best_objective"])
            outer_rows.append(
                {
                    "actual_outer": int(actual_outer),
                    "start_objective": start_obj,
                    "selected_objective": float(best_attempt["best_objective"]),
                    "gap_to_direct_optimum": float(best_attempt["best_objective"] - direct_optimum),
                    "selected_c": int(best_attempt["c"]),
                    "selected_k": int(best_attempt["k"]),
                    "selected_beta": float(best_attempt["best_beta"]),
                    "selected_ck_cost": int(best_attempt["ck_cost"]),
                    "accepted": False,
                    "status": "no_improving_candidate",
                }
            )
            break

        chosen_poses = best_pose_map[(int(selected["c"]), int(selected["k"]))]
        poses = np.asarray(chosen_poses, dtype=float).copy()
        pose_history.append(poses.copy())
        improvement = float(start_obj - selected["best_objective"])
        outer_rows.append(
            {
                "actual_outer": int(actual_outer),
                "start_objective": start_obj,
                "selected_objective": float(selected["best_objective"]),
                "gap_to_direct_optimum": float(selected["best_objective"] - direct_optimum),
                "selected_c": int(selected["c"]),
                "selected_k": int(selected["k"]),
                "selected_beta": float(selected["best_beta"]),
                "selected_ck_cost": int(selected["ck_cost"]),
                "improvement": improvement,
                "accepted": True,
                "status": "accepted",
            }
        )
        print(
            f"outer={actual_outer:02d} start={start_obj:.6f} "
            f"selected={selected['best_objective']:.6f} "
            f"c={selected['c']} k={selected['k']} beta={selected['best_beta']:.6f}",
            flush=True,
        )
        if improvement < float(stop_when_improvement_below):
            break

    return {
        "config": {
            "num_outer": int(num_outer),
            "ck_schedule": [[int(c), int(k)] for c, k in ck_schedule],
            "beta_candidates": [float(beta) for beta in beta_candidates],
            "policy": policy,
            "min_abs_improvement": float(min_abs_improvement),
            "stop_when_improvement_below": float(stop_when_improvement_below),
            "beta_accept_threshold": float(beta_accept_threshold),
            "basis_source": basis_source,
            "group_size": int(group_size),
            "r_reduced": int(r_reduced),
        },
        "outer_rows": outer_rows,
        "candidate_rows": candidate_rows_all,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scan_actual_outer", "run_policy"], required=True)
    parser.add_argument("--actual-outer", type=int, default=12)
    parser.add_argument(
        "--ck-schedule",
        type=str,
        default="1x50,1x100,2x100,2x200,2x300,3x200,3x300,5x200,5x300",
    )
    parser.add_argument(
        "--beta-candidates",
        type=str,
        default="1,0.7,0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001",
    )
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--policy", choices=["first_success", "beta_guarded", "best_success"], default="first_success")
    parser.add_argument("--min-abs-improvement", type=float, default=1e-12)
    parser.add_argument("--stop-when-improvement-below", type=float, default=1e-4)
    parser.add_argument("--beta-accept-threshold", type=float, default=0.2)
    parser.add_argument("--basis-source", type=str, default="message_conditioned_information")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()

    ck_schedule = parse_ck_schedule(args.ck_schedule)
    beta_candidates = parse_float_list(args.beta_candidates)

    if args.mode == "scan_actual_outer":
        problem = parse_g2o_se2(G2O_PATH)
        base_poses = baseline_pose_for_actual_outer(args.actual_outer)
        all_rows, pair_rows, _ = evaluate_candidates_for_base_pose(
            problem=problem,
            base_poses=base_poses,
            actual_outer=args.actual_outer,
            ck_schedule=ck_schedule,
            beta_candidates=beta_candidates,
            basis_source=args.basis_source,
            group_size=args.group_size,
            r_reduced=args.r_reduced,
            stop_on_first_success=False,
        )
        stem = (
            f"intel_g2o_actual_outer{args.actual_outer}"
            f"_ckbeta_scan_{args.basis_source}"
            f"_{args.tag}"
        )
        all_csv = RESULT_DIR / f"{stem}_all.csv"
        pair_csv = RESULT_DIR / f"{stem}_pairs.csv"
        json_path = RESULT_DIR / f"{stem}.json"
        save_csv(all_rows, all_csv)
        save_csv(pair_rows, pair_csv)
        json_path.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "actual_outer": int(args.actual_outer),
                    "ck_schedule": [[int(c), int(k)] for c, k in ck_schedule],
                    "beta_candidates": beta_candidates,
                    "basis_source": args.basis_source,
                    "group_size": int(args.group_size),
                    "r_reduced": int(args.r_reduced),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"all_csv": str(all_csv), "pair_csv": str(pair_csv), "json": str(json_path)}, indent=2))
        return

    result = run_policy(
        num_outer=args.num_outer,
        ck_schedule=ck_schedule,
        beta_candidates=beta_candidates,
        policy=args.policy,
        min_abs_improvement=args.min_abs_improvement,
        stop_when_improvement_below=args.stop_when_improvement_below,
        beta_accept_threshold=args.beta_accept_threshold,
        basis_source=args.basis_source,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
    )
    stem = (
        f"intel_g2o_{args.policy}_policy_outer{args.num_outer}"
        f"_{args.basis_source}_{args.tag}"
    )
    outer_csv = RESULT_DIR / f"{stem}_outer.csv"
    cand_csv = RESULT_DIR / f"{stem}_candidates.csv"
    traj_npz = RESULT_DIR / f"{stem}_trajectories.npz"
    json_path = RESULT_DIR / f"{stem}.json"
    save_csv(result["outer_rows"], outer_csv)
    save_csv(result["candidate_rows"], cand_csv)
    np.savez_compressed(
        traj_npz,
        pose_history=np.asarray(result["pose_history"], dtype=float),
        final_poses=np.asarray(result["final_poses"], dtype=float),
    )
    json_path.write_text(json.dumps(result["config"], indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "outer_csv": str(outer_csv),
                "candidate_csv": str(cand_csv),
                "trajectories": str(traj_npz),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
