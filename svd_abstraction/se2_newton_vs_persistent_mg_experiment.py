from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages
from svd_abstraction.persistent_state_exact_coarse_experiment import mean_vector
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices
from svd_abstraction.se2_utils import apply_pose_deltas
from svd_abstraction.se2_utils import build_linearized_local_graph
from svd_abstraction.se2_utils import build_se2_problem
from svd_abstraction.se2_utils import mean_pose_log_error
from svd_abstraction.se2_utils import nonlinear_objective
from svd_abstraction.se2_utils import poses_to_nodes
from svd_abstraction.se2_utils import rms_angle_error
from svd_abstraction.se2_utils import rms_translation_error
from svd_abstraction.se2_utils import stack_pose_errors


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def basis_suffix(basis_source: str) -> str:
    if basis_source == "joint_covariance":
        return ""
    if basis_source == "message_conditioned_information":
        return "_conditioned_information"
    return f"_{basis_source}"


def pose_metrics(problem, poses: np.ndarray) -> dict[str, float]:
    return {
        "nonlinear_objective": float(nonlinear_objective(problem, poses)),
        "mean_pose_log_error": float(mean_pose_log_error(problem.gt_poses, poses)),
        "rms_translation_error": float(rms_translation_error(problem.gt_poses, poses)),
        "rms_angle_error": float(rms_angle_error(problem.gt_poses, poses)),
    }


def pairwise_pose_gap(poses_a: np.ndarray, poses_b: np.ndarray) -> float:
    errs = stack_pose_errors(np.asarray(poses_a, dtype=float), np.asarray(poses_b, dtype=float))
    if errs.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(errs.reshape(-1, 3), axis=1)))


def exact_local_solve(problem, base_poses: np.ndarray) -> dict[str, object]:
    local_graph = build_linearized_local_graph(problem, base_poses)
    eta, lam = local_graph.joint_distribution_inf_absolute()
    e_star = np.linalg.solve(0.5 * (lam + lam.T), eta)
    lin_res = float(np.linalg.norm(eta - lam @ e_star))
    next_poses = apply_pose_deltas(base_poses, e_star)
    out = {
        "local_graph": local_graph,
        "eta": eta,
        "lam": lam,
        "e_star": e_star,
        "e_norm": float(np.linalg.norm(e_star)),
        "linear_residual_norm": lin_res,
        "next_poses": next_poses,
    }
    out.update({f"after_{k}": v for k, v in pose_metrics(problem, next_poses).items()})
    return out


def approx_local_persistent_mg_solve(
    problem,
    base_poses: np.ndarray,
    inner_cycles: int,
    pre_sweeps: int,
    group_size: int,
    r_reduced: int,
    basis_source: str = "joint_covariance",
) -> dict[str, object]:
    template_graph = build_linearized_local_graph(problem, base_poses)
    eta, lam = template_graph.joint_distribution_inf_absolute()
    e_star = np.linalg.solve(0.5 * (lam + lam.T), eta)

    residual_graph = build_linearized_local_graph(problem, base_poses)
    zero = np.zeros(template_graph.n_var_nodes * 3, dtype=float)
    reset_residual_graph(residual_graph, template_graph, zero, var_slices(template_graph))

    groups = group_list(
        nodes=poses_to_nodes(base_poses),
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
    for cyc in range(1, inner_cycles + 1):
        for _ in range(pre_sweeps):
            residual_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_e = level.prolongate(delta_z)
        inject_correction_keep_messages(residual_graph, delta_e)

        e_now = mean_vector(residual_graph)
        inner_history.append(
            {
                "inner_cycle": int(cyc),
                "e_rel_to_exact": float(np.linalg.norm(e_now - e_star) / max(np.linalg.norm(e_star), 1e-15)),
                "linear_residual_norm": float(np.linalg.norm(eta - lam @ e_now)),
                "e_norm": float(np.linalg.norm(e_now)),
            }
        )

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
        "inner_history": inner_history,
        "num_groups": int(len(groups)),
        "coarse_dim": int(level.total_reduced_dim),
        "next_pose_gap_to_exact": float(pairwise_pose_gap(exact_next_poses, next_poses)),
        "next_objective_gap_to_exact": float(approx_after["nonlinear_objective"] - exact_after["nonlinear_objective"]),
    }
    out.update({f"after_{k}": v for k, v in approx_after.items()})
    return out


def run_direct_baseline(problem, num_outer: int) -> dict[str, object]:
    poses = problem.init_poses.copy()
    history: list[dict[str, float]] = []
    history.append({"outer": 0, **pose_metrics(problem, poses)})

    for outer in range(1, num_outer + 1):
        step = exact_local_solve(problem, poses)
        poses = step["next_poses"]
        history.append(
            {
                "outer": int(outer),
                "linear_step_norm": float(step["e_norm"]),
                "linear_residual_norm": float(step["linear_residual_norm"]),
                **pose_metrics(problem, poses),
            }
        )

    return {
        "config": {"num_outer": int(num_outer)},
        "history": history,
    }


def run_mg_outer(
    problem,
    num_outer: int,
    inner_cycles: int,
    pre_sweeps: int,
    group_size: int,
    r_reduced: int,
    basis_source: str = "joint_covariance",
) -> dict[str, object]:
    poses = problem.init_poses.copy()
    history: list[dict[str, float]] = []
    history.append({"outer": 0, **pose_metrics(problem, poses)})

    for outer in range(1, num_outer + 1):
        step = approx_local_persistent_mg_solve(
            problem=problem,
            base_poses=poses,
            inner_cycles=inner_cycles,
            pre_sweeps=pre_sweeps,
            group_size=group_size,
            r_reduced=r_reduced,
            basis_source=basis_source,
        )
        poses = step["next_poses"]
        row = {
            "outer": int(outer),
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
        history.append(row)

    return {
        "config": {
            "num_outer": int(num_outer),
            "inner_cycles": int(inner_cycles),
            "pre_sweeps": int(pre_sweeps),
            "group_size": int(group_size),
            "r_reduced": int(r_reduced),
            "basis_source": basis_source,
        },
        "history": history,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-outer", type=int, default=15)
    parser.add_argument("--inner-cycles", type=int, default=10)
    parser.add_argument("--pre-sweeps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--basis-source", type=str, default="joint_covariance")
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
    mg = run_mg_outer(
        problem,
        num_outer=args.num_outer,
        inner_cycles=args.inner_cycles,
        pre_sweeps=args.pre_sweeps,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        basis_source=args.basis_source,
    )

    out = {
        "config": {
            "n": int(args.n),
            "seed": int(args.seed),
            "num_outer": int(args.num_outer),
            "inner_cycles": int(args.inner_cycles),
            "pre_sweeps": int(args.pre_sweeps),
            "group_size": int(args.group_size),
            "r_reduced": int(args.r_reduced),
            "basis_source": args.basis_source,
        },
        "initial_metrics": pose_metrics(problem, problem.init_poses),
        "gt_objective": float(nonlinear_objective(problem, problem.gt_poses)),
        "direct_newton": direct,
        "persistent_residual_mg": mg,
    }

    stem = (
        f"se2_newton_vs_persistent_mg_n{args.n}_seed{args.seed}"
        f"_outer{args.num_outer}_c{args.inner_cycles}_k{args.pre_sweeps}"
    )
    stem += basis_suffix(args.basis_source)
    json_path = OUT_DIR / f"{stem}.json"
    direct_csv = OUT_DIR / f"{stem}_direct.csv"
    mg_csv = OUT_DIR / f"{stem}_mg.csv"

    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_csv(direct["history"], direct_csv)
    save_csv(mg["history"], mg_csv)

    print(json.dumps(
        {
            "json": str(json_path),
            "direct_csv": str(direct_csv),
            "mg_csv": str(mg_csv),
            "direct_final": direct["history"][-1],
            "mg_final": mg["history"][-1],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
