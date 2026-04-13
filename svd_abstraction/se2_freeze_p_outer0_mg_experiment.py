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
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import exact_local_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import pose_metrics
from svd_abstraction.se2_utils import build_linearized_local_graph
from svd_abstraction.se2_utils import build_se2_problem
from svd_abstraction.se2_utils import poses_to_nodes


OUTPUT_DIR = pathlib.Path(
    "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def copy_frozen_basis(dst: SVDResidualAbstraction, src: SVDResidualAbstraction) -> None:
    dst.Bs = [np.asarray(b, dtype=float).copy() for b in src.Bs]
    dst.group_dims = list(src.group_dims)
    dst.group_full_dofs = [list(g) for g in src.group_full_dofs]
    dst.group_reduced_slices = list(src.group_reduced_slices)
    dst.total_reduced_dim = int(src.total_reduced_dim)
    dst.P = np.asarray(src.P, dtype=float).copy()
    dst.bases_initialized = True
    dst._refresh_group_maps()
    dst.coarse_graph = None
    dst.coarse_var_nodes = []
    dst._coarse_prior_eta_terms = []
    dst._coarse_factor_eta_terms = []


def build_level(problem, base_poses, group_size: int, r_reduced: int, frozen_basis_level: SVDResidualAbstraction | None):
    template_graph = build_linearized_local_graph(problem, base_poses)
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
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=False,
    )
    if frozen_basis_level is None:
        level.initialize_bases(force=True)
    else:
        copy_frozen_basis(level, frozen_basis_level)
    level.build_coarse_graph(force=True)
    return template_graph, residual_graph, level


def run_mg_outer(problem, num_outer: int, inner_cycles: int, pre_sweeps: int, group_size: int, r_reduced: int, freeze_from_outer0: bool):
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    history: list[dict[str, float]] = [{"outer": 0, **pose_metrics(problem, poses)}]
    frozen_level = None

    for outer in range(1, num_outer + 1):
        exact = exact_local_solve(problem, poses)
        template_graph, residual_graph, level = build_level(
            problem=problem,
            base_poses=poses,
            group_size=group_size,
            r_reduced=r_reduced,
            frozen_basis_level=frozen_level if freeze_from_outer0 else None,
        )
        if freeze_from_outer0 and frozen_level is None:
            frozen_level = level

        inner_history = []
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
                    "e_rel_to_exact": float(np.linalg.norm(e_now - exact["e_star"]) / max(np.linalg.norm(exact["e_star"]), 1e-15)),
                    "linear_residual_norm": float(np.linalg.norm(exact["eta"] - exact["lam"] @ e_now)),
                    "e_norm": float(np.linalg.norm(e_now)),
                }
            )

        e_hat = mean_vector(residual_graph)
        poses = exact["next_poses"] if inner_cycles == -1 else poses  # dead branch; keep type checker calm
        from svd_abstraction.se2_utils import apply_pose_deltas
        poses = apply_pose_deltas(poses, e_hat)
        row = {
            "outer": int(outer),
            "inner_cycles": int(inner_cycles),
            "pre_sweeps": int(pre_sweeps),
            "e_hat_norm": float(np.linalg.norm(e_hat)),
            "e_star_norm": float(np.linalg.norm(exact["e_star"])),
            "e_rel_to_exact": float(np.linalg.norm(e_hat - exact["e_star"]) / max(np.linalg.norm(exact["e_star"]), 1e-15)),
            "linear_residual_exact": float(exact["linear_residual_norm"]),
            "linear_residual_approx": float(np.linalg.norm(exact["eta"] - exact["lam"] @ e_hat)),
            "num_groups": int(len(level.groups)),
            "coarse_dim": int(level.total_reduced_dim),
            **pose_metrics(problem, poses),
        }
        history.append(row)

    return {"history": history, "final_poses": poses.tolist()}


def save_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--inner-cycles", type=int, default=10)
    parser.add_argument("--pre-sweeps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = build_se2_problem(n=args.n, seed=args.seed)
    frozen = run_mg_outer(
        problem=problem,
        num_outer=args.num_outer,
        inner_cycles=args.inner_cycles,
        pre_sweeps=args.pre_sweeps,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        freeze_from_outer0=True,
    )
    refreshed = run_mg_outer(
        problem=problem,
        num_outer=args.num_outer,
        inner_cycles=args.inner_cycles,
        pre_sweeps=args.pre_sweeps,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        freeze_from_outer0=False,
    )

    payload = {
        "config": {
            "n": int(args.n),
            "seed": int(args.seed),
            "num_outer": int(args.num_outer),
            "inner_cycles": int(args.inner_cycles),
            "pre_sweeps": int(args.pre_sweeps),
            "group_size": int(args.group_size),
            "r_reduced": int(args.r_reduced),
        },
        "frozen_outer0": frozen,
        "refreshed_each_outer": refreshed,
    }

    stem = f"synthetic_se2_freeze_p0_vs_refreshed_n{args.n}_outer{args.num_outer}_c{args.inner_cycles}_k{args.pre_sweeps}"
    json_path = OUTPUT_DIR / f"{stem}.json"
    frozen_csv = OUTPUT_DIR / f"{stem}_frozen.csv"
    refreshed_csv = OUTPUT_DIR / f"{stem}_refreshed.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_csv(frozen["history"], frozen_csv)
    save_csv(refreshed["history"], refreshed_csv)

    print(json.dumps(payload["config"], indent=2))


if __name__ == "__main__":
    main()
