from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import approx_local_persistent_mg_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import exact_local_solve
from svd_abstraction.se2_utils import build_linearized_local_graph
from svd_abstraction.se2_utils import build_se2_problem
from svd_abstraction.se2_utils import nonlinear_objective
from svd_abstraction.se2_utils import poses_to_nodes


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_level(problem, base_poses: np.ndarray, basis_source: str) -> tuple[SVDResidualAbstraction, np.ndarray]:
    template_graph = build_linearized_local_graph(problem, base_poses)
    residual_graph = build_linearized_local_graph(problem, base_poses)
    zero = np.zeros(template_graph.n_var_nodes * 3, dtype=float)
    reset_residual_graph(residual_graph, template_graph, zero, var_slices(template_graph))

    groups = group_list(
        nodes=poses_to_nodes(base_poses),
        graph=template_graph,
        method="order",
        group_size=20,
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
        r_reduced=4,
        basis_source=basis_source,
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=False,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)
    return level, groups


def projection_relative_error(P: np.ndarray, e_star: np.ndarray) -> float:
    e_star = np.asarray(e_star, dtype=float).reshape(-1)
    denom = max(float(np.linalg.norm(e_star)), 1e-15)
    z, *_ = np.linalg.lstsq(P, e_star, rcond=None)
    residual = e_star - P @ z
    return float(np.linalg.norm(residual) / denom)


def analyze_config(problem, basis_source: str, inner_cycles: int, pre_sweeps: int, num_outer: int) -> list[dict[str, object]]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    history: list[dict[str, object]] = []

    for outer in range(0, num_outer + 1):
        exact = exact_local_solve(problem, poses)
        level, groups = build_level(problem, poses, basis_source=basis_source)
        proj_rel = projection_relative_error(level.P, exact["e_star"])

        history.append(
            {
                "basis_source": basis_source,
                "inner_cycles": int(inner_cycles),
                "pre_sweeps": int(pre_sweeps),
                "outer": int(outer),
                "nonlinear_objective": float(nonlinear_objective(problem, poses)),
                "e_star_norm": float(np.linalg.norm(exact["e_star"])),
                "proj_rel": float(proj_rel),
                "num_groups": int(len(groups)),
                "coarse_dim": int(level.total_reduced_dim),
            }
        )

        if outer == num_outer:
            break

        step = approx_local_persistent_mg_solve(
            problem=problem,
            base_poses=poses,
            inner_cycles=inner_cycles,
            pre_sweeps=pre_sweeps,
            group_size=20,
            r_reduced=4,
            basis_source=basis_source,
        )
        poses = np.asarray(step["next_poses"], dtype=float)

    return history


def main() -> None:
    problem = build_se2_problem(
        n=64,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        seed=0,
    )

    all_rows: list[dict[str, object]] = []
    configs = [
        ("joint_covariance", 1, 50),
        ("message_conditioned_information", 1, 50),
        ("joint_covariance", 2, 50),
        ("message_conditioned_information", 2, 50),
    ]

    for basis_source, inner_cycles, pre_sweeps in configs:
        all_rows.extend(
            analyze_config(
                problem=problem,
                basis_source=basis_source,
                inner_cycles=inner_cycles,
                pre_sweeps=pre_sweeps,
                num_outer=20,
            )
        )

    selected = []
    keep_outers = {0, 1, 2, 3, 5, 10, 20}
    for row in all_rows:
        if int(row["outer"]) in keep_outers:
            selected.append(row)

    stem = "se2_n64_proj_rel_outer20_k50_c1_c2_basis_compare"
    json_path = OUT_DIR / f"{stem}.json"
    csv_path = OUT_DIR / f"{stem}.csv"
    selected_csv_path = OUT_DIR / f"{stem}_selected.csv"

    json_path.write_text(json.dumps({"rows": all_rows}, indent=2), encoding="utf-8")
    save_csv(all_rows, csv_path)
    save_csv(selected, selected_csv_path)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "selected_csv": str(selected_csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
