from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import build_linearized_local_graph_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import run_direct_newton_g2o
from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.intel_g2o_persistent_residual_mg import exact_local_solve_g2o
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


G2O_PATH = pathlib.Path("/home/yuzhou/Desktop/input_INTEL_g2o.g2o")
OUTPUT_DIR = pathlib.Path(
    "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def projection_residual(vec: np.ndarray, basis: np.ndarray) -> float:
    coeffs, *_ = np.linalg.lstsq(basis, vec, rcond=None)
    fit = basis @ coeffs
    return float(np.linalg.norm(vec - fit) / max(np.linalg.norm(vec), 1e-15))


def ideal_coarse_rel(a: np.ndarray, b: np.ndarray, e_star: np.ndarray, basis: np.ndarray) -> float:
    ac = basis.T @ a @ basis
    rc = basis.T @ b
    yc = np.linalg.solve(ac, rc)
    e_c = basis @ yc
    return float(np.linalg.norm(e_star - e_c) / max(np.linalg.norm(e_star), 1e-15))


def build_basis(problem, base_poses: np.ndarray, group_size: int, r_reduced: int) -> tuple[np.ndarray, dict[str, float]]:
    graph = build_linearized_local_graph_g2o(problem, base_poses)
    groups = group_list(
        nodes=[
            {"data": {"id": str(i)}, "position": {"x": float(p[0]), "y": float(p[1])}}
            for i, p in enumerate(np.asarray(base_poses, dtype=float))
        ],
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
        basis_source="joint_covariance",
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=False,
    )
    t0 = time.time()
    level.initialize_bases(force=True)
    build_time = time.time() - t0
    return level.P.copy(), {
        "num_groups": int(len(groups)),
        "coarse_dim": int(level.total_reduced_dim),
        "build_time_sec": float(build_time),
    }


def main() -> None:
    problem = parse_g2o_se2(G2O_PATH)
    direct = run_direct_newton_g2o(problem, num_outer=20, rel_obj_tol=1e-12, step_tol=1e-10)

    poses_hist = [np.asarray(problem.init_poses, dtype=float)]
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    for _ in range(20):
        step = exact_local_solve_g2o(problem, poses)
        poses = np.asarray(step["next_poses"], dtype=float)
        poses_hist.append(poses.copy())

    p0, p0_stats = build_basis(problem, poses_hist[0], group_size=20, r_reduced=4)

    rows: list[dict[str, float | int]] = []
    for outer, poses_t in enumerate(poses_hist):
        exact = exact_local_solve_g2o(problem, poses_t)
        p_t, p_t_stats = build_basis(problem, poses_t, group_size=20, r_reduced=4)
        a = exact["A"].toarray()
        b = np.asarray(exact["b"], dtype=float)
        e_star = np.asarray(exact["e_star"], dtype=float)

        rows.append(
            {
                "outer": int(outer),
                "nonlinear_objective": float(direct["history"][outer]["nonlinear_objective"]),
                "step_norm": float(np.linalg.norm(e_star)),
                "frozen_p0_proj_rel": projection_residual(e_star, p0),
                "refreshed_pt_proj_rel": projection_residual(e_star, p_t),
                "frozen_p0_ideal_coarse_rel": ideal_coarse_rel(a, b, e_star, p0),
                "refreshed_pt_ideal_coarse_rel": ideal_coarse_rel(a, b, e_star, p_t),
            }
        )

    payload = {
        "config": {
            "group_size": 20,
            "r_reduced": 4,
            "basis_source": "joint_covariance",
            "freeze_from_outer": 0,
            "num_outer": 20,
        },
        "frozen_basis_stats": p0_stats,
        "rows": rows,
    }

    json_path = OUTPUT_DIR / "intel_g2o_freeze_p_outer0_proj_rel.json"
    csv_path = OUTPUT_DIR / "intel_g2o_freeze_p_outer0_proj_rel.csv"
    selected_csv_path = OUTPUT_DIR / "intel_g2o_freeze_p_outer0_proj_rel_selected.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    header = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in header) + "\n")

    keep = {0, 1, 2, 3, 4, 5, 10, 15, 20}
    with selected_csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            if int(row["outer"]) in keep:
                f.write(",".join(str(row[k]) for k in header) + "\n")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
