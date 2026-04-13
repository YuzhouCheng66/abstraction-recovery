from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.se2_utils import apply_pose_deltas


G2O_PATH = pathlib.Path("/home/yuzhou/Desktop/input_INTEL_g2o.g2o")
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
DIRECT_TRAJ = RESULT_DIR / "intel_g2o_direct_newton_damping0p1_outer300_trajectories.npz"
DIRECT_META = RESULT_DIR / "intel_g2o_direct_newton_damping0p1_outer300.json"


def _run_candidate(*args, **kwargs):
    from svd_abstraction.intel_g2o_ck_beta_policy_experiment import run_candidate

    return run_candidate(*args, **kwargs)


def _exact_local_solve_g2o(*args, **kwargs):
    from svd_abstraction.intel_g2o_persistent_residual_mg import exact_local_solve_g2o

    return exact_local_solve_g2o(*args, **kwargs)


def xy_stack(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    return poses[:, :2].reshape(-1)


def xy_rel_to_ref(poses: np.ndarray, ref_poses: np.ndarray) -> float:
    x = xy_stack(poses)
    x_ref = xy_stack(ref_poses)
    denom = max(float(np.linalg.norm(x_ref)), 1e-15)
    return float(np.linalg.norm(x - x_ref) / denom)


def direct_reference_poses() -> np.ndarray:
    data = np.load(DIRECT_TRAJ)
    return np.asarray(data["final_poses"], dtype=float)


def direct_optimum_objective() -> float:
    return float(json.loads(DIRECT_META.read_text(encoding="utf-8"))["best_objective"])


def evaluate_betas(
    problem,
    base_poses: np.ndarray,
    e_hat: np.ndarray,
    beta_candidates: list[float],
    ref_poses: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start_obj = float(nonlinear_objective_g2o(problem, base_poses))
    start_x_rel = float(xy_rel_to_ref(base_poses, ref_poses))
    for beta in beta_candidates:
        beta = float(beta)
        next_poses = apply_pose_deltas(base_poses, beta * np.asarray(e_hat, dtype=float))
        post_obj = float(nonlinear_objective_g2o(problem, next_poses))
        post_x_rel = float(xy_rel_to_ref(next_poses, ref_poses))
        rows.append(
            {
                "beta": beta,
                "post_objective": post_obj,
                "objective_improvement": float(start_obj - post_obj),
                "x_rel_to_direct": post_x_rel,
                "x_rel_improvement": float(start_x_rel - post_x_rel),
                "poses": next_poses,
            }
        )
    return rows


def select_largest_improving(
    beta_rows: list[dict[str, Any]],
    start_obj: float,
    min_improvement: float = 1e-12,
) -> dict[str, Any] | None:
    for row in beta_rows:
        if float(row["post_objective"]) <= start_obj - float(min_improvement):
            return row
    return None


def select_best_improving(
    beta_rows: list[dict[str, Any]],
    start_obj: float,
    min_improvement: float = 1e-12,
) -> dict[str, Any] | None:
    improving = [
        row
        for row in beta_rows
        if float(row["post_objective"]) <= start_obj - float(min_improvement)
    ]
    if not improving:
        return None
    return min(improving, key=lambda row: float(row["post_objective"]))


def select_best_improving_with_beta_ceiling(
    beta_rows: list[dict[str, Any]],
    start_obj: float,
    min_improvement: float,
    beta_ceiling: float,
) -> dict[str, Any] | None:
    improving = [
        row
        for row in beta_rows
        if float(row["beta"]) <= float(beta_ceiling)
        and float(row["post_objective"]) <= start_obj - float(min_improvement)
    ]
    if not improving:
        return None
    return min(improving, key=lambda row: float(row["post_objective"]))


def select_largest_under_cap(
    beta_rows: list[dict[str, Any]],
    start_obj: float,
    cap_multiplier: float,
    min_beta: float,
) -> dict[str, Any] | None:
    cap = float(start_obj) * float(cap_multiplier)
    for row in beta_rows:
        if float(row["beta"]) < float(min_beta):
            continue
        if float(row["post_objective"]) <= cap:
            return row
    return None


def select_largest_improving_or_capped(
    beta_rows: list[dict[str, Any]],
    start_obj: float,
    min_improvement: float,
    cap_multiplier: float,
    beta_floor: float,
) -> dict[str, Any] | None:
    for row in beta_rows:
        if float(row["beta"]) < float(beta_floor):
            continue
        if float(row["post_objective"]) <= start_obj - float(min_improvement):
            return row
    cap = float(start_obj) * float(cap_multiplier)
    for row in beta_rows:
        if float(row["beta"]) < float(beta_floor):
            continue
        if float(row["post_objective"]) <= cap:
            return row
    return select_largest_improving(beta_rows, start_obj=start_obj, min_improvement=min_improvement)


def select_best_improving_or_capped(
    beta_rows: list[dict[str, Any]],
    start_obj: float,
    min_improvement: float,
    cap_multiplier: float,
    beta_floor: float,
) -> dict[str, Any] | None:
    for row in beta_rows:
        if float(row["beta"]) < float(beta_floor):
            continue
        if float(row["post_objective"]) <= start_obj - float(min_improvement):
            return row
    cap = float(start_obj) * float(cap_multiplier)
    for row in beta_rows:
        if float(row["beta"]) < float(beta_floor):
            continue
        if float(row["post_objective"]) <= cap:
            return row
    return select_best_improving(beta_rows, start_obj=start_obj, min_improvement=min_improvement)


def candidate_plan_for_phase(start_obj: float, preset: str) -> tuple[str, list[dict[str, Any]]]:
    if preset == "bridge_v1":
        if start_obj > 1.0e4:
            return (
                "geometry_high",
                [
                    {"c": 5, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.5},
                    {"c": 3, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.3},
                    {"c": 5, "k": 100, "selector": "cap", "cap_multiplier": 10.0, "min_beta": 0.2},
                ],
            )
        if start_obj > 2.0e2:
            return (
                "geometry_mid",
                [
                    {"c": 3, "k": 50, "selector": "improve_or_cap", "cap_multiplier": 2.0, "beta_floor": 0.3},
                    {"c": 2, "k": 100, "selector": "improve_or_cap", "cap_multiplier": 1.5, "beta_floor": 0.2},
                    {"c": 5, "k": 100, "selector": "improve_or_cap", "cap_multiplier": 1.25, "beta_floor": 0.2},
                    {"c": 2, "k": 300, "selector": "improve", "min_improvement": 1e-12},
                ],
            )
        return (
            "objective_low",
            [
                {"c": 5, "k": 200, "selector": "improve", "min_improvement": 1e-12},
                {"c": 2, "k": 300, "selector": "improve", "min_improvement": 1e-12},
                {"c": 4, "k": 200, "selector": "improve", "min_improvement": 1e-12},
                {"c": 5, "k": 300, "selector": "improve", "min_improvement": 1e-12},
            ],
        )

    if preset == "bridge_v2":
        if start_obj > 5.0e4:
            return (
                "geometry_high",
                [
                    {"c": 5, "k": 50, "selector": "cap", "cap_multiplier": 30.0, "min_beta": 0.5},
                    {"c": 5, "k": 100, "selector": "cap", "cap_multiplier": 15.0, "min_beta": 0.3},
                    {"c": 3, "k": 50, "selector": "cap", "cap_multiplier": 15.0, "min_beta": 0.3},
                ],
            )
        if start_obj > 3.0e2:
            return (
                "geometry_mid",
                [
                    {"c": 3, "k": 50, "selector": "improve_or_cap", "cap_multiplier": 1.8, "beta_floor": 0.5},
                    {"c": 5, "k": 100, "selector": "improve_or_cap", "cap_multiplier": 1.4, "beta_floor": 0.3},
                    {"c": 2, "k": 100, "selector": "improve_or_cap", "cap_multiplier": 1.4, "beta_floor": 0.2},
                ],
            )
        return (
            "objective_low",
            [
                {"c": 5, "k": 200, "selector": "improve", "min_improvement": 1e-12},
                {"c": 5, "k": 300, "selector": "improve", "min_improvement": 1e-12},
                {"c": 4, "k": 200, "selector": "improve", "min_improvement": 1e-12},
                {"c": 2, "k": 300, "selector": "improve", "min_improvement": 1e-12},
            ],
        )

    if preset in {"bridge_v3", "formal_v1"}:
        if start_obj > 1.0e4:
            return (
                "geometry_high",
                [
                    {"c": 5, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.5},
                    {"c": 3, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.3},
                    {"c": 5, "k": 100, "selector": "cap", "cap_multiplier": 10.0, "min_beta": 0.2},
                ],
            )
        if start_obj > 2.0e2:
            return (
                "geometry_mid",
                [
                    {"c": 3, "k": 50, "selector": "improve_or_cap", "cap_multiplier": 2.0, "beta_floor": 0.3},
                    {"c": 2, "k": 100, "selector": "improve_or_cap", "cap_multiplier": 1.5, "beta_floor": 0.2},
                    {"c": 5, "k": 100, "selector": "improve_or_cap", "cap_multiplier": 1.25, "beta_floor": 0.2},
                    {"c": 2, "k": 300, "selector": "improve", "min_improvement": 1e-12},
                ],
            )
        return (
            "objective_low",
            [
                {"c": 5, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 4, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 5, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
            ],
        )

    if preset == "formal_v2":
        if start_obj > 1.0e4:
            return (
                "geometry_high",
                [
                    {"c": 5, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.5},
                    {"c": 3, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.3},
                    {"c": 5, "k": 100, "selector": "cap", "cap_multiplier": 10.0, "min_beta": 0.2},
                ],
            )
        if start_obj > 2.0e2:
            return (
                "geometry_mid",
                [
                    {
                        "c": 3,
                        "k": 50,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.0,
                        "beta_floor": 0.3,
                    },
                    {
                        "c": 2,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.5,
                        "beta_floor": 0.2,
                    },
                    {
                        "c": 5,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.25,
                        "beta_floor": 0.2,
                    },
                    {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                ],
            )
        return (
            "objective_low",
            [
                {"c": 5, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 4, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 5, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
            ],
        )

    if preset == "formal_v3":
        if start_obj > 1.0e4:
            return (
                "geometry_high",
                [
                    {"c": 5, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.5},
                    {"c": 3, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.3},
                    {"c": 5, "k": 100, "selector": "cap", "cap_multiplier": 10.0, "min_beta": 0.2},
                ],
            )
        if start_obj > 2.3e2:
            return (
                "geometry_mid_entry",
                [
                    {
                        "c": 4,
                        "k": 200,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.0,
                        "beta_floor": 0.5,
                    },
                    {
                        "c": 5,
                        "k": 200,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.5,
                        "beta_floor": 0.5,
                    },
                    {
                        "c": 5,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.5,
                        "beta_floor": 0.2,
                    },
                    {
                        "c": 3,
                        "k": 50,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.0,
                        "beta_floor": 0.3,
                    },
                    {
                        "c": 2,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.5,
                        "beta_floor": 0.2,
                    },
                    {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                ],
            )
        if start_obj > 2.0e2:
            return (
                "geometry_mid",
                [
                    {
                        "c": 3,
                        "k": 50,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.0,
                        "beta_floor": 0.3,
                    },
                    {
                        "c": 2,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.5,
                        "beta_floor": 0.2,
                    },
                    {
                        "c": 5,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.25,
                        "beta_floor": 0.2,
                    },
                    {"c": 5, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                    {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                ],
            )
        return (
            "objective_low",
            [
                {"c": 4, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 5, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 5, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
            ],
        )

    if preset == "formal_v4":
        if start_obj > 1.0e4:
            return (
                "geometry_high",
                [
                    {"c": 5, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.5},
                    {"c": 3, "k": 50, "selector": "cap", "cap_multiplier": 20.0, "min_beta": 0.3},
                    {"c": 5, "k": 100, "selector": "cap", "cap_multiplier": 10.0, "min_beta": 0.2},
                ],
            )
        if start_obj > 2.3e2:
            return (
                "geometry_mid_entry",
                [
                    {
                        "c": 5,
                        "k": 200,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.5,
                        "beta_floor": 0.5,
                    },
                    {
                        "c": 4,
                        "k": 200,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.0,
                        "beta_floor": 0.5,
                    },
                    {
                        "c": 3,
                        "k": 50,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.0,
                        "beta_floor": 0.3,
                    },
                    {
                        "c": 2,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.5,
                        "beta_floor": 0.2,
                    },
                    {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                ],
            )
        if start_obj > 2.0e2:
            return (
                "geometry_mid",
                [
                    {
                        "c": 3,
                        "k": 50,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 2.0,
                        "beta_floor": 0.3,
                    },
                    {
                        "c": 5,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.25,
                        "beta_floor": 0.2,
                    },
                    {
                        "c": 2,
                        "k": 100,
                        "selector": "best_improve_or_cap",
                        "cap_multiplier": 1.5,
                        "beta_floor": 0.2,
                    },
                    {"c": 5, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                    {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                ],
            )
        return (
            "objective_low",
            [
                {"c": 5, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 4, "k": 200, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 2, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
                {"c": 5, "k": 300, "selector": "best_improve", "min_improvement": 1e-12},
            ],
        )

    if preset in {"formal_v5", "formal_v6"}:
        return candidate_plan_for_phase(start_obj=start_obj, preset="formal_v4")

    raise ValueError(f"Unsupported preset={preset!r}")


def kick_gate_specs_for_preset(preset: str, start_obj: float) -> list[dict[str, Any]]:
    if preset not in {"formal_v5", "formal_v6"}:
        return []
    if start_obj > 1.10e2:
        return []
    return [
        {"c": 5, "k": 200, "max_e_rel_to_exact": 0.8, "objective_cap_multiplier": 2.5},
        {"c": 4, "k": 200, "max_e_rel_to_exact": 0.8, "objective_cap_multiplier": 2.5},
        {"c": 2, "k": 300, "max_e_rel_to_exact": 0.8, "objective_cap_multiplier": 2.5},
        {"c": 5, "k": 50, "max_e_rel_to_exact": 0.8, "objective_cap_multiplier": 2.5},
    ]


def post_kick_recovery_plan_for_preset(
    preset: str,
    history: list[dict[str, Any]],
    start_obj: float,
) -> tuple[str, list[dict[str, Any]]] | None:
    if preset != "formal_v6":
        return None
    if not history:
        return None

    consecutive_recovery_steps = 0
    for row in reversed(history):
        phase = str(row.get("phase", ""))
        if phase in {"kick_gate", "post_kick_recovery"}:
            consecutive_recovery_steps += 1
            continue
        break

    if consecutive_recovery_steps <= 0 or consecutive_recovery_steps > 2:
        return None
    if start_obj <= 1.09e2:
        return None

    if start_obj > 1.50e2:
        return (
            "post_kick_recovery",
            [
                {"c": 5, "k": 50, "selector": "best_improve_beta_ceiling", "beta_ceiling": 0.05},
                {"c": 5, "k": 100, "selector": "best_improve_beta_ceiling", "beta_ceiling": 0.05},
                {"c": 2, "k": 100, "selector": "best_improve_beta_ceiling", "beta_ceiling": 0.02},
                {"c": 5, "k": 200, "selector": "best_improve_beta_ceiling", "beta_ceiling": 0.02},
            ],
        )

    return (
        "post_kick_recovery",
        [
            {"c": 5, "k": 50, "selector": "best_improve_beta_ceiling", "beta_ceiling": 0.05},
            {"c": 5, "k": 100, "selector": "best_improve_beta_ceiling", "beta_ceiling": 0.02},
            {"c": 2, "k": 100, "selector": "best_improve_beta_ceiling", "beta_ceiling": 0.02},
            {"c": 5, "k": 200, "selector": "best_improve_beta_ceiling", "beta_ceiling": 0.01},
        ],
    )


def select_beta_from_spec(
    beta_rows: list[dict[str, Any]],
    start_obj: float,
    spec: dict[str, Any],
) -> dict[str, Any] | None:
    selector = str(spec["selector"])
    if selector == "improve":
        return select_largest_improving(
            beta_rows,
            start_obj=start_obj,
            min_improvement=float(spec.get("min_improvement", 1e-12)),
        )
    if selector == "best_improve":
        return select_best_improving(
            beta_rows,
            start_obj=start_obj,
            min_improvement=float(spec.get("min_improvement", 1e-12)),
        )
    if selector == "best_improve_beta_ceiling":
        return select_best_improving_with_beta_ceiling(
            beta_rows,
            start_obj=start_obj,
            min_improvement=float(spec.get("min_improvement", 1e-12)),
            beta_ceiling=float(spec["beta_ceiling"]),
        )
    if selector == "cap":
        return select_largest_under_cap(
            beta_rows,
            start_obj=start_obj,
            cap_multiplier=float(spec["cap_multiplier"]),
            min_beta=float(spec["min_beta"]),
        )
    if selector == "improve_or_cap":
        return select_largest_improving_or_capped(
            beta_rows,
            start_obj=start_obj,
            min_improvement=float(spec.get("min_improvement", 1e-12)),
            cap_multiplier=float(spec["cap_multiplier"]),
            beta_floor=float(spec["beta_floor"]),
        )
    if selector == "best_improve_or_cap":
        return select_best_improving_or_capped(
            beta_rows,
            start_obj=start_obj,
            min_improvement=float(spec.get("min_improvement", 1e-12)),
            cap_multiplier=float(spec["cap_multiplier"]),
            beta_floor=float(spec["beta_floor"]),
        )
    raise ValueError(f"Unsupported selector={selector!r}")


def run_adaptive_policy_outer_g2o(
    problem,
    num_outer: int,
    basis_source: str,
    group_size: int,
    r_reduced: int,
    preset: str,
    beta_candidates: list[float],
    ref_poses: np.ndarray | None = None,
    direct_optimum: float | None = None,
) -> dict[str, Any]:
    if ref_poses is None:
        ref_poses = direct_reference_poses()
    if direct_optimum is None:
        direct_optimum = direct_optimum_objective()

    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    history: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []

    initial_obj = float(nonlinear_objective_g2o(problem, poses))
    history.append(
        {
            "outer": 0,
            "phase": "initial",
            "nonlinear_objective": initial_obj,
            "gap_to_direct_optimum": float(initial_obj - direct_optimum),
            "x_rel_to_direct": float(xy_rel_to_ref(poses, ref_poses)),
            "accepted": True,
        }
    )

    for outer in range(1, int(num_outer) + 1):
        start_obj = float(nonlinear_objective_g2o(problem, poses))
        start_x_rel = float(xy_rel_to_ref(poses, ref_poses))
        exact = _exact_local_solve_g2o(problem, poses)
        kick_specs = kick_gate_specs_for_preset(preset=preset, start_obj=start_obj)
        if kick_specs:
            kick_accepts: list[dict[str, Any]] = []
            for attempt_idx, spec in enumerate(kick_specs, start=1):
                candidate = _run_candidate(
                    problem=problem,
                    base_poses=poses,
                    exact=exact,
                    inner_cycles=int(spec["c"]),
                    pre_sweeps=int(spec["k"]),
                    group_size=group_size,
                    r_reduced=r_reduced,
                    basis_source=basis_source,
                )
                full_step_poses = apply_pose_deltas(poses, np.asarray(candidate["e_hat"], dtype=float))
                full_step_obj = float(nonlinear_objective_g2o(problem, full_step_poses))
                full_step_x_rel = float(xy_rel_to_ref(full_step_poses, ref_poses))
                passes_gate = (
                    float(candidate["e_rel_to_exact"]) <= float(spec["max_e_rel_to_exact"])
                    and full_step_obj <= start_obj * float(spec["objective_cap_multiplier"])
                )
                attempt_rows.append(
                    {
                        "outer": int(outer),
                        "phase": "kick_gate_probe",
                        "attempt_idx": int(attempt_idx),
                        "start_objective": start_obj,
                        "start_x_rel_to_direct": start_x_rel,
                        "c": int(spec["c"]),
                        "k": int(spec["k"]),
                        "selector": "kick_gate_full_step",
                        "beta": 1.0,
                        "post_objective": full_step_obj,
                        "objective_improvement": float(start_obj - full_step_obj),
                        "x_rel_to_direct": full_step_x_rel,
                        "x_rel_improvement": float(start_x_rel - full_step_x_rel),
                        "selected": False,
                        "passes_gate": bool(passes_gate),
                        "max_e_rel_to_exact": float(spec["max_e_rel_to_exact"]),
                        "objective_cap_multiplier": float(spec["objective_cap_multiplier"]),
                        "linear_residual_approx": float(candidate["linear_residual_approx"]),
                        "e_rel_to_exact": float(candidate["e_rel_to_exact"]),
                        "raw_full_step_objective": float(candidate["raw_full_step_objective"]),
                        "last_cycle_pre_residual_ratio": float(
                            candidate["cycle_rows"][-1]["pre_residual_ratio_vs_cycle_start"]
                        ),
                        "last_cycle_post_residual_ratio": float(
                            candidate["cycle_rows"][-1]["post_residual_ratio_vs_cycle_start"]
                        ),
                    }
                )
                if passes_gate:
                    kick_accepts.append(
                        {
                            "spec": spec,
                            "candidate": candidate,
                            "poses": full_step_poses,
                            "post_objective": full_step_obj,
                            "x_rel_to_direct": full_step_x_rel,
                        }
                    )
            if kick_accepts:
                accepted_kick = min(
                    kick_accepts,
                    key=lambda row: (
                        float(row["candidate"]["e_rel_to_exact"]),
                        float(row["post_objective"]),
                    ),
                )
                selected_spec = accepted_kick["spec"]
                poses = np.asarray(accepted_kick["poses"], dtype=float).copy()
                pose_history.append(poses.copy())
                for row in reversed(attempt_rows):
                    if (
                        int(row["outer"]) == int(outer)
                        and str(row["phase"]) == "kick_gate_probe"
                        and int(row["c"]) == int(selected_spec["c"])
                        and int(row["k"]) == int(selected_spec["k"])
                    ):
                        row["selected"] = True
                        break
                history.append(
                    {
                        "outer": int(outer),
                        "phase": "kick_gate",
                        "nonlinear_objective": float(accepted_kick["post_objective"]),
                        "gap_to_direct_optimum": float(accepted_kick["post_objective"] - direct_optimum),
                        "x_rel_to_direct": float(accepted_kick["x_rel_to_direct"]),
                        "objective_improvement": float(start_obj - accepted_kick["post_objective"]),
                        "x_rel_improvement": float(start_x_rel - accepted_kick["x_rel_to_direct"]),
                        "accepted": True,
                        "selected_c": int(selected_spec["c"]),
                        "selected_k": int(selected_spec["k"]),
                        "selected_beta": 1.0,
                        "kick_gate_max_e_rel_to_exact": float(selected_spec["max_e_rel_to_exact"]),
                        "kick_gate_objective_cap_multiplier": float(selected_spec["objective_cap_multiplier"]),
                        "linear_residual_approx": float(accepted_kick["candidate"]["linear_residual_approx"]),
                        "e_rel_to_exact": float(accepted_kick["candidate"]["e_rel_to_exact"]),
                        "raw_full_step_objective": float(accepted_kick["candidate"]["raw_full_step_objective"]),
                        "last_cycle_pre_residual_ratio": float(
                            accepted_kick["candidate"]["cycle_rows"][-1]["pre_residual_ratio_vs_cycle_start"]
                        ),
                        "last_cycle_post_residual_ratio": float(
                            accepted_kick["candidate"]["cycle_rows"][-1]["post_residual_ratio_vs_cycle_start"]
                        ),
                    }
                )
                print(
                    f"outer={outer:02d} phase=kick_gate     obj={history[-1]['nonlinear_objective']:.6f} "
                    f"xrel={history[-1]['x_rel_to_direct']:.6f} "
                    f"c={selected_spec['c']} k={selected_spec['k']} beta=1.000 "
                    f"e_rel={accepted_kick['candidate']['e_rel_to_exact']:.3f}",
                    flush=True,
                )
                continue
        recovery_plan = post_kick_recovery_plan_for_preset(
            preset=preset,
            history=history,
            start_obj=start_obj,
        )
        if recovery_plan is not None:
            phase, plan = recovery_plan
        else:
            phase, plan = candidate_plan_for_phase(start_obj=start_obj, preset=preset)

        accepted = None
        for attempt_idx, spec in enumerate(plan, start=1):
            candidate = _run_candidate(
                problem=problem,
                base_poses=poses,
                exact=exact,
                inner_cycles=int(spec["c"]),
                pre_sweeps=int(spec["k"]),
                group_size=group_size,
                r_reduced=r_reduced,
                basis_source=basis_source,
            )
            beta_rows = evaluate_betas(
                problem=problem,
                base_poses=poses,
                e_hat=np.asarray(candidate["e_hat"], dtype=float),
                beta_candidates=beta_candidates,
                ref_poses=ref_poses,
            )
            selected = select_beta_from_spec(beta_rows=beta_rows, start_obj=start_obj, spec=spec)
            for beta_row in beta_rows:
                attempt_rows.append(
                    {
                        "outer": int(outer),
                        "phase": phase,
                        "attempt_idx": int(attempt_idx),
                        "start_objective": start_obj,
                        "start_x_rel_to_direct": start_x_rel,
                        "c": int(spec["c"]),
                        "k": int(spec["k"]),
                        "selector": str(spec["selector"]),
                        "beta": float(beta_row["beta"]),
                        "post_objective": float(beta_row["post_objective"]),
                        "objective_improvement": float(beta_row["objective_improvement"]),
                        "x_rel_to_direct": float(beta_row["x_rel_to_direct"]),
                        "x_rel_improvement": float(beta_row["x_rel_improvement"]),
                        "selected": bool(
                            selected is not None and np.isclose(float(beta_row["beta"]), float(selected["beta"]))
                        ),
                        "linear_residual_approx": float(candidate["linear_residual_approx"]),
                        "e_rel_to_exact": float(candidate["e_rel_to_exact"]),
                        "raw_full_step_objective": float(candidate["raw_full_step_objective"]),
                        "last_cycle_pre_residual_ratio": float(
                            candidate["cycle_rows"][-1]["pre_residual_ratio_vs_cycle_start"]
                        ),
                        "last_cycle_post_residual_ratio": float(
                            candidate["cycle_rows"][-1]["post_residual_ratio_vs_cycle_start"]
                        ),
                    }
                )
            if selected is None:
                continue
            accepted = {
                "spec": spec,
                "candidate": candidate,
                "selected": selected,
            }
            break

        if accepted is None:
            best_attempt = min(
                [row for row in attempt_rows if int(row["outer"]) == int(outer)],
                key=lambda row: float(row["post_objective"]),
            )
            history.append(
                {
                    "outer": int(outer),
                    "phase": phase,
                    "nonlinear_objective": float(best_attempt["post_objective"]),
                    "gap_to_direct_optimum": float(best_attempt["post_objective"] - direct_optimum),
                    "x_rel_to_direct": float(best_attempt["x_rel_to_direct"]),
                    "accepted": False,
                    "selected_c": int(best_attempt["c"]),
                    "selected_k": int(best_attempt["k"]),
                    "selected_beta": float(best_attempt["beta"]),
                    "status": "no_rule_acceptance",
                }
            )
            break

        poses = np.asarray(accepted["selected"]["poses"], dtype=float).copy()
        pose_history.append(poses.copy())
        history.append(
            {
                "outer": int(outer),
                "phase": phase,
                "nonlinear_objective": float(accepted["selected"]["post_objective"]),
                "gap_to_direct_optimum": float(accepted["selected"]["post_objective"] - direct_optimum),
                "x_rel_to_direct": float(accepted["selected"]["x_rel_to_direct"]),
                "objective_improvement": float(accepted["selected"]["objective_improvement"]),
                "x_rel_improvement": float(accepted["selected"]["x_rel_improvement"]),
                "accepted": True,
                "selected_c": int(accepted["spec"]["c"]),
                "selected_k": int(accepted["spec"]["k"]),
                "selected_beta": float(accepted["selected"]["beta"]),
                "linear_residual_approx": float(accepted["candidate"]["linear_residual_approx"]),
                "e_rel_to_exact": float(accepted["candidate"]["e_rel_to_exact"]),
                "raw_full_step_objective": float(accepted["candidate"]["raw_full_step_objective"]),
                "last_cycle_pre_residual_ratio": float(
                    accepted["candidate"]["cycle_rows"][-1]["pre_residual_ratio_vs_cycle_start"]
                ),
                "last_cycle_post_residual_ratio": float(
                    accepted["candidate"]["cycle_rows"][-1]["post_residual_ratio_vs_cycle_start"]
                ),
            }
        )
        print(
            f"outer={outer:02d} phase={phase:<13} obj={history[-1]['nonlinear_objective']:.6f} "
            f"xrel={history[-1]['x_rel_to_direct']:.6f} "
            f"c={accepted['spec']['c']} k={accepted['spec']['k']} beta={accepted['selected']['beta']:.3f}",
            flush=True,
        )

    return {
        "config": {
            "num_outer": int(num_outer),
            "basis_source": basis_source,
            "group_size": int(group_size),
            "r_reduced": int(r_reduced),
            "preset": preset,
            "beta_candidates": [float(beta) for beta in beta_candidates],
        },
        "history": history,
        "attempt_rows": attempt_rows,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
    }


def parse_beta_candidates(text: str) -> list[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def supported_policy_presets() -> tuple[str, ...]:
    return (
        "bridge_v1",
        "bridge_v2",
        "bridge_v3",
        "formal_v1",
        "formal_v2",
        "formal_v3",
        "formal_v4",
        "formal_v5",
        "formal_v6",
    )


def build_default_problem():
    return parse_g2o_se2(G2O_PATH)
