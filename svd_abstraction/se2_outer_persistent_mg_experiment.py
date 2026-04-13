from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.persistent_state_exact_coarse_experiment import mean_vector
from svd_abstraction.residual_abstraction import SVDResidualAbstraction
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import reset_residual_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import var_slices
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import exact_local_solve
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import pairwise_pose_gap
from svd_abstraction.se2_newton_vs_persistent_mg_experiment import pose_metrics
from svd_abstraction.se2_utils import apply_pose_deltas
from svd_abstraction.se2_utils import build_linearized_local_graph
from svd_abstraction.se2_utils import build_se2_problem
from svd_abstraction.se2_utils import poses_to_nodes


OUTPUT_DIR = pathlib.Path(
    "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def basis_suffix(basis_source: str) -> str:
    if basis_source == "joint_covariance":
        return ""
    if basis_source == "message_conditioned_information":
        return "_conditioned_information"
    return f"_{basis_source}"


def xy_rel_to_ref(poses: np.ndarray, ref_poses: np.ndarray) -> float:
    xy = np.asarray(poses, dtype=float)[:, :2].reshape(-1)
    xy_ref = np.asarray(ref_poses, dtype=float)[:, :2].reshape(-1)
    denom = max(float(np.linalg.norm(xy_ref)), 1e-15)
    return float(np.linalg.norm(xy - xy_ref) / denom)


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


def grouped_order_ids(base_poses: np.ndarray, template_graph, group_size: int) -> list[list[int]]:
    return group_list(
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


def rebuild_level_projection(level: SVDResidualAbstraction) -> None:
    total_full_dim = sum(var.dofs for var in level.base_graph.var_nodes[: level.base_graph.n_var_nodes])
    level.P = np.zeros((total_full_dim, level.total_reduced_dim), dtype=float)
    for group_var_ids, basis, reduced_slice in zip(level.groups, level.Bs, level.group_reduced_slices):
        full_indices = []
        for var_id in group_var_ids:
            sl = level.var_slices[int(var_id)]
            full_indices.extend(range(sl.start, sl.stop))
        level.P[np.asarray(full_indices, dtype=int), reduced_slice] = np.asarray(basis, dtype=float)
    level.bases_initialized = True
    level._refresh_group_maps()
    level.coarse_graph = None
    level.coarse_var_nodes = []
    level._coarse_prior_eta_terms = []
    level._coarse_factor_eta_terms = []


def blend_level_basis_with_previous(
    level: SVDResidualAbstraction,
    prev_level: SVDResidualAbstraction,
    current_weight: float,
) -> None:
    weight = float(np.clip(current_weight, 0.0, 1.0))
    new_bases: list[np.ndarray] = []
    for cur_basis, prev_basis in zip(level.Bs, prev_level.Bs):
        if cur_basis.shape != prev_basis.shape:
            new_bases.append(np.asarray(cur_basis, dtype=float).copy())
            continue
        mix = (
            weight * (np.asarray(cur_basis, dtype=float) @ np.asarray(cur_basis, dtype=float).T)
            + (1.0 - weight) * (np.asarray(prev_basis, dtype=float) @ np.asarray(prev_basis, dtype=float).T)
        )
        eigvals, eigvecs = np.linalg.eigh(0.5 * (mix + mix.T))
        order = np.argsort(eigvals)[::-1]
        r_local = cur_basis.shape[1]
        new_bases.append(np.asarray(eigvecs[:, order[:r_local]], dtype=float))
    level.Bs = new_bases
    rebuild_level_projection(level)


def snapshot_message_state(graph) -> dict[int, list[dict[str, np.ndarray]]]:
    state: dict[int, list[dict[str, np.ndarray]]] = {}
    for factor in graph.factors[: graph.n_factor_nodes]:
        state[int(factor.factorID)] = [
            {
                "eta": np.asarray(msg.eta, dtype=float).copy(),
                "lam": np.asarray(msg.lam, dtype=float).copy(),
            }
            for msg in factor.messages
        ]
    return state


def restore_message_state(graph, state: dict[int, list[dict[str, np.ndarray]]], keep_eta: bool, keep_lam: bool) -> None:
    for factor in graph.factors[: graph.n_factor_nodes]:
        saved = state.get(int(factor.factorID))
        if saved is None or len(saved) != len(factor.messages):
            continue
        for msg, saved_msg in zip(factor.messages, saved):
            if keep_eta and saved_msg["eta"].shape == np.asarray(msg.eta).shape:
                msg.eta = np.asarray(saved_msg["eta"], dtype=float).copy()
            else:
                msg.eta = np.zeros_like(msg.eta)
            if keep_lam and saved_msg["lam"].shape == np.asarray(msg.lam).shape:
                msg.lam = np.asarray(saved_msg["lam"], dtype=float).copy()
            else:
                msg.lam = np.zeros_like(msg.lam)


def refresh_residual_graph_from_template(
    residual_graph,
    template_graph,
    x: np.ndarray,
    slices: dict[int, slice],
    *,
    keep_eta: bool,
    keep_lam: bool,
    settle_sweeps: int,
) -> None:
    old_state = snapshot_message_state(residual_graph)

    residual_graph.var_heap.clear()
    residual_graph.var_residual.clear()

    for orig_var, var in zip(
        template_graph.var_nodes[: template_graph.n_var_nodes],
        residual_graph.var_nodes[: residual_graph.n_var_nodes],
    ):
        xi = np.asarray(x[slices[int(orig_var.variableID)]], dtype=float).reshape(-1)
        var.prior.lam = np.asarray(orig_var.prior.lam, dtype=float).copy()
        var.prior.eta = (
            np.asarray(orig_var.prior.eta, dtype=float).reshape(-1)
            - np.asarray(orig_var.prior.lam, dtype=float) @ xi
        )
        var.mu = np.zeros(var.dofs, dtype=float)
        var.belief.lam = np.asarray(var.prior.lam, dtype=float).copy()
        var.belief.eta = np.zeros(var.dofs, dtype=float)
        var.Sigma = np.eye(var.dofs, dtype=float) * 1e12
        var.residual = np.zeros(var.dofs, dtype=float)

    for orig_factor, factor in zip(
        template_graph.factors[: template_graph.n_factor_nodes],
        residual_graph.factors[: residual_graph.n_factor_nodes],
    ):
        local_x = np.concatenate(
            [
                np.asarray(x[slices[int(orig_var.variableID)]], dtype=float).reshape(-1)
                for orig_var in orig_factor.adj_var_nodes
            ]
        )
        abs_eta, abs_lam = orig_factor.compute_factor_absolute(update_self=False)
        factor.factor.lam = np.asarray(abs_lam, dtype=float).copy()
        factor.factor.eta = (
            np.asarray(abs_eta, dtype=float).reshape(-1)
            - np.asarray(abs_lam, dtype=float) @ local_x
        )
        factor.residual = None
        for idx, adj_var in enumerate(factor.adj_var_nodes):
            factor.adj_beliefs[idx].eta = np.zeros_like(factor.adj_beliefs[idx].eta)
            factor.adj_beliefs[idx].lam = np.asarray(adj_var.prior.lam, dtype=float).copy()

    restore_message_state(
        residual_graph,
        state=old_state,
        keep_eta=keep_eta,
        keep_lam=keep_lam,
    )
    residual_graph.update_all_beliefs()
    for _ in range(int(settle_sweeps)):
        residual_graph.synchronous_iteration(local_relin=False)


def build_level(
    base_graph,
    groups: list[list[int]],
    *,
    r_reduced: int,
    basis_source: str,
    frozen_basis_level: SVDResidualAbstraction | None,
    coarse_state: dict[int, list[dict[str, np.ndarray]]] | None,
    carry_coarse_state: bool,
    coarse_settle_sweeps: int,
) -> SVDResidualAbstraction:
    level = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups,
        r_reduced=r_reduced,
        basis_source=basis_source,
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
    if carry_coarse_state and coarse_state is not None:
        restore_message_state(
            level.coarse_graph,
            state=coarse_state,
            keep_eta=True,
            keep_lam=True,
        )
        level.coarse_graph.update_all_beliefs()
        for _ in range(int(coarse_settle_sweeps)):
            level.coarse_graph.synchronous_iteration(local_relin=False)
    return level


def run_direct_baseline_with_traj(problem, num_outer: int) -> dict[str, Any]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    history: list[dict[str, float]] = [{"outer": 0, **pose_metrics(problem, poses)}]
    for outer in range(1, int(num_outer) + 1):
        step = exact_local_solve(problem, poses)
        poses = np.asarray(step["next_poses"], dtype=float).copy()
        pose_history.append(poses.copy())
        history.append(
            {
                "outer": int(outer),
                "linear_step_norm": float(step["e_norm"]),
                "linear_residual_norm": float(step["linear_residual_norm"]),
                **pose_metrics(problem, poses),
            }
        )
    return {
        "history": history,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
    }


def run_outer_persistent_mg(
    problem,
    *,
    num_outer: int,
    inner_cycles: int,
    pre_sweeps: int,
    group_size: int,
    r_reduced: int,
    basis_source: str,
    carry_message_eta: bool,
    carry_message_lam: bool,
    carry_coarse_state: bool,
    settle_sweeps: int,
    coarse_settle_sweeps: int,
    freeze_basis_from_outer0: bool,
    carry_basis_prev_outer: bool,
    blend_current_basis_weight: float | None,
    ref_poses: np.ndarray | None,
) -> dict[str, Any]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    history: list[dict[str, float]] = []
    trajectory_key = {
        "carry_message_eta": bool(carry_message_eta),
        "carry_message_lam": bool(carry_message_lam),
        "carry_coarse_state": bool(carry_coarse_state),
        "settle_sweeps": int(settle_sweeps),
        "coarse_settle_sweeps": int(coarse_settle_sweeps),
        "freeze_basis_from_outer0": bool(freeze_basis_from_outer0),
        "carry_basis_prev_outer": bool(carry_basis_prev_outer),
        "blend_current_basis_weight": (
            float(blend_current_basis_weight) if blend_current_basis_weight is not None else None
        ),
    }

    residual_graph = None
    groups = None
    frozen_basis_level = None
    prev_basis_level = None
    coarse_state = None

    initial_row = {"outer": 0, **trajectory_key, **pose_metrics(problem, poses)}
    if ref_poses is not None:
        initial_row["xy_rel_to_direct"] = float(xy_rel_to_ref(poses, ref_poses))
    history.append(initial_row)

    for outer in range(1, int(num_outer) + 1):
        exact = exact_local_solve(problem, poses)
        template_graph = build_linearized_local_graph(problem, poses)
        zero = np.zeros(template_graph.n_var_nodes * 3, dtype=float)

        if residual_graph is None:
            residual_graph = build_linearized_local_graph(problem, poses)
            reset_residual_graph(residual_graph, template_graph, zero, var_slices(template_graph))
            groups = grouped_order_ids(poses, template_graph, group_size=group_size)
        else:
            refresh_residual_graph_from_template(
                residual_graph,
                template_graph,
                zero,
                var_slices(template_graph),
                keep_eta=carry_message_eta,
                keep_lam=carry_message_lam,
                settle_sweeps=settle_sweeps,
            )

        level = build_level(
            base_graph=residual_graph,
            groups=groups,
            r_reduced=r_reduced,
            basis_source=basis_source,
            frozen_basis_level=(
                frozen_basis_level
                if freeze_basis_from_outer0
                else None
                if (carry_basis_prev_outer and blend_current_basis_weight is not None and prev_basis_level is not None)
                else prev_basis_level
                if carry_basis_prev_outer
                else None
            ),
            coarse_state=coarse_state,
            carry_coarse_state=carry_coarse_state,
            coarse_settle_sweeps=coarse_settle_sweeps,
        )
        if (
            carry_basis_prev_outer
            and blend_current_basis_weight is not None
            and prev_basis_level is not None
        ):
            blend_level_basis_with_previous(
                level,
                prev_level=prev_basis_level,
                current_weight=blend_current_basis_weight,
            )
            level.build_coarse_graph(force=True)
        if freeze_basis_from_outer0 and frozen_basis_level is None:
            frozen_basis_level = level
        if carry_basis_prev_outer:
            prev_basis_level = level

        inner_rows = []
        for cyc in range(1, int(inner_cycles) + 1):
            for _ in range(int(pre_sweeps)):
                residual_graph.synchronous_iteration()
            level.update_coarse_residual_eta()
            delta_z = level.direct_solve_coarse_graph()
            delta_e = level.prolongate(delta_z)
            x_before = mean_vector(residual_graph)
            x_after = x_before + delta_e
            offset = 0
            for var in residual_graph.var_nodes[: residual_graph.n_var_nodes]:
                sl = slice(offset, offset + var.dofs)
                var.mu = np.asarray(x_after[sl], dtype=float).copy()
                var.belief.eta = var.belief.lam @ var.mu
                for factor in var.adj_factors:
                    belief_ix = factor.var_index[var.variableID]
                    factor.adj_beliefs[belief_ix].eta = np.asarray(var.belief.eta, dtype=float).copy()
                    factor.adj_beliefs[belief_ix].lam = np.asarray(var.belief.lam, dtype=float).copy()
                offset += var.dofs
            e_now = mean_vector(residual_graph)
            inner_rows.append(
                {
                    "inner_cycle": int(cyc),
                    "e_rel_to_exact": float(
                        np.linalg.norm(e_now - exact["e_star"]) / max(np.linalg.norm(exact["e_star"]), 1e-15)
                    ),
                    "linear_residual_norm": float(np.linalg.norm(exact["eta"] - exact["lam"] @ e_now)),
                    "e_norm": float(np.linalg.norm(e_now)),
                }
            )

        e_hat = mean_vector(residual_graph)
        poses = apply_pose_deltas(poses, e_hat)
        pose_history.append(np.asarray(poses, dtype=float).copy())
        coarse_state = snapshot_message_state(level.coarse_graph) if carry_coarse_state else None

        row = {
            "outer": int(outer),
            **trajectory_key,
            "inner_cycles": int(inner_cycles),
            "pre_sweeps": int(pre_sweeps),
            "e_hat_norm": float(np.linalg.norm(e_hat)),
            "e_star_norm": float(np.linalg.norm(exact["e_star"])),
            "e_rel_to_exact": float(np.linalg.norm(e_hat - exact["e_star"]) / max(np.linalg.norm(exact["e_star"]), 1e-15)),
            "linear_residual_exact": float(exact["linear_residual_norm"]),
            "linear_residual_approx": float(np.linalg.norm(exact["eta"] - exact["lam"] @ e_hat)),
            "next_pose_gap_to_exact": float(pairwise_pose_gap(np.asarray(exact["next_poses"], dtype=float), poses)),
            "next_objective_gap_to_exact": float(
                pose_metrics(problem, poses)["nonlinear_objective"] - pose_metrics(problem, np.asarray(exact["next_poses"], dtype=float))["nonlinear_objective"]
            ),
            "num_groups": int(len(groups)),
            "coarse_dim": int(level.total_reduced_dim),
            **pose_metrics(problem, poses),
        }
        if ref_poses is not None:
            row["xy_rel_to_direct"] = float(xy_rel_to_ref(poses, ref_poses))
        history.append(row)
        print(
            f"outer={outer:02d} obj={row['nonlinear_objective']:.6f} "
            f"xrel={row.get('xy_rel_to_direct', float('nan')):.6f} "
            f"e_rel={row['e_rel_to_exact']:.6f} carry_eta={carry_message_eta} "
            f"carry_lam={carry_message_lam} coarse={carry_coarse_state}",
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
            **trajectory_key,
        },
        "history": history,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--inner-cycles", type=int, default=1)
    parser.add_argument("--pre-sweeps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--basis-source", type=str, default="message_conditioned_information")
    parser.add_argument("--carry-message-eta", action="store_true")
    parser.add_argument("--carry-message-lam", action="store_true")
    parser.add_argument("--carry-coarse-state", action="store_true")
    parser.add_argument("--settle-sweeps", type=int, default=0)
    parser.add_argument("--coarse-settle-sweeps", type=int, default=0)
    parser.add_argument("--freeze-basis-from-outer0", action="store_true")
    parser.add_argument("--carry-basis-prev-outer", action="store_true")
    parser.add_argument("--blend-current-basis-weight", type=float, default=None)
    args = parser.parse_args()

    problem = build_se2_problem(
        n=args.n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        seed=args.seed,
    )
    direct = run_direct_baseline_with_traj(problem, num_outer=args.num_outer)
    ref_poses = np.asarray(direct["final_poses"], dtype=float)
    mg = run_outer_persistent_mg(
        problem,
        num_outer=args.num_outer,
        inner_cycles=args.inner_cycles,
        pre_sweeps=args.pre_sweeps,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
        basis_source=args.basis_source,
        carry_message_eta=args.carry_message_eta,
        carry_message_lam=args.carry_message_lam,
        carry_coarse_state=args.carry_coarse_state,
        settle_sweeps=args.settle_sweeps,
        coarse_settle_sweeps=args.coarse_settle_sweeps,
        freeze_basis_from_outer0=args.freeze_basis_from_outer0,
        carry_basis_prev_outer=args.carry_basis_prev_outer,
        blend_current_basis_weight=args.blend_current_basis_weight,
        ref_poses=ref_poses,
    )

    stem = (
        f"se2_outer_persistent_mg_n{args.n}_seed{args.seed}"
        f"_outer{args.num_outer}_c{args.inner_cycles}_k{args.pre_sweeps}"
        f"_carryeta{int(args.carry_message_eta)}"
        f"_carrylam{int(args.carry_message_lam)}"
        f"_carrycoarse{int(args.carry_coarse_state)}"
        f"_settle{args.settle_sweeps}"
        f"_coarsesettle{args.coarse_settle_sweeps}"
        f"_freezebasis{int(args.freeze_basis_from_outer0)}"
        f"_carryprevbasis{int(args.carry_basis_prev_outer)}"
        f"_blendcur{str(args.blend_current_basis_weight).replace('.', 'p') if args.blend_current_basis_weight is not None else 'none'}"
        f"{basis_suffix(args.basis_source)}"
    )
    json_path = OUTPUT_DIR / f"{stem}.json"
    direct_csv = OUTPUT_DIR / f"{stem}_direct.csv"
    mg_csv = OUTPUT_DIR / f"{stem}_mg.csv"
    traj_npz = OUTPUT_DIR / f"{stem}_trajectories.npz"

    payload = {
        "config": {
            "n": int(args.n),
            "seed": int(args.seed),
            "num_outer": int(args.num_outer),
            "inner_cycles": int(args.inner_cycles),
            "pre_sweeps": int(args.pre_sweeps),
            "group_size": int(args.group_size),
            "r_reduced": int(args.r_reduced),
            "basis_source": args.basis_source,
            "carry_message_eta": bool(args.carry_message_eta),
            "carry_message_lam": bool(args.carry_message_lam),
            "carry_coarse_state": bool(args.carry_coarse_state),
            "settle_sweeps": int(args.settle_sweeps),
            "coarse_settle_sweeps": int(args.coarse_settle_sweeps),
            "freeze_basis_from_outer0": bool(args.freeze_basis_from_outer0),
            "carry_basis_prev_outer": bool(args.carry_basis_prev_outer),
            "blend_current_basis_weight": args.blend_current_basis_weight,
        },
        "direct_newton": direct,
        "outer_persistent_mg": mg,
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(direct["history"], direct_csv)
    save_csv(mg["history"], mg_csv)
    np.savez_compressed(
        traj_npz,
        direct_pose_history=np.asarray(direct["pose_history"], dtype=float),
        mg_pose_history=np.asarray(mg["pose_history"], dtype=float),
        direct_final_poses=np.asarray(direct["final_poses"], dtype=float),
        mg_final_poses=np.asarray(mg["final_poses"], dtype=float),
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "direct_csv": str(direct_csv),
                "mg_csv": str(mg_csv),
                "traj_npz": str(traj_npz),
                "mg_final": mg["history"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
