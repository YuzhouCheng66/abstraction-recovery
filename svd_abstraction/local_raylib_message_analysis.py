from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

LOCAL_RAYLIB_ROOT = SCRIPT_DIR / "raylib_gbp_local"
if str(LOCAL_RAYLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_RAYLIB_ROOT))

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(1, str(WORKSPACE_ROOT))

from svd_abstraction.raylib_local_eta_prolongation_validation import build_slam_graph


def _message_diag_trace(factor, belief_ix):
    msg_lam = factor.messages[belief_ix].lam
    return float(np.sum(np.abs(np.diagonal(msg_lam))))


def _factor_diag_trace(factor, belief_ix, dofs):
    start = belief_ix * dofs
    stop = start + dofs
    lam_block = factor.factor.lam[start:stop, start:stop]
    return float(np.sum(np.abs(np.diagonal(lam_block))))


def snapshot_message_state(graph):
    state = {}
    for var in graph.var_nodes[: graph.n_var_nodes]:
        entries = []
        for factor in var.adj_factors:
            belief_ix = factor.adj_vIDs.index(var.variableID)
            entries.append(
                {
                    "factor_id": factor.factorID,
                    "eta": factor.messages[belief_ix].eta.copy(),
                    "lam": factor.messages[belief_ix].lam.copy(),
                    "msg_trace": _message_diag_trace(factor, belief_ix),
                    "factor_trace": _factor_diag_trace(factor, belief_ix, var.dofs),
                }
            )
        state[var.variableID] = entries
    return state


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize_iteration(prev_state, curr_state):
    eta_norms = []
    lam_traces = []
    delta_eta_norms = []
    delta_lam_norms = []

    weight_l1_uniform = []
    weight_l1_msg = []
    weight_l1_factor = []
    negative_actual_count = 0
    usable_weight_sets = 0

    for var_id, prev_entries in prev_state.items():
        curr_entries = curr_state[var_id]
        prev_by_factor = {entry["factor_id"]: entry for entry in prev_entries}
        curr_by_factor = {entry["factor_id"]: entry for entry in curr_entries}

        delta_messages = []
        msg_weights = []
        factor_weights = []

        for factor_id, prev_entry in prev_by_factor.items():
            curr_entry = curr_by_factor[factor_id]
            eta_norms.append(float(np.linalg.norm(curr_entry["eta"])))
            lam_traces.append(curr_entry["msg_trace"])
            delta_eta = curr_entry["eta"] - prev_entry["eta"]
            delta_lam = curr_entry["lam"] - prev_entry["lam"]
            delta_eta_norms.append(float(np.linalg.norm(delta_eta)))
            delta_lam_norms.append(float(np.linalg.norm(delta_lam)))

            delta_messages.append(delta_eta)
            msg_weights.append(max(prev_entry["msg_trace"], 1e-12))
            factor_weights.append(max(prev_entry["factor_trace"], 1e-12))

        if len(delta_messages) <= 1:
            continue

        total_delta = np.sum(delta_messages, axis=0)
        total_norm_sq = float(total_delta @ total_delta)
        if total_norm_sq < 1e-14:
            continue

        actual_weights = np.array(
            [float(delta @ total_delta) / total_norm_sq for delta in delta_messages],
            dtype=float,
        )
        uniform_weights = np.full(len(delta_messages), 1.0 / len(delta_messages), dtype=float)
        msg_weights = np.asarray(msg_weights, dtype=float)
        msg_weights = msg_weights / np.sum(msg_weights)
        factor_weights = np.asarray(factor_weights, dtype=float)
        factor_weights = factor_weights / np.sum(factor_weights)

        weight_l1_uniform.append(float(np.sum(np.abs(actual_weights - uniform_weights))))
        weight_l1_msg.append(float(np.sum(np.abs(actual_weights - msg_weights))))
        weight_l1_factor.append(float(np.sum(np.abs(actual_weights - factor_weights))))
        negative_actual_count += int(np.sum(actual_weights < -1e-9))
        usable_weight_sets += 1

    return {
        "eta_norm_mean": float(np.mean(eta_norms)) if eta_norms else np.nan,
        "eta_norm_median": float(np.median(eta_norms)) if eta_norms else np.nan,
        "lam_trace_mean": float(np.mean(lam_traces)) if lam_traces else np.nan,
        "delta_eta_mean": float(np.mean(delta_eta_norms)) if delta_eta_norms else np.nan,
        "delta_eta_median": float(np.median(delta_eta_norms)) if delta_eta_norms else np.nan,
        "delta_lam_mean": float(np.mean(delta_lam_norms)) if delta_lam_norms else np.nan,
        "delta_lam_median": float(np.median(delta_lam_norms)) if delta_lam_norms else np.nan,
        "corr_eta_lam": _safe_corr(eta_norms, lam_traces),
        "corr_delta_eta_delta_lam": _safe_corr(delta_eta_norms, delta_lam_norms),
        "weight_l1_uniform": float(np.mean(weight_l1_uniform)) if weight_l1_uniform else np.nan,
        "weight_l1_msg": float(np.mean(weight_l1_msg)) if weight_l1_msg else np.nan,
        "weight_l1_factor": float(np.mean(weight_l1_factor)) if weight_l1_factor else np.nan,
        "negative_actual_weights": negative_actual_count,
        "usable_weight_sets": usable_weight_sets,
    }


def print_summary_line(iteration, stats):
    print(
        f"iter {iteration:>3d} | "
        f"eta_mean {stats['eta_norm_mean']:.3e} | "
        f"lam_mean {stats['lam_trace_mean']:.3e} | "
        f"deta_med {stats['delta_eta_median']:.3e} | "
        f"dlam_med {stats['delta_lam_median']:.3e} | "
        f"corr(eta,lam) {stats['corr_eta_lam']:.3f} | "
        f"corr(deta,dlam) {stats['corr_delta_eta_delta_lam']:.3f} | "
        f"L1[u/msg/f] {stats['weight_l1_uniform']:.3f}/{stats['weight_l1_msg']:.3f}/{stats['weight_l1_factor']:.3f} | "
        f"neg {stats['negative_actual_weights']}/{stats['usable_weight_sets']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--report-iters",
        type=int,
        nargs="*",
        default=[1, 2, 5, 10, 20, 50, 100],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    graph = build_slam_graph(n=args.n, seed=args.seed)
    prev_state = snapshot_message_state(graph)

    report_iters = set(args.report_iters)
    for iteration in range(1, args.iters + 1):
        graph.synchronous_iteration(level=0)
        curr_state = snapshot_message_state(graph)
        stats = summarize_iteration(prev_state, curr_state)
        if iteration in report_iters:
            print_summary_line(iteration, stats)
        prev_state = curr_state


if __name__ == "__main__":
    main()
