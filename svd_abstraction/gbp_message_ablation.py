"""Ablations for message-state management in geometric GBP multigrid.

We study how to manage Gaussian BP messages across V-cycles on a fixed
geometric hierarchy. The focus is the long 1D chain where the current
fresh-direct smoother fails to achieve "jump propagation".

The main ablations are:
  * whether precision messages p are reset, carried, or fixed/preconverged
  * whether eta/mean messages h are reset, carried, damped, or projected

The current linear xy chain decouples exactly into two identical scalar
channels, so scalar ablations are faithful for the chain and much cheaper.
We then re-run the best candidates on the full 2D block system.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.block_gabp import BlockGaBPLevel
from svd_abstraction.block_gabp import BlockGaBPSmootherState
from svd_abstraction.block_gabp import _sum_matrix_messages
from svd_abstraction.block_gabp import _sum_vector_messages
from svd_abstraction.block_gabp import block_gabp_eta_sweep
from svd_abstraction.block_gabp import block_gabp_full_sweep
from svd_abstraction.block_gabp import block_gabp_mean
from svd_abstraction.block_gabp import block_gabp_precision_sweep
from svd_abstraction.block_gabp import build_block_gabp_level
from svd_abstraction.block_gabp import converge_block_gabp_precision
from svd_abstraction.chain_same_system_benchmark import build_line_levels_from_matrix
from svd_abstraction.chain_same_system_benchmark import extract_scalar_channel
from svd_abstraction.chain_same_system_benchmark import system_from_chain
from svd_abstraction.poisson_multigrid_benchmark import Level
from svd_abstraction.poisson_multigrid_benchmark import weighted_jacobi


@dataclass
class ManagedState:
    p_msg: np.ndarray
    h_msg: np.ndarray
    prev_rhs: np.ndarray | None = None


def zero_state(level: BlockGaBPLevel) -> ManagedState:
    return ManagedState(
        p_msg=np.zeros((level.src.size, level.block_dofs, level.block_dofs), dtype=float),
        h_msg=np.zeros((level.src.size, level.block_dofs), dtype=float),
        prev_rhs=None,
    )


def clone_state(state: ManagedState) -> ManagedState:
    return ManagedState(
        p_msg=np.array(state.p_msg, copy=True),
        h_msg=np.array(state.h_msg, copy=True),
        prev_rhs=None if state.prev_rhs is None else np.array(state.prev_rhs, copy=True),
    )


def global_project_h(level: BlockGaBPLevel, h_msg: np.ndarray, rhs_new: np.ndarray, rhs_prev: np.ndarray | None) -> np.ndarray:
    if rhs_prev is None:
        return np.zeros_like(h_msg)
    denom = float(rhs_prev @ rhs_prev)
    if denom <= 1e-30:
        return np.zeros_like(h_msg)
    alpha = float(rhs_new @ rhs_prev) / denom
    alpha = float(np.clip(alpha, 0.0, 1.5))
    return alpha * h_msg


def local_project_h(level: BlockGaBPLevel, h_msg: np.ndarray, rhs_new: np.ndarray, rhs_prev: np.ndarray | None) -> np.ndarray:
    if rhs_prev is None:
        return np.zeros_like(h_msg)

    new_blocks = rhs_new.reshape(level.n_nodes, level.block_dofs)
    old_blocks = rhs_prev.reshape(level.n_nodes, level.block_dofs)
    scales = np.zeros(level.n_nodes, dtype=float)
    for i in range(level.n_nodes):
        denom = float(old_blocks[i] @ old_blocks[i])
        if denom <= 1e-30:
            scales[i] = 0.0
            continue
        scales[i] = float(np.clip((new_blocks[i] @ old_blocks[i]) / denom, 0.0, 1.5))

    projected = np.array(h_msg, copy=True)
    for edge_idx, src in enumerate(level.src.tolist()):
        projected[edge_idx] *= scales[src]
    return projected


def warmup_precision(level: BlockGaBPLevel, sweeps: int) -> np.ndarray:
    p_msg = np.zeros((level.src.size, level.block_dofs, level.block_dofs), dtype=float)
    for _ in range(sweeps):
        p_msg = block_gabp_precision_sweep(level, p_msg)
    return p_msg


def parse_variant(variant: str) -> tuple[str, bool, tuple[str, int | None]]:
    """Return (core_variant, reset_each_cycle, p_mode)."""
    reset_each_cycle = variant.startswith("cycle_")
    core = variant[len("cycle_") :] if reset_each_cycle else variant

    if core.startswith("fixed_p1_"):
        return "fixed_p_" + core[len("fixed_p1_") :], reset_each_cycle, ("warm", 1)
    if core.startswith("fixed_p2_"):
        return "fixed_p_" + core[len("fixed_p2_") :], reset_each_cycle, ("warm", 2)
    if core.startswith("fixed_p4_"):
        return "fixed_p_" + core[len("fixed_p4_") :], reset_each_cycle, ("warm", 4)
    if core.startswith("fixed_p_"):
        return core, reset_each_cycle, ("converged", None)
    return core, reset_each_cycle, ("none", None)


def init_state_for_rhs(
    level: BlockGaBPLevel,
    cache: ManagedState,
    rhs: np.ndarray,
    variant: str,
    p_fixed: np.ndarray | None = None,
) -> ManagedState:
    state = clone_state(cache)

    if variant == "fresh_direct":
        state = zero_state(level)
    elif variant == "carry_p_zero_h":
        state.h_msg.fill(0.0)
    elif variant == "fixed_p_zero_h":
        if p_fixed is None:
            raise ValueError("fixed_p_zero_h requires p_fixed")
        state.p_msg = np.array(p_fixed, copy=True)
        state.h_msg.fill(0.0)
    elif variant == "fixed_p_keep_h":
        if p_fixed is None:
            raise ValueError("fixed_p_keep_h requires p_fixed")
        state.p_msg = np.array(p_fixed, copy=True)
    elif variant == "fixed_p_damp_h_05":
        if p_fixed is None:
            raise ValueError("fixed_p_damp_h_05 requires p_fixed")
        state.p_msg = np.array(p_fixed, copy=True)
        state.h_msg *= 0.5
    elif variant == "fixed_p_project_h_global":
        if p_fixed is None:
            raise ValueError("fixed_p_project_h_global requires p_fixed")
        state.p_msg = np.array(p_fixed, copy=True)
        state.h_msg = global_project_h(level, state.h_msg, rhs, state.prev_rhs)
    elif variant == "fixed_p_project_h_local":
        if p_fixed is None:
            raise ValueError("fixed_p_project_h_local requires p_fixed")
        state.p_msg = np.array(p_fixed, copy=True)
        state.h_msg = local_project_h(level, state.h_msg, rhs, state.prev_rhs)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return state


def smooth_once(
    level_mat: Level,
    level_gabp: BlockGaBPLevel,
    cache: ManagedState,
    x: np.ndarray,
    b: np.ndarray,
    sweeps: int,
    variant: str,
    p_fixed: np.ndarray | None = None,
) -> tuple[np.ndarray, ManagedState]:
    rhs = b - level_mat.a @ x
    state = init_state_for_rhs(level_gabp, cache, rhs, variant, p_fixed=p_fixed)

    if variant.startswith("fixed_p_"):
        for _ in range(sweeps):
            state.h_msg = block_gabp_eta_sweep(level_gabp, state.p_msg, state.h_msg, rhs)
    else:
        for _ in range(sweeps):
            block_gabp_full_sweep(level_gabp, state, rhs)

    correction = block_gabp_mean(level_gabp, state.p_msg, state.h_msg, rhs)
    state.prev_rhs = np.array(rhs, copy=True)
    return x + correction, state


def v_cycle_managed(
    levels: list[Level],
    gabp_levels: list[BlockGaBPLevel],
    caches: list[ManagedState],
    p_fixed_list: list[np.ndarray | None],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    pre_sweeps: int,
    post_sweeps: int,
    variant: str,
) -> np.ndarray:
    level = levels[level_idx]
    if level_idx == len(levels) - 1:
        from scipy.sparse.linalg import spsolve

        return spsolve(level.a, b)

    x, caches[level_idx] = smooth_once(
        level,
        gabp_levels[level_idx],
        caches[level_idx],
        x,
        b,
        pre_sweeps,
        variant,
        p_fixed=p_fixed_list[level_idx],
    )
    residual = b - level.a @ x
    coarse_rhs = level.r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_managed(
        levels,
        gabp_levels,
        caches,
        p_fixed_list,
        level_idx + 1,
        coarse_error,
        coarse_rhs,
        pre_sweeps,
        post_sweeps,
        variant,
    )
    x = x + level.p @ coarse_error
    x, caches[level_idx] = smooth_once(
        level,
        gabp_levels[level_idx],
        caches[level_idx],
        x,
        b,
        post_sweeps,
        variant,
        p_fixed=p_fixed_list[level_idx],
    )
    return x


def run_mg_variant(
    levels: list[Level],
    block_dofs: int,
    b: np.ndarray,
    x_star: np.ndarray,
    variant: str,
    pre_sweeps: int,
    post_sweeps: int,
    max_cycles: int,
) -> dict[str, object]:
    core_variant, reset_each_cycle, p_mode = parse_variant(variant)
    gabp_levels = [build_block_gabp_level(level.a, block_dofs=block_dofs) for level in levels[:-1]]
    caches = [zero_state(level) for level in gabp_levels]
    p_fixed_list: list[np.ndarray | None] = [None] * len(gabp_levels)

    if p_mode[0] == "converged":
        for i, level in enumerate(gabp_levels):
            p_fixed, _ = converge_block_gabp_precision(level, tol=1e-10, max_iters=1000)
            p_fixed_list[i] = p_fixed
    elif p_mode[0] == "warm":
        for i, level in enumerate(gabp_levels):
            p_fixed_list[i] = warmup_precision(level, p_mode[1])

    x = np.zeros_like(b)
    initial = float(np.linalg.norm(x - x_star))
    error_history = [initial]
    for cycle in range(1, max_cycles + 1):
        if reset_each_cycle:
            caches = [zero_state(level) for level in gabp_levels]
        x = v_cycle_managed(
            levels,
            gabp_levels,
            caches,
            p_fixed_list,
            0,
            x,
            b,
            pre_sweeps,
            post_sweeps,
            core_variant,
        )
        error_history.append(float(np.linalg.norm(x - x_star)))
    return {"iterations": max_cycles, "x": x, "error_history": error_history}


def damping_table_chain(n: int) -> None:
    nodes, edges, eta, lam, x_star = system_from_chain(n, 25.0, 1.0, 1.0, 0)
    b, a = extract_scalar_channel(eta, lam, 0)
    vals_l, vecs_l = eigsh(a, k=5, which="LM")
    target_lambda = float(vals_l[-2])
    v = vecs_l[:, -2]
    v = v / np.linalg.norm(v)
    level0 = build_line_levels_from_matrix(n, a, block_dofs=1, max_levels=2)[0]
    gabp_level = build_block_gabp_level(a, 1)
    print(f"Representative high-frequency mode lambda={target_lambda:.6f}")
    j = weighted_jacobi(level0, v.copy(), np.zeros_like(v), omega=2.0 / 3.0, iterations=1)
    print(f"Jacobi(1): {np.linalg.norm(j):.6f}")

    variants = [
        "fresh_direct",
        "cycle_carry_p_zero_h",
        "cycle_fixed_p1_zero_h",
        "cycle_fixed_p2_zero_h",
        "cycle_fixed_p4_zero_h",
        "cycle_fixed_p1_damp_h_05",
        "cycle_fixed_p1_project_h_local",
    ]
    for variant in variants:
        core_variant, _, p_mode = parse_variant(variant)
        cache = zero_state(gabp_level)
        p_fixed = None
        if p_mode[0] == "converged":
            p_fixed, _ = converge_block_gabp_precision(gabp_level, tol=1e-10, max_iters=1000)
        elif p_mode[0] == "warm":
            p_fixed = warmup_precision(gabp_level, p_mode[1])
        x1, cache = smooth_once(
            level0,
            gabp_level,
            cache,
            v.copy(),
            np.zeros_like(v),
            sweeps=1,
            variant=core_variant,
            p_fixed=p_fixed,
        )
        print(f"{variant}(1): {np.linalg.norm(x1):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--mode", choices=["scalar", "block"], default="scalar")
    parser.add_argument("--max-cycles", type=int, default=200)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--show-damping", action="store_true")
    args = parser.parse_args()

    if args.show_damping:
        damping_table_chain(256)
        print("")

    nodes, edges, eta, lam, x_star_full = system_from_chain(args.n, 25.0, 1.0, 1.0, 0)

    variants = [
        "fresh_direct",
        "cycle_carry_p_zero_h",
        "cycle_fixed_p1_zero_h",
        "cycle_fixed_p2_zero_h",
        "cycle_fixed_p4_zero_h",
        "cycle_fixed_p1_damp_h_05",
        "cycle_fixed_p1_project_h_local",
    ]

    if args.mode == "scalar":
        b, a = extract_scalar_channel(eta, lam, 0)
        x_star = x_star_full[0::2]
        two = build_line_levels_from_matrix(args.n, a, block_dofs=1, max_levels=2)
        multi = build_line_levels_from_matrix(args.n, a, block_dofs=1, max_levels=None)
        print(f"Scalar chain ablation: n={args.n}")
        for variant in variants:
            for label, levels in [("two", two), ("multi", multi)]:
                res = run_mg_variant(
                    levels,
                    block_dofs=1,
                    b=b,
                    x_star=x_star,
                    variant=variant,
                    pre_sweeps=args.pre,
                    post_sweeps=args.post,
                    max_cycles=args.max_cycles,
                )
                rel = res["error_history"][-1] / max(res["error_history"][0], 1e-15)
                print(f"{variant:24s} {label:5s} cycles={args.max_cycles:4d} rel={rel:.6e}")
    else:
        two = build_line_levels_from_matrix(args.n, lam, block_dofs=2, max_levels=2)
        multi = build_line_levels_from_matrix(args.n, lam, block_dofs=2, max_levels=None)
        print(f"Block-2 chain ablation: n={args.n}")
        shortlist = [
            "fresh_direct",
            "cycle_carry_p_zero_h",
            "cycle_fixed_p1_zero_h",
            "cycle_fixed_p2_zero_h",
            "cycle_fixed_p1_project_h_local",
        ]
        for variant in shortlist:
            for label, levels in [("two", two), ("multi", multi)]:
                res = run_mg_variant(
                    levels,
                    block_dofs=2,
                    b=eta,
                    x_star=x_star_full,
                    variant=variant,
                    pre_sweeps=args.pre,
                    post_sweeps=args.post,
                    max_cycles=args.max_cycles,
                )
                rel = res["error_history"][-1] / max(res["error_history"][0], 1e-15)
                print(f"{variant:24s} {label:5s} cycles={args.max_cycles:4d} rel={rel:.6e}")


if __name__ == "__main__":
    main()
