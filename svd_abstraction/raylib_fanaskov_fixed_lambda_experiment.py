from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOCAL_RAYLIB_ROOT = SCRIPT_DIR / "raylib_gbp_local"

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(LOCAL_RAYLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_RAYLIB_ROOT))

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(1, str(WORKSPACE_ROOT))

from svd_abstraction.raylib_baware_interp_analysis import odom_tiny_init_base
from svd_abstraction.raylib_fanaskov_twolevel_experiment import (
    fanaskov_converged_lam_edge,
    fanaskov_edge_solve,
    fanaskov_fixed_lam_msg_solve,
    fanaskov_zero_msg_state,
    relative_error_vec,
)
from svd_abstraction.raylib_grouped_svd_benchmark import (
    build_grouped_svd_basis,
    group_list,
)
from svd_abstraction.raylib_local_eta_prolongation_validation import (
    build_slam_graph,
    exact_mean,
    mean_vector,
)
from svd_abstraction.raylib_variance_freeze_experiment import lam_state


def fixed_problem(
    n: int,
    seed: int,
    prior_prop: float,
    variance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    graph = build_slam_graph(n=n, seed=seed, prior_prop=prior_prop)
    mu_star = exact_mean(graph)
    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)

    prev = lam_state(graph, [0])
    freeze_step = None
    for step in range(1, 1000):
        graph.synchronous_iteration(level=0)
        curr = lam_state(graph, [0])
        delta = float(np.max(np.abs(curr - prev)))
        if delta < variance_threshold:
            freeze_step = step
            break
        prev = curr
    if freeze_step is None:
        raise RuntimeError("Base lam did not settle within 1000 iterations")

    odom_tiny_init_base(graph, n=n)
    x0 = mean_vector(graph)
    return a0, b0, mu_star, x0, freeze_step


def grouped_basis(
    n: int,
    seed: int,
    prior_prop: float,
    a0: np.ndarray,
    grouping: str,
    group_size: int,
    r_reduced: int,
) -> np.ndarray:
    graph = build_slam_graph(n=n, seed=seed, prior_prop=prior_prop)
    groups = group_list(
        graph,
        method=grouping,
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
    return build_grouped_svd_basis(
        graph,
        a0=a0,
        groups=groups,
        r_reduced=r_reduced,
        basis_source="joint_covariance",
    )


def maybe_record(points: set[int], idx: int, rel: float, out: dict[int, float]) -> None:
    if idx in points:
        out[idx] = float(rel)


def run_base_zero_lam(
    a0: np.ndarray,
    b0: np.ndarray,
    mu_star: np.ndarray,
    x0: np.ndarray,
    max_iters: int,
    tol: float,
    points: list[int],
) -> tuple[int | None, dict[int, float]]:
    x = x0.copy()
    point_set = set(points)
    rels = {}
    maybe_record(point_set, 0, relative_error_vec(x, mu_star), rels)
    conv = 0 if rels.get(0, np.inf) < tol else None

    for it in range(1, max_iters + 1):
        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a0, residual, n_sweeps=1, mode="parallel")
        rel = relative_error_vec(x, mu_star)
        maybe_record(point_set, it, rel, rels)
        if conv is None and rel < tol:
            conv = it
            break
        if not np.isfinite(rel) or rel > 1e12:
            break
    return conv, rels


def run_base_fixed_lam(
    a0: np.ndarray,
    b0: np.ndarray,
    mu_star: np.ndarray,
    x0: np.ndarray,
    lam_edge_star: dict[tuple[int, int], float],
    m_sweeps: int,
    persistent_msg: bool,
    msg_damping: float,
    max_iters: int,
    tol: float,
    points: list[int],
) -> tuple[int | None, dict[int, float]]:
    x = x0.copy()
    point_set = set(points)
    rels = {}
    maybe_record(point_set, 0, relative_error_vec(x, mu_star), rels)
    conv = 0 if rels.get(0, np.inf) < tol else None
    msg_state = fanaskov_zero_msg_state(a0)

    for it in range(1, max_iters + 1):
        residual = b0 - a0 @ x
        init_msg = msg_state if persistent_msg else None
        e, msg_state_new = fanaskov_fixed_lam_msg_solve(
            a0,
            residual,
            lam_edge_star,
            n_sweeps=m_sweeps,
            mode="parallel",
            init_msg=init_msg,
            msg_damping=msg_damping,
            return_state=True,
        )
        if persistent_msg:
            msg_state = msg_state_new
        x = x + e
        rel = relative_error_vec(x, mu_star)
        maybe_record(point_set, it, rel, rels)
        if conv is None and rel < tol:
            conv = it
            break
        if not np.isfinite(rel) or rel > 1e12:
            break
    return conv, rels


def run_twolevel_exact_zero_lam(
    a0: np.ndarray,
    b0: np.ndarray,
    mu_star: np.ndarray,
    x0: np.ndarray,
    p: np.ndarray,
    max_cycles: int,
    tol: float,
    pre_post: bool,
    points: list[int],
) -> tuple[int | None, dict[int, float]]:
    ac = p.T @ a0 @ p
    x = x0.copy()
    point_set = set(points)
    rels = {}
    maybe_record(point_set, 0, relative_error_vec(x, mu_star), rels)
    conv = 0 if rels.get(0, np.inf) < tol else None

    for cyc in range(1, max_cycles + 1):
        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a0, residual, n_sweeps=1, mode="parallel")

        residual = b0 - a0 @ x
        yc = np.linalg.solve(ac, p.T @ residual)
        x = x + p @ yc

        if pre_post:
            residual = b0 - a0 @ x
            x = x + fanaskov_edge_solve(a0, residual, n_sweeps=1, mode="parallel")

        rel = relative_error_vec(x, mu_star)
        maybe_record(point_set, cyc, rel, rels)
        if conv is None and rel < tol:
            conv = cyc
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, rels


def run_twolevel_exact_fixed_lam(
    a0: np.ndarray,
    b0: np.ndarray,
    mu_star: np.ndarray,
    x0: np.ndarray,
    p: np.ndarray,
    lam_edge_star: dict[tuple[int, int], float],
    m_sweeps: int,
    persistent_msg: bool,
    msg_damping: float,
    max_cycles: int,
    tol: float,
    pre_post: bool,
    points: list[int],
) -> tuple[int | None, dict[int, float]]:
    ac = p.T @ a0 @ p
    x = x0.copy()
    point_set = set(points)
    rels = {}
    maybe_record(point_set, 0, relative_error_vec(x, mu_star), rels)
    conv = 0 if rels.get(0, np.inf) < tol else None
    msg_state = fanaskov_zero_msg_state(a0)

    for cyc in range(1, max_cycles + 1):
        residual = b0 - a0 @ x
        init_msg = msg_state if persistent_msg else None
        e_pre, msg_pre = fanaskov_fixed_lam_msg_solve(
            a0,
            residual,
            lam_edge_star,
            n_sweeps=m_sweeps,
            mode="parallel",
            init_msg=init_msg,
            msg_damping=msg_damping,
            return_state=True,
        )
        if persistent_msg:
            msg_state = msg_pre
        x = x + e_pre

        residual = b0 - a0 @ x
        yc = np.linalg.solve(ac, p.T @ residual)
        x = x + p @ yc

        if pre_post:
            residual = b0 - a0 @ x
            init_msg = msg_state if persistent_msg else None
            e_post, msg_post = fanaskov_fixed_lam_msg_solve(
                a0,
                residual,
                lam_edge_star,
                n_sweeps=m_sweeps,
                mode="parallel",
                init_msg=init_msg,
                msg_damping=msg_damping,
                return_state=True,
            )
            if persistent_msg:
                msg_state = msg_post
            x = x + e_post

        rel = relative_error_vec(x, mu_star)
        maybe_record(point_set, cyc, rel, rels)
        if conv is None and rel < tol:
            conv = cyc
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, rels


def fmt_points(label: str, rels: dict[int, float], points: list[int]) -> str:
    vals = ", ".join(f"{pt}:{rels[pt]:.6g}" for pt in points if pt in rels)
    return f"{label}: {vals}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument("--grouping", type=str, default="order")
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--max-base-iters", type=int, default=2000)
    parser.add_argument("--max-cycles", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--m-sweeps", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    parser.add_argument("--msg-damping", type=float, default=0.0)
    parser.add_argument("--points", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 50, 100, 200])
    args = parser.parse_args()

    a0, b0, mu_star, x0, freeze_step = fixed_problem(
        n=args.n,
        seed=args.seed,
        prior_prop=args.prior_prop,
        variance_threshold=args.variance_threshold,
    )
    p = grouped_basis(
        n=args.n,
        seed=args.seed,
        prior_prop=args.prior_prop,
        a0=a0,
        grouping=args.grouping,
        group_size=args.group_size,
        r_reduced=args.r_reduced,
    )
    lam_edge_star, lam_sweeps, lam_delta = fanaskov_converged_lam_edge(
        a0, mode="parallel", tol=1e-12, max_sweeps=10000
    )

    print(f"freeze_step={freeze_step}")
    print(f"lam_edge_converged_sweeps={lam_sweeps} lam_edge_final_delta={lam_delta:.3e}")
    print(f"coarse_dim={p.shape[1]}")

    conv, rels = run_base_zero_lam(
        a0, b0, mu_star, x0, args.max_base_iters, args.tol, args.points
    )
    print(f"base_zero_lam conv={conv}")
    print(fmt_points("base_zero_lam_points", rels, args.points))

    for m_sweeps in args.m_sweeps:
        conv, rels = run_base_fixed_lam(
            a0, b0, mu_star, x0, lam_edge_star, m_sweeps,
            persistent_msg=False, msg_damping=args.msg_damping,
            max_iters=args.max_base_iters, tol=args.tol, points=args.points
        )
        print(f"base_fixed_lam_reset_m sweeps={m_sweeps} conv={conv}")
        print(fmt_points("points", rels, args.points))

    for m_sweeps in args.m_sweeps:
        conv, rels = run_base_fixed_lam(
            a0, b0, mu_star, x0, lam_edge_star, m_sweeps,
            persistent_msg=True, msg_damping=args.msg_damping,
            max_iters=args.max_base_iters, tol=args.tol, points=args.points
        )
        print(f"base_fixed_lam_persistent_m sweeps={m_sweeps} conv={conv}")
        print(fmt_points("points", rels, args.points))

    for pre_post in [False, True]:
        label = "pre_post" if pre_post else "pre_only"
        conv, rels = run_twolevel_exact_zero_lam(
            a0, b0, mu_star, x0, p, args.max_cycles, args.tol, pre_post, args.points
        )
        print(f"twolevel_zero_lam_exact {label} conv={conv}")
        print(fmt_points("points", rels, args.points))

    for pre_post in [False, True]:
        label = "pre_post" if pre_post else "pre_only"
        for m_sweeps in args.m_sweeps:
            conv, rels = run_twolevel_exact_fixed_lam(
                a0, b0, mu_star, x0, p, lam_edge_star, m_sweeps,
                persistent_msg=False, msg_damping=args.msg_damping,
                max_cycles=args.max_cycles, tol=args.tol,
                pre_post=pre_post, points=args.points
            )
            print(f"twolevel_fixed_lam_reset_m exact {label} sweeps={m_sweeps} conv={conv}")
            print(fmt_points("points", rels, args.points))

    for pre_post in [False, True]:
        label = "pre_post" if pre_post else "pre_only"
        for m_sweeps in args.m_sweeps:
            conv, rels = run_twolevel_exact_fixed_lam(
                a0, b0, mu_star, x0, p, lam_edge_star, m_sweeps,
                persistent_msg=True, msg_damping=args.msg_damping,
                max_cycles=args.max_cycles, tol=args.tol,
                pre_post=pre_post, points=args.points
            )
            print(f"twolevel_fixed_lam_persistent_m exact {label} sweeps={m_sweeps} conv={conv}")
            print(fmt_points("points", rels, args.points))


if __name__ == "__main__":
    main()
