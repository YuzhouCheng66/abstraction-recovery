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

from svd_abstraction.raylib_fanaskov_twolevel_experiment import fanaskov_edge_solve
from svd_abstraction.raylib_fanaskov_twolevel_experiment import relative_error_vec
from svd_abstraction.raylib_local_eta_prolongation_validation import build_slam_graph
from svd_abstraction.raylib_local_eta_prolongation_validation import exact_mean
from svd_abstraction.raylib_local_eta_prolongation_validation import mean_vector


def build_spectral_levels(a0: np.ndarray, ranks: list[int]) -> list[dict[str, np.ndarray | None]]:
    a_level = np.asarray(a0, dtype=float)
    levels: list[dict[str, np.ndarray | None]] = [{"a": a_level, "p": None, "r": None}]

    for rank in ranks:
        eigvals, eigvecs = np.linalg.eigh(a_level)
        p = eigvecs[:, :rank]
        r = p.T
        levels[-1]["p"] = p
        levels[-1]["r"] = r
        a_level = r @ a_level @ p
        levels.append({"a": a_level, "p": None, "r": None})

    return levels


def odom_tiny_init_vector(graph, n: int, tiny: float = 1e-12) -> np.ndarray:
    base_vars = [var for var in graph.multigrid_vars[0] if getattr(var, "type", None) != "dead"]
    base_factors = [factor for factor in graph.multigrid_factors[0] if getattr(factor, "type", None) != "dead"]

    chain_meas = {}
    for factor in base_factors:
        if getattr(factor, "type", None) != "odometry":
            continue
        i, j = [int(v) for v in factor.adj_vIDs]
        if j == i + 1:
            chain_meas[(i, j)] = np.asarray(factor.measurement, dtype=float)

    mus = {0: np.asarray(base_vars[0].GT, dtype=float).copy()}
    for i in range(n - 1):
        mus[i + 1] = mus[i] + chain_meas[(i, i + 1)]

    for var in base_vars:
        vid = int(var.variableID)
        mu = mus[vid].copy()
        var.mu = mu
        var.prior.eta = np.asarray(var.prior.eta, dtype=float) + tiny * mu
        var.belief.eta = np.asarray(var.belief.lam, dtype=float) @ mu

    for factor in base_factors:
        for belief_idx, var in enumerate(factor.adj_var_nodes):
            if getattr(var, "type", None) == "dead":
                continue
            factor.adj_beliefs[belief_idx].eta = np.asarray(var.belief.eta, dtype=float).copy()
        for msg in factor.messages:
            msg.eta = np.zeros_like(msg.eta)

    return mean_vector(graph)


def run_base_fanaskov_parallel_odomtiny(
    n: int,
    prior_prop: float,
    rank: int,
    max_iters: int,
    tol: float,
    base_sweeps: int,
) -> tuple[int | None, list[float], np.ndarray]:
    graph = build_slam_graph(n=n, seed=0, prior_prop=prior_prop)
    mu_star = exact_mean(graph)
    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)

    x = odom_tiny_init_vector(graph, n=n)
    relerrs = [relative_error_vec(x, mu_star)]
    conv = 0 if relerrs[-1] < tol else None

    for it in range(1, max_iters + 1):
        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a=a0, b=residual, n_sweeps=base_sweeps, mode="parallel")
        rel = relative_error_vec(x, mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = it
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, relerrs, a0


def run_two_level_spectral_fanaskov_parallel_odomtiny(
    n: int,
    prior_prop: float,
    rank: int,
    max_cycles: int,
    tol: float,
    base_sweeps: int,
    coarse_sweeps: int,
) -> tuple[int | None, list[float], np.ndarray]:
    graph = build_slam_graph(n=n, seed=0, prior_prop=prior_prop)
    mu_star = exact_mean(graph)
    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)

    eigvals, eigvecs = np.linalg.eigh(a0)
    p = eigvecs[:, :rank]
    a1 = p.T @ a0 @ p

    x = odom_tiny_init_vector(graph, n=n)
    relerrs = [relative_error_vec(x, mu_star)]
    conv = 0 if relerrs[-1] < tol else None

    for cyc in range(1, max_cycles + 1):
        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a=a0, b=residual, n_sweeps=base_sweeps, mode="parallel")

        residual = b0 - a0 @ x
        coarse_rhs = p.T @ residual
        coarse_err = fanaskov_edge_solve(a=a1, b=coarse_rhs, n_sweeps=coarse_sweeps, mode="parallel")
        x = x + p @ coarse_err

        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a=a0, b=residual, n_sweeps=base_sweeps, mode="parallel")

        rel = relative_error_vec(x, mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = cyc
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, relerrs, eigvals


def v_cycle_multilevel_spectral_fanaskov(
    levels: list[dict[str, np.ndarray | None]],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    base_sweeps: int,
    coarse_sweeps: int,
) -> np.ndarray:
    a = np.asarray(levels[level_idx]["a"], dtype=float)
    is_coarsest = level_idx == len(levels) - 1
    if is_coarsest:
        return np.linalg.solve(a, b)

    x = x + fanaskov_edge_solve(a=a, b=b - a @ x, n_sweeps=base_sweeps, mode="parallel")

    residual = b - a @ x
    r = np.asarray(levels[level_idx]["r"], dtype=float)
    p = np.asarray(levels[level_idx]["p"], dtype=float)
    coarse_rhs = r @ residual
    coarse_x = np.zeros_like(coarse_rhs)
    if level_idx + 1 == len(levels) - 1:
        coarse_x = coarse_x + fanaskov_edge_solve(
            a=np.asarray(levels[level_idx + 1]["a"], dtype=float),
            b=coarse_rhs,
            n_sweeps=coarse_sweeps,
            mode="parallel",
        )
    else:
        coarse_x = v_cycle_multilevel_spectral_fanaskov(
            levels=levels,
            level_idx=level_idx + 1,
            x=coarse_x,
            b=coarse_rhs,
            base_sweeps=base_sweeps,
            coarse_sweeps=coarse_sweeps,
        )
    x = x + p @ coarse_x

    x = x + fanaskov_edge_solve(a=a, b=b - a @ x, n_sweeps=base_sweeps, mode="parallel")
    return x


def run_multilevel_spectral_fanaskov_parallel_odomtiny(
    n: int,
    prior_prop: float,
    ranks: list[int],
    max_cycles: int,
    tol: float,
    base_sweeps: int,
    coarse_sweeps: int,
) -> tuple[int | None, list[float], np.ndarray]:
    graph = build_slam_graph(n=n, seed=0, prior_prop=prior_prop)
    mu_star = exact_mean(graph)
    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)

    levels = build_spectral_levels(a0, ranks)
    eigvals = np.linalg.eigvalsh(a0)

    x = odom_tiny_init_vector(graph, n=n)
    relerrs = [relative_error_vec(x, mu_star)]
    conv = 0 if relerrs[-1] < tol else None

    for cyc in range(1, max_cycles + 1):
        x = v_cycle_multilevel_spectral_fanaskov(
            levels=levels,
            level_idx=0,
            x=x,
            b=b0,
            base_sweeps=base_sweeps,
            coarse_sweeps=coarse_sweeps,
        )

        rel = relative_error_vec(x, mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = cyc
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, relerrs, eigvals


def print_summary(name: str, conv: int | None, relerrs: list[float], points: list[int]) -> None:
    print(f"{name} conv {conv}")
    for point in points:
        if point < len(relerrs):
            print(f"{name} {point} {relerrs[point]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--multilevel-ranks", type=int, nargs="+", default=None)
    parser.add_argument("--base-sweeps", type=int, default=1)
    parser.add_argument("--coarse-sweeps", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000],
    )
    args = parser.parse_args()

    multilevel_ranks = args.multilevel_ranks if args.multilevel_ranks is not None else [args.rank, 4, 2]

    base_conv, base_relerrs, a0 = run_base_fanaskov_parallel_odomtiny(
        n=args.n,
        prior_prop=args.prior_prop,
        rank=args.rank,
        max_iters=args.max_iters,
        tol=args.tol,
        base_sweeps=args.base_sweeps,
    )
    mg_conv, mg_relerrs, eigvals = run_two_level_spectral_fanaskov_parallel_odomtiny(
        n=args.n,
        prior_prop=args.prior_prop,
        rank=args.rank,
        max_cycles=args.max_cycles,
        tol=args.tol,
        base_sweeps=args.base_sweeps,
        coarse_sweeps=args.coarse_sweeps,
    )
    multi_conv, multi_relerrs, _ = run_multilevel_spectral_fanaskov_parallel_odomtiny(
        n=args.n,
        prior_prop=args.prior_prop,
        ranks=multilevel_ranks,
        max_cycles=args.max_cycles,
        tol=args.tol,
        base_sweeps=args.base_sweeps,
        coarse_sweeps=args.coarse_sweeps,
    )

    print(f"smallest eigvals[:{args.rank}] {eigvals[:args.rank]}")
    print(f"spectral rank {args.rank}")
    print(f"multilevel ranks {multilevel_ranks}")
    print_summary("base_fan_parallel_odomtiny", base_conv, base_relerrs, args.points)
    print_summary("twolevel_spectral_fan_parallel_odomtiny", mg_conv, mg_relerrs, args.points)
    print_summary("multilevel_spectral_fan_parallel_odomtiny", multi_conv, multi_relerrs, args.points)


if __name__ == "__main__":
    main()
