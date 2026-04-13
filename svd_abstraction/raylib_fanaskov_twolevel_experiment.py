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

from svd_abstraction.raylib_local_eta_prolongation_validation import build_hierarchy
from svd_abstraction.raylib_local_eta_prolongation_validation import build_slam_graph
from svd_abstraction.raylib_local_eta_prolongation_validation import exact_mean
from svd_abstraction.raylib_same_hierarchy_jacobi_experiment import build_transfer_operators
from svd_abstraction.raylib_same_hierarchy_jacobi_experiment import build_exact_rap_levels


def base_mean_vector(graph) -> np.ndarray:
    base_vars = [var for var in graph.multigrid_vars[0] if var.type != "dead"]
    return np.concatenate([var.mu for var in base_vars])


def relative_error_vec(x: np.ndarray, x_star: np.ndarray) -> float:
    return float(np.linalg.norm(x - x_star) / max(np.linalg.norm(x_star), 1e-15))


def strip_to_two_levels(graph):
    if len(graph.multigrid_vars) <= 2:
        return graph

    keep_vars = [
        v for level in range(2) for v in graph.multigrid_vars[level] if getattr(v, "type", None) != "dead"
    ]
    keep_factors = [
        f for level in range(2) for f in graph.multigrid_factors[level] if getattr(f, "type", None) != "dead"
    ]
    keep_var_ids = {v.variableID for v in keep_vars}
    keep_factor_ids = {f.factorID for f in keep_factors}

    graph.var_nodes = [v for v in graph.var_nodes if v.variableID in keep_var_ids]
    graph.factors = [f for f in graph.factors if f.factorID in keep_factor_ids]
    graph.multigrid_vars = [
        [v for v in graph.multigrid_vars[0] if getattr(v, "type", None) != "dead"],
        [v for v in graph.multigrid_vars[1] if getattr(v, "type", None) != "dead"],
    ]
    graph.multigrid_factors = [
        [f for f in graph.multigrid_factors[0] if getattr(f, "type", None) != "dead"],
        [f for f in graph.multigrid_factors[1] if getattr(f, "type", None) != "dead"],
    ]
    graph.n_var_nodes = len(graph.var_nodes)
    graph.n_factor_nodes = len(graph.factors)

    for var in graph.var_nodes:
        var.adj_factors = [f for f in var.adj_factors if f.factorID in keep_factor_ids]
        if var.multigrid.level >= 1:
            var.multigrid.parent = None
            var.multigrid.restriction_vars = []
            var.multigrid.restriction_coefficients = []

    return graph


def fanaskov_edge_solve(
    a: np.ndarray,
    b: np.ndarray,
    n_sweeps: int,
    mode: str = "sequential",
    init_lam_edge: dict[tuple[int, int], float] | None = None,
    init_msg: dict[tuple[int, int], float] | None = None,
    return_state: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]

    neighbors = [np.flatnonzero(np.abs(a[i]) > 1e-12).tolist() for i in range(n)]
    for i in range(n):
        if i in neighbors[i]:
            neighbors[i].remove(i)

    lam_edge = {(j, i): 0.0 for j in range(n) for i in neighbors[j]}
    msg = {(j, i): 0.0 for j in range(n) for i in neighbors[j]}
    if init_lam_edge is not None:
        for key in lam_edge:
            if key in init_lam_edge:
                lam_edge[key] = float(init_lam_edge[key])
    if init_msg is not None:
        for key in msg:
            if key in init_msg:
                msg[key] = float(init_msg[key])

    def estimate(cur_lam, cur_msg):
        x = np.zeros(n, dtype=float)
        for i in range(n):
            num = b[i] + sum(cur_msg[(j, i)] for j in neighbors[i])
            den = a[i, i] + sum(cur_lam[(j, i)] * a[j, i] for j in neighbors[i])
            x[i] = num / den
        return x

    if mode not in {"parallel", "sequential"}:
        raise ValueError(f"Unknown Fanaskov mode: {mode}")

    for _ in range(int(n_sweeps)):
        if mode == "parallel":
            new_lam = {}
            new_msg = {}
            for j in range(n):
                for i in neighbors[j]:
                    denom = a[j, j] + sum(
                        lam_edge[(k, j)] * a[k, j] for k in neighbors[j] if k != i
                    )
                    new_l = -a[i, j] / denom
                    new_m = new_l * (
                        b[j] + sum(msg[(k, j)] for k in neighbors[j] if k != i)
                    )
                    new_lam[(j, i)] = new_l
                    new_msg[(j, i)] = new_m
            lam_edge = new_lam
            msg = new_msg
        else:
            for j in range(n):
                for i in neighbors[j]:
                    denom = a[j, j] + sum(
                        lam_edge[(k, j)] * a[k, j] for k in neighbors[j] if k != i
                    )
                    new_l = -a[i, j] / denom
                    new_m = new_l * (
                        b[j] + sum(msg[(k, j)] for k in neighbors[j] if k != i)
                    )
                    lam_edge[(j, i)] = new_l
                    msg[(j, i)] = new_m

    x = estimate(lam_edge, msg)
    if return_state:
        return x, lam_edge, msg
    return x


def fanaskov_converged_lam_edge(
    a: np.ndarray,
    mode: str = "parallel",
    tol: float = 1e-12,
    max_sweeps: int = 10000,
) -> tuple[dict[tuple[int, int], float], int, float]:
    a = np.asarray(a, dtype=float)
    n = a.shape[0]

    neighbors = [np.flatnonzero(np.abs(a[i]) > 1e-12).tolist() for i in range(n)]
    for i in range(n):
        if i in neighbors[i]:
            neighbors[i].remove(i)

    lam_edge = {(j, i): 0.0 for j in range(n) for i in neighbors[j]}
    if mode not in {"parallel", "sequential"}:
        raise ValueError(f"Unknown Fanaskov mode: {mode}")

    delta = np.inf
    for sweep in range(1, int(max_sweeps) + 1):
        if mode == "parallel":
            new_lam = {}
            for j in range(n):
                for i in neighbors[j]:
                    denom = a[j, j] + sum(
                        lam_edge[(k, j)] * a[k, j] for k in neighbors[j] if k != i
                    )
                    new_lam[(j, i)] = -a[i, j] / denom
            delta = max(abs(new_lam[key] - lam_edge[key]) for key in lam_edge) if lam_edge else 0.0
            lam_edge = new_lam
        else:
            delta = 0.0
            for j in range(n):
                for i in neighbors[j]:
                    denom = a[j, j] + sum(
                        lam_edge[(k, j)] * a[k, j] for k in neighbors[j] if k != i
                    )
                    new_l = -a[i, j] / denom
                    delta = max(delta, abs(new_l - lam_edge[(j, i)]))
                    lam_edge[(j, i)] = new_l
        if delta < tol:
            return lam_edge, sweep, float(delta)

    return lam_edge, int(max_sweeps), float(delta)


def fanaskov_zero_msg_state(a: np.ndarray) -> dict[tuple[int, int], float]:
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    neighbors = [np.flatnonzero(np.abs(a[i]) > 1e-12).tolist() for i in range(n)]
    for i in range(n):
        if i in neighbors[i]:
            neighbors[i].remove(i)
    return {(j, i): 0.0 for j in range(n) for i in neighbors[j]}


def fanaskov_fixed_lam_msg_solve(
    a: np.ndarray,
    b: np.ndarray,
    lam_edge: dict[tuple[int, int], float],
    n_sweeps: int,
    mode: str = "parallel",
    init_msg: dict[tuple[int, int], float] | None = None,
    msg_damping: float = 0.0,
    return_state: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[tuple[int, int], float]]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]

    neighbors = [np.flatnonzero(np.abs(a[i]) > 1e-12).tolist() for i in range(n)]
    for i in range(n):
        if i in neighbors[i]:
            neighbors[i].remove(i)

    msg = {(j, i): 0.0 for j in range(n) for i in neighbors[j]}
    if init_msg is not None:
        for key in msg:
            if key in init_msg:
                msg[key] = float(init_msg[key])

    if mode not in {"parallel", "sequential"}:
        raise ValueError(f"Unknown Fanaskov mode: {mode}")
    if not (0.0 <= msg_damping < 1.0):
        raise ValueError(f"msg_damping must be in [0, 1), got {msg_damping}")

    for _ in range(int(n_sweeps)):
        if mode == "parallel":
            new_msg = {}
            for j in range(n):
                for i in neighbors[j]:
                    candidate = lam_edge[(j, i)] * (
                        b[j] + sum(msg[(k, j)] for k in neighbors[j] if k != i)
                    )
                    new_msg[(j, i)] = (1.0 - msg_damping) * candidate + msg_damping * msg[(j, i)]
            msg = new_msg
        else:
            for j in range(n):
                for i in neighbors[j]:
                    candidate = lam_edge[(j, i)] * (
                        b[j] + sum(msg[(k, j)] for k in neighbors[j] if k != i)
                    )
                    msg[(j, i)] = (1.0 - msg_damping) * candidate + msg_damping * msg[(j, i)]

    x = np.zeros(n, dtype=float)
    for i in range(n):
        num = b[i] + sum(msg[(j, i)] for j in neighbors[i])
        den = a[i, i] + sum(lam_edge[(j, i)] * a[j, i] for j in neighbors[i])
        x[i] = num / den

    if return_state:
        return x, msg
    return x


def fanaskov_block_solve(
    a: np.ndarray,
    b: np.ndarray,
    n_sweeps: int,
    block_size: int = 2,
    mode: str = "sequential",
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]
    if n % block_size != 0:
        raise ValueError(f"Dimension {n} is not divisible by block_size={block_size}")

    n_blocks = n // block_size
    blocks = [slice(i * block_size, (i + 1) * block_size) for i in range(n_blocks)]

    def block(i: int, j: int) -> np.ndarray:
        return a[blocks[i], blocks[j]]

    neighbors = []
    for i in range(n_blocks):
        nbrs = []
        for j in range(n_blocks):
            if i == j:
                continue
            if np.linalg.norm(block(i, j)) > 1e-12:
                nbrs.append(j)
        neighbors.append(nbrs)

    edge_gain = {
        (j, i): np.zeros((block_size, block_size), dtype=float)
        for j in range(n_blocks)
        for i in neighbors[j]
    }
    edge_msg = {
        (j, i): np.zeros(block_size, dtype=float)
        for j in range(n_blocks)
        for i in neighbors[j]
    }

    def estimate(cur_gain, cur_msg):
        x = np.zeros(n, dtype=float)
        for i in range(n_blocks):
            num = b[blocks[i]].copy()
            den = block(i, i).copy()
            for j in neighbors[i]:
                num += cur_msg[(j, i)]
                den += cur_gain[(j, i)] @ block(j, i)
            den = 0.5 * (den + den.T)
            x[blocks[i]] = np.linalg.solve(den, num)
        return x

    if mode not in {"parallel", "sequential"}:
        raise ValueError(f"Unknown Fanaskov mode: {mode}")

    for _ in range(int(n_sweeps)):
        if mode == "parallel":
            new_gain = {}
            new_msg = {}
            for j in range(n_blocks):
                for i in neighbors[j]:
                    den = block(j, j).copy()
                    rhs = b[blocks[j]].copy()
                    for k in neighbors[j]:
                        if k == i:
                            continue
                        den += edge_gain[(k, j)] @ block(k, j)
                        rhs += edge_msg[(k, j)]
                    den = 0.5 * (den + den.T)
                    gain = -np.linalg.solve(den.T, block(i, j).T).T
                    new_gain[(j, i)] = gain
                    new_msg[(j, i)] = gain @ rhs
            edge_gain = new_gain
            edge_msg = new_msg
        else:
            for j in range(n_blocks):
                for i in neighbors[j]:
                    den = block(j, j).copy()
                    rhs = b[blocks[j]].copy()
                    for k in neighbors[j]:
                        if k == i:
                            continue
                        den += edge_gain[(k, j)] @ block(k, j)
                        rhs += edge_msg[(k, j)]
                    den = 0.5 * (den + den.T)
                    gain = -np.linalg.solve(den.T, block(i, j).T).T
                    edge_gain[(j, i)] = gain
                    edge_msg[(j, i)] = gain @ rhs

    return estimate(edge_gain, edge_msg)


def run_two_level_fanaskov(
    coarse_mode: str,
    n: int,
    warmup: int,
    max_cycles: int,
    tol: float,
    coarse_sweeps: int,
) -> tuple[int | None, list[float]]:
    graph = build_slam_graph(n=n, seed=0)
    mu_star = exact_mean(graph)
    build_hierarchy(graph)
    strip_to_two_levels(graph)

    for _ in range(warmup):
        graph.synchronous_iteration(level=0)

    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    r, p = build_transfer_operators(graph, coarse_level=1)
    a1 = r @ a0 @ p

    relerrs = [relative_error_vec(base_mean_vector(graph), mu_star)]
    conv = 0 if relerrs[-1] < tol else None

    for cyc in range(1, max_cycles + 1):
        graph.synchronous_iteration(level=0)
        x = base_mean_vector(graph)
        residual = b0 - a0 @ x
        coarse_rhs = r @ residual
        coarse_err = fanaskov_edge_solve(
            a=a1,
            b=coarse_rhs,
            n_sweeps=coarse_sweeps,
            mode=coarse_mode,
        )
        graph.set_level_mean_vector(0, x + p @ coarse_err)

        rel = relative_error_vec(base_mean_vector(graph), mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = cyc
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, relerrs


def run_base_fanaskov_error_solver(
    mode: str,
    n: int,
    max_iters: int,
    tol: float,
    base_sweeps: int,
) -> tuple[int | None, list[float]]:
    graph = build_slam_graph(n=n, seed=0)
    mu_star = exact_mean(graph)
    build_hierarchy(graph)

    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)

    x = base_mean_vector(graph)
    relerrs = [relative_error_vec(x, mu_star)]
    conv = 0 if relerrs[-1] < tol else None

    for it in range(1, max_iters + 1):
        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a=a0, b=residual, n_sweeps=base_sweeps, mode=mode)
        rel = relative_error_vec(x, mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = it
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, relerrs


def run_two_level_fanaskov_error_mg(
    mode: str,
    n: int,
    max_cycles: int,
    tol: float,
    base_sweeps: int,
    coarse_sweeps: int,
) -> tuple[int | None, list[float]]:
    graph = build_slam_graph(n=n, seed=0)
    mu_star = exact_mean(graph)
    build_hierarchy(graph)
    strip_to_two_levels(graph)

    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    r, p = build_transfer_operators(graph, coarse_level=1)
    a1 = r @ a0 @ p

    x = base_mean_vector(graph)
    relerrs = [relative_error_vec(x, mu_star)]
    conv = 0 if relerrs[-1] < tol else None

    for cyc in range(1, max_cycles + 1):
        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a=a0, b=residual, n_sweeps=base_sweeps, mode=mode)

        residual = b0 - a0 @ x
        coarse_rhs = r @ residual
        coarse_err = fanaskov_edge_solve(a=a1, b=coarse_rhs, n_sweeps=coarse_sweeps, mode=mode)
        x = x + p @ coarse_err

        residual = b0 - a0 @ x
        x = x + fanaskov_edge_solve(a=a0, b=residual, n_sweeps=base_sweeps, mode=mode)

        rel = relative_error_vec(x, mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = cyc
            break
        if not np.isfinite(rel) or rel > 1e12:
            break

    return conv, relerrs


def v_cycle_multilevel_fanaskov(
    levels: list[dict[str, np.ndarray | None]],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    mode: str,
    base_sweeps: int,
    coarse_sweeps: int,
) -> np.ndarray:
    a = np.asarray(levels[level_idx]["a"], dtype=float)
    is_coarsest = level_idx == len(levels) - 1
    if is_coarsest:
        return np.linalg.solve(a, b)

    x = x + fanaskov_edge_solve(a=a, b=b - a @ x, n_sweeps=base_sweeps, mode=mode)

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
            mode=mode,
        )
    else:
        coarse_x = v_cycle_multilevel_fanaskov(
            levels=levels,
            level_idx=level_idx + 1,
            x=coarse_x,
            b=coarse_rhs,
            mode=mode,
            base_sweeps=base_sweeps,
            coarse_sweeps=coarse_sweeps,
        )
    x = x + p @ coarse_x

    x = x + fanaskov_edge_solve(a=a, b=b - a @ x, n_sweeps=base_sweeps, mode=mode)
    return x


def run_multilevel_fanaskov_error_mg(
    mode: str,
    n: int,
    max_cycles: int,
    tol: float,
    base_sweeps: int,
    coarse_sweeps: int,
) -> tuple[int | None, list[float]]:
    graph = build_slam_graph(n=n, seed=0)
    mu_star = exact_mean(graph)
    build_hierarchy(graph)

    b0, _a0 = graph.joint_distribution_inf_level(0)
    b0 = np.asarray(b0, dtype=float)
    levels = build_exact_rap_levels(graph)

    x = base_mean_vector(graph)
    relerrs = [relative_error_vec(x, mu_star)]
    conv = 0 if relerrs[-1] < tol else None

    for cyc in range(1, max_cycles + 1):
        x = v_cycle_multilevel_fanaskov(
            levels=levels,
            level_idx=0,
            x=x,
            b=b0,
            mode=mode,
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

    return conv, relerrs


def print_summary(name: str, conv: int | None, relerrs: list[float], points: list[int]) -> None:
    print(f"{name} conv {conv}")
    for point in points:
        if point < len(relerrs):
            print(f"{name} {point} {relerrs[point]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=800)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--base-sweeps", type=int, default=1)
    parser.add_argument("--coarse-sweeps", type=int, default=2)
    parser.add_argument(
        "--experiment",
        default="raylib_fine_fanaskov_coarse",
        choices=[
            "raylib_fine_fanaskov_coarse",
            "fanaskov_base",
            "fanaskov_twolevel_error_mg",
            "fanaskov_multilevel_error_mg",
        ],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["parallel", "sequential"],
        choices=["parallel", "sequential"],
    )
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 500],
    )
    args = parser.parse_args()

    for mode in args.modes:
        if args.experiment == "raylib_fine_fanaskov_coarse":
            conv, relerrs = run_two_level_fanaskov(
                coarse_mode=mode,
                n=args.n,
                warmup=args.warmup,
                max_cycles=args.max_cycles,
                tol=args.tol,
                coarse_sweeps=args.coarse_sweeps,
            )
            print_summary(f"two_level_fanaskov_{mode}", conv, relerrs, args.points)
        elif args.experiment == "fanaskov_base":
            conv, relerrs = run_base_fanaskov_error_solver(
                mode=mode,
                n=args.n,
                max_iters=args.max_iters,
                tol=args.tol,
                base_sweeps=args.base_sweeps,
            )
            print_summary(f"base_fanaskov_{mode}", conv, relerrs, args.points)
        else:
            if args.experiment == "fanaskov_twolevel_error_mg":
                conv, relerrs = run_two_level_fanaskov_error_mg(
                    mode=mode,
                    n=args.n,
                    max_cycles=args.max_cycles,
                    tol=args.tol,
                    base_sweeps=args.base_sweeps,
                    coarse_sweeps=args.coarse_sweeps,
                )
                print_summary(f"two_level_fanaskov_error_mg_{mode}", conv, relerrs, args.points)
            else:
                conv, relerrs = run_multilevel_fanaskov_error_mg(
                    mode=mode,
                    n=args.n,
                    max_cycles=args.max_cycles,
                    tol=args.tol,
                    base_sweeps=args.base_sweeps,
                    coarse_sweeps=args.coarse_sweeps,
                )
                print_summary(f"multilevel_fanaskov_error_mg_{mode}", conv, relerrs, args.points)


if __name__ == "__main__":
    main()
