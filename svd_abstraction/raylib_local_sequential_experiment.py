from __future__ import annotations

import argparse
import pathlib
import sys


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOCAL_RAYLIB_ROOT = SCRIPT_DIR / "raylib_gbp_local"

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(LOCAL_RAYLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_RAYLIB_ROOT))

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(1, str(WORKSPACE_ROOT))

from svd_abstraction.raylib_local_eta_prolongation_validation import (
    build_hierarchy,
    build_slam_graph,
    exact_mean,
    relative_error,
)


def run_base(mode: str, n: int, warmup: int, max_iters: int, tol: float):
    graph = build_slam_graph(n=n, seed=0)
    mu_star = exact_mean(graph)

    for _ in range(warmup):
        graph.synchronous_iteration(level=0)

    relerrs = [relative_error(graph, mu_star)]
    conv = None
    for it in range(1, max_iters + 1):
        if mode == "synchronous":
            graph.synchronous_iteration(level=0)
        elif mode == "sequential":
            graph.sequential_iteration(level=0)
        else:
            raise ValueError(f"Unknown base mode: {mode}")
        rel = relative_error(graph, mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = it
            break
    return conv, relerrs


def run_two_level(base_mode: str, coarse_mode: str, n: int, warmup: int, max_cycles: int, tol: float):
    graph = build_slam_graph(n=n, seed=0)
    mu_star = exact_mean(graph)
    build_hierarchy(graph)

    for _ in range(warmup):
        graph.synchronous_iteration(level=0)

    relerrs = [relative_error(graph, mu_star)]
    conv = None
    for cyc in range(1, max_cycles + 1):
        graph.vcycle_step(base_smoother=base_mode, coarse_smoother=coarse_mode)
        rel = relative_error(graph, mu_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = cyc
            break
    return conv, relerrs


def print_summary(name: str, conv, relerrs, points):
    print(f"{name} conv {conv}")
    for p in points:
        if p < len(relerrs):
            print(f"{name} {p} {relerrs[p]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=1200)
    parser.add_argument("--max-cycles", type=int, default=600)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=[0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400],
    )
    args = parser.parse_args()

    print("Base-only:")
    for mode in ["synchronous", "sequential"]:
        conv, relerrs = run_base(
            mode=mode,
            n=args.n,
            warmup=args.warmup,
            max_iters=args.max_iters,
            tol=args.tol,
        )
        print_summary(f"base_{mode}", conv, relerrs, args.points)

    print("\nTwo-level:")
    cases = [
        ("sync_sync", "synchronous", "synchronous"),
        ("seq_sync", "sequential", "synchronous"),
        ("sync_seq", "synchronous", "sequential"),
        ("seq_seq", "sequential", "sequential"),
    ]
    for name, base_mode, coarse_mode in cases:
        conv, relerrs = run_two_level(
            base_mode=base_mode,
            coarse_mode=coarse_mode,
            n=args.n,
            warmup=args.warmup,
            max_cycles=args.max_cycles,
            tol=args.tol,
        )
        print_summary(name, conv, relerrs, args.points)


if __name__ == "__main__":
    main()
