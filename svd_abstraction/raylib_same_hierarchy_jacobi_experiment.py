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


def base_mean_vector(graph) -> np.ndarray:
    base_vars = [var for var in graph.multigrid_vars[0] if var.type != "dead"]
    return np.concatenate([var.mu for var in base_vars])


def relative_error_vec(x: np.ndarray, x_star: np.ndarray) -> float:
    return float(np.linalg.norm(x - x_star) / max(np.linalg.norm(x_star), 1e-15))


def build_block_diag_inv(a: np.ndarray, block_dof: int) -> np.ndarray:
    if a.shape[0] % block_dof != 0:
        raise ValueError(f"Matrix dimension {a.shape[0]} not divisible by block size {block_dof}")

    n_blocks = a.shape[0] // block_dof
    inv_blocks = np.zeros((n_blocks, block_dof, block_dof), dtype=float)
    for block_idx in range(n_blocks):
        sl = slice(block_idx * block_dof, (block_idx + 1) * block_dof)
        inv_blocks[block_idx] = np.linalg.inv(a[sl, sl])
    return inv_blocks


def build_transfer_operators(graph, coarse_level: int = 1) -> tuple[np.ndarray, np.ndarray]:
    fine_vars = [var for var in graph.multigrid_vars[coarse_level - 1] if var.type != "dead"]
    coarse_vars = [var for var in graph.multigrid_vars[coarse_level] if var.type != "dead"]
    dof = 2

    fine_ix = {var.variableID: idx for idx, var in enumerate(fine_vars)}
    coarse_ix = {var.variableID: idx for idx, var in enumerate(coarse_vars)}

    r = np.zeros((len(coarse_vars) * dof, len(fine_vars) * dof), dtype=float)
    p = np.zeros((len(fine_vars) * dof, len(coarse_vars) * dof), dtype=float)

    for fine_var in fine_vars:
        fi = fine_ix[fine_var.variableID]
        for coarse_var, coeff in zip(
            fine_var.multigrid.restriction_vars,
            fine_var.multigrid.restriction_coefficients,
        ):
            if coarse_var.type == "dead":
                continue
            ci = coarse_ix[coarse_var.variableID]
            block = np.asarray(coeff, dtype=float)
            r[ci * dof : (ci + 1) * dof, fi * dof : (fi + 1) * dof] = block

    for coarse_var in coarse_vars:
        ci = coarse_ix[coarse_var.variableID]
        for coeff, fine_var in zip(
            coarse_var.multigrid.interpolation_coefficients,
            coarse_var.multigrid.interpolation_vars,
        ):
            if fine_var.type == "dead":
                continue
            fi = fine_ix[fine_var.variableID]
            block = np.asarray(coeff, dtype=float)
            p[fi * dof : (fi + 1) * dof, ci * dof : (ci + 1) * dof] = block

    return r, p


def build_exact_rap_levels(graph) -> list[dict[str, np.ndarray | None]]:
    _, a0 = graph.joint_distribution_inf_level(0)
    a_level = np.asarray(a0, dtype=float)
    levels: list[dict[str, np.ndarray | None]] = [{"a": a_level, "r": None, "p": None}]

    for coarse_level in range(1, len(graph.multigrid_vars)):
        r, p = build_transfer_operators(graph, coarse_level=coarse_level)
        levels[-1]["r"] = r
        levels[-1]["p"] = p
        a_level = r @ a_level @ p
        levels.append({"a": a_level, "r": None, "p": None})

    return levels


def jacobi_sweep(a: np.ndarray, b: np.ndarray, x: np.ndarray, diag_inv: np.ndarray, omega: float) -> np.ndarray:
    return x + omega * (diag_inv * (b - a @ x))


def block_jacobi_sweep(
    a: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    inv_blocks: np.ndarray,
    block_dof: int,
    omega: float,
) -> np.ndarray:
    residual = b - a @ x
    delta = np.zeros_like(x)
    for block_idx, inv_block in enumerate(inv_blocks):
        sl = slice(block_idx * block_dof, (block_idx + 1) * block_dof)
        delta[sl] = inv_block @ residual[sl]
    return x + omega * delta


def run_base_jacobi(
    a0: np.ndarray,
    b0: np.ndarray,
    x0: np.ndarray,
    x_star: np.ndarray,
    omega: float,
    max_iters: int,
    tol: float,
    block_dof: int | None = None,
) -> tuple[int | None, list[float]]:
    x = x0.copy()
    diag_inv = None if block_dof is not None else (1.0 / np.diag(a0))
    inv_blocks = None if block_dof is None else build_block_diag_inv(a0, block_dof)
    relerrs = [relative_error_vec(x, x_star)]
    conv = 0 if relerrs[-1] < tol else None
    for it in range(1, max_iters + 1):
        if block_dof is None:
            x = jacobi_sweep(a0, b0, x, diag_inv, omega)
        else:
            x = block_jacobi_sweep(a0, b0, x, inv_blocks, block_dof, omega)
        rel = relative_error_vec(x, x_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = it
            break
    return conv, relerrs


def run_two_level_jacobi(
    a0: np.ndarray,
    b0: np.ndarray,
    a1: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
    x0: np.ndarray,
    x_star: np.ndarray,
    omega: float,
    max_cycles: int,
    tol: float,
    coarse_sweeps: int = 2,
    block_dof: int | None = None,
) -> tuple[int | None, list[float]]:
    x = x0.copy()
    d0_inv = None if block_dof is not None else (1.0 / np.diag(a0))
    d1_inv = None if block_dof is not None else (1.0 / np.diag(a1))
    inv0_blocks = None if block_dof is None else build_block_diag_inv(a0, block_dof)
    inv1_blocks = None if block_dof is None else build_block_diag_inv(a1, block_dof)

    relerrs = [relative_error_vec(x, x_star)]
    conv = 0 if relerrs[-1] < tol else None

    for cyc in range(1, max_cycles + 1):
        if block_dof is None:
            x = jacobi_sweep(a0, b0, x, d0_inv, omega)
        else:
            x = block_jacobi_sweep(a0, b0, x, inv0_blocks, block_dof, omega)

        residual = b0 - a0 @ x
        coarse_rhs = r @ residual
        coarse_x = np.zeros_like(coarse_rhs)
        for _ in range(coarse_sweeps):
            if block_dof is None:
                coarse_x = jacobi_sweep(a1, coarse_rhs, coarse_x, d1_inv, omega)
            else:
                coarse_x = block_jacobi_sweep(a1, coarse_rhs, coarse_x, inv1_blocks, block_dof, omega)

        x = x + p @ coarse_x
        if block_dof is None:
            x = jacobi_sweep(a0, b0, x, d0_inv, omega)
        else:
            x = block_jacobi_sweep(a0, b0, x, inv0_blocks, block_dof, omega)

        rel = relative_error_vec(x, x_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = cyc
            break

    return conv, relerrs


def v_cycle_multilevel(
    levels: list[dict[str, np.ndarray | None]],
    level_idx: int,
    x: np.ndarray,
    b: np.ndarray,
    omega: float,
    block_dof: int | None,
) -> np.ndarray:
    a = np.asarray(levels[level_idx]["a"], dtype=float)
    is_coarsest = level_idx == len(levels) - 1
    if is_coarsest:
        return np.linalg.solve(a, b)

    if block_dof is None:
        d_inv = 1.0 / np.diag(a)
        x = jacobi_sweep(a, b, x, d_inv, omega)
    else:
        inv_blocks = build_block_diag_inv(a, block_dof)
        x = block_jacobi_sweep(a, b, x, inv_blocks, block_dof, omega)

    residual = b - a @ x
    r = np.asarray(levels[level_idx]["r"], dtype=float)
    p = np.asarray(levels[level_idx]["p"], dtype=float)
    coarse_rhs = r @ residual
    coarse_error = np.zeros_like(coarse_rhs)
    coarse_error = v_cycle_multilevel(
        levels=levels,
        level_idx=level_idx + 1,
        x=coarse_error,
        b=coarse_rhs,
        omega=omega,
        block_dof=block_dof,
    )
    x = x + p @ coarse_error

    if block_dof is None:
        x = jacobi_sweep(a, b, x, d_inv, omega)
    else:
        x = block_jacobi_sweep(a, b, x, inv_blocks, block_dof, omega)

    return x


def run_multilevel_jacobi(
    levels: list[dict[str, np.ndarray | None]],
    b0: np.ndarray,
    x0: np.ndarray,
    x_star: np.ndarray,
    omega: float,
    max_cycles: int,
    tol: float,
    block_dof: int | None = None,
) -> tuple[int | None, list[float]]:
    x = x0.copy()
    relerrs = [relative_error_vec(x, x_star)]
    conv = 0 if relerrs[-1] < tol else None

    for cyc in range(1, max_cycles + 1):
        x = v_cycle_multilevel(
            levels=levels,
            level_idx=0,
            x=x,
            b=b0,
            omega=omega,
            block_dof=block_dof,
        )
        rel = relative_error_vec(x, x_star)
        relerrs.append(rel)
        if conv is None and rel < tol:
            conv = cyc
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
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--max-cycles", type=int, default=1500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--coarse-sweeps", type=int, default=2)
    parser.add_argument(
        "--coarse-operator",
        type=str,
        default="graph",
        choices=["graph", "rap"],
    )
    parser.add_argument(
        "--jacobi-mode",
        type=str,
        default="point",
        choices=["point", "block2", "both"],
    )
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 800, 1200],
    )
    args = parser.parse_args()

    graph = build_slam_graph(n=args.n, seed=0)
    x0 = base_mean_vector(graph)
    x_star = exact_mean(graph)

    build_hierarchy(graph)
    _, a0 = graph.joint_distribution_inf_level(0)
    b0, _ = graph.joint_distribution_inf_level(0)
    _, a1_graph = graph.joint_distribution_inf_level(1)
    r, p = build_transfer_operators(graph, coarse_level=1)
    rap_levels = build_exact_rap_levels(graph)

    a0 = np.asarray(a0, dtype=float)
    a1_graph = np.asarray(a1_graph, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    a1 = a1_graph if args.coarse_operator == "graph" else (r @ a0 @ p)

    print(f"levels {[len(level) for level in graph.multigrid_vars]}")
    print(f"A0 {a0.shape} A1 {a1.shape} R {r.shape} P {p.shape}")
    print(f"initial_rel {relative_error_vec(x0, x_star)}")
    print(
        "coarse_diff_rel "
        f"{np.linalg.norm((r @ a0 @ p) - a1_graph) / max(np.linalg.norm(a1_graph), 1e-15)}"
    )

    configs: list[tuple[str, int | None]] = []
    if args.jacobi_mode in {"point", "both"}:
        configs.append(("point", None))
    if args.jacobi_mode in {"block2", "both"}:
        configs.append(("block2", 2))

    for mode_name, block_dof in configs:
        base_conv, base_relerrs = run_base_jacobi(
            a0=a0,
            b0=b0,
            x0=x0,
            x_star=x_star,
            omega=args.omega,
            max_iters=args.max_iters,
            tol=args.tol,
            block_dof=block_dof,
        )
        mg_conv, mg_relerrs = run_two_level_jacobi(
            a0=a0,
            b0=b0,
            a1=a1,
            r=r,
            p=p,
            x0=x0,
            x_star=x_star,
            omega=args.omega,
            max_cycles=args.max_cycles,
            tol=args.tol,
            coarse_sweeps=args.coarse_sweeps,
            block_dof=block_dof,
        )
        multi_conv, multi_relerrs = run_multilevel_jacobi(
            levels=rap_levels,
            b0=b0,
            x0=x0,
            x_star=x_star,
            omega=args.omega,
            max_cycles=args.max_cycles,
            tol=args.tol,
            block_dof=block_dof,
        )

        print_summary(f"base_jacobi_{mode_name}", base_conv, base_relerrs, args.points)
        print_summary(f"two_level_jacobi_{mode_name}", mg_conv, mg_relerrs, args.points)
        print_summary(f"multilevel_jacobi_{mode_name}", multi_conv, multi_relerrs, args.points)


if __name__ == "__main__":
    main()
