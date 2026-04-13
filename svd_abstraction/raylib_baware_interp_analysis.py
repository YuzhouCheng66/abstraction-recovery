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
from svd_abstraction.raylib_local_eta_prolongation_validation import mean_vector
from svd_abstraction.raylib_same_hierarchy_jacobi_experiment import build_transfer_operators
from svd_abstraction.raylib_variance_freeze_experiment import eta_only_iteration
from svd_abstraction.raylib_variance_freeze_experiment import lam_state


def odom_tiny_init_base(graph, n: int, tiny: float = 1e-12) -> None:
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


def collect_fixed_lam_errors(graph, mu_star: np.ndarray, sample_steps: list[int]) -> dict[int, np.ndarray]:
    errors = {}
    max_step = max(sample_steps)
    for step in range(max_step + 1):
        if step in sample_steps:
            errors[step] = (mean_vector(graph) - mu_star).copy()
        if step < max_step:
            eta_only_iteration(graph, 0)
    return errors


def projection_residuals(vectors: dict[int, np.ndarray], basis: np.ndarray) -> dict[int, float]:
    if basis.size == 0:
        return {step: 1.0 for step in vectors}

    coeff_map = np.linalg.pinv(basis.T @ basis) @ basis.T
    residuals = {}
    for step, vec in vectors.items():
        fit = basis @ (coeff_map @ vec)
        residuals[step] = float(np.linalg.norm(vec - fit) / max(np.linalg.norm(vec), 1e-15))
    return residuals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--sample-steps",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 2000],
    )
    parser.add_argument(
        "--eig-ranks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 10, 20, 40, 80, 166],
    )
    args = parser.parse_args()

    graph = build_slam_graph(n=args.n, seed=args.seed, prior_prop=args.prior_prop)
    mu_star = exact_mean(graph)
    _, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)

    build_hierarchy(graph)
    _, p = build_transfer_operators(graph, coarse_level=1)

    prev = lam_state(graph, [0])
    freeze_step = None
    for step in range(1, 1000):
        graph.synchronous_iteration(level=0)
        curr = lam_state(graph, [0])
        delta = float(np.max(np.abs(curr - prev)))
        if delta < args.variance_threshold:
            freeze_step = step
            break
        prev = curr

    if freeze_step is None:
        raise RuntimeError("Base lam did not settle within 1000 iterations")

    odom_tiny_init_base(graph, n=args.n)
    errors = collect_fixed_lam_errors(graph, mu_star, args.sample_steps)
    current_p_residuals = projection_residuals(errors, p)

    error_matrix = np.column_stack([errors[step] for step in args.sample_steps])
    left_modes, singular_values, _ = np.linalg.svd(error_matrix, full_matrices=False)

    eigvals, eigvecs = np.linalg.eigh(a0)

    print(f"n={args.n} seed={args.seed} prior_prop={args.prior_prop}")
    print(f"freeze_step={freeze_step}")
    print(f"P shape={p.shape}")
    print("current_P_projection_residuals")
    for step in args.sample_steps:
        print(f"  step {step}: {current_p_residuals[step]}")

    print(f"slow_error_singular_values={singular_values}")
    print(f"smallest_A_eigvals={eigvals[: min(10, len(eigvals))]}")

    print("A_low_mode_projection_residuals")
    for rank in args.eig_ranks:
        basis = eigvecs[:, :rank]
        residuals = projection_residuals(errors, basis)
        print(f"  rank {rank}: {[residuals[step] for step in args.sample_steps]}")

    print("slow_error_mode_vs_A_low_subspace")
    for mode_idx in range(left_modes.shape[1]):
        mode = left_modes[:, mode_idx]
        rel6 = float(np.linalg.norm(mode - eigvecs[:, :6] @ (eigvecs[:, :6].T @ mode)))
        rel20 = float(np.linalg.norm(mode - eigvecs[:, :20] @ (eigvecs[:, :20].T @ mode)))
        rel80 = float(np.linalg.norm(mode - eigvecs[:, :80] @ (eigvecs[:, :80].T @ mode)))
        print(f"  mode {mode_idx}: low6={rel6} low20={rel20} low80={rel80}")


if __name__ == "__main__":
    main()
