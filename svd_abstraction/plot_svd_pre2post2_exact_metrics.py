from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs, group_list
from svd_abstraction.plot_map_base_multigrid_geometry import objective_value
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import (
    build_joint_cov_basis,
    mean_vector,
    odom_tiny_init_graph,
    relative_error_vec,
    residual_block_step,
    var_slices,
)


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def residual_norm(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - a @ x))


def energy_gap(a: np.ndarray, x: np.ndarray, x_star: np.ndarray) -> float:
    e = np.asarray(x - x_star, dtype=float)
    return float(0.5 * e @ (a @ e))


def factor_objective(graph, x: np.ndarray, slices: dict[int, slice]) -> float:
    total = 0.0
    for factor in graph.factors[: graph.n_factor_nodes]:
        local_x = np.concatenate(
            [np.asarray(x[slices[int(var.variableID)]]).reshape(-1) for var in factor.adj_var_nodes]
        )
        pred_measurement = factor.meas_fn(local_x, *factor.args)
        for lam_z, meas, pred in zip(factor.measurement_lambda, factor.measurement, pred_measurement):
            res = np.asarray(meas, dtype=float).reshape(-1) - np.asarray(pred, dtype=float).reshape(-1)
            total += 0.5 * float(res @ (np.asarray(lam_z, dtype=float) @ res))
    return float(total)


def run_pre2post2_exact_metrics(max_cycles: int = 20) -> dict[str, list[float] | dict[str, int] | str]:
    n = 512
    seed = 0
    group_size = 20
    r_reduced = 4

    nodes, edges, exact_graph, base_graph, mg_graph = build_graphs(
        n=n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=seed,
    )

    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    b0, a0 = exact_graph.joint_distribution_inf_absolute()

    groups = group_list(
        nodes=nodes,
        graph=mg_graph,
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
    p = build_joint_cov_basis(mg_graph, a0, groups, r_reduced=r_reduced)
    ac = p.T @ a0 @ p

    odom_tiny_init_graph(base_graph, n=n)
    x = mean_vector(base_graph)
    slices = var_slices(base_graph)

    residual_graph = build_graphs(
        n=n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=seed,
    )[2]  # exact_graph clone with same system structure

    cycles: list[int] = [0]
    relerrs: list[float] = [relative_error_vec(x, x_star)]
    residuals: list[float] = [residual_norm(a0, b0, x)]
    energies: list[float] = [energy_gap(a0, x, x_star)]
    objectives: list[float] = [factor_objective(exact_graph, x, slices)]

    for cyc in range(1, max_cycles + 1):
        x = x + residual_block_step(
            residual_graph,
            base_graph,
            x,
            slices,
            n_sweeps=2,
            preserve_lam=False,
            scheduler="sync",
        )
        yc = np.linalg.solve(ac, p.T @ (b0 - a0 @ x))
        x = x + p @ yc
        x = x + residual_block_step(
            residual_graph,
            base_graph,
            x,
            slices,
            n_sweeps=2,
            preserve_lam=False,
            scheduler="sync",
        )

        cycles.append(cyc)
        relerrs.append(relative_error_vec(x, x_star))
        residuals.append(residual_norm(a0, b0, x))
        energies.append(energy_gap(a0, x, x_star))
        objectives.append(factor_objective(exact_graph, x, slices))

    return {
        "title": "SVD GBP: pre2+post2, exact coarse solve, first 20 cycles",
        "config": {
            "N": n,
            "seed": seed,
            "prior_prop": 0,
            "grouping": "order",
            "group_size": group_size,
            "r_reduced": r_reduced,
            "base_scheduler": "sync",
            "base_sweeps_pre": 2,
            "base_sweeps_post": 2,
            "coarse_solver": "exact",
        },
        "cycles": cycles,
        "relative_state_error": relerrs,
        "algebraic_residual": residuals,
        "energy_gap": energies,
        "factor_objective": objectives,
    }


def main() -> None:
    data = run_pre2post2_exact_metrics(max_cycles=20)
    cycles = np.asarray(data["cycles"], dtype=int)
    relerrs = np.asarray(data["relative_state_error"], dtype=float)
    residuals = np.asarray(data["algebraic_residual"], dtype=float)
    energies = np.asarray(data["energy_gap"], dtype=float)
    objectives = np.asarray(data["factor_objective"], dtype=float)

    (OUT_DIR / "svd_pre2post2_exact_first20_metrics.json").write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    def draw(ax, y, title, ylabel, color):
        ax.plot(cycles, y, marker="o", linewidth=2.2, markersize=5.5, color=color)
        ax.set_yscale("log")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)

    draw(ax1, relerrs, "Relative State Error", r"$\|x_k-x^\star\|_2 / \|x^\star\|_2$", "#1d4ed8")
    draw(ax2, residuals, "Algebraic Residual", r"$\|b-Ax_k\|_2$", "#dc2626")
    draw(ax3, energies, "Energy Gap", r"$\frac{1}{2}(x_k-x^\star)^\top A (x_k-x^\star)$", "#059669")
    draw(ax4, objectives, "Factor Objective", r"$\sum_t \frac{1}{2} r_t(x_k)^\top \Lambda_t r_t(x_k)$", "#7c3aed")

    fig.suptitle(data["title"], fontsize=16)
    fig.savefig(OUT_DIR / "svd_pre2post2_exact_first20_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
