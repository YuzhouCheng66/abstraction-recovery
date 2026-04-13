from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.raylib_baware_interp_analysis import odom_tiny_init_base
from svd_abstraction.raylib_grouped_svd_benchmark import (
    build_grouped_svd_basis,
    group_list,
    ideal_coarse_correction,
    projection_residual,
    relative_error_vec,
)
from svd_abstraction.raylib_local_eta_prolongation_validation import (
    build_hierarchy,
    build_slam_graph,
    exact_mean,
    mean_vector,
)
from svd_abstraction.raylib_same_hierarchy_jacobi_experiment import build_transfer_operators


def build_metrics():
    graph = build_slam_graph(n=512, seed=0, prior_prop=0.0)
    mu_star = exact_mean(graph)
    b0, a0 = graph.joint_distribution_inf_level(0)
    a0 = np.asarray(a0, dtype=float)
    b0 = np.asarray(b0, dtype=float)

    build_hierarchy(graph)
    _, p_raylib = build_transfer_operators(graph, coarse_level=1)

    odom_tiny_init_base(graph, n=512)
    x0 = mean_vector(graph)
    e0 = x0 - mu_star

    graph_svd = build_slam_graph(n=512, seed=0, prior_prop=0.0)
    groups = group_list(
        graph_svd,
        method="order",
        group_size=20,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    p_svd = build_grouped_svd_basis(
        graph_svd,
        a0=a0,
        groups=groups,
        r_reduced=2,
        basis_source="joint_covariance",
    )

    rel_before = relative_error_vec(x0, mu_star)

    out = []
    for name, p in [("Raylib P", p_raylib), ("Grouped-SVD P", p_svd)]:
        proj = projection_residual(e0, p)
        x_corr = ideal_coarse_correction(a0, b0, x0, p)
        rel_after = relative_error_vec(x_corr, mu_star)
        out.append(
            {
                "name": name,
                "shape": p.shape,
                "coarse_dim": int(p.shape[1]),
                "nnz": int(np.count_nonzero(np.abs(p) > 0.0)),
                "proj_resid": float(proj),
                "rel_before": float(rel_before),
                "rel_after": float(rel_after),
                "improve": float(rel_before / max(rel_after, 1e-15)),
            }
        )
    return out


def make_plot(metrics: list[dict], output: pathlib.Path) -> None:
    raylib = metrics[0]
    svd = metrics[1]

    fig = plt.figure(figsize=(14.5, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_text = fig.add_subplot(gs[:, 0])
    ax_proj = fig.add_subplot(gs[0, 1:])
    ax_corr = fig.add_subplot(gs[1, 1:])

    ax_text.axis("off")
    title = "Odometry-Init Prolongation Expressiveness (N=512, seed=0)"
    ax_text.text(
        0.0,
        1.0,
        title,
        va="top",
        ha="left",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Sans",
    )

    top_block = [
        r"Top metric: projection expressiveness",
        r"$e_0 = x_{\mathrm{odom}} - x^\star$",
        r"$\rho(P) = \min_y \frac{\|e_0 - P y\|_2}{\|e_0\|_2}$",
        "Interpretation: how much of the odometry error lies in Range(P).",
        "Lower is better.",
    ]
    ax_text.text(
        0.0,
        0.88,
        "\n\n".join(top_block),
        va="top",
        ha="left",
        fontsize=15,
        linespacing=1.45,
        family="DejaVu Sans",
    )

    bottom_block = [
        r"Bottom metric: one ideal exact coarse correction",
        r"$x^+ = x_{\mathrm{odom}} + P(P^\top A P)^{-1}P^\top(b-Ax_{\mathrm{odom}})$",
        r"$\epsilon^+(P)=\frac{\|x^+-x^\star\|_2}{\|x^\star\|_2}$",
        "Interpretation: how much one exact coarse correction can reduce the",
        "state error when using this P.",
        "Lower is better.",
        f"Odometry initial error = {raylib['rel_before']:.6f}",
    ]
    ax_text.text(
        0.0,
        0.36,
        "\n\n".join(bottom_block),
        va="top",
        ha="left",
        fontsize=15,
        linespacing=1.45,
        family="DejaVu Sans",
    )

    labels = [raylib["name"], svd["name"]]
    colors = ["#d04f33", "#1f77b4"]
    proj_vals = [raylib["proj_resid"], svd["proj_resid"]]
    corr_vals = [raylib["rel_after"], svd["rel_after"]]

    bars1 = ax_proj.bar(labels, proj_vals, color=colors, alpha=0.92)
    ax_proj.set_title("Projection Residual on Odometry Error", fontsize=15, pad=12)
    ax_proj.set_ylabel(r"$\rho(P)$", fontsize=14)
    ax_proj.tick_params(axis="x", labelsize=12)
    ax_proj.tick_params(axis="y", labelsize=12)
    ax_proj.grid(axis="y", alpha=0.25)
    for bar, m in zip(bars1, metrics):
        ax_proj.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{m['proj_resid']:.4f}\n(dim={m['coarse_dim']})",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    bars2 = ax_corr.bar(labels, corr_vals, color=colors, alpha=0.92)
    ax_corr.set_title("Error After One Ideal Exact Coarse Correction", fontsize=15, pad=12)
    ax_corr.set_ylabel(r"$\epsilon^+(P)$", fontsize=14)
    ax_corr.tick_params(axis="x", labelsize=12)
    ax_corr.tick_params(axis="y", labelsize=12)
    ax_corr.grid(axis="y", alpha=0.25)
    for bar, m in zip(bars2, metrics):
        ax_corr.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{m['rel_after']:.5f}\n({m['improve']:.2f}x better)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    output = pathlib.Path("svd_abstraction/out/odometry_P_expressiveness_comparison.png")
    metrics = build_metrics()
    make_plot(metrics, output)
    for item in metrics:
        print(item)
    print(f"saved_plot={output.resolve()}")


if __name__ == "__main__":
    main()
