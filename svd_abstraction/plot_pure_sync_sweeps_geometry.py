from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs
from svd_abstraction.pose_graph import build_noisy_pose_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import (
    mean_vector,
    odom_tiny_init_graph,
    relative_error_vec,
    reset_residual_graph,
    var_slices,
)


def vector_to_xy(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(n, 2)


def build_residual_graph(nodes, edges, seed: int):
    return build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=1.0,
        odom_sigma=1.0,
        tiny_prior=1e-12,
        seed=seed,
    )


def run_pure_sync_checkpoints(
    nodes,
    edges,
    template_graph,
    x0: np.ndarray,
    slices: dict[int, slice],
    sweeps_list: list[int],
    seed: int,
) -> dict[int, np.ndarray]:
    checkpoints = sorted(set(int(s) for s in sweeps_list))
    if not checkpoints:
        return {}

    residual_graph = build_residual_graph(nodes, edges, seed)
    reset_residual_graph(residual_graph, template_graph, x0, slices)

    out: dict[int, np.ndarray] = {}
    for sweep in range(1, checkpoints[-1] + 1):
        residual_graph.synchronous_iteration(fixed_lam=False)
        if sweep in checkpoints:
            e = mean_vector(residual_graph)
            out[sweep] = x0 + e
    return out


def plot_geometry(
    x_map: np.ndarray,
    x_odom: np.ndarray,
    odom_rel_err: float,
    pure_sync_curves: list[tuple[int, np.ndarray, float]],
    n: int,
    output: pathlib.Path,
    title: str,
) -> None:
    xy_map = vector_to_xy(x_map, n)
    xy_odom = vector_to_xy(x_odom, n)

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    ax.plot(xy_map[:, 0], xy_map[:, 1], color="#111111", lw=2.8, label="MAP")
    ax.plot(
        xy_odom[:, 0],
        xy_odom[:, 1],
        color="#6f6f6f",
        lw=1.8,
        alpha=0.95,
        linestyle="--",
        label=f"after odometry, rel={odom_rel_err:.4f}",
    )

    colors = ["#d04f33", "#cc5fa8", "#1f77b4"]
    for color, (sweeps, x_curve, rel_err) in zip(colors, pure_sync_curves):
        xy = vector_to_xy(x_curve, n)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            lw=1.8,
            alpha=0.92,
            label=f"pure sync T={sweeps}, rel={rel_err:.4f}",
        )

    marker_ids = [0, 100, 200, 300, 400, n - 1]
    for idx in marker_ids:
        ax.scatter(xy_map[idx, 0], xy_map[idx, 1], s=18, color="#111111")
        ax.text(xy_map[idx, 0], xy_map[idx, 1], f" {idx}", fontsize=8, color="#111111")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    all_xy = [xy_map, xy_odom] + [vector_to_xy(x_curve, n) for _, x_curve, _ in pure_sync_curves]
    all_pts = np.vstack(all_xy)
    x_min, y_min = np.min(all_pts, axis=0)
    x_max, y_max = np.max(all_pts, axis=0)
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    zoom_x0 = x_min + 0.58 * x_span
    zoom_x1 = x_min + 0.92 * x_span
    zoom_y0 = y_min + 0.38 * y_span
    zoom_y1 = y_min + 0.72 * y_span

    axins = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=1.4)
    axins.plot(xy_map[:, 0], xy_map[:, 1], color="#111111", lw=2.2)
    axins.plot(
        xy_odom[:, 0],
        xy_odom[:, 1],
        color="#6f6f6f",
        lw=1.5,
        alpha=0.95,
        linestyle="--",
    )
    for color, (_, x_curve, _) in zip(colors, pure_sync_curves):
        xy = vector_to_xy(x_curve, n)
        axins.plot(xy[:, 0], xy[:, 1], color=color, lw=1.6, alpha=0.95)
    axins.set_xlim(zoom_x0, zoom_x1)
    axins.set_ylim(zoom_y0, zoom_y1)
    axins.set_aspect("equal", adjustable="box")
    axins.grid(alpha=0.25)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=0.9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sweeps",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000],
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("svd_abstraction/out/pure_sync_sweeps_geometry.png"),
    )
    args = parser.parse_args()

    nodes, edges, exact_graph, base_graph, _ = build_graphs(
        n=args.n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=args.seed,
    )
    x_star, _ = exact_graph.joint_distribution_cov_absolute()

    odom_tiny_init_graph(base_graph, n=args.n)
    x0 = mean_vector(base_graph)
    odom_rel_err = relative_error_vec(x0, x_star)
    slices = var_slices(base_graph)

    x_by_sweeps = run_pure_sync_checkpoints(
        nodes=nodes,
        edges=edges,
        template_graph=base_graph,
        x0=x0,
        slices=slices,
        sweeps_list=args.sweeps,
        seed=args.seed,
    )

    pure_sync_curves: list[tuple[int, np.ndarray, float]] = []
    for sweeps in sorted(int(s) for s in args.sweeps):
        x_curve = x_by_sweeps[int(sweeps)]
        rel_err = relative_error_vec(x_curve, x_star)
        pure_sync_curves.append((int(sweeps), x_curve, rel_err))
        print(f"pure_sync T={int(sweeps)} relative_state_error={rel_err}")

    title = (
        f"N={args.n}, seed={args.seed}\n"
        f"MAP vs pure synchronous GBP on residual problem"
    )
    plot_geometry(
        x_map=x_star,
        x_odom=x0,
        odom_rel_err=odom_rel_err,
        pure_sync_curves=pure_sync_curves,
        n=args.n,
        output=args.output,
        title=title,
    )
    print(f"after_odometry relative_state_error={odom_rel_err}")
    print(f"saved_plot={args.output}")


if __name__ == "__main__":
    main()
