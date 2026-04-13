from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs, group_list
from svd_abstraction.pose_graph import build_noisy_pose_graph
from svd_abstraction.residual_base_gbp_direct_coarse_experiment import (
    build_joint_cov_basis,
    mean_vector,
    odom_tiny_init_graph,
    relative_error_vec,
    residual_block_step,
    var_slices,
)


def vector_to_xy(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(n, 2)


def residual_norm(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(b - a @ x))


def objective_value(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x @ (a @ x) - b @ x)


def weaken_loop_factors(graph, sigma_ratio: float) -> int:
    if sigma_ratio <= 1.0:
        return 0

    weakened = 0
    for factor in graph.factors[: graph.n_factor_nodes]:
        if len(getattr(factor, "adj_vIDs", [])) != 2:
            continue
        i, j = [int(v) for v in factor.adj_vIDs]
        if abs(i - j) <= 1:
            continue

        factor.measurement_lambda = [
            np.asarray(lam, dtype=float) / (sigma_ratio**2) for lam in factor.measurement_lambda
        ]
        linpoint = np.concatenate([np.asarray(v.GT).reshape(-1) for v in factor.adj_var_nodes])
        factor.compute_factor(linpoint=linpoint, update_self=True)
        weakened += 1
    return weakened


def run_twolevel_final(
    residual_graph,
    template_graph,
    a0: np.ndarray,
    b0: np.ndarray,
    p: np.ndarray,
    x0: np.ndarray,
    x_star: np.ndarray,
    slices: dict[int, slice],
    base_sweeps: int,
    pre_post: bool,
    max_cycles: int,
    base_scheduler: str,
) -> tuple[np.ndarray, list[float]]:
    x = x0.copy()
    ac = p.T @ a0 @ p
    hist = [relative_error_vec(x, x_star)]
    for _ in range(int(max_cycles)):
        x = x + residual_block_step(
            residual_graph,
            template_graph,
            x,
            slices,
            n_sweeps=base_sweeps,
            preserve_lam=False,
            scheduler=base_scheduler,
        )
        residual = b0 - a0 @ x
        yc = np.linalg.solve(ac, p.T @ residual)
        x = x + p @ yc
        if pre_post:
            x = x + residual_block_step(
                residual_graph,
                template_graph,
                x,
                slices,
                n_sweeps=base_sweeps,
                preserve_lam=False,
                scheduler=base_scheduler,
            )
        hist.append(relative_error_vec(x, x_star))
    return x, hist


def build_residual_graph(nodes, edges, seed: int):
    return build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=1.0,
        odom_sigma=1.0,
        tiny_prior=1e-12,
        seed=seed,
    )


def plot_geometry(
    x_map: np.ndarray,
    x_base: np.ndarray,
    x_mg: np.ndarray,
    n: int,
    output: pathlib.Path,
    title: str,
) -> None:
    xy_map = vector_to_xy(x_map, n)
    xy_base = vector_to_xy(x_base, n)
    xy_mg = vector_to_xy(x_mg, n)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.plot(xy_map[:, 0], xy_map[:, 1], color="#111111", lw=2.5, label="MAP")
    ax.plot(
        xy_base[:, 0],
        xy_base[:, 1],
        color="#d04f33",
        lw=1.8,
        alpha=0.9,
        label="base-only sync T=1000",
    )
    ax.plot(
        xy_mg[:, 0],
        xy_mg[:, 1],
        color="#1f77b4",
        lw=1.8,
        alpha=0.9,
        label="two-level pre2+post2 cycle100",
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
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--base-t", type=int, default=1000)
    parser.add_argument("--mg-cycles", type=int, default=100)
    parser.add_argument("--loop-sigma-ratio", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("svd_abstraction/out/map_base_multigrid_geometry.png"),
    )
    args = parser.parse_args()

    nodes, edges, exact_graph, base_graph, mg_graph = build_graphs(
        n=args.n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=args.seed,
    )
    weakened = weaken_loop_factors(exact_graph, args.loop_sigma_ratio)
    weaken_loop_factors(base_graph, args.loop_sigma_ratio)
    weaken_loop_factors(mg_graph, args.loop_sigma_ratio)
    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    b0, a0 = exact_graph.joint_distribution_inf_absolute()

    groups = group_list(
        nodes=nodes,
        graph=mg_graph,
        method="order",
        group_size=args.group_size,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    p = build_joint_cov_basis(mg_graph, a0, groups, r_reduced=args.r_reduced)

    odom_tiny_init_graph(base_graph, n=args.n)
    x0 = mean_vector(base_graph)
    slices = var_slices(base_graph)

    residual_base_graph = build_residual_graph(nodes, edges, args.seed)
    weaken_loop_factors(residual_base_graph, args.loop_sigma_ratio)
    e_base = residual_block_step(
        residual_base_graph,
        base_graph,
        x0,
        slices,
        n_sweeps=args.base_t,
        preserve_lam=False,
        scheduler="sync",
    )
    x_base = x0 + e_base

    residual_mg_graph = build_residual_graph(nodes, edges, args.seed)
    weaken_loop_factors(residual_mg_graph, args.loop_sigma_ratio)
    x_mg, hist_mg = run_twolevel_final(
        residual_graph=residual_mg_graph,
        template_graph=base_graph,
        a0=a0,
        b0=b0,
        p=p,
        x0=x0,
        x_star=x_star,
        slices=slices,
        base_sweeps=2,
        pre_post=True,
        max_cycles=args.mg_cycles,
        base_scheduler="sync",
    )

    title = (
        f"N={args.n}, seed={args.seed}\n"
        f"MAP vs base-only sync T={args.base_t} vs two-level cycle{args.mg_cycles}"
    )
    if args.loop_sigma_ratio > 1.0:
        title += f"\nloop sigma ratio={args.loop_sigma_ratio:g}"
    plot_geometry(
        x_map=x_star,
        x_base=x_base,
        x_mg=x_mg,
        n=args.n,
        output=args.output,
        title=title,
    )

    print(f"saved_plot={args.output}")
    print(f"loop_factors_weakened={weakened}")
    print(f"x0_rel={relative_error_vec(x0, x_star)}")
    print(
        "base_sync "
        f"x_rel={relative_error_vec(x_base, x_star)} "
        f"residual={residual_norm(a0, b0, x_base)} "
        f"objective={objective_value(a0, b0, x_base)}"
    )
    print(
        "two_level "
        f"x_rel={relative_error_vec(x_mg, x_star)} "
        f"residual={residual_norm(a0, b0, x_mg)} "
        f"objective={objective_value(a0, b0, x_mg)} "
        f"hist_last={hist_mg[-1]}"
    )


if __name__ == "__main__":
    main()
