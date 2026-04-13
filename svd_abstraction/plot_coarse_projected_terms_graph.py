from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs
from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


def group_centers_from_gt(graph, groups: list[list[int]]) -> np.ndarray:
    centers = []
    for group in groups:
        pts = [np.asarray(graph.var_nodes[int(var_id)].GT, dtype=float).reshape(-1) for var_id in group]
        centers.append(np.mean(np.vstack(pts), axis=0))
    return np.vstack(centers)


def unique_binary_pairs(coarse_graph) -> Counter:
    pairs = []
    for factor in coarse_graph.factors[: coarse_graph.n_factor_nodes]:
        if len(factor.adj_var_nodes) != 2:
            continue
        i, j = [int(v.variableID) for v in factor.adj_var_nodes]
        pairs.append(tuple(sorted((i, j))))
    return Counter(pairs)


def plot_coarse_graph(
    fine_gt: np.ndarray,
    coarse_centers: np.ndarray,
    pair_counter: Counter,
    output: pathlib.Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)

    ax.plot(
        fine_gt[:, 0],
        fine_gt[:, 1],
        color="#c7ced6",
        lw=1.2,
        alpha=0.9,
        label="fine GT trajectory",
        zorder=1,
    )

    if pair_counter:
        max_mult = max(pair_counter.values())
        for (i, j), mult in pair_counter.items():
            p0 = coarse_centers[i]
            p1 = coarse_centers[j]
            width = 0.8 + 2.8 * (mult / max_mult)
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="#f28e2b",
                alpha=0.55,
                lw=width,
                zorder=2,
            )
            mid = 0.5 * (p0 + p1)
            ax.text(
                mid[0],
                mid[1],
                str(mult),
                fontsize=7,
                color="#7a4a12",
                ha="center",
                va="center",
                zorder=4,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.65},
            )

    ax.scatter(
        coarse_centers[:, 0],
        coarse_centers[:, 1],
        s=120,
        color="#1f77b4",
        edgecolors="white",
        linewidths=1.2,
        label="coarse variables",
        zorder=3,
    )
    for idx, xy in enumerate(coarse_centers):
        ax.text(
            xy[0],
            xy[1],
            str(idx),
            fontsize=8,
            color="white",
            ha="center",
            va="center",
            zorder=5,
            fontweight="bold",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grouping", type=str, default="order", choices=["order", "grid", "kmeans", "loop_aware", "degree_aware"])
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--gx", type=int, default=8)
    parser.add_argument("--gy", type=int, default=4)
    parser.add_argument("--kmeans-k", type=int, default=26)
    parser.add_argument("--target-groups", type=int, default=None)
    parser.add_argument("--loop-window", type=int, default=2)
    parser.add_argument("--loop-boost", type=float, default=3.0)
    parser.add_argument("--degree-boost", type=float, default=1.0)
    parser.add_argument("--loop-sep-min", type=int, default=2)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("svd_abstraction/out/coarse_projected_terms_graph_r4.png"),
    )
    args = parser.parse_args()

    nodes, _, _, _, mg_graph = build_graphs(
        n=args.n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=args.seed,
    )

    groups = group_list(
        nodes=nodes,
        graph=mg_graph,
        method=args.grouping,
        group_size=args.group_size,
        gx=args.gx,
        gy=args.gy,
        kmeans_k=args.kmeans_k,
        target_groups=args.target_groups,
        loop_window=args.loop_window,
        loop_boost=args.loop_boost,
        degree_boost=args.degree_boost,
        loop_sep_min=args.loop_sep_min,
    )

    abstraction = SVDResidualAbstraction(
        base_graph=mg_graph,
        groups=groups,
        r_reduced=args.r_reduced,
        basis_source="joint_covariance",
        freeze_basis=True,
        eta_assignment_mode="projected_terms",
        absolute_system=True,
    )
    abstraction.initialize_bases(force=True)
    abstraction.build_coarse_graph(force=True)

    fine_gt = np.vstack(
        [np.asarray(var.GT, dtype=float).reshape(-1) for var in mg_graph.var_nodes[: mg_graph.n_var_nodes]]
    )
    coarse_centers = group_centers_from_gt(mg_graph, groups)
    pair_counter = unique_binary_pairs(abstraction.coarse_graph)

    plot_coarse_graph(
        fine_gt=fine_gt,
        coarse_centers=coarse_centers,
        pair_counter=pair_counter,
        output=args.output,
        title=(
            f"Coarse Graph (projected_terms): {len(groups)} vars, "
            f"{len(pair_counter)} unique between-edges, r={args.r_reduced}"
        ),
    )

    print(f"output={args.output}")
    print(f"num_groups={len(groups)}")
    print(f"coarse_total_dim={sum(v.dofs for v in abstraction.coarse_graph.var_nodes[:abstraction.coarse_graph.n_var_nodes])}")
    print(f"num_unique_between_edges={len(pair_counter)}")
    print(f"num_total_binary_factors={sum(pair_counter.values())}")
    print(f"max_parallel_mult={max(pair_counter.values()) if pair_counter else 0}")


if __name__ == "__main__":
    main()
