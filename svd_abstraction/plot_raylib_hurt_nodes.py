"""Plot raylib multigrid node-level hurt statistics on the toy pose graph."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.pose_graph import make_slam_like_graph
from svd_abstraction.raylib_recursive_experiment import build_multigrid_graph


def base_vars(graph):
    return [var for var in graph.multigrid_vars[0] if var.type != "dead"]


def run_standard_cycle_with_level10_hook(graph, callback=None):
    graph.synchronous_iteration(level=0)

    if len(graph.multigrid_vars) <= 1:
        return

    graph.update_all_residual_etas(level=1)
    graph.update_all_beliefs(level=1)
    graph.synchronous_iteration(level=1)

    for level in range(2, len(graph.multigrid_vars)):
        graph.update_all_residual_etas(level=level)
        graph.update_all_beliefs(level=level)
        graph.synchronous_iteration(level=level)

    for level in range(len(graph.multigrid_vars) - 1, 0, -1):
        graph.synchronous_iteration(level=level)
        if level == 1 and callback is not None:
            callback(graph)
        else:
            graph.prolongate_corrections(level=level)


def collect_level10_hurt_stats(graph, analysis_cycles):
    vars0 = base_vars(graph)
    per_node_ratios = defaultdict(list)

    for _ in range(int(analysis_cycles)):

        def level10_probe(active_graph):
            before = {
                var.variableID: float(np.linalg.norm(var.compute_residual()))
                for var in vars0
            }
            active_graph.prolongate_corrections(level=1)
            after = {
                var.variableID: float(np.linalg.norm(var.compute_residual()))
                for var in vars0
            }
            for var in vars0:
                vid = var.variableID
                ratio = after[vid] / max(before[vid], 1e-15)
                per_node_ratios[vid].append(ratio)

        run_standard_cycle_with_level10_hook(graph, callback=level10_probe)

    rows = []
    for var in vars0:
        ratios = np.array(per_node_ratios[var.variableID], dtype=float)
        rows.append(
            {
                "id": var.variableID,
                "xy": np.array(var.GT, dtype=float),
                "classification": var.multigrid.classification,
                "frac_hurt": float(np.mean(ratios > 1.0)),
                "avg_ratio": float(np.mean(ratios)),
                "num_l1_parents": len(var.multigrid.restriction_vars),
            }
        )

    rows.sort(key=lambda row: (-row["frac_hurt"], -row["avg_ratio"], row["id"]))
    return rows


def plot_hurt_map(nodes, edges, node_stats, output_path, label_top_k):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = {
        int(node["data"]["id"]): np.array(
            [node["position"]["x"], node["position"]["y"]], dtype=float
        )
        for node in nodes
    }

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    for edge in edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]
        if dst in {"prior", "anchor"}:
            continue
        i = int(src)
        j = int(dst)
        xy_i = pos[i]
        xy_j = pos[j]
        ax.plot(
            [xy_i[0], xy_j[0]],
            [xy_i[1], xy_j[1]],
            color="#c8c8c8",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )

    coarse_rows = [row for row in node_stats if row["classification"] == "coarse"]
    fine_rows = [row for row in node_stats if row["classification"] == "fine"]
    cmap = plt.get_cmap("inferno")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    for rows, marker, label, size in [
        (fine_rows, "o", "fine", 52),
        (coarse_rows, "s", "coarse", 72),
    ]:
        if not rows:
            continue
        xy = np.vstack([row["xy"] for row in rows])
        colors = [row["frac_hurt"] for row in rows]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=colors,
            cmap=cmap,
            norm=norm,
            s=size,
            marker=marker,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.95,
            label=label,
            zorder=3,
        )

    for row in node_stats[: int(label_top_k)]:
        x, y = row["xy"]
        ax.text(
            x + 1.0,
            y + 1.0,
            f"{row['id']}",
            fontsize=8,
            color="black",
            zorder=4,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("frac_hurt on level 1 -> 0")

    ax.legend(loc="best", frameon=True)
    ax.set_title("Raylib node-level hurt map (standard V-cycle)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    note = (
        "Color: frac_hurt; marker: coarse vs fine; "
        f"labels show top {int(label_top_k)} hurt nodes"
    )
    ax.text(
        0.01,
        0.01,
        note,
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
    )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=15.0)
    parser.add_argument("--loop-prob", type=float, default=0.08)
    parser.add_argument("--loop-radius", type=float, default=40.0)
    parser.add_argument("--prior-prop", type=float, default=0.15)
    parser.add_argument("--prior-sigma", type=float, default=6.0)
    parser.add_argument("--odom-sigma", type=float, default=3.0)
    parser.add_argument("--analysis-cycles", type=int, default=20)
    parser.add_argument("--max-total-levels", type=int, default=3)
    parser.add_argument("--label-top-k", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/artifacts/raylib_hurt_map.png"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    nodes, edges = make_slam_like_graph(
        N=args.N,
        step_size=args.step_size,
        loop_prob=args.loop_prob,
        loop_radius=args.loop_radius,
        prior_prop=args.prior_prop,
        seed=args.seed,
    )
    raylib_args = SimpleNamespace(
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
    )
    graph = build_multigrid_graph(
        nodes,
        edges,
        max_total_levels=args.max_total_levels,
        args=raylib_args,
    )
    node_stats = collect_level10_hurt_stats(graph, analysis_cycles=args.analysis_cycles)
    plot_hurt_map(
        nodes,
        edges,
        node_stats,
        output_path=args.output,
        label_top_k=args.label_top_k,
    )

    print(f"Saved plot to {args.output}")
    print("Top hurt nodes:")
    for row in node_stats[: min(12, len(node_stats))]:
        print(
            f"id={row['id']:>3d} class={row['classification']:<6s} "
            f"frac_hurt={row['frac_hurt']:.3f} avg_ratio={row['avg_ratio']:.3f} "
            f"num_l1_parents={row['num_l1_parents']}"
        )


if __name__ == "__main__":
    main()
