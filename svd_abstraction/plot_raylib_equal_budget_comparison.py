from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    budgets = np.array([0, 20, 100, 500, 1000, 2000, 5000, 10000], dtype=float)

    base_sync = np.array(
        [
            0.034997922253041436,
            0.03422715374501403,
            0.03217397019672841,
            0.029883941543540283,
            0.02903784353918452,
            0.027522981626200142,
            0.023421334597726662,
            0.01789783463759962,
        ],
        dtype=float,
    )

    two_level_standard = np.array(
        [
            0.034997922253041436,
            0.03298286741792501,
            0.03032105770943911,
            0.02765311594594706,
            0.024894609076933406,
            0.020178799193611586,
            0.010746463586786699,
            0.003760247732189085,
        ],
        dtype=float,
    )

    two_level_pre2post2 = np.array(
        [
            0.034997922253041436,
            0.033727354303912715,
            0.031239907315114616,
            0.029128963742194484,
            0.027787320816106836,
            0.02530494196430895,
            0.019112750637598108,
            0.011972581675738751,
        ],
        dtype=float,
    )

    payload = {
        "metric": "relative_state_error",
        "formula": "||x_k - x*||_2 / ||x*||_2",
        "budgets_base_sweeps": budgets.tolist(),
        "base_sync": base_sync.tolist(),
        "two_level_standard": two_level_standard.tolist(),
        "two_level_pre2post2": two_level_pre2post2.tolist(),
        "two_level_levels": [512, 291],
        "notes": {
            "standard_cycle_cost": "1 base sweep / cycle",
            "pre2post2_cycle_cost": "4 base sweeps / cycle",
            "budget_match": "10000 base sweeps = 10000 standard cycles = 2500 pre2+post2 cycles",
            "multilevel_status": "still running separately",
        },
    }

    (OUT_DIR / "raylib_equal_budget_comparison.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig = plt.figure(figsize=(12.5, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.2], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    x_plot = budgets.copy()
    x_plot[0] = 10.0

    ax.plot(
        x_plot,
        base_sync,
        marker="o",
        linewidth=2.6,
        markersize=7,
        color="#6b7280",
        label="Base sync",
    )
    ax.plot(
        x_plot,
        two_level_standard,
        marker="s",
        linewidth=2.8,
        markersize=7,
        color="#1d4ed8",
        label="Two-level standard",
    )
    ax.plot(
        x_plot,
        two_level_pre2post2,
        marker="^",
        linewidth=2.8,
        markersize=7,
        color="#dc2626",
        label="Two-level pre2+post2",
    )

    ax.set_xscale("log")
    ax.set_xlim(10, 12000)
    ax.set_ylim(0.0, 0.0375)
    ax.set_xticks([10, 20, 100, 500, 1000, 2000, 5000, 10000])
    ax.set_xticklabels(["0", "20", "100", "500", "1000", "2000", "5000", "10000"])
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", alpha=0.12)
    ax.set_xlabel("Base-Sweep Budget")
    ax.set_ylabel(r"Relative State Error")
    ax.set_title("Raylib Equal-Budget Comparison on prior_prop = 0")
    ax.legend(loc="upper right", frameon=True)

    ax.text(
        0.03,
        0.08,
        "Same graph, same hierarchy, same odometry init\n"
        "Two-level hierarchy sizes: [512, 291]",
        transform=ax.transAxes,
        fontsize=11.5,
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="0.85"),
    )

    explanation = (
        r"$\mathrm{metric}=\|x_k-x^\star\|_2/\|x^\star\|_2$" "\n"
        r"$\mathrm{standard}:$ 1 base sweep / cycle" "\n"
        r"$\mathrm{pre2+post2}:$ 4 base sweeps / cycle" "\n"
        r"$10000$ base sweeps $=$ $10000$ standard cycles $=$ $2500$ pre2+post2 cycles" "\n"
        r"Lower is better."
    )
    ax_text.text(
        0.02,
        0.82,
        explanation,
        va="top",
        ha="left",
        fontsize=14,
    )
    ax_text.text(
        0.73,
        0.82,
        "Multilevel fair-budget run\n"
        "is still running separately.",
        va="top",
        ha="left",
        fontsize=12.5,
        color="#7c2d12",
    )

    fig.savefig(OUT_DIR / "raylib_equal_budget_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
