from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_JSON = pathlib.Path(
    "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results/"
    "se2_newton_vs_base_sync_n512_seed0_outer20_s1000.json"
)


def main() -> None:
    data = json.loads(INPUT_JSON.read_text())
    gt = np.asarray(data["gt_poses"], dtype=float)
    odom = np.asarray(data["init_poses"], dtype=float)
    direct = np.asarray(data["direct_newton"]["final_poses"], dtype=float)
    base_sync = np.asarray(data["base_sync_only"]["final_poses"], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)

    ax.plot(gt[:, 0], gt[:, 1], color="black", linewidth=1.6, label="GT", zorder=1)
    ax.plot(
        odom[:, 0],
        odom[:, 1],
        color="#8f8f8f",
        linewidth=1.0,
        linestyle=(0, (10, 3, 2, 3)),
        label="Initial odometry",
        zorder=2,
    )
    ax.plot(
        direct[:, 0],
        direct[:, 1],
        color="#1f77b4",
        linewidth=1.25,
        label="Direct Newton, outer=20",
        zorder=3,
    )
    ax.plot(
        base_sync[:, 0],
        base_sync[:, 1],
        color="#d62728",
        linewidth=1.05,
        linestyle=(0, (4, 2)),
        label="Base sync GBP, 1000 sweeps/outer",
        zorder=4,
    )

    ax.scatter([gt[0, 0]], [gt[0, 1]], color="black", s=20, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SE(2) Trajectories, N=512, Base Sync Baseline")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True, loc="best")

    out_path = OUT_DIR / "se2_n512_gt_odom_direct20_base_sync1000_outer20.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
