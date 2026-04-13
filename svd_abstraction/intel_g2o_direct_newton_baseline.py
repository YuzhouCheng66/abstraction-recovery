from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import run_direct_newton_g2o


G2O_PATH = pathlib.Path("/home/yuzhou/Desktop/input_INTEL_g2o.g2o")
OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")


def plot_initial_vs_direct(problem, direct_poses: np.ndarray, out_path: pathlib.Path) -> pathlib.Path:
    init = np.asarray(problem.init_poses, dtype=float)
    direct = np.asarray(direct_poses, dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)

    for edge in problem.edges:
        if edge.kind != "loop":
            continue
        pi = init[edge.i, :2]
        pj = init[edge.j, :2]
        ax.plot(
            [pi[0], pj[0]],
            [pi[1], pj[1]],
            color="#8a8a8a",
            alpha=0.25,
            linewidth=0.30,
            zorder=1,
        )

    ax.plot(init[:, 0], init[:, 1], color="#8f8f8f", linewidth=1.0, linestyle=(0, (10, 3, 2, 3)), label="Initial", zorder=2)
    ax.plot(direct[:, 0], direct[:, 1], color="#1f77b4", linewidth=1.25, label="Direct Newton", zorder=3)
    ax.scatter([init[0, 0]], [init[0, 1]], color="black", s=18, label="Anchor/start", zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("INTEL g2o: Initial vs Direct Newton")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    problem = parse_g2o_se2(G2O_PATH)
    direct = run_direct_newton_g2o(problem, num_outer=20, rel_obj_tol=1e-12, step_tol=1e-10)
    direct_poses = np.asarray(direct["final_poses"], dtype=float)

    plot_path = plot_initial_vs_direct(
        problem,
        direct_poses,
        OUT_DIR / "intel_g2o_initial_vs_direct_newton.png",
    )

    out = {
        "config": {"path": str(G2O_PATH), "anchor_precision": 1e8},
        "initial_objective": float(nonlinear_objective_g2o(problem, problem.init_poses)),
        "direct_newton": direct,
    }
    json_path = RESULT_DIR / "intel_g2o_direct_newton_baseline.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "json": str(json_path),
                "plot": str(plot_path),
                "initial_objective": out["initial_objective"],
                "final_row": direct["history"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
