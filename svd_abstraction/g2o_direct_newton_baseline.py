from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import nonlinear_objective_g2o
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import run_direct_newton_g2o


OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(key, "")) for key in keys) + "\n")


def stem_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return slug.strip("_") or "g2o"


def plot_initial_vs_direct(problem, direct_poses: np.ndarray, out_path: pathlib.Path, title: str) -> pathlib.Path:
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

    ax.plot(
        init[:, 0],
        init[:, 1],
        color="#8f8f8f",
        linewidth=1.0,
        linestyle=(0, (10, 3, 2, 3)),
        label="Initial",
        zorder=2,
    )
    ax.plot(
        direct[:, 0],
        direct[:, 1],
        color="#1f77b4",
        linewidth=1.25,
        label="Direct Newton",
        zorder=3,
    )
    ax.scatter([init[0, 0]], [init[0, 1]], color="black", s=18, label="Anchor/start", zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2o-path", type=pathlib.Path, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--num-outer", type=int, default=20)
    parser.add_argument("--rel-obj-tol", type=float, default=1e-10)
    parser.add_argument("--step-tol", type=float, default=1e-8)
    args = parser.parse_args()

    label = args.label or args.g2o_path.stem
    label_stem = f"{stem_slug(label)}_g2o"
    stem = f"{label_stem}_direct_newton_baseline"

    problem = parse_g2o_se2(args.g2o_path)
    direct = run_direct_newton_g2o(
        problem,
        num_outer=args.num_outer,
        rel_obj_tol=args.rel_obj_tol,
        step_tol=args.step_tol,
    )
    direct_poses = np.asarray(direct["final_poses"], dtype=float)

    plot_path = plot_initial_vs_direct(
        problem,
        direct_poses,
        OUT_DIR / f"{label_stem}_initial_vs_direct_newton.png",
        f"{label} g2o: Initial vs Direct Newton",
    )
    json_path = RESULT_DIR / f"{stem}.json"
    history_csv = RESULT_DIR / f"{stem}_history.csv"
    traj_path = RESULT_DIR / f"{stem}_trajectories.npz"

    out = {
        "config": {
            "path": str(args.g2o_path),
            "label": label,
            "anchor_precision": 1e8,
            "num_outer": int(args.num_outer),
            "rel_obj_tol": float(args.rel_obj_tol),
            "step_tol": float(args.step_tol),
        },
        "initial_objective": float(nonlinear_objective_g2o(problem, problem.init_poses)),
        "direct_newton": direct,
    }
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_csv(direct["history"], history_csv)
    np.savez_compressed(
        traj_path,
        initial_poses=np.asarray(problem.init_poses, dtype=float),
        final_poses=direct_poses,
        pose_history=np.asarray(direct["pose_history"], dtype=float),
    )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "history_csv": str(history_csv),
                "trajectories": str(traj_path),
                "plot": str(plot_path),
                "initial_objective": out["initial_objective"],
                "final_row": direct["history"][-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
