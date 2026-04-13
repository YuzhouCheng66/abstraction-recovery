from __future__ import annotations

import json
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.g2o_se2 import plot_initial_pose_graph
from svd_abstraction.g2o_se2 import summarize_g2o_se2


G2O_PATH = pathlib.Path("/home/yuzhou/Desktop/input_INTEL_g2o.g2o")
OUT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/out")
RESULT_DIR = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results")


def main() -> None:
    problem = parse_g2o_se2(G2O_PATH)
    summary = summarize_g2o_se2(problem)

    plot_path = plot_initial_pose_graph(
        problem,
        OUT_DIR / "intel_g2o_initial_pose_graph.png",
        loop_alpha=0.30,
        loop_linewidth=0.30,
    )
    summary_path = RESULT_DIR / "intel_g2o_initial_pose_graph_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "plot": str(plot_path),
                "summary_json": str(summary_path),
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
