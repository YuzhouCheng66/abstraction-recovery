from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from svd_abstraction.g2o_se2_landmark import (
    parse_g2o_se2_landmark,
    parse_victoria_dataset_xylmk,
    parse_victoria_park_slampp,
    plot_initial_vs_direct_pose_landmark,
    run_direct_newton_g2o_se2_landmark,
    summarize_g2o_se2_landmark,
    write_victoria_park_g2o,
)


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "output_results" / "victoria_park_direct_newton"


def _load_problem(source_format: str, input_path: Path, g2o_out: Path) -> tuple[object, Path]:
    if source_format == "slampp_raw":
        raw_problem = parse_victoria_park_slampp(input_path)
        write_victoria_park_g2o(raw_problem, g2o_out)
        return parse_g2o_se2_landmark(g2o_out), g2o_out
    if source_format == "xylmk_txt":
        raw_problem = parse_victoria_dataset_xylmk(input_path)
        write_victoria_park_g2o(raw_problem, g2o_out)
        return parse_g2o_se2_landmark(g2o_out), g2o_out
    if source_format == "g2o":
        return parse_g2o_se2_landmark(input_path), input_path
    raise ValueError(f"Unsupported source_format {source_format!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct Newton on a Victoria Park pose+landmark graph.")
    parser.add_argument("--source-format", choices=["slampp_raw", "xylmk_txt", "g2o"], required=True)
    parser.add_argument("--input", required=True, help="Input path for the selected source format.")
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--g2o-out", default=None, help="Where to write the converted g2o, if conversion is needed.")
    parser.add_argument("--tag", default=None, help="Optional tag used in output filenames.")
    parser.add_argument("--num-outer", type=int, default=15)
    parser.add_argument("--ridge", type=float, default=1e-9)
    parser.add_argument("--rel-obj-tol", type=float, default=None)
    parser.add_argument("--step-tol", type=float, default=None)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or input_path.stem
    g2o_out = Path(args.g2o_out).resolve() if args.g2o_out else (output_dir / f"{tag}.g2o")

    problem, resolved_g2o_path = _load_problem(args.source_format, input_path, g2o_out)
    result = run_direct_newton_g2o_se2_landmark(
        problem,
        num_outer=args.num_outer,
        rel_obj_tol=args.rel_obj_tol,
        step_tol=args.step_tol,
        ridge=args.ridge,
    )

    final_poses = np.asarray(result["final_poses"], dtype=float)
    final_landmarks = np.asarray(result["final_landmarks"], dtype=float)
    plot_path = output_dir / f"{tag}_direct_newton.png"
    plot_initial_vs_direct_pose_landmark(
        problem,
        final_poses,
        final_landmarks,
        plot_path,
        title=f"Victoria Park Direct Newton ({tag})",
    )

    payload = {
        "source_format": args.source_format,
        "input_path": str(input_path),
        "resolved_g2o_path": str(resolved_g2o_path),
        "problem_summary": summarize_g2o_se2_landmark(problem),
        "result": result,
        "plot_path": str(plot_path),
    }
    json_path = output_dir / f"{tag}_direct_newton.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"g2o_path={resolved_g2o_path}")
    print(f"plot_path={plot_path}")
    print(f"json_path={json_path}")
    print(f"problem_summary={payload['problem_summary']}")
    print(f"final_history_row={result['history'][-1]}")


if __name__ == "__main__":
    main()
