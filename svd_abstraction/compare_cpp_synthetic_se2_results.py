from __future__ import annotations

import argparse
import json
import math
import pathlib


def max_abs_diff(rows_a, rows_b, key: str) -> float:
    diffs = []
    for row_a, row_b in zip(rows_a, rows_b):
        diffs.append(abs(float(row_a[key]) - float(row_b[key])))
    return max(diffs) if diffs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--cpp", type=pathlib.Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()

    ref = json.loads(args.reference.read_text(encoding="utf-8"))
    cpp = json.loads(args.cpp.read_text(encoding="utf-8"))

    summary = {}
    for key in ["nonlinear_objective", "linear_step_norm", "linear_residual_norm"]:
        summary[f"direct_{key}"] = max_abs_diff(ref["direct_history"], cpp["direct_history"], key)
    for key in [
        "nonlinear_objective",
        "e_hat_norm",
        "e_star_norm",
        "e_rel_to_exact",
        "linear_residual_exact",
        "linear_residual_approx",
    ]:
        summary[f"mg_{key}"] = max_abs_diff(ref["mg_history"], cpp["mg_history"], key)

    passed = all(math.isfinite(value) and value <= args.tol for value in summary.values())
    for key, value in summary.items():
        print(f"{key}={value:.3e}")
    print(f"passed={passed}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
