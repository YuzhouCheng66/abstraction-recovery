from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.persistent_residual_fixed_problem_experiment import (
    FixedResidualSetup,
    build_setup,
    current_metrics,
    reset_fixed_residual_state,
)
from svd_abstraction.persistent_state_exact_coarse_experiment import inject_correction_keep_messages
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


OUTPUT_DIR = pathlib.Path(
    "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_level(setup: FixedResidualSetup, basis_source: str) -> SVDResidualAbstraction:
    level = SVDResidualAbstraction(
        base_graph=setup.residual_graph,
        groups=setup.level.groups,
        r_reduced=4,
        basis_source=basis_source,
        freeze_basis=True,
        ridge=1e-10,
        eta_assignment_mode="projected_terms",
        absolute_system=False,
    )
    level.initialize_bases(force=True)
    level.build_coarse_graph(force=True)
    return level


def run_fixed_k_with_basis(
    setup: FixedResidualSetup,
    basis_source: str,
    k: int,
    cycles: int = 100,
) -> dict[str, object]:
    reset_fixed_residual_state(setup)
    level = build_level(setup, basis_source=basis_source)
    history = [{"cycle": 0, "k_used": 0, **current_metrics(setup)}]

    for cyc in range(1, cycles + 1):
        for _ in range(k):
            setup.residual_graph.synchronous_iteration()

        level.update_coarse_residual_eta()
        delta_z = level.direct_solve_coarse_graph()
        delta_e = level.prolongate(delta_z)
        inject_correction_keep_messages(setup.residual_graph, delta_e)

        history.append({"cycle": cyc, "k_used": int(k), **current_metrics(setup)})
        if (
            not np.isfinite(history[-1]["relative_state_error"])
            or history[-1]["relative_state_error"] > 1e12
        ):
            break

    rel_hist = [row["relative_state_error"] for row in history]
    best_cycle = int(np.argmin(rel_hist))
    return {
        "config": {"basis_source": basis_source, "k": int(k), "cycles": int(cycles)},
        "summary": {
            "final_cycle": int(history[-1]["cycle"]),
            "final_relerr": float(history[-1]["relative_state_error"]),
            "final_residual": float(history[-1]["algebraic_residual"]),
            "final_fixed_residual_norm": float(history[-1]["fixed_residual_norm"]),
            "best_relerr": float(rel_hist[best_cycle]),
            "best_cycle": int(best_cycle),
        },
        "history": history,
    }


def save_history_csv(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def main() -> None:
    setup = build_setup()

    experiments = [
        ("joint_covariance", 50),
        ("message_conditioned_information", 50),
        ("joint_covariance", 20),
        ("message_conditioned_information", 20),
    ]

    results = []
    summary_rows = []
    for basis_source, k in experiments:
        result = run_fixed_k_with_basis(setup, basis_source=basis_source, k=k, cycles=100)
        results.append(result)
        summary_rows.append(
            {
                "basis_source": basis_source,
                "k": int(k),
                **result["summary"],
            }
        )
        stem = f"linear512_{basis_source}_k{k}_cycle100"
        save_history_csv(
            result["history"],
            OUTPUT_DIR / f"{stem}.csv",
        )

    summary_csv = OUTPUT_DIR / "linear512_message_conditioned_mg_summary.csv"
    json_path = OUTPUT_DIR / "linear512_message_conditioned_mg_summary.json"

    save_history_csv(summary_rows, summary_csv)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(json.dumps({"summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
