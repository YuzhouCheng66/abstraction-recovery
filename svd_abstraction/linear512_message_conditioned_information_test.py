from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.linear512_proj_rel_basis_compare import projection_residual
from svd_abstraction.persistent_residual_fixed_problem_experiment import build_setup
from svd_abstraction.persistent_residual_fixed_problem_experiment import reset_fixed_residual_state
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


OUTPUT_DIR = pathlib.Path(
    "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def max_message_lam_delta(graph, prev_lams):
    max_delta = 0.0
    new_lams: list[np.ndarray] = []
    idx = 0
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg in factor.messages:
            lam = np.asarray(msg.lam, dtype=float).copy()
            if prev_lams is not None:
                max_delta = max(max_delta, float(np.max(np.abs(lam - prev_lams[idx]))))
            new_lams.append(lam)
            idx += 1
    return max_delta, new_lams


def warmup_lam(graph, tol: float = 1e-8, max_sweeps: int = 500):
    _, prev_lams = max_message_lam_delta(graph, None)
    delta = float("inf")
    for sweep in range(1, max_sweeps + 1):
        graph.synchronous_iteration()
        delta, prev_lams = max_message_lam_delta(graph, prev_lams)
        if delta < tol:
            return {"sweeps": int(sweep), "final_delta": float(delta)}
    return {"sweeps": int(max_sweeps), "final_delta": float(delta)}


def build_basis(setup, basis_source: str):
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
    t0 = time.time()
    level.initialize_bases(force=True)
    build_time = time.time() - t0
    return level, build_time


def main() -> None:
    setup = build_setup()
    rows = []

    reset_fixed_residual_state(setup)
    level_init, build_time_init = build_basis(setup, "message_conditioned_information")
    rows.append(
        {
            "case": "linear512_fixed_residual",
            "basis_source": "message_conditioned_information_initial",
            "full_dim": int(setup.a.shape[0]),
            "coarse_dim": int(level_init.total_reduced_dim),
            "compression_ratio": float(level_init.total_reduced_dim / setup.a.shape[0]),
            "proj_rel": projection_residual(setup.e_star, level_init.P),
            "build_time_sec": float(build_time_init),
            "lam_warmup_sweeps": 0,
            "lam_final_delta": float("nan"),
        }
    )

    reset_fixed_residual_state(setup)
    warmup = warmup_lam(setup.residual_graph, tol=1e-8, max_sweeps=500)
    level_conv, build_time_conv = build_basis(setup, "message_conditioned_information")
    rows.append(
        {
            "case": "linear512_fixed_residual",
            "basis_source": "message_conditioned_information_lam_converged",
            "full_dim": int(setup.a.shape[0]),
            "coarse_dim": int(level_conv.total_reduced_dim),
            "compression_ratio": float(level_conv.total_reduced_dim / setup.a.shape[0]),
            "proj_rel": projection_residual(setup.e_star, level_conv.P),
            "build_time_sec": float(build_time_conv),
            "lam_warmup_sweeps": int(warmup["sweeps"]),
            "lam_final_delta": float(warmup["final_delta"]),
        }
    )

    csv_path = OUTPUT_DIR / "linear512_message_conditioned_information_test.csv"
    json_path = OUTPUT_DIR / "linear512_message_conditioned_information_test.json"

    header = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in header) + "\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
