from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import ArpackNoConvergence

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.g2o_se2 import G2OSE2Problem
from svd_abstraction.g2o_se2 import linearize_g2o_problem
from svd_abstraction.g2o_se2 import parse_g2o_se2
from svd_abstraction.intel_g2o_persistent_residual_mg import G2O_PATH
from svd_abstraction.intel_g2o_persistent_residual_mg import RESULT_DIR
from svd_abstraction.persistent_residual_fixed_problem_experiment import build_setup


def oracle_proj_rel_from_A(A, e_star: np.ndarray, topk: int) -> dict[str, object]:
    n = int(A.shape[0])
    if sp.issparse(A):
        try:
            vals, vecs = spla.eigsh(A, k=topk, which="SM", tol=1e-8, maxiter=max(20000, 20 * n))
        except ArpackNoConvergence:
            vals, vecs = spla.eigsh(A, k=topk, sigma=0.0, which="LM", tol=1e-8)
    else:
        vals, vecs = scipy.linalg.eigh(A)
        vals = vals[:topk]
        vecs = vecs[:, :topk]
    order = np.argsort(vals)
    vals = np.asarray(vals[order], dtype=float)
    vecs = np.asarray(vecs[:, order], dtype=float)

    coeff = vecs.T @ e_star
    e_proj = vecs @ coeff
    proj_rel = float(np.linalg.norm(e_star - e_proj) / max(np.linalg.norm(e_star), 1e-15))
    captured = float(np.linalg.norm(e_proj) / max(np.linalg.norm(e_star), 1e-15))
    return {
        "topk": int(topk),
        "dim": n,
        "eigvals": vals.tolist(),
        "proj_rel": proj_rel,
        "captured_norm_ratio": captured,
    }


def synthetic_case(topk: int) -> dict[str, object]:
    setup = build_setup()
    A = np.asarray(setup.a, dtype=float)
    e_star = np.asarray(setup.e_star, dtype=float).reshape(-1)
    oracle = oracle_proj_rel_from_A(A, e_star, topk=topk)
    return {
        "name": "synthetic_fixed_residual",
        "topk": int(topk),
        "full_dim": int(A.shape[0]),
        "e_star_norm": float(np.linalg.norm(e_star)),
        **oracle,
    }


def intel_case(topk: int) -> dict[str, object]:
    problem: G2OSE2Problem = parse_g2o_se2(G2O_PATH)
    A, b = linearize_g2o_problem(problem, problem.init_poses)
    A = A + 1e-10 * sp.eye(A.shape[0], format="csc")
    e_star = np.asarray(spla.spsolve(A, b), dtype=float).reshape(-1)
    oracle = oracle_proj_rel_from_A(A, e_star, topk=topk)
    return {
        "name": "intel_initial_residual",
        "topk": int(topk),
        "full_dim": int(A.shape[0]),
        "e_star_norm": float(np.linalg.norm(e_star)),
        **oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    syn = synthetic_case(args.topk)
    intel = intel_case(args.topk)

    payload = {
        "config": {"topk": int(args.topk)},
        "synthetic": syn,
        "intel": intel,
    }
    out_json = RESULT_DIR / f"oracle_top{int(args.topk)}_eig_proj_rel.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(json.dumps({"json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
