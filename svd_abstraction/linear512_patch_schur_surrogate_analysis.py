from __future__ import annotations

import json
import pathlib
import sys
import time
from collections import deque

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.linear512_proj_rel_basis_compare import projection_residual
from svd_abstraction.persistent_residual_fixed_problem_experiment import build_setup


OUTPUT_DIR = pathlib.Path(
    "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_variable_adjacency(graph) -> dict[int, set[int]]:
    adj = {int(v.variableID): set() for v in graph.var_nodes[: graph.n_var_nodes]}
    for factor in graph.factors[: graph.n_factor_nodes]:
        vids = [int(v.variableID) for v in factor.adj_var_nodes]
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                adj[vids[i]].add(vids[j])
                adj[vids[j]].add(vids[i])
    return adj


def patch_variable_ids(group: list[int], adj: dict[int, set[int]], hops: int) -> list[int]:
    seen = {int(v) for v in group}
    frontier = set(seen)
    for _ in range(int(hops)):
        nxt: set[int] = set()
        for vid in frontier:
            nxt.update(adj[vid])
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return sorted(seen)


def patch_schur_basis(
    a0: np.ndarray,
    group: list[int],
    patch_vars: list[int],
    slices: dict[int, slice],
    r_reduced: int,
    ridge: float = 1e-10,
) -> np.ndarray:
    group_set = {int(v) for v in group}

    group_idx: list[int] = []
    for vid in group:
        sl = slices[int(vid)]
        group_idx.extend(range(sl.start, sl.stop))
    group_idx_arr = np.array(group_idx, dtype=int)

    interior_vars = [vid for vid in patch_vars if int(vid) not in group_set]
    if not interior_vars:
        block = a0[np.ix_(group_idx_arr, group_idx_arr)]
        eigvals, eigvecs = np.linalg.eigh(block)
        return eigvecs[:, : min(int(r_reduced), block.shape[0])]

    interior_idx: list[int] = []
    for vid in interior_vars:
        sl = slices[int(vid)]
        interior_idx.extend(range(sl.start, sl.stop))
    interior_idx_arr = np.array(interior_idx, dtype=int)

    a_gg = a0[np.ix_(group_idx_arr, group_idx_arr)]
    a_gi = a0[np.ix_(group_idx_arr, interior_idx_arr)]
    a_ig = a0[np.ix_(interior_idx_arr, group_idx_arr)]
    a_ii = a0[np.ix_(interior_idx_arr, interior_idx_arr)]
    a_ii_reg = a_ii + ridge * np.eye(a_ii.shape[0], dtype=float)

    try:
        solve_ii = np.linalg.solve(a_ii_reg, a_ig)
    except np.linalg.LinAlgError:
        solve_ii = np.linalg.pinv(a_ii_reg) @ a_ig

    schur = a_gg - a_gi @ solve_ii
    schur = 0.5 * (schur + schur.T)
    eigvals, eigvecs = np.linalg.eigh(schur)
    return eigvecs[:, : min(int(r_reduced), schur.shape[0])]


def build_patch_basis(
    a0: np.ndarray,
    groups: list[list[int]],
    slices: dict[int, slice],
    adj: dict[int, set[int]],
    patch_hops: int,
    r_reduced: int,
) -> tuple[np.ndarray, dict[str, float]]:
    total_dim = a0.shape[0]
    total_reduced = 0
    local_bases: list[np.ndarray] = []
    full_indices_per_group: list[np.ndarray] = []
    patch_sizes: list[int] = []
    patch_dofs: list[int] = []

    for group in groups:
        full_indices: list[int] = []
        for var_id in group:
            sl = slices[int(var_id)]
            full_indices.extend(range(sl.start, sl.stop))
        full_indices_arr = np.array(full_indices, dtype=int)
        full_indices_per_group.append(full_indices_arr)

        patch_vars = patch_variable_ids(group, adj=adj, hops=patch_hops)
        patch_sizes.append(len(patch_vars))
        patch_dofs.append(sum(slices[int(v)].stop - slices[int(v)].start for v in patch_vars))

        basis_local = patch_schur_basis(
            a0=a0,
            group=group,
            patch_vars=patch_vars,
            slices=slices,
            r_reduced=r_reduced,
        )
        local_bases.append(basis_local)
        total_reduced += basis_local.shape[1]

    p = np.zeros((total_dim, total_reduced), dtype=float)
    offset = 0
    for full_indices_arr, basis_local in zip(full_indices_per_group, local_bases):
        r_local = basis_local.shape[1]
        p[np.ix_(full_indices_arr, np.arange(offset, offset + r_local))] = basis_local
        offset += r_local

    stats = {
        "patch_hops": int(patch_hops),
        "mean_patch_vars": float(np.mean(patch_sizes)),
        "max_patch_vars": int(np.max(patch_sizes)),
        "min_patch_vars": int(np.min(patch_sizes)),
        "mean_patch_dofs": float(np.mean(patch_dofs)),
        "max_patch_dofs": int(np.max(patch_dofs)),
        "min_patch_dofs": int(np.min(patch_dofs)),
    }
    return p, stats


def main() -> None:
    setup = build_setup()
    a0 = setup.a
    e_star = setup.e_star
    groups = setup.level.groups
    slices = setup.base_slices
    adj = build_variable_adjacency(setup.residual_graph)

    rows: list[dict[str, float | int | str]] = []

    for hops in [0, 1, 2, 3, 4]:
        t0 = time.time()
        basis, stats = build_patch_basis(
            a0=a0,
            groups=groups,
            slices=slices,
            adj=adj,
            patch_hops=hops,
            r_reduced=4,
        )
        build_time = time.time() - t0
        rows.append(
            {
                "case": "linear512_fixed_residual",
                "basis_source": f"patch_schur_h{hops}",
                "full_dim": int(a0.shape[0]),
                "coarse_dim": int(basis.shape[1]),
                "compression_ratio": float(basis.shape[1] / a0.shape[0]),
                "proj_rel": projection_residual(e_star, basis),
                "build_time_sec": float(build_time),
                **stats,
            }
        )

    csv_path = OUTPUT_DIR / "linear512_patch_schur_surrogate_analysis.csv"
    json_path = OUTPUT_DIR / "linear512_patch_schur_surrogate_analysis.json"

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
