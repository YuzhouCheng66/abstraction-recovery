from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.persistent_residual_fixed_problem_experiment import build_setup


OUTPUT_DIR = pathlib.Path(
    "/home/yuzhou/Desktop/abstraction-recovery/svd_abstraction/output_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def projection_residual(vec: np.ndarray, basis: np.ndarray) -> float:
    coeffs, *_ = np.linalg.lstsq(basis, vec, rcond=None)
    fit = basis @ coeffs
    return float(np.linalg.norm(vec - fit) / max(np.linalg.norm(vec), 1e-15))


def group_boundary_vars(graph, group: list[int]) -> tuple[list[int], list[int]]:
    group_set = {int(v) for v in group}
    boundary: list[int] = []
    interior: list[int] = []
    base_vars = {
        int(var.variableID): var
        for var in graph.var_nodes[: graph.n_var_nodes]
        if getattr(var, "active", True)
    }
    for var_id in group:
        var = base_vars[int(var_id)]
        touches_outside = False
        for factor in var.adj_factors:
            for adj_var in factor.adj_var_nodes:
                if int(adj_var.variableID) not in group_set:
                    touches_outside = True
                    break
            if touches_outside:
                break
        if touches_outside:
            boundary.append(int(var_id))
        else:
            interior.append(int(var_id))
    return boundary, interior


def local_boundary_modes(
    graph,
    a0: np.ndarray,
    group: list[int],
    slices: dict[int, slice],
    r_reduced: int,
    ridge: float = 1e-10,
) -> np.ndarray:
    boundary_vars, interior_vars = group_boundary_vars(graph, group)
    if not boundary_vars or not interior_vars:
        full_indices: list[int] = []
        for var_id in group:
            sl = slices[int(var_id)]
            full_indices.extend(range(sl.start, sl.stop))
        full_indices_arr = np.array(full_indices, dtype=int)
        a_sub = a0[np.ix_(full_indices_arr, full_indices_arr)]
        eigvals, eigvecs = np.linalg.eigh(a_sub)
        return eigvecs[:, : min(int(r_reduced), a_sub.shape[0])]

    boundary_idx: list[int] = []
    for var_id in boundary_vars:
        sl = slices[int(var_id)]
        boundary_idx.extend(range(sl.start, sl.stop))
    boundary_idx_arr = np.array(boundary_idx, dtype=int)

    interior_idx: list[int] = []
    for var_id in interior_vars:
        sl = slices[int(var_id)]
        interior_idx.extend(range(sl.start, sl.stop))
    interior_idx_arr = np.array(interior_idx, dtype=int)

    a_bb = a0[np.ix_(boundary_idx_arr, boundary_idx_arr)]
    a_bi = a0[np.ix_(boundary_idx_arr, interior_idx_arr)]
    a_ib = a0[np.ix_(interior_idx_arr, boundary_idx_arr)]
    a_ii = a0[np.ix_(interior_idx_arr, interior_idx_arr)]
    a_ii_reg = a_ii + ridge * np.eye(a_ii.shape[0], dtype=float)

    try:
        solve_ii = np.linalg.solve(a_ii_reg, a_ib)
    except np.linalg.LinAlgError:
        solve_ii = np.linalg.pinv(a_ii_reg) @ a_ib

    schur = a_bb - a_bi @ solve_ii
    schur = 0.5 * (schur + schur.T)

    eigvals, eigvecs = np.linalg.eigh(schur)
    n_modes = min(int(r_reduced), schur.shape[0])
    boundary_modes = eigvecs[:, :n_modes]
    interior_modes = -solve_ii @ boundary_modes

    full_indices: list[int] = []
    for var_id in group:
        sl = slices[int(var_id)]
        full_indices.extend(range(sl.start, sl.stop))
    full_indices_arr = np.array(full_indices, dtype=int)

    local_pos = {idx: pos for pos, idx in enumerate(full_indices_arr.tolist())}
    basis = np.zeros((full_indices_arr.shape[0], n_modes), dtype=float)
    for row, global_idx in enumerate(boundary_idx_arr.tolist()):
        basis[local_pos[global_idx], :] = boundary_modes[row, :]
    for row, global_idx in enumerate(interior_idx_arr.tolist()):
        basis[local_pos[global_idx], :] = interior_modes[row, :]

    q, _ = np.linalg.qr(basis, mode="reduced")
    return q[:, :n_modes]


def build_grouped_basis(
    graph,
    a0: np.ndarray,
    groups: list[list[int]],
    slices: dict[int, slice],
    r_reduced: int,
    basis_source: str,
) -> np.ndarray:
    total_dim = a0.shape[0]
    total_reduced = 0
    local_bases: list[np.ndarray] = []
    full_indices_per_group: list[np.ndarray] = []

    if basis_source == "joint_covariance":
        source_matrix = np.linalg.inv(a0)
    elif basis_source == "joint_information":
        source_matrix = a0
    else:
        source_matrix = None

    for group in groups:
        full_indices: list[int] = []
        for var_id in group:
            sl = slices[int(var_id)]
            full_indices.extend(range(sl.start, sl.stop))
        full_indices_arr = np.array(full_indices, dtype=int)
        full_indices_per_group.append(full_indices_arr)

        if basis_source in {"joint_covariance", "joint_information"}:
            block = source_matrix[np.ix_(full_indices_arr, full_indices_arr)]
            eigvals, eigvecs = np.linalg.eigh(block)
            if basis_source.endswith("covariance"):
                order = np.argsort(eigvals)[::-1]
            else:
                order = np.argsort(eigvals)
            r_local = min(int(r_reduced), block.shape[0])
            basis_local = eigvecs[:, order[:r_local]]
        elif basis_source == "local_sub_covariance":
            block = np.linalg.inv(a0[np.ix_(full_indices_arr, full_indices_arr)])
            eigvals, eigvecs = np.linalg.eigh(block)
            order = np.argsort(eigvals)[::-1]
            basis_local = eigvecs[:, order[: min(int(r_reduced), block.shape[0])]]
        elif basis_source == "local_sub_information":
            block = a0[np.ix_(full_indices_arr, full_indices_arr)]
            eigvals, eigvecs = np.linalg.eigh(block)
            order = np.argsort(eigvals)
            basis_local = eigvecs[:, order[: min(int(r_reduced), block.shape[0])]]
        elif basis_source == "local_boundary_modes":
            basis_local = local_boundary_modes(
                graph=graph,
                a0=a0,
                group=group,
                slices=slices,
                r_reduced=r_reduced,
            )
        else:
            raise ValueError(f"Unknown basis source: {basis_source}")

        local_bases.append(basis_local)
        total_reduced += basis_local.shape[1]

    p = np.zeros((total_dim, total_reduced), dtype=float)
    offset = 0
    for full_indices_arr, basis_local in zip(full_indices_per_group, local_bases):
        r_local = basis_local.shape[1]
        p[np.ix_(full_indices_arr, np.arange(offset, offset + r_local))] = basis_local
        offset += r_local
    return p


def main() -> None:
    setup = build_setup()
    basis_sources = [
        "joint_covariance",
        "joint_information",
        "local_sub_covariance",
        "local_sub_information",
        "local_boundary_modes",
    ]

    rows: list[dict[str, float | int | str]] = []
    for basis_source in basis_sources:
        t0 = time.time()
        basis = build_grouped_basis(
            graph=setup.residual_graph,
            a0=setup.a,
            groups=setup.level.groups,
            slices=setup.base_slices,
            r_reduced=4,
            basis_source=basis_source,
        )
        build_time = time.time() - t0
        rows.append(
            {
                "case": "linear512_fixed_residual",
                "basis_source": basis_source,
                "full_dim": int(setup.a.shape[0]),
                "coarse_dim": int(basis.shape[1]),
                "compression_ratio": float(basis.shape[1] / setup.a.shape[0]),
                "proj_rel": projection_residual(setup.e_star, basis),
                "build_time_sec": float(build_time),
            }
        )

    csv_path = OUTPUT_DIR / "linear512_proj_rel_basis_compare.csv"
    json_path = OUTPUT_DIR / "linear512_proj_rel_basis_compare.json"

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
