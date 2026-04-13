from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import pathlib
import sys

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.csgraph import connected_components


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.raylib_fanaskov_fixed_lambda_experiment import fixed_problem
from svd_abstraction.raylib_fanaskov_twolevel_experiment import fanaskov_converged_lam_edge


def directed_message_structure(a: np.ndarray) -> tuple[list[list[int]], list[tuple[int, int]]]:
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    neighbors = [np.flatnonzero(np.abs(a[i]) > 1e-12).tolist() for i in range(n)]
    for i in range(n):
        if i in neighbors[i]:
            neighbors[i].remove(i)
    msg_nodes = [(j, i) for j in range(n) for i in neighbors[j]]
    return neighbors, msg_nodes


def assemble_i_minus_g(
    a: np.ndarray,
    lam_edge: dict[tuple[int, int], float],
) -> tuple[csr_matrix, csr_matrix, list[tuple[int, int]]]:
    neighbors, msg_nodes = directed_message_structure(a)
    msg_index = {edge: k for k, edge in enumerate(msg_nodes)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    patt_rows: list[int] = []
    patt_cols: list[int] = []

    for (j, i), lam in lam_edge.items():
        row = msg_index[(j, i)]
        for k in neighbors[j]:
            if k == i:
                continue
            col = msg_index[(k, j)]
            rows.append(row)
            cols.append(col)
            vals.append(float(lam))
            patt_rows.append(row)
            patt_cols.append(col)

    m = len(msg_nodes)
    g = csr_matrix((vals, (rows, cols)), shape=(m, m))
    g_pattern = csr_matrix((np.ones(len(patt_rows), dtype=int), (patt_rows, patt_cols)), shape=(m, m))
    i_minus_g = identity(m, format="csr", dtype=float) - g
    return i_minus_g, g_pattern, msg_nodes


def analyze_scc(
    g_pattern: csr_matrix,
    msg_nodes: list[tuple[int, int]],
) -> dict[str, object]:
    # SCC should be computed on the directed off-diagonal dependency graph.
    num_scc, labels = connected_components(g_pattern, directed=True, connection="strong")
    size_by_label = Counter(labels)
    nontrivial_labels = {lab for lab, size in size_by_label.items() if size > 1}

    nontrivial_sizes = sorted((size for size in size_by_label.values() if size > 1), reverse=True)
    cyclic_core_msg = sum(nontrivial_sizes)
    cyclic_core_msg_frac = cyclic_core_msg / max(len(msg_nodes), 1)

    label_nodes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge, lab in zip(msg_nodes, labels):
        label_nodes[int(lab)].append(edge)

    core_edges = [edge for edge, lab in zip(msg_nodes, labels) if int(lab) in nontrivial_labels]
    core_scalar_vars = sorted({u for edge in core_edges for u in edge})
    core_pose_vars = sorted({u // 2 for u in core_scalar_vars})
    all_scalar_vars = sorted({u for edge in msg_nodes for u in edge})
    all_pose_vars = sorted({u // 2 for u in all_scalar_vars})
    pose_outside_core = [pose for pose in all_pose_vars if pose not in set(core_pose_vars)]

    size_hist = Counter(size_by_label.values())
    largest_components = []
    for size, lab in sorted(((size, lab) for lab, size in size_by_label.items()), reverse=True)[:10]:
        nodes = label_nodes[int(lab)]
        src_parity = Counter(edge[0] % 2 for edge in nodes)
        dst_parity = Counter(edge[1] % 2 for edge in nodes)
        largest_components.append(
            {
                "label": int(lab),
                "size": int(size),
                "src_parity": dict(src_parity),
                "dst_parity": dict(dst_parity),
                "src_minmax": (min(edge[0] for edge in nodes), max(edge[0] for edge in nodes)),
                "dst_minmax": (min(edge[1] for edge in nodes), max(edge[1] for edge in nodes)),
            }
        )

    return {
        "num_scc": int(num_scc),
        "num_nontrivial_scc": int(len(nontrivial_sizes)),
        "largest_scc": int(max(size_by_label.values(), default=0)),
        "nontrivial_sizes_top10": nontrivial_sizes[:10],
        "size_hist": dict(sorted(size_hist.items())),
        "cyclic_core_msg": int(cyclic_core_msg),
        "cyclic_core_msg_frac": float(cyclic_core_msg_frac),
        "scalar_vars_in_core": int(len(core_scalar_vars)),
        "scalar_vars_total": int(len(all_scalar_vars)),
        "pose_vars_in_core": int(len(core_pose_vars)),
        "pose_vars_total": int(len(all_pose_vars)),
        "pose_outside_core_count": int(len(pose_outside_core)),
        "pose_outside_core": pose_outside_core,
        "largest_components": largest_components,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-prop", type=float, default=0.0)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument("--lam-tol", type=float, default=1e-12)
    args = parser.parse_args()

    a0, b0, mu_star, x0, freeze_step = fixed_problem(
        n=args.n,
        seed=args.seed,
        prior_prop=args.prior_prop,
        variance_threshold=args.variance_threshold,
    )
    lam_edge_star, lam_sweeps, lam_delta = fanaskov_converged_lam_edge(
        a0,
        mode="parallel",
        tol=args.lam_tol,
        max_sweeps=10000,
    )

    i_minus_g, g_pattern, msg_nodes = assemble_i_minus_g(a0, lam_edge_star)
    scc = analyze_scc(g_pattern, msg_nodes)

    print(f"n={args.n} seed={args.seed} prior_prop={args.prior_prop}")
    print(f"A_shape={a0.shape}")
    print(f"freeze_step={freeze_step}")
    print(f"lam_sweeps={lam_sweeps} lam_delta={lam_delta}")
    print(f"num_directed_messages={len(msg_nodes)}")
    print(f"G_nnz={g_pattern.nnz}")
    print(f"I_minus_G_shape={i_minus_g.shape}")
    print(f"I_minus_G_nnz={i_minus_g.nnz}")
    print(f"num_scc={scc['num_scc']}")
    print(f"num_nontrivial_scc={scc['num_nontrivial_scc']}")
    print(f"largest_scc={scc['largest_scc']}")
    print(f"nontrivial_sizes_top10={scc['nontrivial_sizes_top10']}")
    print(f"size_hist={scc['size_hist']}")
    print(f"cyclic_core_msg={scc['cyclic_core_msg']}")
    print(f"cyclic_core_msg_frac={scc['cyclic_core_msg_frac']}")
    print(
        "scalar_core="
        f"{scc['scalar_vars_in_core']}/{scc['scalar_vars_total']} "
        f"pose_core={scc['pose_vars_in_core']}/{scc['pose_vars_total']}"
    )
    print(f"pose_outside_core_count={scc['pose_outside_core_count']}")
    print(f"pose_outside_core={scc['pose_outside_core']}")
    print("largest_components")
    for comp in scc["largest_components"]:
        print(f"  {comp}")


if __name__ == "__main__":
    main()
