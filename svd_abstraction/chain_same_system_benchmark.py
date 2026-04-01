"""Same-system benchmark on a long linear chain pose-graph.

We compare on the exact same scalar channel system extracted from a 2D linear
chain pose graph:
  * base Jacobi
  * geometric 1D multigrid + Jacobi smoother
  * base scalar GaBP
  * geometric 1D multigrid + GaBP smoother
  * raylib base GBP
  * raylib two-level
  * raylib multilevel

This is meant to answer whether multigrid can really "jump propagate" on a
long chain, reducing the cycle count well below the chain length.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.poisson_multigrid_benchmark import Level
from svd_abstraction.poisson_multigrid_benchmark import run_jacobi
from svd_abstraction.poisson_multigrid_benchmark import run_multigrid
from svd_abstraction.pose_graph import make_slam_like_graph
from svd_abstraction.raylib_recursive_experiment import build_multigrid_graph
from svd_abstraction.raylib_recursive_experiment import build_raylib_graph
from svd_abstraction.raylib_recursive_experiment import relative_error
from svd_abstraction.raylib_recursive_experiment import run_base_until_converged
from svd_abstraction.raylib_recursive_experiment import run_recursive_standard_until_converged
from svd_abstraction.scalar_gabp import build_scalar_gabp_level
from svd_abstraction.scalar_gabp import run_multigrid_with_gabp
from svd_abstraction.scalar_gabp import run_scalar_gabp


@dataclass
class RaylibArgs:
    prior_sigma: float
    odom_sigma: float
    seed: int
    theta: float
    split_mode: str
    interp_mode: str
    disable_second_pass_coarse_match: bool


def extract_scalar_channel(
    eta: np.ndarray,
    lam: sparse.spmatrix | np.ndarray,
    channel: int,
    block_dofs: int = 2,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    lam_csr = sparse.csr_matrix(lam)
    idx = np.arange(channel, lam_csr.shape[0], block_dofs)
    return eta[idx], lam_csr[idx][:, idx].tocsr()


def choose_coarse_indices(n_fine: int) -> np.ndarray:
    coarse = np.arange(0, n_fine, 2, dtype=int)
    if coarse[-1] != n_fine - 1:
        coarse = np.r_[coarse, n_fine - 1]
    return coarse


def prolongation_1d_nodes(n_fine: int) -> sparse.csr_matrix:
    if n_fine <= 1:
        raise ValueError("n_fine must be at least 2")

    coarse_idx = choose_coarse_indices(n_fine)
    n_coarse = coarse_idx.size
    rows = []
    cols = []
    data = []

    coarse_pos = {int(idx): j for j, idx in enumerate(coarse_idx.tolist())}
    for i in range(n_fine):
        if i in coarse_pos:
            rows.append(i)
            cols.append(coarse_pos[i])
            data.append(1.0)
            continue

        right_pos = int(np.searchsorted(coarse_idx, i))
        left_pos = right_pos - 1
        left_idx = int(coarse_idx[left_pos])
        right_idx = int(coarse_idx[right_pos])
        gap = max(right_idx - left_idx, 1)
        t = (i - left_idx) / gap
        rows.extend([i, i])
        cols.extend([left_pos, right_pos])
        data.extend([1.0 - t, t])

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_fine, n_coarse))


def build_line_levels_from_matrix(
    n: int,
    a: sparse.spmatrix | np.ndarray,
    block_dofs: int = 1,
    max_levels: int | None = None,
    min_coarse_n: int = 2,
) -> list[Level]:
    a_level = sparse.csr_matrix(a)
    expected_dim = n * block_dofs
    if a_level.shape != (expected_dim, expected_dim):
        raise ValueError(f"Expected {(expected_dim, expected_dim)} operator, got {a_level.shape}")

    levels: list[Level] = []
    n_level = n
    level_count = 0

    while True:
        levels.append(
            Level(
                n=n_level,
                a=a_level.tocsr(),
                r=None,
                p=None,
                diag_inv=1.0 / a_level.diagonal(),
                nx=n_level,
                ny=1,
                block_dofs=block_dofs,
            )
        )
        level_count += 1

        if n_level <= min_coarse_n:
            break
        if max_levels is not None and level_count >= max_levels:
            break

        p_scalar = prolongation_1d_nodes(n_level).tocsr()
        if block_dofs == 1:
            p = p_scalar
            r = (0.5 * p.transpose()).tocsr()
            n_coarse = p_scalar.shape[1]
        else:
            eye = sparse.eye(block_dofs, format="csr")
            p = sparse.kron(p_scalar, eye, format="csr")
            r = sparse.kron(0.5 * p_scalar.transpose(), eye, format="csr")
            n_coarse = p_scalar.shape[1]
        levels[-1].p = p
        levels[-1].r = r
        a_level = (r @ a_level @ p).tocsr()
        n_level = n_coarse

    return levels


def system_from_chain(
    n: int,
    step_size: float,
    prior_sigma: float,
    odom_sigma: float,
    seed: int,
):
    nodes, edges = make_slam_like_graph(
        N=n,
        step_size=step_size,
        loop_prob=0.0,
        loop_radius=1.0,
        prior_prop=0.0,
        seed=seed,
    )
    graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=prior_sigma,
        odom_sigma=odom_sigma,
        tiny_prior=1e-12,
        seed=seed,
    )
    eta, lam = graph.joint_distribution_inf()
    x_star, _ = graph.joint_distribution_cov()
    return nodes, edges, eta, sparse.csr_matrix(lam), x_star


def run_raylib_family(nodes, edges, x_star, args: RaylibArgs, tol: float, max_base_iters: int, max_v_cycles: int):
    base_graph = build_raylib_graph(
        nodes,
        edges,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    base_iters, base_err = run_base_until_converged(
        base_graph,
        x_star,
        tol=tol,
        max_iters=max_base_iters,
    )
    results = {
        "raylib_base": {
            "iterations": base_iters,
            "rel_error": base_err,
        }
    }

    for label, max_levels in (("raylib_two_level", 2), ("raylib_multilevel", None)):
        mg_graph = build_multigrid_graph(
            nodes,
            edges,
            max_total_levels=max_levels,
            args=args,
        )
        sizes = [len(level_vars) for level_vars in mg_graph.multigrid_vars]
        cycles, err = run_recursive_standard_until_converged(
            mg_graph,
            x_star,
            tol=tol,
            max_cycles=max_v_cycles,
        )
        results[label] = {
            "iterations": cycles,
            "rel_error": err,
            "sizes": sizes,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--step-size", type=float, default=25.0)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--channel", choices=["x", "y"], default="x")
    parser.add_argument("--omega", type=float, default=2.0 / 3.0)
    parser.add_argument("--pre", type=int, default=1)
    parser.add_argument("--post", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-jacobi", type=int, default=5000)
    parser.add_argument("--max-gabp", type=int, default=5000)
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--max-base-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--split-mode", choices=["rs", "pmis", "pmis2"], default="pmis2")
    parser.add_argument(
        "--interp-mode",
        choices=["direct", "extended_if_needed", "extended_all"],
        default="extended_if_needed",
    )
    parser.add_argument("--disable-second-pass-coarse-match", action="store_true", default=True)
    args = parser.parse_args()

    nodes, edges, eta, lam, x_star_full = system_from_chain(
        n=args.n,
        step_size=args.step_size,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
    )
    channel_idx = 0 if args.channel == "x" else 1
    b, a = extract_scalar_channel(eta, lam, channel_idx)
    x_star = x_star_full[channel_idx::2]

    jacobi_base = run_jacobi(
        Level(n=args.n, a=a, r=None, p=None, diag_inv=1.0 / a.diagonal(), nx=args.n, ny=1, block_dofs=1),
        b,
        x_star,
        omega=args.omega,
        tol=args.tol,
        max_iters=args.max_jacobi,
    )
    two_levels = build_line_levels_from_matrix(args.n, a, max_levels=2)
    multilevel = build_line_levels_from_matrix(args.n, a, max_levels=None)
    jacobi_two = run_multigrid(
        two_levels,
        b,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )
    jacobi_multi = run_multigrid(
        multilevel,
        b,
        x_star,
        omega=args.omega,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )

    gabp_base = run_scalar_gabp(
        build_scalar_gabp_level(a),
        b,
        x_star,
        tol=args.tol,
        max_iters=args.max_gabp,
    )
    gabp_two = run_multigrid_with_gabp(
        two_levels,
        b,
        x_star,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )
    gabp_multi = run_multigrid_with_gabp(
        multilevel,
        b,
        x_star,
        pre_sweeps=args.pre,
        post_sweeps=args.post,
        tol=args.tol,
        max_cycles=args.max_cycles,
    )

    ray_args = RaylibArgs(
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=args.seed,
        theta=args.theta,
        split_mode=args.split_mode,
        interp_mode=args.interp_mode,
        disable_second_pass_coarse_match=args.disable_second_pass_coarse_match,
    )
    raylib = run_raylib_family(
        nodes,
        edges,
        x_star_full,
        args=ray_args,
        tol=args.tol,
        max_base_iters=args.max_base_iters,
        max_v_cycles=args.max_cycles,
    )

    init_norm = np.linalg.norm(x_star)
    if init_norm == 0.0:
        init_norm = 1.0

    def summarize(name: str, result: dict[str, object], is_mg: bool = False) -> None:
        rel_error = result["error_history"][-1] / init_norm if "error_history" in result else result["rel_error"]
        extra = ""
        if "elapsed_time" in result:
            extra += f", time={result['elapsed_time']:.3f}s"
        print(f"{name}: iterations={result['iterations']}, rel_error={rel_error:.3e}{extra}")
        if is_mg:
            print(
                f"  smoothing_work={result['smoothing_work']:.2f}, "
                f"coarse_solves={result['coarse_solves']:.0f}"
            )

    print(f"Chain same-system benchmark: n={args.n}, channel={args.channel}, edges={len(edges)}")
    print(f"Geometric two-level sizes: {[lvl.n for lvl in two_levels]}")
    print(f"Geometric multilevel sizes: {[lvl.n for lvl in multilevel]}")
    print(
        f"Raylib hierarchy: split_mode={args.split_mode}; interp_mode={args.interp_mode}; "
        f"second_pass={'off' if args.disable_second_pass_coarse_match else 'on'}"
    )
    print("")
    summarize("Jacobi(base)", jacobi_base)
    summarize("Jacobi(two-level)", jacobi_two, is_mg=True)
    summarize("Jacobi(multilevel)", jacobi_multi, is_mg=True)
    summarize("GaBP(base)", gabp_base)
    summarize("GaBP(two-level)", gabp_two, is_mg=True)
    summarize("GaBP(multilevel)", gabp_multi, is_mg=True)
    summarize("Raylib(base)", raylib["raylib_base"])
    print(
        f"Raylib(two-level): sizes={raylib['raylib_two_level']['sizes']}, "
        f"iterations={raylib['raylib_two_level']['iterations']}, "
        f"rel_error={raylib['raylib_two_level']['rel_error']:.3e}"
    )
    print(
        f"Raylib(multilevel): sizes={raylib['raylib_multilevel']['sizes']}, "
        f"iterations={raylib['raylib_multilevel']['iterations']}, "
        f"rel_error={raylib['raylib_multilevel']['rel_error']:.3e}"
    )


if __name__ == "__main__":
    main()
