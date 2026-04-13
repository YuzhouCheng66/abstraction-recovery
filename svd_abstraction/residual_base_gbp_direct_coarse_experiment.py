from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouped_svd_gbp_benchmark import build_graphs
from svd_abstraction.grouped_svd_gbp_benchmark import group_list
from svd_abstraction.pose_graph import build_noisy_pose_graph


def mean_vector(graph):
    return np.concatenate([np.asarray(v.mu).reshape(-1) for v in graph.var_nodes[: graph.n_var_nodes]])


def relative_error_vec(x: np.ndarray, x_star: np.ndarray) -> float:
    denom = max(np.linalg.norm(x_star), 1e-15)
    return float(np.linalg.norm(x - x_star) / denom)


def var_slices(graph):
    mapping = {}
    offset = 0
    for var in graph.var_nodes[: graph.n_var_nodes]:
        mapping[int(var.variableID)] = slice(offset, offset + var.dofs)
        offset += var.dofs
    return mapping


def build_joint_cov_basis(graph, a0: np.ndarray, groups: list[list[int]], r_reduced: int) -> np.ndarray:
    slices = var_slices(graph)
    cov = np.linalg.inv(a0)
    total_dim = a0.shape[0]

    total_reduced = 0
    full_indices_per_group = []
    local_bases = []
    for group in groups:
        full_indices = []
        for var_id in group:
            sl = slices[int(var_id)]
            full_indices.extend(range(sl.start, sl.stop))
        full_indices = np.array(full_indices, dtype=int)
        block = cov[np.ix_(full_indices, full_indices)]
        eigvals, eigvecs = np.linalg.eigh(block)
        order = np.argsort(eigvals)[::-1]
        r_local = min(int(r_reduced), block.shape[0])
        basis_local = eigvecs[:, order[:r_local]]
        full_indices_per_group.append(full_indices)
        local_bases.append(basis_local)
        total_reduced += r_local

    p = np.zeros((total_dim, total_reduced), dtype=float)
    offset = 0
    for full_indices, basis_local in zip(full_indices_per_group, local_bases):
        r_local = basis_local.shape[1]
        p[np.ix_(full_indices, np.arange(offset, offset + r_local))] = basis_local
        offset += r_local
    return p


def odom_tiny_init_graph(graph, n: int, tiny: float = 1e-12) -> None:
    chain_meas = {}
    for factor in graph.factors[: graph.n_factor_nodes]:
        if getattr(factor, "type", None) != "odometry":
            continue
        i, j = [int(v) for v in factor.adj_vIDs]
        if j == i + 1:
            meas = factor.measurement[0] if isinstance(factor.measurement, list) else factor.measurement
            chain_meas[(i, j)] = np.asarray(meas, dtype=float).reshape(-1)

    mus = {0: np.asarray(graph.var_nodes[0].GT, dtype=float).copy()}
    for i in range(n - 1):
        mus[i + 1] = mus[i] + chain_meas[(i, i + 1)]

    for var in graph.var_nodes[: graph.n_var_nodes]:
        mu = mus[int(var.variableID)].copy()
        var.mu = mu
        var.prior.eta = np.asarray(var.prior.eta, dtype=float).reshape(-1) + tiny * mu
        var.belief.eta = np.asarray(var.belief.lam, dtype=float) @ mu

    for factor in graph.factors[: graph.n_factor_nodes]:
        for adj_var, adj_belief in zip(factor.adj_var_nodes, factor.adj_beliefs):
            adj_belief.eta = np.asarray(adj_var.belief.eta, dtype=float).copy()
            adj_belief.lam = np.asarray(adj_var.belief.lam, dtype=float).copy()
        for msg in factor.messages:
            msg.eta = np.zeros_like(msg.eta)


def reset_residual_graph(residual_graph, template_graph, x: np.ndarray, slices: dict[int, slice]) -> None:
    residual_graph.var_heap.clear()
    residual_graph.var_residual.clear()
    for orig_var, var in zip(
        template_graph.var_nodes[: template_graph.n_var_nodes],
        residual_graph.var_nodes[: residual_graph.n_var_nodes],
    ):
        xi = np.asarray(x[slices[int(orig_var.variableID)]]).reshape(-1)
        var.prior.lam = np.asarray(orig_var.prior.lam, dtype=float).copy()
        var.prior.eta = (
            np.asarray(orig_var.prior.eta, dtype=float).reshape(-1)
            - np.asarray(orig_var.prior.lam, dtype=float) @ xi
        )
        var.mu = np.zeros(var.dofs, dtype=float)
        var.belief.lam = var.prior.lam.copy()
        var.belief.eta = np.zeros(var.dofs, dtype=float)

    for orig_factor, factor in zip(
        template_graph.factors[: template_graph.n_factor_nodes],
        residual_graph.factors[: residual_graph.n_factor_nodes],
    ):
        local_x = np.concatenate(
            [
                np.asarray(x[slices[int(orig_var.variableID)]]).reshape(-1)
                for orig_var in orig_factor.adj_var_nodes
            ]
        )
        abs_eta, abs_lam = orig_factor.compute_factor_absolute(update_self=False)
        factor.factor.lam = np.asarray(abs_lam, dtype=float).copy()
        factor.factor.eta = (
            np.asarray(abs_eta, dtype=float).reshape(-1)
            - np.asarray(abs_lam, dtype=float) @ local_x
        )
        for msg in factor.messages:
            msg.eta = np.zeros_like(msg.eta)
            msg.lam = np.zeros_like(msg.lam)
        for adj_var, adj_belief in zip(factor.adj_var_nodes, factor.adj_beliefs):
            adj_belief.eta = np.zeros_like(adj_belief.eta)
            adj_belief.lam = np.asarray(adj_var.prior.lam, dtype=float).copy()


def reset_residual_graph_eta_only(residual_graph, template_graph, x: np.ndarray, slices: dict[int, slice]) -> None:
    residual_graph.var_heap.clear()
    residual_graph.var_residual.clear()
    for orig_var, var in zip(
        template_graph.var_nodes[: template_graph.n_var_nodes],
        residual_graph.var_nodes[: residual_graph.n_var_nodes],
    ):
        xi = np.asarray(x[slices[int(orig_var.variableID)]]).reshape(-1)
        var.prior.eta = (
            np.asarray(orig_var.prior.eta, dtype=float).reshape(-1)
            - np.asarray(orig_var.prior.lam, dtype=float) @ xi
        )
        var.mu = np.zeros(var.dofs, dtype=float)
        var.belief.eta = np.zeros(var.dofs, dtype=float)

    for orig_factor, factor in zip(
        template_graph.factors[: template_graph.n_factor_nodes],
        residual_graph.factors[: residual_graph.n_factor_nodes],
    ):
        local_x = np.concatenate(
            [
                np.asarray(x[slices[int(orig_var.variableID)]]).reshape(-1)
                for orig_var in orig_factor.adj_var_nodes
            ]
        )
        abs_eta, _ = orig_factor.compute_factor_absolute(update_self=False)
        factor.factor.eta = (
            np.asarray(abs_eta, dtype=float).reshape(-1)
            - np.asarray(factor.factor.lam, dtype=float) @ local_x
        )
        for msg in factor.messages:
            msg.eta = np.zeros_like(msg.eta)
        for adj_belief in factor.adj_beliefs:
            adj_belief.eta = np.zeros_like(adj_belief.eta)


def lam_state_messages(graph) -> np.ndarray:
    pieces = []
    for factor in graph.factors[: graph.n_factor_nodes]:
        for msg in factor.messages:
            pieces.append(np.asarray(msg.lam, dtype=float).reshape(-1))
    if not pieces:
        return np.zeros(0, dtype=float)
    return np.concatenate(pieces)


def warmup_residual_lam(
    residual_graph,
    template_graph,
    x: np.ndarray,
    slices: dict[int, slice],
    max_sweeps: int,
    tol: float,
) -> tuple[int | None, float]:
    reset_residual_graph(residual_graph, template_graph, x, slices)
    prev = lam_state_messages(residual_graph)
    for sweep in range(1, int(max_sweeps) + 1):
        residual_graph.synchronous_iteration()
        curr = lam_state_messages(residual_graph)
        delta = float(np.max(np.abs(curr - prev))) if curr.size else 0.0
        if delta < tol:
            return sweep, delta
        prev = curr
    return None, delta


def residual_block_step(
    residual_graph,
    template_graph,
    x: np.ndarray,
    slices: dict[int, slice],
    n_sweeps: int,
    preserve_lam: bool = False,
    scheduler: str = "sync",
) -> np.ndarray:
    if preserve_lam:
        reset_residual_graph_eta_only(residual_graph, template_graph, x, slices)
    else:
        reset_residual_graph(residual_graph, template_graph, x, slices)
    for _ in range(int(n_sweeps)):
        if scheduler == "sync":
            residual_graph.synchronous_iteration(fixed_lam=preserve_lam)
        elif scheduler == "residual":
            residual_graph.residual_iteration_var_heap(
                max_updates=residual_graph.n_var_nodes,
                fixed_lam=preserve_lam,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
    return mean_vector(residual_graph)


def run_twolevel(
    residual_graph,
    template_graph,
    a0: np.ndarray,
    b0: np.ndarray,
    p: np.ndarray,
    x0: np.ndarray,
    x_star: np.ndarray,
    slices: dict[int, slice],
    base_sweeps: int,
    pre_post: bool,
    max_cycles: int,
    preserve_lam: bool,
    base_scheduler: str,
) -> list[float]:
    x = x0.copy()
    ac = p.T @ a0 @ p
    hist = [relative_error_vec(x, x_star)]
    for _ in range(int(max_cycles)):
        x = x + residual_block_step(
            residual_graph,
            template_graph,
            x,
            slices,
            n_sweeps=base_sweeps,
            preserve_lam=preserve_lam,
            scheduler=base_scheduler,
        )
        residual = b0 - a0 @ x
        yc = np.linalg.solve(ac, p.T @ residual)
        x = x + p @ yc
        if pre_post:
            x = x + residual_block_step(
                residual_graph,
                template_graph,
                x,
                slices,
                n_sweeps=base_sweeps,
                preserve_lam=preserve_lam,
                scheduler=base_scheduler,
            )
        hist.append(relative_error_vec(x, x_star))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e12:
            break
    return hist


def print_points(name: str, hist: list[float], points: list[int]) -> None:
    for point in points:
        if point < len(hist):
            print(f"{name} {point} {hist[point]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r-reduced", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--base-sweeps", type=int, default=1)
    parser.add_argument("--base-scheduler", type=str, default="sync", choices=["sync", "residual"])
    parser.add_argument("--pre-post", action="store_true")
    parser.add_argument("--eta-damping", type=float, default=0.0)
    parser.add_argument("--preserve-lam", action="store_true")
    parser.add_argument("--lam-warmup-max", type=int, default=200)
    parser.add_argument("--lam-tol", type=float, default=1e-8)
    parser.add_argument("--points", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20])
    args = parser.parse_args()

    nodes, edges, exact_graph, base_graph, mg_graph = build_graphs(
        n=args.n,
        step_size=25.0,
        loop_prob=0.05,
        loop_radius=50.0,
        prior_prop=0.0,
        prior_sigma=1.0,
        odom_sigma=1.0,
        seed=args.seed,
    )
    x_star, _ = exact_graph.joint_distribution_cov_absolute()
    b0, a0 = exact_graph.joint_distribution_inf_absolute()

    groups = group_list(
        nodes=nodes,
        graph=mg_graph,
        method="order",
        group_size=args.group_size,
        gx=8,
        gy=4,
        kmeans_k=26,
        target_groups=None,
        loop_window=2,
        loop_boost=3.0,
        degree_boost=1.0,
        loop_sep_min=2,
    )
    p = build_joint_cov_basis(mg_graph, a0, groups, r_reduced=args.r_reduced)

    odom_tiny_init_graph(base_graph, n=args.n)
    x0 = mean_vector(base_graph)

    residual_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=1.0,
        odom_sigma=1.0,
        tiny_prior=1e-12,
        seed=args.seed,
    )
    residual_graph.eta_damping = float(args.eta_damping)

    lam_warmup_sweeps = None
    lam_warmup_delta = None
    if args.preserve_lam:
        lam_warmup_sweeps, lam_warmup_delta = warmup_residual_lam(
            residual_graph=residual_graph,
            template_graph=base_graph,
            x=x0,
            slices=var_slices(base_graph),
            max_sweeps=args.lam_warmup_max,
            tol=args.lam_tol,
        )

    hist = run_twolevel(
        residual_graph=residual_graph,
        template_graph=base_graph,
        a0=a0,
        b0=b0,
        p=p,
        x0=x0,
        x_star=x_star,
        slices=var_slices(base_graph),
        base_sweeps=args.base_sweeps,
        pre_post=args.pre_post,
        max_cycles=args.max_cycles,
        preserve_lam=args.preserve_lam,
        base_scheduler=args.base_scheduler,
    )

    print(
        f"n={args.n} seed={args.seed} group_size={args.group_size} "
        f"r={args.r_reduced} base_sweeps={args.base_sweeps} "
        f"base_scheduler={args.base_scheduler} pre_post={args.pre_post} "
        f"eta_damping={args.eta_damping} preserve_lam={args.preserve_lam} P_shape={p.shape}"
    )
    if args.preserve_lam:
        print(f"lam_warmup_sweeps={lam_warmup_sweeps} lam_warmup_delta={lam_warmup_delta}")
    print(f"x0_rel={relative_error_vec(x0, x_star)}")
    print_points("residual_block_gbp_direct_coarse", hist, args.points)


if __name__ == "__main__":
    main()
