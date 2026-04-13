from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svd_abstraction.raylib_local_eta_prolongation_validation import (  # noqa: E402
    base_residual_norm,
    build_hierarchy,
    build_slam_graph,
    exact_mean,
    mean_vector,
    relative_error,
)


def lam_state(graph, levels):
    out = []
    for level in levels:
        for var in graph.multigrid_vars[level]:
            if getattr(var, "type", None) == "dead":
                continue
            out.append(np.asarray(var.belief.lam, dtype=float).ravel())
        for factor in graph.multigrid_factors[level]:
            if getattr(factor, "type", None) == "dead":
                continue
            for msg in factor.messages:
                out.append(np.asarray(msg.lam, dtype=float).ravel())
            for belief in factor.adj_beliefs:
                out.append(np.asarray(belief.lam, dtype=float).ravel())
    return np.concatenate(out)


def eta_only_iteration(graph, level):
    for factor in graph.multigrid_factors[level]:
        if getattr(factor, "type", None) != "dead" and factor.active:
            factor.compute_messages_eta_only_fixed_lam(graph.eta_damping)
    graph.update_all_beliefs_eta_only_fixed_lam(level=level)


def eta_only_vcycle_step(graph):
    eta_only_iteration(graph, 0)
    top = len(graph.multigrid_vars) - 1

    for level in range(1, top + 1):
        graph.update_all_residual_etas(level=level)
        graph.update_all_beliefs_eta_only_fixed_lam(level=level)
        eta_only_iteration(graph, level)
        graph.update_all_residuals(level=level)

    for level in range(top, 0, -1):
        eta_only_iteration(graph, level)
        graph.prolongate_corrections(level=level)


def zero_dynamic_mean_state(graph, levels):
    for level in levels:
        for var in graph.multigrid_vars[level]:
            if getattr(var, "type", None) == "dead":
                continue
            var.mu = np.zeros_like(var.mu)
            var.belief.eta = np.zeros_like(var.belief.eta)
            for factor in var.adj_factors:
                belief_ix = factor.adj_var_nodes.index(var)
                factor.adj_beliefs[belief_ix].eta = np.zeros_like(factor.adj_beliefs[belief_ix].eta)

        for factor in graph.multigrid_factors[level]:
            if getattr(factor, "type", None) == "dead":
                continue
            for msg in factor.messages:
                msg.eta = np.zeros_like(msg.eta)
            for belief in factor.adj_beliefs:
                belief.eta = np.zeros_like(belief.eta)

    # Coarse prior.eta is dynamic residual state, so reset it too.
    # Base prior.eta defines the original linear system and is kept unchanged.
    for level in levels:
        if level == 0:
            continue
        for var in graph.multigrid_vars[level]:
            if getattr(var, "type", None) == "dead":
                continue
            var.prior.eta = np.zeros_like(var.prior.eta)


def normal_run_until_variance_settles(graph, mu_star, mode: str, threshold: float, max_steps: int):
    if mode == "base":
        levels = [0]
    elif mode == "multilevel":
        levels = list(range(len(graph.multigrid_vars)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    prev = lam_state(graph, levels)
    lam_deltas = []
    relerrs = [relative_error(graph, mu_star)]
    residuals = [base_residual_norm(graph)]

    freeze_step = None
    for step in range(1, max_steps + 1):
        if mode == "base":
            graph.synchronous_iteration(level=0)
        else:
            graph.vcycle_step()

        curr = lam_state(graph, levels)
        lam_delta = float(np.max(np.abs(curr - prev)))
        lam_deltas.append(lam_delta)
        prev = curr
        relerrs.append(relative_error(graph, mu_star))
        residuals.append(base_residual_norm(graph))
        if freeze_step is None and lam_delta < threshold:
            freeze_step = step
            break

    return {
        "mu_star": mu_star,
        "levels": levels,
        "freeze_step": freeze_step,
        "lam_deltas": lam_deltas,
        "relerrs": relerrs,
        "residuals": residuals,
    }


def frozen_run_from_zero_mean(graph, mu_star, levels, mode: str, tol: float, max_steps: int):
    zero_dynamic_mean_state(graph, levels)

    relerrs = [float(np.linalg.norm(mean_vector(graph) - mu_star) / np.linalg.norm(mu_star))]
    residuals = [base_residual_norm(graph)]
    conv = None
    for step in range(1, max_steps + 1):
        if mode == "base":
            eta_only_iteration(graph, 0)
        else:
            eta_only_vcycle_step(graph)

        rel = relative_error(graph, mu_star)
        relerrs.append(rel)
        residuals.append(base_residual_norm(graph))
        if conv is None and rel < tol:
            conv = step
            break

    return {
        "conv": conv,
        "relerrs": relerrs,
        "residuals": residuals,
    }


def metric_points(relerrs, points):
    return {pt: relerrs[pt] for pt in points if pt < len(relerrs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument("--freeze-max-steps", type=int, default=500)
    parser.add_argument("--post-max-base-iters", type=int, default=3000)
    parser.add_argument("--post-max-vcycles", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000],
    )
    args = parser.parse_args()

    print(f"Variance-freeze experiment on N={args.n}")
    print(f"variance threshold: {args.variance_threshold:.1e}")
    print("definition: max abs change over all dynamic belief/message lam blocks")
    print()

    for mode in ["base", "multilevel"]:
        graph = build_slam_graph(n=args.n, seed=0)
        mu_star = exact_mean(graph)
        if mode == "multilevel":
            build_hierarchy(graph)

        normal = normal_run_until_variance_settles(
            graph=graph,
            mu_star=mu_star,
            mode=mode,
            threshold=args.variance_threshold,
            max_steps=args.freeze_max_steps,
        )

        freeze_step = normal["freeze_step"]
        if freeze_step is None:
            print(f"{mode}: variance did not settle within {args.freeze_max_steps} steps")
            continue

        print(f"{mode}: variance settled at step {freeze_step}")
        print(
            f"  mean relerr at freeze: {normal['relerrs'][freeze_step]:.12g}"
        )
        print(
            f"  base residual at freeze: {normal['residuals'][freeze_step]:.12g}"
        )
        print(
            f"  lam delta at freeze: {normal['lam_deltas'][freeze_step - 1]:.12g}"
        )

        frozen = frozen_run_from_zero_mean(
            graph=graph,
            mu_star=normal["mu_star"],
            levels=normal["levels"],
            mode=mode,
            tol=args.tol,
            max_steps=args.post_max_base_iters if mode == "base" else args.post_max_vcycles,
        )

        print(f"  after reset relerr: {frozen['relerrs'][0]:.12g}")
        print(f"  after reset base residual: {frozen['residuals'][0]:.12g}")
        print(f"  frozen-lam convergence steps: {frozen['conv']}")
        print(f"  frozen-lam relerr points: {metric_points(frozen['relerrs'], args.points)}")
        print()


if __name__ == "__main__":
    main()
