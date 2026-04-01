"""Ablate lam-side reuse in the reference GBP core.

This script keeps the graph/operator fixed and changes only the right-hand side
across outer residual-correction steps. It compares:

* fresh: rebuild a reference GBP graph on each new defect equation
* keep_lam_side: preserve all lam-side state (factor-message lam, variable
  belief lam, structural prior/factor lam) while clearing eta/mu-side state

The current linear xy chain decouples into two identical scalar channels, so
the scalar chain is an exact and cheap diagnostic target here.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

WORKSPACE_ROOT = pathlib.Path("/home/yuzhou/Desktop/abstraction-recovery")
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from svd_abstraction.chain_same_system_benchmark import extract_scalar_channel
from svd_abstraction.chain_same_system_benchmark import system_from_chain
from svd_abstraction.gbp_from_operator import build_reference_gbp_graph_from_operator
from svd_abstraction.gbp_from_operator import decompose_block_operator
from svd_abstraction.gbp_from_operator import graph_mean_vector


def reset_keep_lam_side(graph, eta_new: np.ndarray, lam_fixed, block_dofs: int = 1) -> None:
    """Keep the whole lam-side state and clear eta/mu-side state.

    Kept:
      * factor message lam
      * variable belief lam
      * structural prior/factor lam
    Reset:
      * factor message eta
      * variable belief eta
      * variable mu
      * factor adj_belief eta
    """
    unary_eta, _, _ = decompose_block_operator(eta_new, lam_fixed, block_dofs)

    for var, local_eta in zip(graph.var_nodes[: graph.n_var_nodes], unary_eta):
        var.prior.eta = np.array(local_eta, copy=True)

    for var in graph.var_nodes[: graph.n_var_nodes]:
        lam = var.prior.lam.copy()
        for factor in var.adj_factors:
            message_ix = factor.adj_var_nodes.index(var)
            lam += factor.messages[message_ix].lam
        var.belief.lam = lam
        var.belief.eta = np.zeros_like(var.belief.eta)
        var.mu = np.zeros_like(var.mu)
        try:
            var.Sigma = np.linalg.inv(lam)
        except np.linalg.LinAlgError:
            var.Sigma = np.linalg.pinv(lam)

    for factor in graph.factors[: graph.n_factor_nodes]:
        for idx, var in enumerate(factor.adj_var_nodes):
            factor.messages[idx].eta[:] = 0.0
            factor.adj_beliefs[idx].eta[:] = 0.0
            factor.adj_beliefs[idx].lam[:] = var.belief.lam


def run_outer_residual_correction(
    a,
    b: np.ndarray,
    x_star: np.ndarray,
    inner_sweeps: int,
    outer_iters: int,
    variant: str,
) -> dict[str, object]:
    x = np.zeros_like(b)
    initial_error = max(np.linalg.norm(x_star), 1e-15)
    error_history = [float(np.linalg.norm(x - x_star) / initial_error)]
    residual_history = [float(np.linalg.norm(b - a @ x))]

    graph = None
    if variant == "keep_lam_side":
        graph = build_reference_gbp_graph_from_operator(np.zeros_like(b), a, block_dofs=1)

    for _ in range(outer_iters):
        rhs = b - a @ x
        if variant == "fresh":
            graph = build_reference_gbp_graph_from_operator(rhs, a, block_dofs=1)
        elif variant == "keep_lam_side":
            reset_keep_lam_side(graph, rhs, a, block_dofs=1)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        for _ in range(inner_sweeps):
            graph.synchronous_iteration()

        x = x + graph_mean_vector(graph)
        error_history.append(float(np.linalg.norm(x - x_star) / initial_error))
        residual_history.append(float(np.linalg.norm(b - a @ x)))
        if not np.isfinite(error_history[-1]):
            break

    return {
        "iterations": len(error_history) - 1,
        "error_history": error_history,
        "residual_history": residual_history,
    }


def summarize(name: str, result: dict[str, object]) -> str:
    errors = result["error_history"]
    residuals = result["residual_history"]
    checkpoints = []
    for step in (1, 10, 50, len(errors) - 1):
        idx = min(step, len(errors) - 1)
        checkpoints.append(f"err@{idx}={errors[idx]:.3e}")
    return (
        f"{name}: iterations={result['iterations']}, "
        f"final_residual={residuals[-1]:.3e}, "
        + ", ".join(checkpoints)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--step-size", type=float, default=25.0)
    parser.add_argument("--prior-sigma", type=float, default=1.0)
    parser.add_argument("--odom-sigma", type=float, default=1.0)
    parser.add_argument("--inner-sweeps", type=int, default=1)
    parser.add_argument("--outer-iters", type=int, default=120)
    args = parser.parse_args()

    _, _, eta, lam, x_star_full = system_from_chain(
        n=args.n,
        step_size=args.step_size,
        prior_sigma=args.prior_sigma,
        odom_sigma=args.odom_sigma,
        seed=0,
    )
    b, a = extract_scalar_channel(eta, lam, 0)
    x_star = x_star_full[0::2]

    print(f"Reference GBP lam-side ablation on scalar chain n={args.n}")
    print("")

    fresh = run_outer_residual_correction(
        a=a,
        b=b,
        x_star=x_star,
        inner_sweeps=args.inner_sweeps,
        outer_iters=args.outer_iters,
        variant="fresh",
    )
    print(summarize("fresh", fresh))

    keep = run_outer_residual_correction(
        a=a,
        b=b,
        x_star=x_star,
        inner_sweeps=args.inner_sweeps,
        outer_iters=args.outer_iters,
        variant="keep_lam_side",
    )
    print(summarize("keep_lam_side", keep))


if __name__ == "__main__":
    main()
