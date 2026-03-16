"""Minimal end-to-end demo for base -> abs residual correction."""

import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from svd_abstraction.grouping import groups_from_order
from svd_abstraction.pose_graph import build_noisy_pose_graph
from svd_abstraction.pose_graph import make_slam_like_graph
from svd_abstraction.residual_abstraction import SVDResidualAbstraction


def main():
    nodes, edges = make_slam_like_graph(
        N=30,
        step_size=15,
        loop_prob=0.08,
        loop_radius=40,
        prior_prop=0.15,
        seed=0,
    )
    base_graph = build_noisy_pose_graph(
        nodes,
        edges,
        prior_sigma=6.0,
        odom_sigma=3.0,
        tiny_prior=1e-12,
        seed=0,
    )

    abstraction = SVDResidualAbstraction(
        base_graph=base_graph,
        groups=groups_from_order(nodes, group_size=5, tail_heavy=True),
        r_reduced=2,
        basis_source="belief_covariance",
        freeze_basis=True,
    )

    abstraction.warmup(iterations=5, scheduler="sync")
    error_before = abstraction.average_error()
    energy_before = base_graph.energy()

    stats = abstraction.v_cycle(pre_smooth=1, post_smooth=1, scheduler="sync")

    error_after = abstraction.average_error()
    energy_after = base_graph.energy()

    print(f"Average error before: {error_before:.6f}")
    print(f"Average error after : {error_after:.6f}")
    print(f"Energy before       : {energy_before:.6f}")
    print(f"Energy after        : {energy_after:.6f}")
    print(f"Residual before     : {stats.residual_norm_before:.6f}")
    print(f"Residual after      : {stats.residual_norm_after:.6f}")
    print(f"Coarse residual     : {stats.coarse_residual_norm:.6f}")
    print(f"Correction norm     : {stats.correction_norm:.6f}")


if __name__ == "__main__":
    main()
