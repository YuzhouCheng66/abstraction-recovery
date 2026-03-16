"""Experimental package for SVD-based residual abstraction."""

from .grouping import groups_from_grid
from .grouping import groups_from_kmeans
from .grouping import groups_from_order
from .pose_graph import build_noisy_pose_graph
from .pose_graph import make_slam_like_graph
from .residual_abstraction import SVDResidualAbstraction

__all__ = [
    "SVDResidualAbstraction",
    "build_noisy_pose_graph",
    "groups_from_grid",
    "groups_from_kmeans",
    "groups_from_order",
    "make_slam_like_graph",
]
