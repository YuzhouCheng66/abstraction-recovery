"""Grouping utilities for residual abstraction experiments."""

import numpy as np


def _node_id(node):
    return int(str(node["data"]["id"]))


def groups_from_order(nodes, group_size, tail_heavy=True):
    """
    Group nodes in their current order.

    `group_size` means the preferred number of base variables per group.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    ids = [_node_id(node) for node in nodes]
    if not ids:
        return []

    if not tail_heavy:
        return [ids[start : start + group_size] for start in range(0, len(ids), group_size)]

    groups = []
    start = 0
    while start + 2 * group_size <= len(ids):
        groups.append(ids[start : start + group_size])
        start += group_size
    groups.append(ids[start:])
    return [group for group in groups if group]


def groups_from_grid(nodes, gx, gy):
    if gx <= 0 or gy <= 0:
        raise ValueError("gx and gy must be positive")
    if not nodes:
        return []

    positions = np.array([[node["position"]["x"], node["position"]["y"]] for node in nodes], dtype=float)
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)

    cell_w = (xmax - xmin) / gx if gx > 0 else 1.0
    cell_h = (ymax - ymin) / gy if gy > 0 else 1.0
    if cell_w == 0.0:
        cell_w = 1.0
    if cell_h == 0.0:
        cell_h = 1.0

    cell_map = {}
    for node in nodes:
        x = float(node["position"]["x"])
        y = float(node["position"]["y"])
        cx = min(int((x - xmin) / cell_w), gx - 1)
        cy = min(int((y - ymin) / cell_h), gy - 1)
        cell_map.setdefault((cx, cy), []).append(_node_id(node))

    return [cell_map[key] for key in sorted(cell_map)]


def groups_from_kmeans(nodes, k, max_iters=20, tol=1e-6, seed=0):
    if k <= 0:
        raise ValueError("k must be positive")
    if not nodes:
        return []

    positions = np.array([[node["position"]["x"], node["position"]["y"]] for node in nodes], dtype=float)
    ids = np.array([_node_id(node) for node in nodes], dtype=int)

    n = positions.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = positions[init_idx].copy()

    for _ in range(max_iters):
        d2 = ((positions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)

        counts = np.bincount(assign, minlength=k)
        empty_clusters = np.where(counts == 0)[0]
        for cluster_id in empty_clusters:
            largest_cluster = np.argmax(counts)
            largest_members = np.where(assign == largest_cluster)[0]
            moved_idx = largest_members[0]
            assign[moved_idx] = cluster_id
            counts[largest_cluster] -= 1
            counts[cluster_id] += 1

        max_move = 0.0
        for cluster_id in range(k):
            members = np.where(assign == cluster_id)[0]
            new_center = positions[members].mean(axis=0)
            max_move = max(max_move, float(np.linalg.norm(new_center - centers[cluster_id])))
            centers[cluster_id] = new_center
        if max_move < tol:
            break

    d2 = ((positions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = np.argmin(d2, axis=1)

    groups = []
    for cluster_id in range(k):
        members = ids[assign == cluster_id]
        if len(members) > 0:
            groups.append(members.tolist())

    return groups
