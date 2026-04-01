"""Helpers for building simple linear pose-graph GBP problems."""

import numpy as np

from svd_abstraction.gbp.gbp import Factor
from svd_abstraction.gbp.gbp import FactorGraph
from svd_abstraction.gbp.gbp import VariableNode


def make_slam_like_graph(
    N=100,
    step_size=25,
    loop_prob=0.05,
    loop_radius=50,
    prior_prop=0.0,
    seed=None,
):
    rng = np.random.default_rng(seed)

    nodes = []
    edges = []
    positions = []
    x = 0.0
    y = 0.0
    positions.append((x, y))

    for _ in range(1, int(N)):
        dx, dy = rng.standard_normal(2)
        norm = np.sqrt(dx * dx + dy * dy) + 1e-6
        dx = dx / norm * float(step_size)
        dy = dy / norm * float(step_size)
        x += dx
        y += dy
        positions.append((x, y))

    for i, (px, py) in enumerate(positions):
        nodes.append(
            {
                "data": {"id": f"{i}", "layer": 0, "dim": 2, "num_base": 1},
                "position": {"x": float(px), "y": float(py)},
            }
        )

    for i in range(int(N) - 1):
        edges.append({"data": {"source": f"{i}", "target": f"{i + 1}"}})

    for i in range(int(N)):
        for j in range(i + 5, int(N)):
            if rng.random() < float(loop_prob):
                xi, yi = positions[i]
                xj, yj = positions[j]
                if np.hypot(xi - xj, yi - yj) < float(loop_radius):
                    edges.append({"data": {"source": f"{i}", "target": f"{j}"}})

    if prior_prop <= 0.0:
        strong_ids = {0}
    elif prior_prop >= 1.0:
        strong_ids = set(range(int(N)))
    else:
        k = max(1, int(np.floor(prior_prop * N)))
        strong_ids = set(rng.choice(int(N), size=k, replace=False).tolist())

    for i in strong_ids:
        edges.append({"data": {"source": f"{i}", "target": "prior"}})
    edges.append({"data": {"source": "0", "target": "anchor"}})

    return nodes, edges


def make_grid_pose_graph(
    nx=16,
    ny=16,
    spacing=1.0,
    prior_prop=0.0,
    shortcut_prob=0.0,
    shortcut_min_sep=4,
    seed=None,
):
    rng = np.random.default_rng(seed)
    nx = int(nx)
    ny = int(ny)

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")

    nodes = []
    edges = []

    def node_id(ix, iy):
        return iy * nx + ix

    for iy in range(ny):
        for ix in range(nx):
            node_idx = node_id(ix, iy)
            nodes.append(
                {
                    "data": {"id": f"{node_idx}", "layer": 0, "dim": 2, "num_base": 1},
                    "position": {
                        "x": float(ix * spacing),
                        "y": float(iy * spacing),
                    },
                }
            )

    for iy in range(ny):
        for ix in range(nx):
            src = node_id(ix, iy)
            if ix + 1 < nx:
                edges.append({"data": {"source": f"{src}", "target": f"{node_id(ix + 1, iy)}"}})
            if iy + 1 < ny:
                edges.append({"data": {"source": f"{src}", "target": f"{node_id(ix, iy + 1)}"}})

    if shortcut_prob > 0.0:
        existing = {
            (min(int(edge["data"]["source"]), int(edge["data"]["target"])),
             max(int(edge["data"]["source"]), int(edge["data"]["target"])))
            for edge in edges
        }
        for iy0 in range(ny):
            for ix0 in range(nx):
                src = node_id(ix0, iy0)
                for iy1 in range(iy0, ny):
                    for ix1 in range(nx):
                        dst = node_id(ix1, iy1)
                        if dst <= src:
                            continue
                        manhattan = abs(ix1 - ix0) + abs(iy1 - iy0)
                        if manhattan < shortcut_min_sep:
                            continue
                        key = (src, dst)
                        if key in existing:
                            continue
                        if rng.random() < float(shortcut_prob):
                            edges.append({"data": {"source": f"{src}", "target": f"{dst}"}})
                            existing.add(key)

    total_nodes = nx * ny
    if prior_prop <= 0.0:
        strong_ids = set()
    elif prior_prop >= 1.0:
        strong_ids = set(range(total_nodes))
    else:
        k = max(1, int(np.floor(prior_prop * total_nodes)))
        strong_ids = set(rng.choice(total_nodes, size=k, replace=False).tolist())

    for node_idx in strong_ids:
        edges.append({"data": {"source": f"{node_idx}", "target": "prior"}})
    edges.append({"data": {"source": "0", "target": "anchor"}})

    return nodes, edges


def build_noisy_pose_graph(
    nodes,
    edges,
    prior_sigma=10.0,
    odom_sigma=10.0,
    tiny_prior=1e-12,
    seed=None,
):
    fg = FactorGraph(nonlinear_factors=False, eta_damping=0.0)
    rng = np.random.default_rng(seed)
    I2 = np.eye(2, dtype=float)

    prior_noises = {}
    odom_noises = {}

    for edge in edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]
        if dst not in {"prior", "anchor"}:
            odom_noises[(int(src), int(dst))] = rng.normal(0.0, odom_sigma, size=2)
        elif dst == "prior":
            prior_noises[int(src)] = rng.normal(0.0, prior_sigma, size=2)

    var_nodes = []
    for i, node in enumerate(nodes):
        var = VariableNode(i, dofs=2)
        var.GT = np.array([node["position"]["x"], node["position"]["y"]], dtype=float)
        var.prior.lam = tiny_prior * I2
        var.prior.eta = np.zeros(2, dtype=float)
        var_nodes.append(var)

    fg.var_nodes = var_nodes
    fg.n_var_nodes = len(var_nodes)

    def meas_fn_unary(x, *args):
        return [x]

    def jac_fn_unary(x, *args):
        return [np.eye(2)]

    def meas_fn_pair(xy, *args):
        return [xy[2:] - xy[:2]]

    def jac_fn_pair(xy, *args):
        return [np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]], dtype=float)]

    factors = []
    factor_id = 0

    for edge in edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]

        if dst not in {"prior", "anchor"}:
            i = int(src)
            j = int(dst)
            vi = var_nodes[i]
            vj = var_nodes[j]
            meas = (vj.GT - vi.GT) + odom_noises[(i, j)]
            meas_lambda = np.eye(2, dtype=float) / (odom_sigma ** 2)

            factor = Factor(
                factor_id,
                [vi, vj],
                [meas],
                [meas_lambda],
                meas_fn_pair,
                jac_fn_pair,
            )
            factor.type = "odometry"
            factor.compute_factor(linpoint=np.r_[vi.GT, vj.GT], update_self=True)

            factors.append(factor)
            vi.adj_factors.append(factor)
            vj.adj_factors.append(factor)
            factor_id += 1
        elif dst == "prior":
            i = int(src)
            vi = var_nodes[i]
            meas = vi.GT + prior_noises[i]
            meas_lambda = np.eye(2, dtype=float) / (prior_sigma ** 2)

            factor = Factor(
                factor_id,
                [vi],
                [meas],
                [meas_lambda],
                meas_fn_unary,
                jac_fn_unary,
            )
            factor.type = "prior"
            factor.compute_factor(linpoint=vi.GT, update_self=True)

            factors.append(factor)
            vi.adj_factors.append(factor)
            factor_id += 1

    anchor_var = var_nodes[0]
    anchor_meas = anchor_var.GT.copy()
    anchor_lambda = np.eye(2, dtype=float) / ((1e-4) ** 2)
    anchor_factor = Factor(
        factor_id,
        [anchor_var],
        [anchor_meas],
        [anchor_lambda],
        meas_fn_unary,
        jac_fn_unary,
    )
    anchor_factor.type = "anchor"
    anchor_factor.compute_factor(linpoint=anchor_var.GT, update_self=True)
    factors.append(anchor_factor)
    anchor_var.adj_factors.append(anchor_factor)

    fg.factors = factors
    fg.n_factor_nodes = len(factors)

    for var in fg.var_nodes:
        var.update_belief()

    return fg
