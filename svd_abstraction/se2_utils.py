from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from svd_abstraction.gbp.gbp import Factor
from svd_abstraction.gbp.gbp import FactorGraph
from svd_abstraction.gbp.gbp import VariableNode
from svd_abstraction.pose_graph import make_slam_like_graph


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi


def rot2(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def se2_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    out = np.zeros(3, dtype=float)
    out[:2] = a[:2] + rot2(a[2]) @ b[:2]
    out[2] = wrap_angle(a[2] + b[2])
    return out


def se2_inverse(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3)
    r_t = rot2(a[2]).T
    out = np.zeros(3, dtype=float)
    out[:2] = -(r_t @ a[:2])
    out[2] = wrap_angle(-a[2])
    return out


def se2_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return se2_compose(se2_inverse(a), b)


def se2_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float).reshape(3)
    vx, vy, w = xi
    if abs(w) < 1e-12:
        return np.array([vx, vy, 0.0], dtype=float)
    a = np.sin(w) / w
    b = (1.0 - np.cos(w)) / w
    v_mat = np.array([[a, -b], [b, a]], dtype=float)
    t = v_mat @ np.array([vx, vy], dtype=float)
    return np.array([t[0], t[1], wrap_angle(w)], dtype=float)


def se2_log(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).reshape(3)
    tx, ty, w = pose
    if abs(w) < 1e-12:
        return np.array([tx, ty, 0.0], dtype=float)
    a = np.sin(w) / w
    b = (1.0 - np.cos(w)) / w
    denom = a * a + b * b
    v_inv = (1.0 / denom) * np.array([[a, b], [-b, a]], dtype=float)
    v = v_inv @ np.array([tx, ty], dtype=float)
    return np.array([v[0], v[1], wrap_angle(w)], dtype=float)


def se2_plus(base_pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return se2_compose(np.asarray(base_pose, dtype=float).reshape(3), se2_exp(delta))


def pose_error(gt_pose: np.ndarray, est_pose: np.ndarray) -> np.ndarray:
    return se2_log(se2_between(gt_pose, est_pose))


def stack_pose_errors(gt_poses: np.ndarray, est_poses: np.ndarray) -> np.ndarray:
    errs = [pose_error(gt, est) for gt, est in zip(gt_poses, est_poses)]
    return np.concatenate(errs) if errs else np.zeros(0, dtype=float)


def rms_translation_error(gt_poses: np.ndarray, est_poses: np.ndarray) -> float:
    diff = np.asarray(est_poses, dtype=float)[:, :2] - np.asarray(gt_poses, dtype=float)[:, :2]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def rms_angle_error(gt_poses: np.ndarray, est_poses: np.ndarray) -> float:
    diff = wrap_angle(np.asarray(est_poses, dtype=float)[:, 2] - np.asarray(gt_poses, dtype=float)[:, 2])
    return float(np.sqrt(np.mean(diff * diff)))


def mean_pose_log_error(gt_poses: np.ndarray, est_poses: np.ndarray) -> float:
    errs = stack_pose_errors(gt_poses, est_poses).reshape(-1, 3)
    return float(np.mean(np.linalg.norm(errs, axis=1)))


def _numeric_jacobian(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    y0 = np.asarray(fn(x), dtype=float).reshape(-1)
    jac = np.zeros((y0.size, x.size), dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        yp = np.asarray(fn(xp), dtype=float).reshape(-1)
        ym = np.asarray(fn(xm), dtype=float).reshape(-1)
        jac[:, i] = (yp - ym) / (2.0 * eps)
    return jac


@dataclass
class SE2Edge:
    i: int
    j: int
    measurement: np.ndarray
    information: np.ndarray
    kind: str


@dataclass
class SE2Problem:
    nodes: list[dict]
    gt_poses: np.ndarray
    init_poses: np.ndarray
    edges: list[SE2Edge]
    anchor_pose: np.ndarray
    anchor_information: np.ndarray


def _derive_headings(positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    headings = np.zeros(n, dtype=float)
    for i in range(n - 1):
        step = positions[i + 1] - positions[i]
        headings[i] = np.arctan2(step[1], step[0])
    if n >= 2:
        headings[-1] = headings[-2]
    return headings


def build_se2_problem(
    n: int = 64,
    step_size: float = 25.0,
    loop_prob: float = 0.05,
    loop_radius: float = 50.0,
    prior_prop: float = 0.0,
    odom_trans_sigma: float = 1.0,
    odom_rot_sigma: float = 0.05,
    loop_trans_sigma: float = 1.0,
    loop_rot_sigma: float = 0.05,
    seed: int = 0,
) -> SE2Problem:
    rng = np.random.default_rng(seed)
    nodes, raw_edges = make_slam_like_graph(
        N=n,
        step_size=step_size,
        loop_prob=loop_prob,
        loop_radius=loop_radius,
        prior_prop=prior_prop,
        seed=seed,
    )
    positions = np.array([[node["position"]["x"], node["position"]["y"]] for node in nodes], dtype=float)
    headings = _derive_headings(positions)
    gt_poses = np.column_stack([positions, headings])

    eye3 = np.eye(3, dtype=float)
    edges: list[SE2Edge] = []
    chain_meas: dict[tuple[int, int], np.ndarray] = {}

    for edge in raw_edges:
        src = edge["data"]["source"]
        dst = edge["data"]["target"]
        if dst in {"prior", "anchor"}:
            continue

        i = int(src)
        j = int(dst)
        gt_rel = se2_between(gt_poses[i], gt_poses[j])
        is_chain = (j == i + 1)
        trans_sigma = odom_trans_sigma if is_chain else loop_trans_sigma
        rot_sigma = odom_rot_sigma if is_chain else loop_rot_sigma
        noise = np.array(
            [
                rng.normal(0.0, trans_sigma),
                rng.normal(0.0, trans_sigma),
                rng.normal(0.0, rot_sigma),
            ],
            dtype=float,
        )
        meas = se2_compose(gt_rel, se2_exp(noise))
        info = np.diag([1.0 / (trans_sigma**2), 1.0 / (trans_sigma**2), 1.0 / (rot_sigma**2)])
        kind = "odometry" if is_chain else "loop"
        edges.append(SE2Edge(i=i, j=j, measurement=meas, information=info, kind=kind))
        if is_chain:
            chain_meas[(i, j)] = meas.copy()

    init_poses = np.zeros_like(gt_poses)
    init_poses[0] = gt_poses[0].copy()
    for i in range(n - 1):
        init_poses[i + 1] = se2_compose(init_poses[i], chain_meas[(i, i + 1)])

    anchor_pose = gt_poses[0].copy()
    anchor_information = np.diag([1e8, 1e8, 1e8])
    return SE2Problem(
        nodes=nodes,
        gt_poses=gt_poses,
        init_poses=init_poses,
        edges=edges,
        anchor_pose=anchor_pose,
        anchor_information=anchor_information,
    )


def nonlinear_objective(problem: SE2Problem, poses: np.ndarray) -> float:
    total = 0.0
    for edge in problem.edges:
        pred = se2_between(poses[edge.i], poses[edge.j])
        err = se2_log(se2_compose(se2_inverse(edge.measurement), pred))
        total += 0.5 * float(err.T @ edge.information @ err)
    anchor_err = se2_log(se2_compose(se2_inverse(problem.anchor_pose), poses[0]))
    total += 0.5 * float(anchor_err.T @ problem.anchor_information @ anchor_err)
    return total


def poses_to_nodes(poses: np.ndarray) -> list[dict]:
    nodes = []
    for i, pose in enumerate(np.asarray(poses, dtype=float)):
        nodes.append(
            {
                "data": {"id": f"{i}", "layer": 0, "dim": 3, "num_base": 1},
                "position": {"x": float(pose[0]), "y": float(pose[1])},
            }
        )
    return nodes


def build_linearized_local_graph(problem: SE2Problem, base_poses: np.ndarray, tiny_prior: float = 1e-12) -> FactorGraph:
    base_poses = np.asarray(base_poses, dtype=float).reshape(-1, 3)
    graph = FactorGraph(nonlinear_factors=False, eta_damping=0.0)
    vars_ = []
    for i, gt_pose in enumerate(problem.gt_poses):
        var = VariableNode(i, dofs=3)
        var.type = "pose"
        var.GT = np.asarray(gt_pose, dtype=float).copy()
        var.prior.lam = tiny_prior * np.eye(3, dtype=float)
        var.prior.eta = np.zeros(3, dtype=float)
        vars_.append(var)
    graph.var_nodes = vars_
    graph.n_var_nodes = len(vars_)

    factors = []
    factor_id = 0

    for edge in problem.edges:
        vi = vars_[edge.i]
        vj = vars_[edge.j]
        base_i = base_poses[edge.i].copy()
        base_j = base_poses[edge.j].copy()
        z = edge.measurement.copy()
        omega = edge.information.copy()

        def meas_fn_local(x, base_i=base_i, base_j=base_j, z=z):
            ei = np.asarray(x[:3], dtype=float)
            ej = np.asarray(x[3:6], dtype=float)
            xi = se2_plus(base_i, ei)
            xj = se2_plus(base_j, ej)
            pred = se2_between(xi, xj)
            err = se2_log(se2_compose(se2_inverse(z), pred))
            return [err]

        def jac_fn_local(x, base_i=base_i, base_j=base_j, z=z):
            fn = lambda xx: meas_fn_local(xx, base_i=base_i, base_j=base_j, z=z)[0]
            return [_numeric_jacobian(fn, np.asarray(x, dtype=float).reshape(-1))]

        factor = Factor(
            factor_id,
            [vi, vj],
            [np.zeros(3, dtype=float)],
            [omega],
            meas_fn_local,
            jac_fn_local,
        )
        factor.type = edge.kind
        factor.compute_factor(linpoint=np.zeros(6, dtype=float), update_self=True)
        factors.append(factor)
        vi.adj_factors.append(factor)
        vj.adj_factors.append(factor)
        factor_id += 1

    base_anchor = base_poses[0].copy()
    anchor_pose = problem.anchor_pose.copy()
    anchor_info = problem.anchor_information.copy()

    def meas_fn_anchor(x, base_anchor=base_anchor, anchor_pose=anchor_pose):
        ei = np.asarray(x[:3], dtype=float)
        xi = se2_plus(base_anchor, ei)
        err = se2_log(se2_compose(se2_inverse(anchor_pose), xi))
        return [err]

    def jac_fn_anchor(x, base_anchor=base_anchor, anchor_pose=anchor_pose):
        fn = lambda xx: meas_fn_anchor(xx, base_anchor=base_anchor, anchor_pose=anchor_pose)[0]
        return [_numeric_jacobian(fn, np.asarray(x, dtype=float).reshape(-1))]

    anchor = Factor(
        factor_id,
        [vars_[0]],
        [np.zeros(3, dtype=float)],
        [anchor_info],
        meas_fn_anchor,
        jac_fn_anchor,
    )
    anchor.type = "anchor"
    anchor.compute_factor(linpoint=np.zeros(3, dtype=float), update_self=True)
    factors.append(anchor)
    vars_[0].adj_factors.append(anchor)

    graph.factors = factors
    graph.n_factor_nodes = len(factors)
    for var in graph.var_nodes[: graph.n_var_nodes]:
        var.update_belief()
    return graph


def direct_solve_linear_graph(graph: FactorGraph, ridge: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta, lam = graph.joint_distribution_inf_absolute()
    if lam.size == 0:
        return np.zeros(0, dtype=float), eta, lam
    stabilized = 0.5 * (lam + lam.T) + ridge * np.eye(lam.shape[0], dtype=float)
    try:
        chol, lower = scipy.linalg.cho_factor(stabilized, lower=False, check_finite=False)
        mu = scipy.linalg.cho_solve((chol, lower), eta)
    except np.linalg.LinAlgError:
        mu = np.linalg.solve(stabilized, eta)
    return mu, eta, lam


def apply_pose_deltas(base_poses: np.ndarray, delta_vec: np.ndarray) -> np.ndarray:
    base_poses = np.asarray(base_poses, dtype=float).reshape(-1, 3)
    delta_vec = np.asarray(delta_vec, dtype=float).reshape(-1)
    out = np.zeros_like(base_poses)
    for i in range(base_poses.shape[0]):
        out[i] = se2_plus(base_poses[i], delta_vec[3 * i : 3 * i + 3])
    return out

