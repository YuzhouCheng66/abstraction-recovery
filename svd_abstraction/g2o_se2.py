from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from svd_abstraction.gbp.gbp import Factor
from svd_abstraction.gbp.gbp import FactorGraph
from svd_abstraction.gbp.gbp import VariableNode
from svd_abstraction.se2_utils import SE2Edge
from svd_abstraction.se2_utils import apply_pose_deltas
from svd_abstraction.se2_utils import se2_between
from svd_abstraction.se2_utils import se2_compose
from svd_abstraction.se2_utils import se2_inverse
from svd_abstraction.se2_utils import se2_log
from svd_abstraction.se2_utils import se2_plus


@dataclass
class G2OSE2Problem:
    source_path: str
    original_ids: list[int]
    init_poses: np.ndarray
    edges: list[SE2Edge]
    nodes: list[dict]
    anchor_pose: np.ndarray
    anchor_information: np.ndarray


def _make_nodes_from_poses(poses: np.ndarray) -> list[dict]:
    nodes = []
    for i, pose in enumerate(np.asarray(poses, dtype=float)):
        nodes.append(
            {
                "data": {"id": f"{i}", "layer": 0, "dim": 3, "num_base": 1},
                "position": {"x": float(pose[0]), "y": float(pose[1])},
            }
        )
    return nodes


def _info6_to_matrix(vals: list[float]) -> np.ndarray:
    if len(vals) != 6:
        raise ValueError(f"Expected 6 upper-triangular information entries, got {len(vals)}")
    i11, i12, i13, i22, i23, i33 = [float(v) for v in vals]
    return np.array(
        [
            [i11, i12, i13],
            [i12, i22, i23],
            [i13, i23, i33],
        ],
        dtype=float,
    )


def parse_g2o_se2(path: str | Path, anchor_precision: float = 1e8) -> G2OSE2Problem:
    path = Path(path)
    raw_vertices: dict[int, np.ndarray] = {}
    raw_edges: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "VERTEX_SE2":
                if len(parts) != 5:
                    raise ValueError(f"{path}:{lineno}: malformed VERTEX_SE2 line")
                vid = int(parts[1])
                pose = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
                raw_vertices[vid] = pose
            elif tag == "EDGE_SE2":
                if len(parts) != 12:
                    raise ValueError(f"{path}:{lineno}: malformed EDGE_SE2 line")
                vi = int(parts[1])
                vj = int(parts[2])
                meas = np.array([float(parts[3]), float(parts[4]), float(parts[5])], dtype=float)
                info = _info6_to_matrix([float(x) for x in parts[6:12]])
                raw_edges.append((vi, vj, meas, info))
            else:
                raise ValueError(f"{path}:{lineno}: unsupported tag {tag}")

    if not raw_vertices:
        raise ValueError(f"{path}: no VERTEX_SE2 entries found")

    original_ids = sorted(raw_vertices)
    id_to_idx = {vid: idx for idx, vid in enumerate(original_ids)}
    init_poses = np.vstack([raw_vertices[vid] for vid in original_ids]).astype(float)

    edges: list[SE2Edge] = []
    for vi, vj, meas, info in raw_edges:
        if vi not in id_to_idx or vj not in id_to_idx:
            raise ValueError(f"{path}: edge references unknown vertex id {vi}->{vj}")
        ii = id_to_idx[vi]
        jj = id_to_idx[vj]
        kind = "odometry" if abs(vi - vj) == 1 else "loop"
        edges.append(
            SE2Edge(
                i=ii,
                j=jj,
                measurement=meas.copy(),
                information=info.copy(),
                kind=kind,
            )
        )

    anchor_pose = init_poses[0].copy()
    anchor_information = float(anchor_precision) * np.eye(3, dtype=float)

    return G2OSE2Problem(
        source_path=str(path),
        original_ids=original_ids,
        init_poses=init_poses,
        edges=edges,
        nodes=_make_nodes_from_poses(init_poses),
        anchor_pose=anchor_pose,
        anchor_information=anchor_information,
    )


def summarize_g2o_se2(problem: G2OSE2Problem) -> dict[str, float | int | bool]:
    n = int(problem.init_poses.shape[0])
    odom_edges = [e for e in problem.edges if e.kind == "odometry"]
    loop_edges = [e for e in problem.edges if e.kind == "loop"]
    gaps = [abs(e.j - e.i) for e in problem.edges]
    loop_gaps = [abs(e.j - e.i) for e in loop_edges]
    return {
        "num_vertices": n,
        "num_edges": int(len(problem.edges)),
        "num_odometry_edges": int(len(odom_edges)),
        "num_loop_edges": int(len(loop_edges)),
        "original_id_min": int(min(problem.original_ids)),
        "original_id_max": int(max(problem.original_ids)),
        "original_ids_contiguous": problem.original_ids == list(range(problem.original_ids[0], problem.original_ids[-1] + 1)),
        "max_edge_gap": int(max(gaps) if gaps else 0),
        "max_loop_gap": int(max(loop_gaps) if loop_gaps else 0),
    }


def plot_initial_pose_graph(
    problem: G2OSE2Problem,
    out_path: str | Path,
    loop_alpha: float = 0.12,
    loop_linewidth: float = 0.35,
) -> Path:
    poses = np.asarray(problem.init_poses, dtype=float)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=180)

    for edge in problem.edges:
        if edge.kind != "loop":
            continue
        pi = poses[edge.i, :2]
        pj = poses[edge.j, :2]
        ax.plot(
            [pi[0], pj[0]],
            [pi[1], pj[1]],
            color="#8a8a8a",
            alpha=loop_alpha,
            linewidth=loop_linewidth,
            zorder=1,
        )

    ax.plot(
        poses[:, 0],
        poses[:, 1],
        color="#1f77b4",
        linewidth=1.2,
        label="Initial trajectory",
        zorder=2,
    )
    ax.scatter([poses[0, 0]], [poses[0, 1]], color="black", s=18, label="Start", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Initial SE(2) Pose Graph from g2o")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


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


def nonlinear_objective_g2o(problem: G2OSE2Problem, poses: np.ndarray) -> float:
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    total = 0.0
    for edge in problem.edges:
        pred = se2_between(poses[edge.i], poses[edge.j])
        err = se2_log(se2_compose(se2_inverse(edge.measurement), pred))
        total += 0.5 * float(err.T @ edge.information @ err)
    anchor_err = se2_log(se2_compose(se2_inverse(problem.anchor_pose), poses[0]))
    total += 0.5 * float(anchor_err.T @ problem.anchor_information @ anchor_err)
    return total


def poses_to_nodes_g2o(poses: np.ndarray) -> list[dict]:
    return _make_nodes_from_poses(poses)


def linearize_g2o_problem(problem: G2OSE2Problem, base_poses: np.ndarray) -> tuple[sp.csc_matrix, np.ndarray]:
    base_poses = np.asarray(base_poses, dtype=float).reshape(-1, 3)
    n = base_poses.shape[0]
    dim = 3 * n

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(dim, dtype=float)

    def add_block(r0: int, c0: int, block: np.ndarray) -> None:
        rr, cc = block.shape
        for i in range(rr):
            for j in range(cc):
                val = float(block[i, j])
                if val == 0.0:
                    continue
                rows.append(r0 + i)
                cols.append(c0 + j)
                data.append(val)

    for edge in problem.edges:
        i = int(edge.i)
        j = int(edge.j)
        Xi = base_poses[i].copy()
        Xj = base_poses[j].copy()
        z = edge.measurement.copy()
        omega = edge.information.copy()

        def residual_fn_local(x: np.ndarray) -> np.ndarray:
            ei = np.asarray(x[:3], dtype=float)
            ej = np.asarray(x[3:6], dtype=float)
            xi = se2_plus(Xi, ei)
            xj = se2_plus(Xj, ej)
            pred = se2_between(xi, xj)
            return se2_log(se2_compose(se2_inverse(z), pred))

        x0 = np.zeros(6, dtype=float)
        r0 = residual_fn_local(x0)
        J = _numeric_jacobian(residual_fn_local, x0)
        H = J.T @ omega @ J
        g = J.T @ omega @ r0

        si = slice(3 * i, 3 * i + 3)
        sj = slice(3 * j, 3 * j + 3)
        add_block(si.start, si.start, H[:3, :3])
        add_block(si.start, sj.start, H[:3, 3:])
        add_block(sj.start, si.start, H[3:, :3])
        add_block(sj.start, sj.start, H[3:, 3:])
        b[si] += -g[:3]
        b[sj] += -g[3:]

    X0 = base_poses[0].copy()
    Z0 = problem.anchor_pose.copy()
    omega0 = problem.anchor_information.copy()

    def residual_fn_anchor(x: np.ndarray) -> np.ndarray:
        e0 = np.asarray(x[:3], dtype=float)
        x0 = se2_plus(X0, e0)
        return se2_log(se2_compose(se2_inverse(Z0), x0))

    a0 = np.zeros(3, dtype=float)
    r0 = residual_fn_anchor(a0)
    J0 = _numeric_jacobian(residual_fn_anchor, a0)
    H0 = J0.T @ omega0 @ J0
    g0 = J0.T @ omega0 @ r0
    add_block(0, 0, H0)
    b[:3] += -g0

    A = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsc()
    A = 0.5 * (A + A.T)
    return A, b


def build_linearized_local_graph_g2o(
    problem: G2OSE2Problem,
    base_poses: np.ndarray,
    tiny_prior: float = 1e-12,
) -> FactorGraph:
    base_poses = np.asarray(base_poses, dtype=float).reshape(-1, 3)
    graph = FactorGraph(nonlinear_factors=False, eta_damping=0.0)

    vars_: list[VariableNode] = []
    for i, pose in enumerate(base_poses):
        var = VariableNode(i, dofs=3)
        var.type = "pose"
        var.GT = np.asarray(pose, dtype=float).copy()
        var.prior.lam = tiny_prior * np.eye(3, dtype=float)
        var.prior.eta = np.zeros(3, dtype=float)
        vars_.append(var)
    graph.var_nodes = vars_
    graph.n_var_nodes = len(vars_)

    factors: list[Factor] = []
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


def direct_newton_step_g2o(problem: G2OSE2Problem, base_poses: np.ndarray, ridge: float = 1e-10) -> dict[str, object]:
    A, b = linearize_g2o_problem(problem, base_poses)
    A = A + ridge * sp.eye(A.shape[0], format="csc")
    delta = spla.spsolve(A, b)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    lin_res = float(np.linalg.norm(A @ delta - b))
    next_poses = apply_pose_deltas(base_poses, delta)
    return {
        "A": A,
        "b": b,
        "delta": delta,
        "linear_step_norm": float(np.linalg.norm(delta)),
        "linear_residual_norm": lin_res,
        "next_poses": next_poses,
    }


def _solve_lm_system(
    A: sp.csc_matrix,
    b: np.ndarray,
    damping: float,
    ridge: float = 0.0,
) -> dict[str, object]:
    system = A + float(damping) * sp.eye(A.shape[0], format="csc")
    if ridge > 0.0:
        system = system + float(ridge) * sp.eye(A.shape[0], format="csc")
    delta = spla.spsolve(system, b)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    lin_res = float(np.linalg.norm(system @ delta - b))
    scale = float(delta @ (float(damping) * delta + np.asarray(b, dtype=float).reshape(-1)))
    return {
        "delta": delta,
        "linear_residual_norm": lin_res,
        "scale": scale,
        "system": system,
    }


def run_levenberg_marquardt_g2o(
    problem: G2OSE2Problem,
    num_outer: int = 100,
    rel_obj_tol: float | None = None,
    step_tol: float | None = None,
    initial_lambda: float = 0.0,
    tau: float = 1e-5,
    good_step_lower_scale: float = 1.0 / 3.0,
    good_step_upper_scale: float = 2.0 / 3.0,
    max_trials_after_failure: int = 10,
    ridge: float = 0.0,
) -> dict[str, object]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    history: list[dict[str, float | int | bool | str]] = []
    attempt_rows: list[dict[str, float | int | bool]] = []

    obj_prev = float(nonlinear_objective_g2o(problem, poses))
    history.append(
        {
            "outer": 0,
            "nonlinear_objective": obj_prev,
            "accepted": True,
            "lambda": 0.0,
            "attempts": 0,
        }
    )

    damping: float | None = None
    nu = 2.0
    termination_reason = "max_outer"

    for outer in range(1, int(num_outer) + 1):
        A, b = linearize_g2o_problem(problem, poses)
        diag = np.asarray(A.diagonal(), dtype=float).reshape(-1)
        max_diag = float(np.max(np.abs(diag))) if diag.size else 0.0
        if damping is None:
            if float(initial_lambda) > 0.0:
                damping = float(initial_lambda)
            else:
                damping = float(tau) * max(max_diag, 1e-12)

        trial_count = 0
        accepted_row: dict[str, float | int | bool | str] | None = None
        rho = float("-inf")

        while trial_count < int(max_trials_after_failure):
            trial_count += 1
            lm_step = _solve_lm_system(
                A=A,
                b=b,
                damping=float(damping),
                ridge=ridge,
            )
            delta = np.asarray(lm_step["delta"], dtype=float)
            step_norm = float(np.linalg.norm(delta))
            trial_poses = apply_pose_deltas(poses, delta)
            trial_obj = float(nonlinear_objective_g2o(problem, trial_poses))
            actual_reduction = float(obj_prev - trial_obj)
            scale = float(lm_step["scale"]) + 1e-3
            rho = float(actual_reduction / scale)
            accepted = bool(rho > 0.0 and np.isfinite(trial_obj))

            attempt_rows.append(
                {
                    "outer": int(outer),
                    "trial": int(trial_count),
                    "accepted": accepted,
                    "lambda": float(damping),
                    "nonlinear_objective_before": obj_prev,
                    "nonlinear_objective_after": trial_obj,
                    "actual_reduction": actual_reduction,
                    "scale": scale,
                    "rho": rho,
                    "step_norm": step_norm,
                    "linear_residual_norm": float(lm_step["linear_residual_norm"]),
                }
            )

            if accepted:
                poses = trial_poses
                pose_history.append(poses.copy())
                rel_improve = float(actual_reduction / max(abs(obj_prev), 1e-15))
                alpha = 1.0 - (2.0 * rho - 1.0) ** 3
                alpha = min(alpha, float(good_step_upper_scale))
                scale_factor = max(float(good_step_lower_scale), alpha)
                lambda_used = float(damping)
                damping = float(damping) * float(scale_factor)
                nu = 2.0
                accepted_row = {
                    "outer": int(outer),
                    "nonlinear_objective": trial_obj,
                    "accepted": True,
                    "lambda": lambda_used,
                    "lambda_next": float(damping),
                    "attempts": int(trial_count),
                    "step_norm": step_norm,
                    "linear_residual_norm": float(lm_step["linear_residual_norm"]),
                    "scale": scale,
                    "actual_reduction": actual_reduction,
                    "rho": rho,
                    "relative_objective_improvement": rel_improve,
                }
                history.append(accepted_row)
                obj_prev = trial_obj

                if rel_obj_tol is not None and rel_improve < float(rel_obj_tol):
                    termination_reason = "relative_objective_tolerance"
                    break
                if step_tol is not None and step_norm < float(step_tol):
                    termination_reason = "step_tolerance"
                    break
                break

            damping = float(damping) * nu
            nu *= 2.0

        if accepted_row is None:
            termination_reason = "max_trials_after_failure" if trial_count >= int(max_trials_after_failure) else "rho_zero"
            history.append(
                {
                    "outer": int(outer),
                    "nonlinear_objective": obj_prev,
                    "accepted": False,
                    "lambda": float(damping),
                    "attempts": int(trial_count),
                    "step_norm": 0.0,
                    "linear_residual_norm": float("nan"),
                    "scale": 0.0,
                    "actual_reduction": 0.0,
                    "rho": rho,
                    "relative_objective_improvement": 0.0,
                }
            )
            break

        if termination_reason != "max_outer":
            break

    return {
        "config": {
            "num_outer": int(num_outer),
            "rel_obj_tol": None if rel_obj_tol is None else float(rel_obj_tol),
            "step_tol": None if step_tol is None else float(step_tol),
            "initial_lambda": float(initial_lambda),
            "tau": float(tau),
            "good_step_lower_scale": float(good_step_lower_scale),
            "good_step_upper_scale": float(good_step_upper_scale),
            "max_trials_after_failure": int(max_trials_after_failure),
            "ridge": float(ridge),
        },
        "history": history,
        "attempt_rows": attempt_rows,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
        "termination_reason": termination_reason,
    }


def run_gtsam_levenberg_marquardt_g2o(
    problem: G2OSE2Problem,
    num_outer: int = 100,
    relative_error_tol: float = 1e-5,
    absolute_error_tol: float = 1e-5,
    error_tol: float = 0.0,
    lambda_initial: float = 1e-5,
    lambda_factor: float = 10.0,
    lambda_upper_bound: float = 1e5,
    lambda_lower_bound: float = 0.0,
    min_model_fidelity: float = 1e-3,
    diagonal_damping: bool = False,
    use_fixed_lambda_factor: bool = True,
    min_diagonal: float = 1e-6,
    max_diagonal: float = 1e32,
    ridge: float = 0.0,
) -> dict[str, object]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    history: list[dict[str, float | int | bool | str]] = []
    attempt_rows: list[dict[str, float | int | bool]] = []

    obj_prev = float(nonlinear_objective_g2o(problem, poses))
    history.append(
        {
            "outer": 0,
            "nonlinear_objective": obj_prev,
            "accepted": True,
            "lambda": float(lambda_initial),
            "attempts": 0,
        }
    )

    current_lambda = max(float(lambda_lower_bound), float(lambda_initial))
    current_factor = float(lambda_factor)
    total_inner_iterations = 0
    termination_reason = "max_iterations"

    for outer in range(1, int(num_outer) + 1):
        if error_tol > 0.0 and obj_prev <= float(error_tol):
            termination_reason = "error_tolerance"
            break

        A, b = linearize_g2o_problem(problem, poses)
        b = np.asarray(b, dtype=float).reshape(-1)
        damping_diag = None
        if diagonal_damping:
            diag = np.asarray(A.diagonal(), dtype=float).reshape(-1)
            damping_diag = np.clip(np.abs(diag), float(min_diagonal), float(max_diagonal))

        trial_lambda = max(float(lambda_lower_bound), float(current_lambda))
        trial_factor = float(current_factor)
        attempts_this_outer = 0
        accepted_row: dict[str, float | int | bool | str] | None = None

        while True:
            if trial_lambda >= float(lambda_upper_bound):
                termination_reason = "lambda_upper_bound"
                history.append(
                    {
                        "outer": int(outer),
                        "nonlinear_objective": obj_prev,
                        "accepted": False,
                        "lambda": float(trial_lambda),
                        "attempts": int(attempts_this_outer),
                        "step_norm": 0.0,
                        "linear_residual_norm": float("nan"),
                        "linearized_cost_change": 0.0,
                        "actual_reduction": 0.0,
                        "model_fidelity": float("nan"),
                        "relative_objective_improvement": 0.0,
                    }
                )
                break

            system = A + float(trial_lambda) * (
                sp.diags(damping_diag, format="csc")
                if damping_diag is not None
                else sp.eye(A.shape[0], format="csc")
            )
            if ridge > 0.0:
                system = system + float(ridge) * sp.eye(A.shape[0], format="csc")

            delta = spla.spsolve(system, b)
            delta = np.asarray(delta, dtype=float).reshape(-1)
            attempts_this_outer += 1
            total_inner_iterations += 1
            step_norm = float(np.linalg.norm(delta))
            lin_res = float(np.linalg.norm(system @ delta - b))
            a_delta = np.asarray(A @ delta, dtype=float).reshape(-1)
            linearized_cost_change = float(b @ delta - 0.5 * delta @ a_delta)
            old_linearized_error = obj_prev
            linearized_change_significant = linearized_cost_change > np.finfo(float).eps * max(old_linearized_error, 1.0)

            trial_poses = None
            trial_obj = float("inf")
            actual_reduction = float("nan")
            model_fidelity = float("nan")
            accepted = False
            stop_search = False

            if linearized_cost_change >= 0.0:
                trial_poses = apply_pose_deltas(poses, delta)
                trial_obj = float(nonlinear_objective_g2o(problem, trial_poses))
                actual_reduction = float(obj_prev - trial_obj)
                if linearized_change_significant:
                    model_fidelity = float(actual_reduction / linearized_cost_change)
                    accepted = bool(model_fidelity > float(min_model_fidelity))
                cost_change_mag = abs(actual_reduction)
                min_abs_tol = float(relative_error_tol) * obj_prev
                stop_search = bool(cost_change_mag < min_abs_tol or cost_change_mag < float(absolute_error_tol))

            attempt_rows.append(
                {
                    "outer": int(outer),
                    "trial": int(attempts_this_outer),
                    "accepted": accepted,
                    "lambda": float(trial_lambda),
                    "current_factor": float(trial_factor),
                    "nonlinear_objective_before": obj_prev,
                    "nonlinear_objective_after": trial_obj,
                    "linearized_cost_change": linearized_cost_change,
                    "actual_reduction": actual_reduction,
                    "model_fidelity": model_fidelity,
                    "step_norm": step_norm,
                    "linear_residual_norm": lin_res,
                    "stop_search": stop_search,
                }
            )

            if accepted and trial_poses is not None:
                if use_fixed_lambda_factor:
                    next_lambda = float(trial_lambda) / float(trial_factor)
                    next_factor = float(trial_factor)
                else:
                    next_lambda = float(trial_lambda) * max(1.0 / 3.0, 1.0 - (2.0 * model_fidelity - 1.0) ** 3)
                    next_factor = 2.0 * float(trial_factor)
                next_lambda = max(float(lambda_lower_bound), next_lambda)
                rel_improve = float(actual_reduction / max(abs(obj_prev), 1e-15))
                poses = trial_poses
                pose_history.append(poses.copy())
                obj_prev = trial_obj
                current_lambda = next_lambda
                current_factor = next_factor
                accepted_row = {
                    "outer": int(outer),
                    "nonlinear_objective": trial_obj,
                    "accepted": True,
                    "lambda": float(trial_lambda),
                    "lambda_next": float(next_lambda),
                    "current_factor": float(trial_factor),
                    "current_factor_next": float(next_factor),
                    "attempts": int(attempts_this_outer),
                    "step_norm": step_norm,
                    "linear_residual_norm": lin_res,
                    "linearized_cost_change": linearized_cost_change,
                    "actual_reduction": actual_reduction,
                    "model_fidelity": model_fidelity,
                    "relative_objective_improvement": rel_improve,
                    "total_inner_iterations": int(total_inner_iterations),
                }
                history.append(accepted_row)
                if obj_prev <= float(error_tol) and error_tol > 0.0:
                    termination_reason = "error_tolerance"
                elif abs(actual_reduction) < float(absolute_error_tol):
                    termination_reason = "absolute_error_tolerance"
                elif abs(actual_reduction) < float(relative_error_tol) * max(abs(history[-2]["nonlinear_objective"]), 1e-15):
                    termination_reason = "relative_error_tolerance"
                break

            trial_lambda *= trial_factor
            if not use_fixed_lambda_factor:
                trial_factor *= 2.0

            if stop_search:
                termination_reason = "small_cost_change"
                history.append(
                    {
                        "outer": int(outer),
                        "nonlinear_objective": obj_prev,
                        "accepted": False,
                        "lambda": float(trial_lambda),
                        "attempts": int(attempts_this_outer),
                        "step_norm": step_norm,
                        "linear_residual_norm": lin_res,
                        "linearized_cost_change": linearized_cost_change,
                        "actual_reduction": actual_reduction,
                        "model_fidelity": model_fidelity,
                        "relative_objective_improvement": 0.0,
                        "total_inner_iterations": int(total_inner_iterations),
                    }
                )
                break

        if accepted_row is None and termination_reason != "max_iterations":
            break
        if accepted_row is not None and termination_reason != "max_iterations":
            break

    return {
        "config": {
            "num_outer": int(num_outer),
            "relative_error_tol": float(relative_error_tol),
            "absolute_error_tol": float(absolute_error_tol),
            "error_tol": float(error_tol),
            "lambda_initial": float(lambda_initial),
            "lambda_factor": float(lambda_factor),
            "lambda_upper_bound": float(lambda_upper_bound),
            "lambda_lower_bound": float(lambda_lower_bound),
            "min_model_fidelity": float(min_model_fidelity),
            "diagonal_damping": bool(diagonal_damping),
            "use_fixed_lambda_factor": bool(use_fixed_lambda_factor),
            "min_diagonal": float(min_diagonal),
            "max_diagonal": float(max_diagonal),
            "ridge": float(ridge),
        },
        "history": history,
        "attempt_rows": attempt_rows,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
        "termination_reason": termination_reason,
        "total_inner_iterations": int(total_inner_iterations),
    }


def run_direct_newton_g2o(
    problem: G2OSE2Problem,
    num_outer: int = 20,
    rel_obj_tol: float = 1e-12,
    step_tol: float = 1e-10,
) -> dict[str, object]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    pose_history = [poses.copy()]
    history: list[dict[str, float]] = []
    obj_prev = nonlinear_objective_g2o(problem, poses)
    history.append(
        {
            "outer": 0,
            "nonlinear_objective": float(obj_prev),
        }
    )

    for outer in range(1, int(num_outer) + 1):
        step = direct_newton_step_g2o(problem, poses)
        poses = np.asarray(step["next_poses"], dtype=float)
        pose_history.append(poses.copy())
        obj = nonlinear_objective_g2o(problem, poses)
        rel_improve = abs(obj_prev - obj) / max(abs(obj_prev), 1e-15)
        history.append(
            {
                "outer": int(outer),
                "nonlinear_objective": float(obj),
                "linear_step_norm": float(step["linear_step_norm"]),
                "linear_residual_norm": float(step["linear_residual_norm"]),
                "relative_objective_improvement": float(rel_improve),
            }
        )
        if rel_improve < rel_obj_tol or float(step["linear_step_norm"]) < step_tol:
            break
        obj_prev = obj

    return {
        "config": {
            "num_outer": int(num_outer),
            "rel_obj_tol": float(rel_obj_tol),
            "step_tol": float(step_tol),
        },
        "history": history,
        "pose_history": [pose.tolist() for pose in pose_history],
        "final_poses": poses.tolist(),
    }
