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
from svd_abstraction.se2_utils import rot2
from svd_abstraction.se2_utils import se2_between
from svd_abstraction.se2_utils import se2_compose
from svd_abstraction.se2_utils import se2_inverse
from svd_abstraction.se2_utils import se2_log
from svd_abstraction.se2_utils import se2_plus


@dataclass
class SE2XYObservation:
    pose_index: int
    landmark_index: int
    measurement: np.ndarray
    information: np.ndarray
    pose_id: int
    landmark_id: int


@dataclass
class VictoriaParkOdometry:
    src_id: int
    dst_id: int
    measurement: np.ndarray
    information: np.ndarray


@dataclass
class VictoriaParkObservation:
    pose_id: int
    landmark_id: int
    measurement: np.ndarray
    information: np.ndarray


@dataclass
class VictoriaParkRawProblem:
    source_path: str
    pose_ids: list[int]
    landmark_ids: list[int]
    init_pose_map: dict[int, np.ndarray]
    init_landmark_map: dict[int, np.ndarray]
    odometry: list[VictoriaParkOdometry]
    observations: list[VictoriaParkObservation]
    fixed_pose_id: int


@dataclass
class G2OSE2LandmarkProblem:
    source_path: str
    pose_ids: list[int]
    landmark_ids: list[int]
    init_poses: np.ndarray
    init_landmarks: np.ndarray
    odom_edges: list[SE2Edge]
    landmark_observations: list[SE2XYObservation]
    anchor_pose: np.ndarray
    anchor_information: np.ndarray
    fixed_pose_id: int


def _make_nodes_from_pose_landmark_state(
    pose_ids: list[int],
    landmark_ids: list[int],
    poses: np.ndarray,
    landmarks: np.ndarray,
) -> list[dict]:
    nodes = []
    for idx, pose in enumerate(np.asarray(poses, dtype=float)):
        nodes.append(
            {
                "data": {
                    "id": f"{idx}",
                    "layer": 0,
                    "dim": 3,
                    "num_base": 1,
                    "kind": "pose",
                    "original_id": int(pose_ids[idx]),
                },
                "position": {"x": float(pose[0]), "y": float(pose[1])},
            }
        )
    pose_offset = len(pose_ids)
    for idx, landmark in enumerate(np.asarray(landmarks, dtype=float)):
        nodes.append(
            {
                "data": {
                    "id": f"{pose_offset + idx}",
                    "layer": 0,
                    "dim": 2,
                    "num_base": 1,
                    "kind": "landmark",
                    "original_id": int(landmark_ids[idx]),
                },
                "position": {"x": float(landmark[0]), "y": float(landmark[1])},
            }
        )
    return nodes


def _info6_to_matrix(vals: list[float]) -> np.ndarray:
    if len(vals) != 6:
        raise ValueError(f"Expected 6 information entries, got {len(vals)}")
    i11, i12, i13, i22, i23, i33 = [float(v) for v in vals]
    return np.array(
        [
            [i11, i12, i13],
            [i12, i22, i23],
            [i13, i23, i33],
        ],
        dtype=float,
    )


def _info3_to_matrix(vals: list[float]) -> np.ndarray:
    if len(vals) != 3:
        raise ValueError(f"Expected 3 information entries, got {len(vals)}")
    i11, i12, i22 = [float(v) for v in vals]
    return np.array(
        [
            [i11, i12],
            [i12, i22],
        ],
        dtype=float,
    )


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


def parse_victoria_park_slampp(path: str | Path) -> VictoriaParkRawProblem:
    path = Path(path)
    odometry: list[VictoriaParkOdometry] = []
    observations: list[VictoriaParkObservation] = []
    pose_ids: set[int] = set()
    landmark_ids: set[int] = set()

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "ODOMETRY":
                if len(parts) != 12:
                    raise ValueError(f"{path}:{lineno}: malformed ODOMETRY line")
                src_id = int(parts[1])
                dst_id = int(parts[2])
                meas = np.array([float(parts[3]), float(parts[4]), float(parts[5])], dtype=float)
                info = _info6_to_matrix([float(x) for x in parts[6:12]])
                odometry.append(
                    VictoriaParkOdometry(
                        src_id=src_id,
                        dst_id=dst_id,
                        measurement=meas,
                        information=info,
                    )
                )
                pose_ids.add(src_id)
                pose_ids.add(dst_id)
            elif tag == "LANDMARK":
                if len(parts) != 8:
                    raise ValueError(f"{path}:{lineno}: malformed LANDMARK line")
                pose_id = int(parts[1])
                landmark_id = int(parts[2])
                meas = np.array([float(parts[3]), float(parts[4])], dtype=float)
                info = _info3_to_matrix([float(x) for x in parts[5:8]])
                observations.append(
                    VictoriaParkObservation(
                        pose_id=pose_id,
                        landmark_id=landmark_id,
                        measurement=meas,
                        information=info,
                    )
                )
                pose_ids.add(pose_id)
                landmark_ids.add(landmark_id)
            else:
                raise ValueError(f"{path}:{lineno}: unsupported tag {tag!r}")

    if not odometry:
        raise ValueError(f"{path}: no ODOMETRY entries found")
    if not observations:
        raise ValueError(f"{path}: no LANDMARK entries found")

    next_by_src: dict[int, tuple[int, np.ndarray]] = {}
    incoming_count: dict[int, int] = {}
    for edge in odometry:
        if edge.src_id in next_by_src:
            raise ValueError(f"{path}: pose {edge.src_id} has more than one outgoing odometry edge")
        next_by_src[edge.src_id] = (edge.dst_id, edge.measurement.copy())
        incoming_count[edge.dst_id] = incoming_count.get(edge.dst_id, 0) + 1

    starts = sorted(pid for pid in pose_ids if incoming_count.get(pid, 0) == 0)
    if len(starts) != 1:
        raise ValueError(f"{path}: expected a single odometry-chain start, got {starts}")
    fixed_pose_id = starts[0]

    init_pose_map: dict[int, np.ndarray] = {fixed_pose_id: np.zeros(3, dtype=float)}
    order = [fixed_pose_id]
    visited = {fixed_pose_id}
    cur = fixed_pose_id
    while cur in next_by_src:
        dst_id, meas = next_by_src[cur]
        if dst_id in visited:
            raise ValueError(f"{path}: odometry chain contains a loop at pose {dst_id}")
        init_pose_map[dst_id] = se2_compose(init_pose_map[cur], meas)
        order.append(dst_id)
        visited.add(dst_id)
        cur = dst_id

    if visited != pose_ids:
        missing = sorted(pose_ids - visited)
        raise ValueError(f"{path}: odometry chain did not cover all poses, missing {missing[:10]}")

    obs_by_landmark: dict[int, list[VictoriaParkObservation]] = {}
    for obs in observations:
        obs_by_landmark.setdefault(obs.landmark_id, []).append(obs)

    init_landmark_map: dict[int, np.ndarray] = {}
    for landmark_id in sorted(landmark_ids):
        samples = []
        for obs in obs_by_landmark[landmark_id]:
            pose = init_pose_map[obs.pose_id]
            point = pose[:2] + rot2(float(pose[2])) @ np.asarray(obs.measurement, dtype=float)
            samples.append(point)
        init_landmark_map[landmark_id] = np.mean(np.asarray(samples, dtype=float), axis=0)

    return VictoriaParkRawProblem(
        source_path=str(path),
        pose_ids=order,
        landmark_ids=sorted(landmark_ids),
        init_pose_map=init_pose_map,
        init_landmark_map=init_landmark_map,
        odometry=odometry,
        observations=observations,
        fixed_pose_id=fixed_pose_id,
    )


def parse_victoria_dataset_xylmk(path: str | Path) -> VictoriaParkRawProblem:
    path = Path(path)
    pose_ids = [0]
    current_pose_id = 0
    init_pose_map: dict[int, np.ndarray] = {0: np.zeros(3, dtype=float)}
    odometry: list[VictoriaParkOdometry] = []
    temp_observations: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    original_landmark_ids: set[int] = set()

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = [token.strip() for token in line.strip().split(",")]
            if not parts or len(parts) < 2 or not parts[0]:
                continue
            tag = parts[1].lower()
            if tag == "odometry":
                if len(parts) != 8:
                    raise ValueError(f"{path}:{lineno}: malformed odometry line")
                meas = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
                info = np.diag([float(parts[5]), float(parts[6]), float(parts[7])]).astype(float)
                next_pose_id = current_pose_id + 1
                odometry.append(
                    VictoriaParkOdometry(
                        src_id=current_pose_id,
                        dst_id=next_pose_id,
                        measurement=meas,
                        information=info,
                    )
                )
                init_pose_map[next_pose_id] = se2_compose(init_pose_map[current_pose_id], meas)
                current_pose_id = next_pose_id
                pose_ids.append(current_pose_id)
            elif tag == "landmark":
                if len(parts) != 8:
                    raise ValueError(f"{path}:{lineno}: malformed landmark line")
                original_landmark_id = int(parts[2])
                meas = np.array([float(parts[3]), float(parts[4])], dtype=float)
                info = _info3_to_matrix([float(parts[5]), float(parts[6]), float(parts[7])])
                temp_observations.append((current_pose_id, original_landmark_id, meas, info))
                original_landmark_ids.add(original_landmark_id)
            else:
                raise ValueError(f"{path}:{lineno}: unsupported entry type {parts[1]!r}")

    if not odometry:
        raise ValueError(f"{path}: no odometry entries found")
    if not temp_observations:
        raise ValueError(f"{path}: no landmark entries found")

    landmark_offset = max(pose_ids) + 1
    original_to_g2o = {
        orig_id: landmark_offset + idx
        for idx, orig_id in enumerate(sorted(original_landmark_ids))
    }

    observations: list[VictoriaParkObservation] = []
    obs_by_landmark: dict[int, list[VictoriaParkObservation]] = {}
    for pose_id, original_landmark_id, meas, info in temp_observations:
        landmark_id = original_to_g2o[original_landmark_id]
        obs = VictoriaParkObservation(
            pose_id=pose_id,
            landmark_id=landmark_id,
            measurement=meas,
            information=info,
        )
        observations.append(obs)
        obs_by_landmark.setdefault(landmark_id, []).append(obs)

    init_landmark_map: dict[int, np.ndarray] = {}
    for landmark_id, obs_list in obs_by_landmark.items():
        samples = []
        for obs in obs_list:
            pose = init_pose_map[obs.pose_id]
            point = pose[:2] + rot2(float(pose[2])) @ np.asarray(obs.measurement, dtype=float)
            samples.append(point)
        init_landmark_map[landmark_id] = np.mean(np.asarray(samples, dtype=float), axis=0)

    return VictoriaParkRawProblem(
        source_path=str(path),
        pose_ids=pose_ids,
        landmark_ids=sorted(init_landmark_map),
        init_pose_map=init_pose_map,
        init_landmark_map=init_landmark_map,
        odometry=odometry,
        observations=observations,
        fixed_pose_id=0,
    )


def write_victoria_park_g2o(raw_problem: VictoriaParkRawProblem, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# converted from {raw_problem.source_path}\n")
        fh.write("# source format: SLAM++ processed Victoria Park graph derived from RADISH\n")
        for pose_id in raw_problem.pose_ids:
            pose = np.asarray(raw_problem.init_pose_map[pose_id], dtype=float)
            fh.write(
                f"VERTEX_SE2 {pose_id} {pose[0]:.9f} {pose[1]:.9f} {pose[2]:.9f}\n"
            )
        for landmark_id in raw_problem.landmark_ids:
            xy = np.asarray(raw_problem.init_landmark_map[landmark_id], dtype=float)
            fh.write(f"VERTEX_XY {landmark_id} {xy[0]:.9f} {xy[1]:.9f}\n")
        fh.write(f"FIX {raw_problem.fixed_pose_id}\n")
        for edge in raw_problem.odometry:
            info = np.asarray(edge.information, dtype=float)
            fh.write(
                "EDGE_SE2 "
                f"{edge.src_id} {edge.dst_id} "
                f"{edge.measurement[0]:.9f} {edge.measurement[1]:.9f} {edge.measurement[2]:.9f} "
                f"{info[0,0]:.9f} {info[0,1]:.9f} {info[0,2]:.9f} "
                f"{info[1,1]:.9f} {info[1,2]:.9f} {info[2,2]:.9f}\n"
            )
        for obs in raw_problem.observations:
            info = np.asarray(obs.information, dtype=float)
            fh.write(
                "EDGE_SE2_XY "
                f"{obs.pose_id} {obs.landmark_id} "
                f"{obs.measurement[0]:.9f} {obs.measurement[1]:.9f} "
                f"{info[0,0]:.9f} {info[0,1]:.9f} {info[1,1]:.9f}\n"
            )
    return out_path


def parse_g2o_se2_landmark(path: str | Path, anchor_precision: float = 1e8) -> G2OSE2LandmarkProblem:
    path = Path(path)
    pose_vertices: dict[int, np.ndarray] = {}
    landmark_vertices: dict[int, np.ndarray] = {}
    raw_odom_edges: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    raw_landmark_obs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    fixed_ids: list[int] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            tag = parts[0]
            if tag == "VERTEX_SE2":
                if len(parts) != 5:
                    raise ValueError(f"{path}:{lineno}: malformed VERTEX_SE2 line")
                pose_vertices[int(parts[1])] = np.array(
                    [float(parts[2]), float(parts[3]), float(parts[4])],
                    dtype=float,
                )
            elif tag == "VERTEX_XY":
                if len(parts) != 4:
                    raise ValueError(f"{path}:{lineno}: malformed VERTEX_XY line")
                landmark_vertices[int(parts[1])] = np.array(
                    [float(parts[2]), float(parts[3])],
                    dtype=float,
                )
            elif tag == "EDGE_SE2":
                if len(parts) != 12:
                    raise ValueError(f"{path}:{lineno}: malformed EDGE_SE2 line")
                raw_odom_edges.append(
                    (
                        int(parts[1]),
                        int(parts[2]),
                        np.array([float(parts[3]), float(parts[4]), float(parts[5])], dtype=float),
                        _info6_to_matrix([float(x) for x in parts[6:12]]),
                    )
                )
            elif tag == "EDGE_SE2_XY":
                if len(parts) != 8:
                    raise ValueError(f"{path}:{lineno}: malformed EDGE_SE2_XY line")
                raw_landmark_obs.append(
                    (
                        int(parts[1]),
                        int(parts[2]),
                        np.array([float(parts[3]), float(parts[4])], dtype=float),
                        _info3_to_matrix([float(x) for x in parts[5:8]]),
                    )
                )
            elif tag == "FIX":
                if len(parts) != 2:
                    raise ValueError(f"{path}:{lineno}: malformed FIX line")
                fixed_ids.append(int(parts[1]))
            else:
                raise ValueError(f"{path}:{lineno}: unsupported tag {tag!r}")

    if not pose_vertices:
        raise ValueError(f"{path}: no VERTEX_SE2 entries found")
    if not landmark_vertices:
        raise ValueError(f"{path}: no VERTEX_XY entries found")

    pose_ids = sorted(pose_vertices)
    landmark_ids = sorted(landmark_vertices)
    pose_id_to_index = {pid: idx for idx, pid in enumerate(pose_ids)}
    landmark_id_to_index = {lid: idx for idx, lid in enumerate(landmark_ids)}

    odom_edges: list[SE2Edge] = []
    for src_id, dst_id, meas, info in raw_odom_edges:
        odom_edges.append(
            SE2Edge(
                i=pose_id_to_index[src_id],
                j=pose_id_to_index[dst_id],
                measurement=meas.copy(),
                information=info.copy(),
                kind="odometry",
            )
        )

    landmark_observations: list[SE2XYObservation] = []
    for pose_id, landmark_id, meas, info in raw_landmark_obs:
        landmark_observations.append(
            SE2XYObservation(
                pose_index=pose_id_to_index[pose_id],
                landmark_index=landmark_id_to_index[landmark_id],
                measurement=meas.copy(),
                information=info.copy(),
                pose_id=pose_id,
                landmark_id=landmark_id,
            )
        )

    fixed_pose_id = min(fixed_ids) if fixed_ids else pose_ids[0]
    anchor_pose = np.asarray(pose_vertices[fixed_pose_id], dtype=float).copy()
    anchor_information = float(anchor_precision) * np.eye(3, dtype=float)

    return G2OSE2LandmarkProblem(
        source_path=str(path),
        pose_ids=pose_ids,
        landmark_ids=landmark_ids,
        init_poses=np.vstack([pose_vertices[pid] for pid in pose_ids]).astype(float),
        init_landmarks=np.vstack([landmark_vertices[lid] for lid in landmark_ids]).astype(float),
        odom_edges=odom_edges,
        landmark_observations=landmark_observations,
        anchor_pose=anchor_pose,
        anchor_information=anchor_information,
        fixed_pose_id=fixed_pose_id,
    )


def summarize_g2o_se2_landmark(problem: G2OSE2LandmarkProblem) -> dict[str, int | float]:
    return {
        "num_poses": int(problem.init_poses.shape[0]),
        "num_landmarks": int(problem.init_landmarks.shape[0]),
        "num_odometry_edges": int(len(problem.odom_edges)),
        "num_landmark_observations": int(len(problem.landmark_observations)),
        "fixed_pose_id": int(problem.fixed_pose_id),
        "total_state_dim": int(3 * problem.init_poses.shape[0] + 2 * problem.init_landmarks.shape[0]),
    }


def states_to_nodes_g2o_se2_landmark(
    problem: G2OSE2LandmarkProblem,
    poses: np.ndarray,
    landmarks: np.ndarray,
) -> list[dict]:
    return _make_nodes_from_pose_landmark_state(
        pose_ids=problem.pose_ids,
        landmark_ids=problem.landmark_ids,
        poses=poses,
        landmarks=landmarks,
    )


def nonlinear_objective_g2o_se2_landmark(
    problem: G2OSE2LandmarkProblem,
    poses: np.ndarray,
    landmarks: np.ndarray,
) -> float:
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)

    total = 0.0
    for edge in problem.odom_edges:
        pred = se2_between(poses[edge.i], poses[edge.j])
        err = se2_log(se2_compose(se2_inverse(edge.measurement), pred))
        total += 0.5 * float(err.T @ edge.information @ err)

    for obs in problem.landmark_observations:
        pose = poses[obs.pose_index]
        landmark = landmarks[obs.landmark_index]
        pred = rot2(float(pose[2])).T @ (landmark - pose[:2])
        err = pred - obs.measurement
        total += 0.5 * float(err.T @ obs.information @ err)

    anchor_idx = problem.pose_ids.index(problem.fixed_pose_id)
    anchor_err = se2_log(se2_compose(se2_inverse(problem.anchor_pose), poses[anchor_idx]))
    total += 0.5 * float(anchor_err.T @ problem.anchor_information @ anchor_err)
    return total


def apply_pose_landmark_deltas(
    problem: G2OSE2LandmarkProblem,
    base_poses: np.ndarray,
    base_landmarks: np.ndarray,
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(delta, dtype=float).reshape(-1)
    n_pose = int(base_poses.shape[0])
    pose_delta = delta[: 3 * n_pose]
    landmark_delta = delta[3 * n_pose :].reshape(-1, 2)
    next_poses = apply_pose_deltas(base_poses, pose_delta)
    next_landmarks = np.asarray(base_landmarks, dtype=float) + landmark_delta
    return next_poses, next_landmarks


def linearize_g2o_se2_landmark_problem(
    problem: G2OSE2LandmarkProblem,
    base_poses: np.ndarray,
    base_landmarks: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    base_poses = np.asarray(base_poses, dtype=float).reshape(-1, 3)
    base_landmarks = np.asarray(base_landmarks, dtype=float).reshape(-1, 2)
    n_pose = int(base_poses.shape[0])
    n_landmark = int(base_landmarks.shape[0])
    dim = 3 * n_pose + 2 * n_landmark

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

    for edge in problem.odom_edges:
        Xi = base_poses[edge.i].copy()
        Xj = base_poses[edge.j].copy()
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

        si = slice(3 * edge.i, 3 * edge.i + 3)
        sj = slice(3 * edge.j, 3 * edge.j + 3)
        add_block(si.start, si.start, H[:3, :3])
        add_block(si.start, sj.start, H[:3, 3:])
        add_block(sj.start, si.start, H[3:, :3])
        add_block(sj.start, sj.start, H[3:, 3:])
        b[si] += -g[:3]
        b[sj] += -g[3:]

    landmark_offset = 3 * n_pose
    for obs in problem.landmark_observations:
        Pi = base_poses[obs.pose_index].copy()
        Lj = base_landmarks[obs.landmark_index].copy()
        z = obs.measurement.copy()
        omega = obs.information.copy()

        def residual_fn_local(x: np.ndarray) -> np.ndarray:
            e_pose = np.asarray(x[:3], dtype=float)
            e_landmark = np.asarray(x[3:5], dtype=float)
            pose = se2_plus(Pi, e_pose)
            landmark = Lj + e_landmark
            pred = rot2(float(pose[2])).T @ (landmark - pose[:2])
            return pred - z

        x0 = np.zeros(5, dtype=float)
        r0 = residual_fn_local(x0)
        J = _numeric_jacobian(residual_fn_local, x0)
        H = J.T @ omega @ J
        g = J.T @ omega @ r0

        spose = slice(3 * obs.pose_index, 3 * obs.pose_index + 3)
        sland = slice(landmark_offset + 2 * obs.landmark_index, landmark_offset + 2 * obs.landmark_index + 2)
        add_block(spose.start, spose.start, H[:3, :3])
        add_block(spose.start, sland.start, H[:3, 3:])
        add_block(sland.start, spose.start, H[3:, :3])
        add_block(sland.start, sland.start, H[3:, 3:])
        b[spose] += -g[:3]
        b[sland] += -g[3:]

    anchor_idx = problem.pose_ids.index(problem.fixed_pose_id)
    X0 = base_poses[anchor_idx].copy()
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
    s0 = slice(3 * anchor_idx, 3 * anchor_idx + 3)
    add_block(s0.start, s0.start, H0)
    b[s0] += -g0

    A = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsc()
    A = 0.5 * (A + A.T)
    return A, b


def build_linearized_local_graph_g2o_se2_landmark(
    problem: G2OSE2LandmarkProblem,
    base_poses: np.ndarray,
    base_landmarks: np.ndarray,
    tiny_prior: float = 1e-12,
) -> FactorGraph:
    base_poses = np.asarray(base_poses, dtype=float).reshape(-1, 3)
    base_landmarks = np.asarray(base_landmarks, dtype=float).reshape(-1, 2)
    n_pose = int(base_poses.shape[0])
    graph = FactorGraph(nonlinear_factors=False, eta_damping=0.0)

    vars_: list[VariableNode] = []
    for idx, pose in enumerate(base_poses):
        var = VariableNode(idx, dofs=3)
        var.type = "pose"
        var.GT = np.asarray(pose, dtype=float).copy()
        var.prior.lam = tiny_prior * np.eye(3, dtype=float)
        var.prior.eta = np.zeros(3, dtype=float)
        vars_.append(var)

    for idx, landmark in enumerate(base_landmarks):
        var = VariableNode(n_pose + idx, dofs=2)
        var.type = "landmark"
        var.GT = np.asarray(landmark, dtype=float).copy()
        var.prior.lam = tiny_prior * np.eye(2, dtype=float)
        var.prior.eta = np.zeros(2, dtype=float)
        vars_.append(var)

    graph.var_nodes = vars_
    graph.n_var_nodes = len(vars_)

    factors: list[Factor] = []
    factor_id = 0

    for edge in problem.odom_edges:
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

        def jac_fn_local(x, _meas_fn=meas_fn_local):
            fn = lambda xx: _meas_fn(xx)[0]
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

    for obs in problem.landmark_observations:
        vi = vars_[obs.pose_index]
        vj = vars_[n_pose + obs.landmark_index]
        base_pose = base_poses[obs.pose_index].copy()
        base_landmark = base_landmarks[obs.landmark_index].copy()
        z = obs.measurement.copy()
        omega = obs.information.copy()

        def meas_fn_local(x, base_pose=base_pose, base_landmark=base_landmark, z=z):
            e_pose = np.asarray(x[:3], dtype=float)
            e_landmark = np.asarray(x[3:5], dtype=float)
            pose = se2_plus(base_pose, e_pose)
            landmark = base_landmark + e_landmark
            pred = rot2(float(pose[2])).T @ (landmark - pose[:2])
            return [pred - z]

        def jac_fn_local(x, _meas_fn=meas_fn_local):
            fn = lambda xx: _meas_fn(xx)[0]
            return [_numeric_jacobian(fn, np.asarray(x, dtype=float).reshape(-1))]

        factor = Factor(
            factor_id,
            [vi, vj],
            [np.zeros(2, dtype=float)],
            [omega],
            meas_fn_local,
            jac_fn_local,
        )
        factor.type = "landmark_xy"
        factor.compute_factor(linpoint=np.zeros(5, dtype=float), update_self=True)
        factors.append(factor)
        vi.adj_factors.append(factor)
        vj.adj_factors.append(factor)
        factor_id += 1

    anchor_idx = problem.pose_ids.index(problem.fixed_pose_id)
    base_anchor = base_poses[anchor_idx].copy()
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
        [vars_[anchor_idx]],
        [np.zeros(3, dtype=float)],
        [anchor_info],
        meas_fn_anchor,
        jac_fn_anchor,
    )
    anchor.type = "anchor"
    anchor.compute_factor(linpoint=np.zeros(3, dtype=float), update_self=True)
    factors.append(anchor)
    vars_[anchor_idx].adj_factors.append(anchor)

    graph.factors = factors
    graph.n_factor_nodes = len(factors)
    for var in graph.var_nodes[: graph.n_var_nodes]:
        var.update_belief()
    return graph


def direct_newton_step_g2o_se2_landmark(
    problem: G2OSE2LandmarkProblem,
    base_poses: np.ndarray,
    base_landmarks: np.ndarray,
    ridge: float = 1e-10,
) -> dict[str, object]:
    A, b = linearize_g2o_se2_landmark_problem(problem, base_poses, base_landmarks)
    A = A + float(ridge) * sp.eye(A.shape[0], format="csc")
    delta = spla.spsolve(A, b)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    lin_res = float(np.linalg.norm(A @ delta - b))
    next_poses, next_landmarks = apply_pose_landmark_deltas(problem, base_poses, base_landmarks, delta)
    return {
        "A": A,
        "b": b,
        "delta": delta,
        "linear_step_norm": float(np.linalg.norm(delta)),
        "linear_residual_norm": lin_res,
        "next_poses": next_poses,
        "next_landmarks": next_landmarks,
    }


def run_direct_newton_g2o_se2_landmark(
    problem: G2OSE2LandmarkProblem,
    num_outer: int = 20,
    rel_obj_tol: float | None = None,
    step_tol: float | None = None,
    ridge: float = 1e-10,
) -> dict[str, object]:
    poses = np.asarray(problem.init_poses, dtype=float).copy()
    landmarks = np.asarray(problem.init_landmarks, dtype=float).copy()
    pose_history = [poses.copy()]
    landmark_history = [landmarks.copy()]
    history: list[dict[str, float | int]] = []

    obj_prev = nonlinear_objective_g2o_se2_landmark(problem, poses, landmarks)
    history.append({"outer": 0, "nonlinear_objective": float(obj_prev)})

    for outer in range(1, int(num_outer) + 1):
        step = direct_newton_step_g2o_se2_landmark(
            problem,
            poses,
            landmarks,
            ridge=ridge,
        )
        poses = np.asarray(step["next_poses"], dtype=float)
        landmarks = np.asarray(step["next_landmarks"], dtype=float)
        pose_history.append(poses.copy())
        landmark_history.append(landmarks.copy())

        obj = float(nonlinear_objective_g2o_se2_landmark(problem, poses, landmarks))
        rel_improve = float(abs(obj_prev - obj) / max(abs(obj_prev), 1e-15))
        row = {
            "outer": int(outer),
            "nonlinear_objective": obj,
            "linear_step_norm": float(step["linear_step_norm"]),
            "linear_residual_norm": float(step["linear_residual_norm"]),
            "relative_objective_improvement": rel_improve,
        }
        history.append(row)
        obj_prev = obj

        if rel_obj_tol is not None and rel_improve < float(rel_obj_tol):
            break
        if step_tol is not None and float(step["linear_step_norm"]) < float(step_tol):
            break

    return {
        "config": {
            "num_outer": int(num_outer),
            "rel_obj_tol": None if rel_obj_tol is None else float(rel_obj_tol),
            "step_tol": None if step_tol is None else float(step_tol),
            "ridge": float(ridge),
        },
        "history": history,
        "pose_history": [poses_i.tolist() for poses_i in pose_history],
        "landmark_history": [landmarks_i.tolist() for landmarks_i in landmark_history],
        "final_poses": poses.tolist(),
        "final_landmarks": landmarks.tolist(),
    }


def plot_initial_vs_direct_pose_landmark(
    problem: G2OSE2LandmarkProblem,
    direct_poses: np.ndarray,
    direct_landmarks: np.ndarray,
    out_path: str | Path,
    title: str,
) -> Path:
    init_poses = np.asarray(problem.init_poses, dtype=float)
    init_landmarks = np.asarray(problem.init_landmarks, dtype=float)
    direct_poses = np.asarray(direct_poses, dtype=float)
    direct_landmarks = np.asarray(direct_landmarks, dtype=float)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.0, 8.5), dpi=180)
    ax.plot(
        init_poses[:, 0],
        init_poses[:, 1],
        color="#8f8f8f",
        linewidth=1.0,
        linestyle=(0, (10, 3, 2, 3)),
        label="Initial poses",
        zorder=2,
    )
    ax.plot(
        direct_poses[:, 0],
        direct_poses[:, 1],
        color="#1f77b4",
        linewidth=1.15,
        label="Direct Newton poses",
        zorder=3,
    )
    ax.scatter(
        init_landmarks[:, 0],
        init_landmarks[:, 1],
        color="#b0b0b0",
        s=10,
        alpha=0.55,
        label="Initial landmarks",
        zorder=1,
    )
    ax.scatter(
        direct_landmarks[:, 0],
        direct_landmarks[:, 1],
        color="#d95f02",
        s=12,
        alpha=0.85,
        label="Direct Newton landmarks",
        zorder=4,
    )
    ax.scatter(
        [init_poses[0, 0]],
        [init_poses[0, 1]],
        color="black",
        s=20,
        label="Anchor/start",
        zorder=5,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
