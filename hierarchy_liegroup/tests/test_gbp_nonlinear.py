import sys
import types

import numpy as np


def _install_fake_manifpy():
    manif_mod = types.ModuleType("manifpy")

    class _FakeSE2Tangent:
        def __init__(self, *args):
            if len(args) == 1:
                vec = np.array(args[0], dtype=float).ravel()
            else:
                vec = np.array(args, dtype=float).ravel()
            if vec.size != 3:
                raise ValueError("Fake SE2 tangent expects 3 entries")
            self._vec = vec

        def coeffs(self):
            return self._vec

        def rjac(self):
            return np.eye(3, dtype=float)

    class _FakeSE2:
        def __init__(self, x=0.0, y=0.0, th=0.0):
            self._vec = np.array([x, y, th], dtype=float)

        def translation(self):
            return self._vec[:2].copy()

        def angle(self):
            return float(self._vec[2])

        def __mul__(self, other):
            return _FakeSE2(*(self._vec + other._vec))

        def __add__(self, tangent):
            return _FakeSE2(*(self._vec + tangent.coeffs()))

        def __sub__(self, other):
            return _FakeSE2Tangent(self._vec - other._vec)

        def minus(self, other, _unused=None, jac=None):
            if jac is not None:
                jac[:, :] = -np.eye(3, dtype=float)
            return _FakeSE2Tangent(self._vec - other._vec)

        def between(self, other, jac1=None, jac2=None):
            if jac1 is not None:
                jac1[:, :] = -np.eye(3, dtype=float)
            if jac2 is not None:
                jac2[:, :] = np.eye(3, dtype=float)
            return _FakeSE2(*(other._vec - self._vec))

        def rminus(self, other, _unused=None, jac=None):
            if jac is not None:
                jac[:, :] = -np.eye(3, dtype=float)
            return _FakeSE2Tangent(self._vec - other._vec)

        def squaredWeightedNorm(self):
            return float(self._vec @ self._vec)

    manif_mod.SE2 = _FakeSE2
    manif_mod.SE2Tangent = _FakeSE2Tangent
    sys.modules["manifpy"] = manif_mod


_install_fake_manifpy()

from gbp.gbp_nonlinear import build_nonlinear_se2_pose_graph


def _simple_chain_graph():
    nodes = []
    for i in range(3):
        nodes.append(
            {
                "data": {"id": str(i), "theta": 0.0},
                "position": {"x": float(i), "y": 0.0},
            }
        )

    edges = [
        {"data": {"source": "0", "target": "1", "kind": "odom", "z": [1.0, 0.0, 0.0]}},
        {"data": {"source": "1", "target": "2", "kind": "odom", "z": [1.0, 0.0, 0.0]}},
    ]
    return nodes, edges


def test_inner_gbp_matches_direct_linear_solve_on_fixed_linearisation():
    nodes, edges = _simple_chain_graph()
    graph = build_nonlinear_se2_pose_graph(
        nodes,
        edges,
        prior_sigma=1.0,
        odom_sigma=1.0,
        loop_sigma=1.0,
        seed=0,
        init_mode="chain",
    )

    graph.relinearise_factors(reset_messages="full")
    direct = graph.direct_solve_delta()

    for _ in range(10):
        graph.inner_iteration()

    np.testing.assert_allclose(graph.current_delta_vector(), direct, atol=1e-8, rtol=0.0)
    assert graph.delta_error_to_direct() < 1e-8
    assert graph.linear_residual_norm() < 1e-8
