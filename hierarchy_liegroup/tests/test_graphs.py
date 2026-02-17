import sys
import types
import numpy as np
import pytest


# -----------------------------
# Test scaffolding: lightweight stubs for gbp.gbp and manifpy
# These stubs avoid heavy third-party deps while preserving the API used by utils/graphs.py
# -----------------------------

def _wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


class _FakeSE2:
    def __init__(self, x=0.0, y=0.0, th=0.0):
        self.x = float(x)
        self.y = float(y)
        self.th = float(th)

    def translation(self):
        return np.array([self.x, self.y], dtype=float)

    def angle(self):
        return float(self.th)

    # Group composition: self * other
    def __mul__(self, other):
        c, s = np.cos(self.th), np.sin(self.th)
        tx = self.x + c * other.x - s * other.y
        ty = self.y + s * other.x + c * other.y
        tth = _wrap_angle(self.th + other.th)
        return _FakeSE2(tx, ty, tth)

    # For readability in debug
    def __repr__(self):
        return f"SE2(x={self.x:.3f}, y={self.y:.3f}, th={self.th:.3f})"


class _FakeMessage:
    def __init__(self, status=None, Lam=None):
        self.status = status if status is not None else _FakeSE2(0, 0, 0)
        self.Lam = Lam if Lam is not None else np.eye(3) * 1e-10


class _FakeVariableNode:
    def __init__(self, variable_id, dofs=3):
        self.dofs = dofs
        self.variableID = variable_id
        self.adj_factors = []
        self.to_factor_messages = []
        self.status = _FakeSE2(0, 0, 0)
        self.GT = None
        self.Lam = np.eye(dofs) * 1e-10


class _FakePriorSE2:
    def __init__(self, factor_id, adj_var_nodes, measurement, measurement_lambda, robustify):
        self.factorID = factor_id
        self.adj_var_nodes = adj_var_nodes
        self.measurement = measurement
        self.measurement_lambda = measurement_lambda
        self.robustify = robustify
        self.linpoints = []
        # created later in graphs.build_noisy_pose_graph
        self.to_var_messages = []

    def update_factor(self):
        # No-op for tests
        return True


class _FakeOdometrySE2(_FakePriorSE2):
    pass


class _FakeFactorGraph:
    def __init__(self, nonlinear_factors=True, eta_damping=0, **kwargs):
        self.var_nodes = []
        self.factors = []
        self.n_var_nodes = 0
        self.n_factor_nodes = 0
        self.nonlinear_factors = nonlinear_factors
        self.eta_damping = eta_damping


def _install_fake_gbp_module(monkeypatch=None):
    # Create a parent package 'gbp' and a submodule 'gbp.gbp'
    gbp_pkg = types.ModuleType('gbp')
    gbp_mod = types.ModuleType('gbp.gbp')

    # minimal 'manifpy as m' surrogate exposed via star-import as symbol 'm'
    manif_mod = types.ModuleType('manifpy')
    setattr(manif_mod, 'SE2', _FakeSE2)
    # Minimal tangent placeholder to satisfy potential accesses (not used by our tests)
    class _FakeSE2Tangent:
        def __init__(self, vec):
            self._vec = np.array(vec, dtype=float).ravel()
        def rjac(self):
            return np.eye(3)
        def coeffs(self):
            return self._vec
    setattr(manif_mod, 'SE2Tangent', _FakeSE2Tangent)

    # Expose in the gbp.gbp namespace the same symbols utils.graphs expects from 'from gbp.gbp import *'
    gbp_mod.m = manif_mod
    gbp_mod.Message = _FakeMessage
    gbp_mod.VariableNode = _FakeVariableNode
    gbp_mod.priorSE2 = _FakePriorSE2
    gbp_mod.odometrySE2 = _FakeOdometrySE2
    gbp_mod.FactorGraph = _FakeFactorGraph

    # Wire up package/module relationship
    gbp_pkg.gbp = gbp_mod

    # Inject into sys.modules before importing utils.graphs
    sys.modules['gbp'] = gbp_pkg
    sys.modules['gbp.gbp'] = gbp_mod
    sys.modules['manifpy'] = manif_mod


@pytest.fixture(autouse=True)
def fake_gbp_module():
    # Autouse across tests to ensure import succeeds deterministically
    _install_fake_gbp_module()
    yield
    # Cleanup is not strictly necessary in pytest process


@pytest.fixture(autouse=True)
def _non_gui_matplotlib(monkeypatch):
    # Prevent actual GUI popups during plotting tests
    try:
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None, raising=False)
    except Exception:
        # If matplotlib isn't available, it's okay—the import in graphs.py may be stubbed externally
        pass


# -----------------------------
# Import target module under test
# -----------------------------
import utils.graphs as graphs


# -----------------------------
# Behaviors
# -----------------------------
# 1) Should create N nodes and N-1 sequential odometry edges with correct connectivity
# 2) Should generate loop edges when probability is 1.0 and radius large
# 3) Should create N-1 prior edges when prior_prop=1.0 (excluding node 0)
# 4) Should build FactorGraph with expected var/factor counts on simple odom-only input
# 5) Should initialize covariance (Lam) of first variable to diag([1e-6, 1e-6, 1e-8])
# 6) Should plot without raising using status positions
# 7) Should compute correct graph diameter for a simple chain


def test_make_slam_like_graph_nodes_and_sequential_edges():
    N = 10
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=1, loop_prob=0.0, prior_prop=0.0, seed=123)

    assert len(nodes) == N

    odom_edges = [e for e in edges if e['data'].get('kind') == 'odom']
    assert len(odom_edges) == N - 1

    # Connectivity i -> i+1
    expected = {(str(i), str(i + 1)) for i in range(N - 1)}
    found = {(e['data']['source'], e['data']['target']) for e in odom_edges}
    assert expected.issubset(found)

    # Measurements have 3 components
    for e in odom_edges:
        z = e['data'].get('z')
        assert isinstance(z, list) and len(z) == 3
        assert all(np.isfinite(z))


def test_make_slam_like_graph_loop_edges_with_prob_one():
    N = 20
    # Force high chance and large radius to create many loops
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=1, loop_prob=1.0, loop_radius=1e9, prior_prop=0.0, seed=7)

    loop_edges = [e for e in edges if e['data'].get('kind') == 'loop']
    assert len(loop_edges) > 0
    # Ensure all flagged as loop
    assert all(e['data'].get('kind') == 'loop' for e in loop_edges)

    # No prior edges expected here
    assert all(e['data'].get('target') != 'prior' for e in loop_edges)


def test_make_slam_like_graph_prior_edges_count():
    N = 10
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=1, loop_prob=0.0, loop_radius=0.0, prior_prop=1.0, seed=0)

    prior_edges = [e for e in edges if e['data'].get('target') == 'prior']
    assert len(prior_edges) == N - 1  # nodes 1..N-1 have priors

    sources = {e['data']['source'] for e in prior_edges}
    assert sources == {str(i) for i in range(1, N)}


def test_build_noisy_pose_graph_counts_and_structure():
    N = 5
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=1, loop_prob=0.0, prior_prop=0.0, seed=11)

    fg = graphs.build_noisy_pose_graph(nodes, edges, seed=0)

    # Var count
    assert fg.n_var_nodes == N
    assert len(fg.var_nodes) == N

    # Factor count: 1 anchor + (N-1) odom (no loops, no priors)
    expected_factors = 1 + (N - 1)
    assert fg.n_factor_nodes == expected_factors
    assert len(fg.factors) == expected_factors


def test_build_noisy_pose_graph_initial_covariance_first_node():
    N = 6
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=1, loop_prob=0.0, prior_prop=0.0, seed=21)
    fg = graphs.build_noisy_pose_graph(nodes, edges, seed=0)

    # The function sets initialize_variable_lambda = diag([1,1,0.01]) * 1e-6
    expected = np.diag([1e-6, 1e-6, 1e-8])
    np.testing.assert_allclose(fg.var_nodes[0].Lam, expected, rtol=0, atol=1e-12)


def test_plot_gbp_graph_xy_runs_without_error(monkeypatch):
    N = 8
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=2, loop_prob=0.0, prior_prop=0.0, seed=5)
    fg = graphs.build_noisy_pose_graph(nodes, edges, seed=0)

    # Ensure no exception during plotting; backend show is no-op from fixture
    graphs.plot_gbp_graph_xy(fg, edges, use='status', show_ids=False, figsize=(4, 3), dpi=100)


def test_graph_diameter_simple_chain():
    # Build a chain 0-1-2-3 => diameter 3
    nodes = [
        {"data": {"id": str(i)}, "position": {"x": i * 1.0, "y": 0.0}}
        for i in range(4)
    ]
    edges = [
        {"data": {"source": str(i), "target": str(i + 1), "kind": "odom", "z": [1.0, 0.0, 0.0]}}
        for i in range(3)
    ]

    d = graphs.graph_diameter(nodes, edges)
    assert d == 3


# -----------------------------
# Additional behaviors and tests
# -----------------------------
# 8) build_noisy_pose_graph uses distinct noise models for odom vs. loop edges
# 9) _sigma_vec: scalar expands to [s, s, s*0.01] and influences prior Lambda
# 10) Sequential init: a prior at node k overrides propagated status
# 11) make_slam_like_graph enforces loop gap >= 5
# 12) graph_diameter ignores prior edges


def test_build_noisy_pose_graph_distinct_noise_models_for_loop_and_odom():
    N = 15
    nodes, edges = graphs.make_slam_like_graph(
        N=N, step_size=1, loop_prob=1.0, loop_radius=1e9, prior_prop=0.0, seed=42
    )
    # Choose clearly different sigmas so Lambdas are distinguishable
    odom_sigma = 0.5
    loop_sigma = 3.0
    fg = graphs.build_noisy_pose_graph(nodes, edges, odom_sigma=odom_sigma, loop_sigma=loop_sigma, seed=0)

    # Expected information diagonals
    theta_ratio = 0.01
    lam_odom = np.diag(1.0 / np.array([odom_sigma, odom_sigma, odom_sigma * theta_ratio]) ** 2)
    lam_loop = np.diag(1.0 / np.array([loop_sigma, loop_sigma, loop_sigma * theta_ratio]) ** 2)

    # Build sets of loop vs odom pairs from edges
    loop_pairs = set()
    odom_pairs = set()
    for e in edges:
        if e["data"]["target"] == "prior":
            continue
        i, j = int(e["data"]["source"]), int(e["data"]["target"])
        if e["data"].get("kind") == "loop":
            loop_pairs.add((i, j))
        elif e["data"].get("kind") == "odom":
            odom_pairs.add((i, j))

    # Check factor measurement lambdas according to their (i,j) membership
    seen_loop = seen_odom = 0
    for f in fg.factors:
        if len(getattr(f, "adj_var_nodes", [])) == 2:
            i = f.adj_var_nodes[0].variableID
            j = f.adj_var_nodes[1].variableID
            pair = (i, j)
            if pair in loop_pairs:
                np.testing.assert_allclose(f.measurement_lambda, lam_loop, rtol=0, atol=1e-12)
                seen_loop += 1
            elif pair in odom_pairs:
                np.testing.assert_allclose(f.measurement_lambda, lam_odom, rtol=0, atol=1e-12)
                seen_odom += 1
    assert seen_loop > 0 and seen_odom > 0


def test_prior_sigma_scalar_expansion_affects_prior_lambda():
    # Force priors on all nodes except 0, then inspect one prior factor
    N = 6
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=1, loop_prob=0.0, prior_prop=1.0, seed=7)
    prior_sigma = 2.0
    fg = graphs.build_noisy_pose_graph(nodes, edges, prior_sigma=prior_sigma, seed=0)

    theta_ratio = 0.01
    expected = np.diag(1.0 / np.array([prior_sigma, prior_sigma, prior_sigma * theta_ratio]) ** 2)

    found = 0
    for f in fg.factors:
        # unary prior factor has one adjacent variable node
        if len(getattr(f, "adj_var_nodes", [])) == 1 and isinstance(f, type(f)):
            # Exclude the anchor by checking its Lambda against expected
            if np.allclose(f.measurement_lambda, expected):
                found += 1
    assert found >= N - 1  # we expect at least one matching prior (excluding anchor)


def test_sequential_initialization_prior_overrides_propagation():
    N = 8
    # Put a prior on node 3 specifically
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=1, loop_prob=0.0, prior_prop=0.0, seed=10)
    edges.append({"data": {"source": "3", "target": "prior", "kind": "prior"}})

    fg = graphs.build_noisy_pose_graph(nodes, edges, seed=123)

    # Find the prior factor attached to node 3 and compare measurement to status
    z_prior = None
    for f in fg.factors:
        if len(getattr(f, "adj_var_nodes", [])) == 1 and f.adj_var_nodes[0].variableID == 3:
            z_prior = f.measurement
            break
    assert z_prior is not None

    v3 = fg.var_nodes[3]
    np.testing.assert_allclose(v3.status.translation(), z_prior.translation(), rtol=0, atol=1e-12)
    assert abs(v3.status.angle() - z_prior.angle()) < 1e-12


def test_make_slam_like_graph_enforces_min_loop_gap():
    N = 25
    nodes, edges = graphs.make_slam_like_graph(N=N, step_size=1, loop_prob=1.0, loop_radius=1e9, prior_prop=0.0, seed=3)
    # All loops must satisfy j - i >= 5
    for e in edges:
        if e["data"].get("kind") == "loop":
            i, j = int(e["data"]["source"]), int(e["data"]["target"])
            assert j - i >= 5


def test_graph_diameter_ignores_prior_edges():
    # Chain of 5 nodes has diameter 4 regardless of number of priors added
    nodes = [
        {"data": {"id": str(i)}, "position": {"x": float(i), "y": 0.0}}
        for i in range(5)
    ]
    edges = [
        {"data": {"source": str(i), "target": str(i + 1), "kind": "odom", "z": [1.0, 0.0, 0.0]}}
        for i in range(4)
    ]
    # Add many priors that should be ignored by graph_diameter
    edges.extend({"data": {"source": str(i), "target": "prior", "kind": "prior"}} for i in range(5))

    d = graphs.graph_diameter(nodes, edges)
    assert d == 4
