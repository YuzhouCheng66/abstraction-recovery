from gbp.gbp import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def make_slam_like_graph(
    N=100, step_size=25, loop_prob=0.05, loop_radius=50, prior_prop=0.0, seed=None
):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    # --- helpers ---
    def wrap_angle(a):
        # (-pi, pi]
        return np.arctan2(np.sin(a), np.cos(a))

    def relpose_SE2(pose_i, pose_j):
        """ Return z_ij = [dx_local, dy_local, dtheta] where 'local' is frame of i """
        xi, yi, thi = pose_i
        xj, yj, thj = pose_j
        Ri = np.array([[np.cos(thi), -np.sin(thi)],
                       [np.sin(thi),  np.cos(thi)]])
        dp = np.array([xj - xi, yj - yi])
        trans_local = Ri.T @ dp
        dth = wrap_angle(thj - thi)
        return np.array([trans_local[0], trans_local[1], dth], dtype=float)

    # --- SE(2) trajectory (smooth heading) ---
    poses = []
    x, y, th = 0.0, 0.0, 0.0
    poses.append((x, y, th))

    TURN_STD = 1  # rad, per step (tune smaller/larger as needed)
    for _ in range(1, int(N)):
        dth = rng.normal(0.0, TURN_STD)
        th = wrap_angle(th + dth)
        x += float(step_size) * np.cos(th)
        y += float(step_size) * np.sin(th)
        poses.append((x, y, th))

    # --- nodes (dim:3); visualization uses x,y ---
    nodes = []
    for i, (px, py, pth) in enumerate(poses):
        nodes.append({
            "data": {"id": f"{i}", "layer": 0, "dim": 3, "theta": float(pth), "num_base": 1},
            "position": {"x": float(px), "y": float(py)}  # for plotting only
        })

    # --- sequential odometry edges; attach measurement z_ij (local frame) ---
    edges = []
    for i in range(int(N) - 1):
        z_ij = relpose_SE2(poses[i], poses[i+1])
        edges.append({
            "data": {"source": f"{i}", "target": f"{i+1}", "kind": "odom", "z": z_ij.tolist()}
        })

    # --- loop-closure edges (proximity-triggered); also attach SE(2) measurements ---
    for i in range(int(N)):
        xi, yi, _ = poses[i]
        for j in range(i + 5, int(N)):  # consider loop closures only when gap >= 5 steps
            if rng.random() < float(loop_prob):
                xj, yj, _ = poses[j]
                if np.hypot(xi - xj, yi - yj) < float(loop_radius):
                    z_ij = relpose_SE2(poses[i], poses[j])
                    edges.append({
                        "data": {"source": f"{i}", "target": f"{j}", "kind": "loop", "z": z_ij.tolist()}
                    })

    # --- strong priors (anchors, etc.); still connect to the virtual "prior" ---
    if prior_prop <= 0.0:
        strong_ids = {}
    elif prior_prop >= 1.0:
        strong_ids = set(range(1,N))
    else:
        k = int(np.floor(prior_prop * N))
        strong_ids = set(rng.choice(np.arange(1, N), size=k, replace=False).tolist())


    for i in strong_ids:
        edges.append({"data": {"source": f"{i}", "target": "prior", "kind": "prior"}})

    return nodes, edges




def build_noisy_pose_graph(
    nodes,
    edges,
    prior_sigma: float | tuple | np.ndarray = 1.0,   # A scalar expands to [s, s, s*THETA_RATIO]
    odom_sigma:  float | tuple | np.ndarray = 1.0,   # Same as above
    loop_sigma:  float | tuple | np.ndarray = 1.0,   # Same as above
    tiny_prior: float = 1e-10,
    seed=None,
):
    """
    Build an SE(2) 2D pose graph (x, y, theta). Binary edges use the SE(2) between nonlinear measurement model.
    Initialization policy: first propagate mu sequentially (reset when encountering a prior), then linearize all factors.
    """

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    THETA_RATIO = 0.01  # A scalar expands to [s, s, s*0.01]

    def _sigma_vec(s):
        v = np.array(s, dtype=float).ravel()
        if v.size == 1:
            s = float(v.item())
            return np.array([s, s, s * THETA_RATIO], dtype=float)
        assert v.size == 3, "If sigma is not scalar, it must be length-3 (x,y,theta)."
        return v

    def wrap_angle(a):
        return np.arctan2(np.sin(a), np.cos(a))


    # ---------- Graph & variables ----------
    fg = FactorGraph(nonlinear_factors=True, eta_damping=0)

    var_nodes = []
    for i, n in enumerate(nodes):
        v = VariableNode(i, dofs=3)
        th = float(n["data"].get("theta", 0.0))  # Written by make_slam_like_graph (radians)
        v.GT = m.SE2(n["position"]["x"], n["position"]["y"], th)
        # Initialize status to zero; will be set by sequential measurements later
        v.status = m.SE2(0, 0, 0)
        var_nodes.append(v)

    fg.var_nodes = var_nodes
    fg.n_var_nodes = len(var_nodes)


    # ---------- Information matrices ----------
    prior_sigma_vec = _sigma_vec(prior_sigma)
    odom_sigma_vec  = _sigma_vec(odom_sigma)
    loop_sigma_vec  = _sigma_vec(loop_sigma)

    Lambda_prior = np.diag(1.0 / (prior_sigma_vec**2))
    Lambda_odom  = np.diag(1.0 / (odom_sigma_vec**2))
    Lambda_loop  = np.diag(1.0 / (loop_sigma_vec**2))
    
    # ---------- Factors; first create noisy measurements (no linearization yet) ----------
    odom_meas = {}   # (i,j) -> z_noisy
    prior_meas = {}  # i -> z_noisy (unary)
    factors = []
    fid = 0

    # Strong anchor: fix global reference (optionally anchor only x,y by relaxing Lambda_anchor[2,2])
    v0 = var_nodes[0]
    z_anchor = v0.GT
    Lambda_anchor = np.diag(1.0 / (np.array([1e-3, 1e-3, 1e-5])**2))

    f0 = priorSE2(fid, [v0], z_anchor, Lambda_anchor, robustify=False)
    f0.linpoints = [v0.GT]
    f0.update_factor()
    factors.append(f0)
    v0.adj_factors.append(f0)
    fid += 1

    for e in edges:
        src = e["data"]["source"]
        dst = e["data"]["target"]

        if dst != "prior":
            i, j = int(src), int(dst)

            # Ground-truth relative pose
            z = np.array(e["data"]["z"], dtype=float).ravel()
            kind = e["data"].get("kind", "between")

            # >>> CHANGED: choose noise model per kind
            if kind == "loop":
                noise_vec = rng.normal(0.0, loop_sigma_vec, size=3)
                this_Lambda = Lambda_loop
            else:
                # default treat as odom
                noise_vec = rng.normal(0.0, odom_sigma_vec, size=3)
                this_Lambda = Lambda_odom

            z_noisy = z.copy()
            z_noisy[:2] += noise_vec[:2]
            z_noisy[2]  = wrap_angle(z_noisy[2] + noise_vec[2])
            z_noisy_SE2 = m.SE2(z_noisy[0], z_noisy[1], z_noisy[2])

            # only store sequential odom for init mu forward-prop
            if kind == "odom" and (j == i + 1):
                odom_meas[(i, j)] = z_noisy_SE2

            vi, vj = var_nodes[i], var_nodes[j]
            
            f = odometrySE2(fid, [vi, vj], z_noisy_SE2, this_Lambda, robustify=False)
            factors.append(f)
            vi.adj_factors.append(f)
            vj.adj_factors.append(f)
            fid += 1

        else:
            # Prior edge: create a noisy measurement for this variable
            i = int(src)
            z = [*var_nodes[i].GT.translation(), var_nodes[i].GT.angle()]
            noise = rng.normal(0.0, prior_sigma_vec, size=3)
            z[:2] += noise[:2]
            z[2]  = wrap_angle(z[2] + noise[2])
            z_SE2 = m.SE2(z[0], z[1], z[2])
            prior_meas[i] = z_SE2

            vi = var_nodes[i]
            f = priorSE2(fid, [vi], z_SE2, Lambda_prior, robustify=False)
            factors.append(f)
            vi.adj_factors.append(f)
            fid += 1


    # ---------- Sequentially initialize mu: forward propagation from node 0; reset when hitting a prior ----------
    N = len(var_nodes)
    # Start: use GT

    var_nodes[0].status = var_nodes[0].GT
    initialize_variable_lambda  = np.diag(_sigma_vec(1))*1e-4
    var_nodes[0].Lam = initialize_variable_lambda

    for i in range(N - 1):
        # First propagate via odometry
        var_nodes[i+1].status = var_nodes[i].status * odom_meas[(i, i+1)]
        # If i+1 has a prior, override with the prior (replace the propagated value)
        if (i + 1) in prior_meas:
            var_nodes[i+1].status = prior_meas[i + 1]
        var_nodes[i+1].Lam = initialize_variable_lambda

    # ---------- Linearize all factors (mu is now in place) ----------
    for f in factors:
        f.linpoints = [vn.status for vn in f.adj_var_nodes]
        f.update_factor()

    fg.factors = factors
    fg.n_factor_nodes = len(factors)


    # ----------- Initialize Messages -----------
    for v in var_nodes:
        v.to_factor_messages = []
        for f in v.adj_factors:
            v.to_factor_messages.append(Message(status=v.status))

    for f in factors:
        f.to_var_messages = []
        for v in f.adj_var_nodes:
            f.to_var_messages.append(Message(status=v.status))
            
    return fg


import matplotlib.pyplot as plt
import networkx as nx

import matplotlib.pyplot as plt
import networkx as nx

def plot_gbp_graph_xy(
    gbp_graph,
    edges,
    use="status",          # "status" | "GT"
    show_ids=False,
    id_font_size=1,
    lw=0.8,
    s=5,
    figsize=(10, 8),        # NEW: figure size in inches
    dpi=200,               # NEW: figure DPI
):
    """
    Plot pose graph using either:
      - var.status.translation()  (GBP estimate)
      - var.GT.translation()      (ground truth)

    Parameters
    ----------
    gbp_graph : FactorGraph
    edges     : list of edge dicts
    use       : "status" or "GT"
    figsize   : (width, height) in inches
    dpi       : dots per inch (resolution)
    """

    assert use in ("status", "GT")

    # ---------- build positions (two possible id systems) ----------
    pos_by_index = {}
    pos_by_vid = {}

    for i, v in enumerate(gbp_graph.var_nodes):
        X = v.status if use == "status" else v.GT
        t = X.translation()
        x, y = float(t[0]), float(t[1])

        pos_by_index[str(i)] = (x, y)
        pos_by_vid[str(v.variableID)] = (x, y)

    # ---------- detect which id system edges use ----------
    edge_ids = set()
    for e in edges:
        u = e["data"]["source"]
        v = e["data"]["target"]
        if u == "prior" or v == "prior":
            continue
        edge_ids.add(str(u))
        edge_ids.add(str(v))

    score_index = sum(1 for k in edge_ids if k in pos_by_index)
    score_vid   = sum(1 for k in edge_ids if k in pos_by_vid)

    pos = pos_by_vid if score_vid > score_index else pos_by_index

    # ---------- build graph ----------
    G = nx.Graph()
    G.add_nodes_from(pos.keys())

    for e in edges:
        u = e["data"]["source"]
        v = e["data"]["target"]
        if u == "prior" or v == "prior":
            continue
        u, v = str(u), str(v)
        if u in pos and v in pos:
            G.add_edge(u, v)

    # ---------- draw ----------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal", adjustable="box")

    nx.draw_networkx_edges(G, pos, ax=ax, width=lw, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=s)

    if show_ids:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=id_font_size)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"GBP pose graph ({use})")

    plt.tight_layout()
    plt.show()


from collections import deque, defaultdict

def graph_diameter(nodes, edges):
    """
    Graph diameter = max shortest-path distance between any two nodes.
    Only between-edges are considered (prior / anchor ignored).
    """

    # ---- node ids ----
    node_ids = [n["data"]["id"] for n in nodes]
    node_set = set(node_ids)

    # ---- adjacency list (undirected, between edges only) ----
    adj = defaultdict(list)
    for e in edges:
        u = e["data"]["source"]
        v = e["data"]["target"]

        # ignore prior / anchor edges
        if u not in node_set or v not in node_set:
            continue

        adj[u].append(v)
        adj[v].append(u)

    # ---- BFS: distances from one node ----
    def bfs_distances(start):
        dist = {start: 0}
        q = deque([start])

        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # ---- diameter = max over all pairs ----
    diameter = 0
    for u in node_ids:
        dists = bfs_distances(u)
        diameter = max(diameter, max(dists.values()))

    return diameter
