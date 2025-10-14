import re
import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_cytoscape as cyto
import numpy as np
from scipy.linalg import block_diag

# ==== GBP import ====
from gbp.gbp import *

app = dash.Dash(__name__)
app.title = "Factor Graph SVD Abs&Recovery"

# -----------------------
# SLAM-like base graph
# -----------------------
def make_slam_like_graph(
    N=100, step_size=25, loop_prob=0.05, loop_radius=50, prior_prop=0.0, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

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

    TURN_STD = 1  # rad, per step（可根据需要调小/调大）
    for _ in range(1, int(N)):
        dth = rng.normal(0.0, TURN_STD)
        th = wrap_angle(th + dth)
        x += float(step_size) * np.cos(th)
        y += float(step_size) * np.sin(th)
        poses.append((x, y, th))

    # --- nodes（dim:3），可视化仍用 x,y ---
    nodes = []
    for i, (px, py, pth) in enumerate(poses):
        nodes.append({
            "data": {"id": f"{i}", "layer": 0, "dim": 3, "theta": float(pth), "num_base": 1},
            "position": {"x": float(px), "y": float(py)}  # 仅用于画图
        })

    # --- sequential odometry edges，附上测量 z_ij（局部系） ---
    edges = []
    for i in range(int(N) - 1):
        z_ij = relpose_SE2(poses[i], poses[i+1])
        edges.append({
            "data": {"source": f"{i}", "target": f"{i+1}", "kind": "odom", "z": z_ij.tolist()}
        })

    # --- loop-closure edges（近邻触发），同样附上 SE(2) 测量 ---
    for i in range(int(N)):
        xi, yi, _ = poses[i]
        for j in range(i + 5, int(N)):  # 间隔>=5步再考虑回环
            if rng.random() < float(loop_prob):
                xj, yj, _ = poses[j]
                if np.hypot(xi - xj, yi - yj) < float(loop_radius):
                    z_ij = relpose_SE2(poses[i], poses[j])
                    edges.append({
                        "data": {"source": f"{i}", "target": f"{j}", "kind": "loop", "z": z_ij.tolist()}
                    })

    # --- strong priors（锚点等）；仍然连接到虚拟 "prior" ---
    if prior_prop <= 0.0:
        strong_ids = {0}
    elif prior_prop >= 1.0:
        strong_ids = set(range(N))
    else:
        k = max(1, int(np.floor(prior_prop * N)))
        strong_ids = set(rng.choice(N, size=k, replace=False).tolist())

    for i in strong_ids:
        edges.append({"data": {"source": f"{i}", "target": "prior", "kind": "prior"}})

    return nodes, edges



# -----------------------
# Grid aggregation
# -----------------------
def fuse_to_super_grid(prev_nodes, prev_edges, gx, gy, layer_idx):
    positions = np.array([[n["position"]["x"], n["position"]["y"]] for n in prev_nodes], dtype=float)
    xmin, ymin = positions.min(axis=0); xmax, ymax = positions.max(axis=0)
    cell_w = (xmax - xmin) / gx if gx > 0 else 1.0
    cell_h = (ymax - ymin) / gy if gy > 0 else 1.0
    if cell_w == 0: cell_w = 1.0
    if cell_h == 0: cell_h = 1.0
    cell_map = {}
    for idx, n in enumerate(prev_nodes):
        x, y = n["position"]["x"], n["position"]["y"]
        cx = min(int((x - xmin) / cell_w), gx - 1)
        cy = min(int((y - ymin) / cell_h), gy - 1)
        cid = cx + cy * gx
        cell_map.setdefault(cid, []).append(idx)
    super_nodes, node_map = [], {}
    for cid, indices in cell_map.items():
        pts = positions[indices]
        mean_x, mean_y = pts.mean(axis=0)
        child_dims = [prev_nodes[i]["data"]["dim"] for i in indices]
        child_nums = [prev_nodes[i]["data"].get("num_base", 1) for i in indices]
        dim_val = int(max(1, sum(child_dims)))
        num_val = int(sum(child_nums))
        nid = str(len(super_nodes))
        super_nodes.append({
            "data": {
                "id": nid,
                "layer": layer_idx,
                "dim": dim_val,
                "num_base": num_val   # Inherit the sum
            },
            "position": {"x": float(mean_x), "y": float(mean_y)}
        })
        for i in indices:
            node_map[prev_nodes[i]["data"]["id"]] = nid
    super_edges, seen = [], set()
    for e in prev_edges:
        u, v = e["data"]["source"], e["data"]["target"]

        if v != "prior":
            su, sv = node_map[u], node_map[v]
            if su != sv:
                eid = tuple(sorted((su, sv)))
                if eid not in seen:
                    super_edges.append({"data": {"source": su, "target": sv}})
                    seen.add(eid)
            elif su == sv:
                eid = tuple(sorted((su, "prior")))
                if eid not in seen:
                    super_edges.append({"data": {"source": su, "target": "prior"}})
                    seen.add(eid)

        elif v == "prior":
            su = node_map[u]
            eid = tuple(sorted((su, v)))
            if eid not in seen:
                super_edges.append({"data": {"source": su, "target": "prior"}})
                seen.add(eid)

    return super_nodes, super_edges, node_map

# -----------------------
# K-Means aggregation
# -----------------------
def fuse_to_super_kmeans(prev_nodes, prev_edges, k, layer_idx, max_iters=20, tol=1e-6, seed=0):
    positions = np.array([[n["position"]["x"], n["position"]["y"]] for n in prev_nodes], dtype=float)
    n = positions.shape[0]
    if k <= 0: 
        k = 1
    k = min(k, n)
    rng = np.random.default_rng(seed)

    # -------- Improved initialization --------
    # Randomly sample k points without replacement to ensure each cluster starts with a distinct point
    init_idx = rng.choice(n, size=k, replace=False)
    centers = positions[init_idx]

    # Lloyd iterations
    for _ in range(max_iters):
        d2 = ((positions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)

        # -------- Empty-cluster fix --------
        counts = np.bincount(assign, minlength=k)
        empty_clusters = np.where(counts == 0)[0]
        for ci in empty_clusters:
            # Find the largest cluster
            big_cluster = np.argmax(counts)
            big_idxs = np.where(assign == big_cluster)[0]
            # Steal one point over
            steal_idx = big_idxs[0]
            assign[steal_idx] = ci
            counts[big_cluster] -= 1
            counts[ci] += 1

        moved = 0.0
        for ci in range(k):
            idxs = np.where(assign == ci)[0]
            new_c = positions[idxs].mean(axis=0)
            moved = max(moved, float(np.linalg.norm(new_c - centers[ci])))
            centers[ci] = new_c
        if moved < tol:
            break

    # Final assign (redo once to be safe)
    d2 = ((positions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = np.argmin(d2, axis=1)

    counts = np.bincount(assign, minlength=k)
    empty_clusters = np.where(counts == 0)[0]
    for ci in empty_clusters:
        big_cluster = np.argmax(counts)
        big_idxs = np.where(assign == big_cluster)[0]
        steal_idx = big_idxs[0]
        assign[steal_idx] = ci
        counts[big_cluster] -= 1
        counts[ci] += 1

    # ---------- Build the super graph ----------
    super_nodes, node_map = [], {}
    for ci in range(k):
        idxs = np.where(assign == ci)[0]
        pts = positions[idxs]
        mean_x, mean_y = pts.mean(axis=0)
        child_dims = [prev_nodes[i]["data"]["dim"] for i in idxs]
        child_nums = [prev_nodes[i]["data"].get("num_base", 1) for i in idxs]
        dim_val = int(max(1, sum(child_dims)))
        num_val = int(sum(child_nums)) 
        nid = f"{ci}"
        super_nodes.append({
            "data": {
                "id": nid,
                "layer": layer_idx,
                "dim": dim_val,
                "num_base": num_val   # Inherit the sum
            },
            "position": {"x": float(mean_x), "y": float(mean_y)}
        })
        for i in idxs:
            node_map[prev_nodes[i]["data"]["id"]] = nid

    super_edges, seen = [], set()
    for e in prev_edges:
        u, v = e["data"]["source"], e["data"]["target"]
        if v != "prior":
            su, sv = node_map[u], node_map[v]
            if su != sv:
                eid = tuple(sorted((su, sv)))
                if eid not in seen:
                    super_edges.append({"data": {"source": su, "target": sv}})
                    seen.add(eid)
            else:
                eid = (su, "prior")
                if eid not in seen:
                    super_edges.append({"data": {"source": su, "target": "prior"}})
                    seen.add(eid)
        else:
            su = node_map[u]
            eid = (su, "prior")
            if eid not in seen:
                super_edges.append({"data": {"source": su, "target": "prior"}})
                seen.add(eid)

    return super_nodes, super_edges, node_map


def copy_to_abs(super_nodes, super_edges, layer_idx):
    abs_nodes = []
    for n in super_nodes:
        nid = n["data"]["id"].replace("s", "a", 1)
        abs_nodes.append({
            "data": {
                "id": nid,
                "layer": layer_idx,
                "dim": n["data"]["dim"],
                "num_base": n["data"].get("num_base", 1)  # Inherit
            },
            "position": {"x": n["position"]["x"], "y": n["position"]["y"]}
        })
    abs_edges = []
    for e in super_edges:
        abs_edges.append({"data": {
            "source": e["data"]["source"].replace("s", "a", 1),
            "target": e["data"]["target"].replace("s", "a", 1)
        }})
    return abs_nodes, abs_edges

# -----------------------
# Sequential merge (tail group absorbs remainder)
# -----------------------
def fuse_to_super_order(prev_nodes, prev_edges, k, layer_idx, tail_heavy=True):
    """
    Sequentially split prev_nodes in current order into k groups; the last group absorbs the remainder (tail_heavy=True).
    Reuse existing rules for aggregating dim/num_base, deduplicating edges, and propagating prior.
    """
    n = len(prev_nodes)
    if k <= 0: k = 1
    k = min(k, n)

    # Group sizes
    base = n // k
    rem  = n %  k
    if tail_heavy:
        sizes = [base]*(k-1) + [base+rem]     # Tail absorbs remainder: ..., last += rem
    else:
        sizes = [base+1]*rem + [base]*(k-rem) # Evenly distribute remainder (optional)

    # Build groups: record indices per group
    groups = []
    start = 0
    for s in sizes:
        groups.append(list(range(start, start+s)))
        start += s

    # ---- Build super_nodes & node_map ----
    positions = np.array([[n["position"]["x"], n["position"]["y"]] for n in prev_nodes], dtype=float)

    super_nodes, node_map = [], {}
    for gi, idxs in enumerate(groups):
        pts = positions[idxs]
        mean_x, mean_y = pts.mean(axis=0)

        child_dims = [prev_nodes[i]["data"]["dim"] for i in idxs]
        child_nums = [prev_nodes[i]["data"].get("num_base", 1) for i in idxs]
        dim_val = int(max(1, sum(child_dims)))
        num_val = int(sum(child_nums))

        nid = f"{gi}"  # Same as kmeans: use group index as id (string)
        super_nodes.append({
            "data": {
                "id": nid,
                "layer": layer_idx,
                "dim": dim_val,
                "num_base": num_val
            },
            "position": {"x": float(mean_x), "y": float(mean_y)}
        })
        # Build base-id -> super-id mapping (note: ids are strings throughout)
        for i in idxs:
            node_map[prev_nodes[i]["data"]["id"]] = nid

    # ---- Super edges: keep and deduplicate inter-group edges; intra-group edges collapse to prior; prior edges roll up to their owning super ----
    super_edges, seen = [], set()
    for e in prev_edges:
        u, v = e["data"]["source"], e["data"]["target"]

        if v != "prior":
            su, sv = node_map[u], node_map[v]
            if su != sv:
                eid = tuple(sorted((su, sv)))
                if eid not in seen:
                    super_edges.append({"data": {"source": su, "target": sv}})
                    seen.add(eid)
            else:
                # Intra-group pairwise edge → group prior (consistent with grid/kmeans handling)
                eid = tuple(sorted((su, "prior")))
                if eid not in seen:
                    super_edges.append({"data": {"source": su, "target": "prior"}})
                    seen.add(eid)
        else:
            su = node_map[u]
            eid = tuple(sorted((su, "prior")))
            if eid not in seen:
                super_edges.append({"data": {"source": su, "target": "prior"}})
                seen.add(eid)

    return super_nodes, super_edges, node_map


# -----------------------
# Tools
# -----------------------
def parse_layer_name(name):
    if name == "base": return ("base", 0)
    m = re.match(r"(super|abs)(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else ("base", 0)

def highest_pair_idx(names):
    hi = 0
    for nm in names:
        kind, k = parse_layer_name(nm)
        if kind in ("super","abs"): hi = max(hi, k)
    return hi


# -----------------------
# Initialization & Boundary
# -----------------------
def init_layers(N=100, step_size=25, loop_prob=0.05, loop_radius=50, prior_prop=0.0, rng=None):
    base_nodes, base_edges = make_slam_like_graph(N, step_size, loop_prob, loop_radius, prior_prop, rng)
    return [{"name": "base", "nodes": base_nodes, "edges": base_edges}]

VIEW_W, VIEW_H = 960, 600
ASPECT = VIEW_W / VIEW_H
AXIS_PAD=20.0

def adjust_bounds_to_aspect(xmin, xmax, ymin, ymax, aspect):
    cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
    dx=xmax-xmin; dy=ymax-ymin
    if dx<=0: dx=1
    if dy<=0: dy=1
    if dx/dy > aspect:
        dy_new=dx/aspect
        return xmin,xmax,cy-dy_new/2,cy+dy_new/2
    else:
        dx_new=dy*aspect
        return cx-dx_new/2,cx+dx_new/2,ymin,ymax

def reset_global_bounds(base_nodes):
    global GLOBAL_XMIN, GLOBAL_XMAX, GLOBAL_YMIN, GLOBAL_YMAX
    global GLOBAL_XMIN_ADJ, GLOBAL_XMAX_ADJ, GLOBAL_YMIN_ADJ, GLOBAL_YMAX_ADJ
    xs=[n["position"]["x"] for n in base_nodes] or [0.0]
    ys=[n["position"]["y"] for n in base_nodes] or [0.0]
    GLOBAL_XMIN,GLOBAL_XMAX=min(xs),max(xs)
    GLOBAL_YMIN,GLOBAL_YMAX=min(ys),max(ys)
    GLOBAL_XMIN_ADJ,GLOBAL_XMAX_ADJ,GLOBAL_YMIN_ADJ,GLOBAL_YMAX_ADJ=adjust_bounds_to_aspect(
        GLOBAL_XMIN,GLOBAL_XMAX,GLOBAL_YMIN,GLOBAL_YMAX,ASPECT)

# ==== Blobal Status ====
layers = init_layers()
pair_idx = 0
reset_global_bounds(layers[0]["nodes"])
gbp_graph = None

# -----------------------
# GBP Graph Construction
# -----------------------
def build_noisy_pose_graph(
    nodes,
    edges,
    prior_sigma: float | tuple | np.ndarray = 1.0,   # 标量更方便；内部会变成 [s, s, s*0.05]
    odom_sigma:  float | tuple | np.ndarray = 1.0,   # 同上
    tiny_prior: float = 1e-10,
    rng=None,
):
    """
    构建 SE(2) 2D pose graph（x, y, theta），二元边使用 SE(2) between 非线性测量模型。
    - 节点 dofs = 3
    - 支持 edges 中自带的 "z"（局部系测量）和 "kind"（'odom'/'loop'/...）
    - prior_sigma / odom_sigma 可为标量或 3 维向量；标量时自动展开为 [s, s, s*THETA_RATIO]
    """

    if rng is None:
        rng = np.random.default_rng()

    THETA_RATIO = 0.01  # 标量展开为 [s, s, s*0.01]

    def _sigma_vec(s):
        v = np.array(s, dtype=float).ravel()
        if v.size == 1:
            s = float(v.item())
            return np.array([s, s, s * THETA_RATIO], dtype=float)
        assert v.size == 3, "If sigma is not scalar, it must be length-3 (x,y,theta)."
        return v

    def wrap_angle(a):
        return np.arctan2(np.sin(a), np.cos(a))

    def relpose_SE2(pose_i, pose_j):
        """z_ij = [dx_local, dy_local, dtheta]，在 i 的坐标系下定义。"""
        xi, yi, thi = pose_i
        xj, yj, thj = pose_j
        c, s = np.cos(thi), np.sin(thi)
        RT = np.array([[ c, s],
                       [-s, c]])    # R(thi)^T
        dp = np.array([xj - xi, yj - yi])
        trans_local = RT @ dp
        dth = wrap_angle(thj - thi)
        return np.array([trans_local[0], trans_local[1], dth], dtype=float)

    # ---------- 图 & 变量 ----------
    fg = FactorGraph(nonlinear_factors=True, eta_damping=0)

    var_nodes = []
    I3 = np.eye(3, dtype=float)
    for i, n in enumerate(nodes):
        v = VariableNode(i, dofs=3)
        th = float(n["data"].get("theta", 0.0))  # 由 make_slam_like_graph 写入（弧度）
        v.GT = np.array([n["position"]["x"], n["position"]["y"], th], dtype=float)

        # tiny prior 防止奇异
        v.prior.lam = tiny_prior * I3
        v.prior.eta = np.zeros(3, dtype=float)

        var_nodes.append(v)

    fg.var_nodes = var_nodes
    fg.n_var_nodes = len(var_nodes)

    # ---------- 测量模型（SE(2) between） ----------
    def meas_fn_se2_between(xij, *args):
        # xij = [xi, yi, thi, xj, yj, thj]
        xi, yi, thi, xj, yj, thj = xij
        c, s = np.cos(thi), np.sin(thi)
        RT = np.array([[ c, s],
                       [-s, c]])
        dp = np.array([xj - xi, yj - yi])
        r  = RT @ dp
        dth = wrap_angle(thj - thi)
        return np.array([r[0], r[1], dth], dtype=float)
    
    def jac_fn_se2_between(xij, *args):
        # J: 3x6 w.r.t [xi,yi,thi,xj,yj,thj]
        xi, yi, thi, xj, yj, thj = xij
        c, s = np.cos(thi), np.sin(thi)
        RT = np.array([[ c, s],
                    [-s, c]])
        dp = np.array([xj - xi, yj - yi])
        r  = RT @ dp                    # [rx, ry]
        dr_dthi = np.array([ r[1], -r[0] ])   # ← 修正后的 ∂r/∂θi

        J = np.zeros((3, 6), dtype=float)
        # wrt i
        J[0:2, 0:2] = -RT
        J[0:2, 2]   = dr_dthi
        # wrt j
        J[0:2, 3:5] = RT
        # angle row
        J[2, 2] = -1.0
        J[2, 5] =  1.0
        return J


    def meas_fn_unary(x, *args):
        return x

    def jac_fn_unary(x, *args):
        return np.eye(3, dtype=float)

    # ---------- 信息矩阵 ----------
    prior_sigma_vec = _sigma_vec(prior_sigma)
    odom_sigma_vec  = _sigma_vec(odom_sigma)

    Lambda_prior = np.diag(1.0 / (prior_sigma_vec**2))
    Lambda_odom  = np.diag(1.0 / (odom_sigma_vec**2))

    # ---------- 因子 ----------
    factors = []
    fid = 0

    # 强锚点：固定全局参考（可仅锚 x,y，把 Lambda_anchor[2,2] 放松）
    v0 = var_nodes[0]
    z_anchor = v0.GT.copy()
    Lambda_anchor = np.diag(1.0 / (np.array([1e-3, 1e-3, 1e-4])**2))
    f0 = Factor(fid, [v0], z_anchor, Lambda_anchor, meas_fn_unary, jac_fn_unary)
    f0.type = "prior"
    f0.compute_factor(linpoint=z_anchor, update_self=True)
    factors.append(f0)
    v0.adj_factors.append(f0)
    fid += 1

    # 其余边
    for e in edges:
        src = e["data"]["source"]
        dst = e["data"]["target"]

        # 二元边：SE(2) between
        if dst != "prior":
            i, j = int(src), int(dst)
            vi, vj = var_nodes[i], var_nodes[j]

            # 若边里已有 z（局部测量），直接用；否则用 GT 生成
            if "z" in e["data"]:
                z = np.array(e["data"]["z"], dtype=float).ravel()
            else:
                z = relpose_SE2(vi.GT, vj.GT)

            # 给测量加噪（这里统一用 odom_sigma；若想区分回环，可按 kind 分支）
            noise = rng.normal(0.0, odom_sigma_vec, size=3)
            z_noisy = z.copy()
            z_noisy[:2] += noise[:2]
            z_noisy[2]   = wrap_angle(z_noisy[2] + noise[2])

            f = Factor(fid, [vi, vj], z_noisy, Lambda_odom, meas_fn_se2_between, jac_fn_se2_between)
            f.type = e["data"].get("kind", "between")
            linpoint = np.r_[vi.mu, vj.mu]  # 线性化点
            f.compute_factor(linpoint=linpoint, update_self=True)

            factors.append(f)
            vi.adj_factors.append(f)
            vj.adj_factors.append(f)
            fid += 1

        # 一元边：额外强先验（来自图中 'prior' 边）
        else:
            i = int(src)
            vi = var_nodes[i]

            z = vi.GT.copy()
            noise = rng.normal(0.0, prior_sigma_vec, size=3)
            z[:2] += noise[:2]
            z[2]   = wrap_angle(z[2] + noise[2])

            f = Factor(fid, [vi], z, Lambda_prior, meas_fn_unary, jac_fn_unary)
            f.type = "prior"
            f.compute_factor(linpoint=vi.mu, update_self=True)

            factors.append(f)
            vi.adj_factors.append(f)
            fid += 1

    fg.factors = factors
    fg.n_factor_nodes = len(factors)
    return fg


def build_super_graph(layers):
    """
    Construct the super graph based on the base graph in layers[-2] and the super-grouping in layers[-1].
    Requirement: layers[-2]["graph"] is an already-built base graph (with unary/binary factors).
    layers[-1]["node_map"]: { base_node_id (str, e.g., 'b12') -> super_node_id (str) }
    """
    # ---------- Extract base & super ----------
    base_graph = layers[-2]["graph"]
    super_nodes = layers[-1]["nodes"]
    super_edges = layers[-1]["edges"]
    node_map    = layers[-1]["node_map"]   # 'bN' -> 'sX_...'

    # base: id(int) -> VariableNode, handy to query dofs and mu
    id2var = {vn.variableID: vn for vn in base_graph.var_nodes}

    # ---------- super_id -> [base_id(int)] ----------
    super_groups = {}
    for b_str, s_id in node_map.items():
        b_int = int(b_str)
        super_groups.setdefault(s_id, []).append(b_int)


    # ---------- For each super group, build a (start, dofs) table ----------
    # local_idx[sid][bid] = (start, dofs), total_dofs[sid] = sum(dofs)
    local_idx   = {}
    total_dofs  = {}
    for sid, group in super_groups.items():
        off = 0
        local_idx[sid] = {}
        for bid in group:
            d = id2var[bid].dofs
            local_idx[sid][bid] = (off, d)
            off += d
        total_dofs[sid] = off


    # ---------- Create super VariableNodes ----------
    fg = FactorGraph(nonlinear_factors=False, eta_damping=0)

    super_var_nodes = {}
    for i, sn in enumerate(super_nodes):
        sid = sn["data"]["id"]
        dofs = total_dofs.get(sid, 0)

        v = VariableNode(i, dofs=dofs)

        # === Stack base GT ===
        gt_vec = np.zeros(dofs)
        for bid, (st, d) in local_idx[sid].items():
            gt_base = getattr(id2var[bid], "GT", None)
            if gt_base is None or len(gt_base) != d:
                gt_base = np.zeros(d)
            gt_vec[st:st+d] = gt_base
        v.GT = gt_vec
        v.prior.lam = 1e-10 * np.eye(dofs, dtype=float)
        v.prior.eta = np.zeros(dofs, dtype=float)

        super_var_nodes[sid] = v
        fg.var_nodes.append(v)

        # === Stack base belief ===
        mu_blocks = []
        Sigma_blocks = []
        for bid, (st, d) in local_idx[sid].items():
            vb = id2var[bid]
            mu_blocks.append(vb.mu)
            Sigma_blocks.append(vb.Sigma)
        mu_super = np.concatenate(mu_blocks) if mu_blocks else np.zeros(dofs)
        Sigma_super = scipy.linalg.block_diag(*Sigma_blocks) if Sigma_blocks else np.eye(dofs)

        lam = np.linalg.inv(Sigma_super)
        eta = lam @ mu_super
        v.mu = mu_super
        v.Sigma = Sigma_super
        v.belief = NdimGaussian(dofs, eta, lam)


    fg.n_var_nodes = len(fg.var_nodes)

    # ---------- Utility: assemble a group's linpoint (using base belief means) ----------
    def make_linpoint_for_group(sid):
        x = np.zeros(total_dofs[sid])
        for bid, (st, d) in local_idx[sid].items():
            mu = getattr(id2var[bid], "mu", None)
            if mu is None or len(mu) != d:
                mu = np.zeros(d)
            x[st:st+d] = mu
        return x

    # ---------- 3) super prior (in-group unary + in-group binary) ----------
    def make_super_prior_factor(sid, base_factors):
        group = super_groups[sid]
        idx_map = local_idx[sid]
        ncols = total_dofs[sid]

        # Select factors whose variables are all within the group (unary or binary)
        in_group = []
        for f in base_factors:
            vids = [v.variableID for v in f.adj_var_nodes]
            if all(vid in group for vid in vids):
                in_group.append(f)

        def meas_fn_super_prior(x_super, *args):
            meas_fn = []
            for f in in_group:
                vids = [v.variableID for v in f.adj_var_nodes]
                # Assemble this factor's local x
                x_loc_list = []
                for vid in vids:
                    st, d = idx_map[vid]
                    x_loc_list.append(x_super[st:st+d])
                x_loc = np.concatenate(x_loc_list) if x_loc_list else np.zeros(0)
                meas_fn.append(f.meas_fn(x_loc))
            return np.concatenate(meas_fn) if meas_fn else np.zeros(0)

        def jac_fn_super_prior(x_super, *args):
            Jrows = []
            for f in in_group:
                vids = [v.variableID for v in f.adj_var_nodes]
                # Build this factor's local x for (potentially) nonlinear Jacobian
                x_loc_list = []
                dims = []
                for vid in vids:
                    st, d = idx_map[vid]
                    dims.append(d)
                    x_loc_list.append(x_super[st:st+d])
                x_loc = np.concatenate(x_loc_list) if x_loc_list else np.zeros(0)

                Jloc = f.jac_fn(x_loc)
                # Map Jloc column blocks back to the super variable columns
                row = np.zeros((Jloc.shape[0], ncols))
                c0 = 0
                for vid, d in zip(vids, dims):
                    st, _ = idx_map[vid]
                    row[:, st:st+d] = Jloc[:, c0:c0+d]
                    c0 += d

                Jrows.append(row)
            return np.vstack(Jrows) if Jrows else np.zeros((0, ncols))

        # z_super: concatenate each base factor's z
        z_list = [f.measurement for f in in_group]
        z_lambda_list = [f.measurement_lambda for f in in_group]
        z_super = np.concatenate(z_list) 
        z_super_lambda = block_diag(*z_lambda_list)

        return meas_fn_super_prior, jac_fn_super_prior, z_super, z_super_lambda 

    # ---------- 4) super between (cross-group binary) ----------
    def make_super_between_factor(sidA, sidB, base_factors):
        groupA, groupB = super_groups[sidA], super_groups[sidB]
        idxA, idxB = local_idx[sidA], local_idx[sidB]
        nA, nB = total_dofs[sidA], total_dofs[sidB]

        cross = []
        for f in base_factors:
            vids = [v.variableID for v in f.adj_var_nodes]
            if len(vids) != 2:
                continue
            i, j = vids
            # One side in A, the other side in B
            if (i in groupA and j in groupB) or (i in groupB and j in groupA):
                cross.append(f)


        def meas_fn_super_between(xAB, *args):
            xA, xB = xAB[:nA], xAB[nA:]
            meas_fn = []
            for f in cross:
                i, j = [v.variableID for v in f.adj_var_nodes]
                if i in groupA:
                    si, di = idxA[i]
                    sj, dj = idxB[j]
                    xi = xA[si:si+di]
                    xj = xB[sj:sj+dj]
                else:
                    si, di = idxB[i]
                    sj, dj = idxA[j]
                    xi = xB[si:si+di]
                    xj = xA[sj:sj+dj]
                x_loc = np.concatenate([xi, xj])
                meas_fn.append(f.meas_fn(x_loc))
            return np.concatenate(meas_fn) 

        def jac_fn_super_between(xAB, *args):
            xA, xB = xAB[:nA], xAB[nA:]
            Jrows = []
            for f in cross:
                i, j = [v.variableID for v in f.adj_var_nodes]
                if i in groupA:
                    si, di = idxA[i]
                    sj, dj = idxB[j]
                    xi = xA[si:si+di]
                    xj = xB[sj:sj+dj]
                    left_start, right_start = si, nA + sj
                else:
                    si, di = idxB[i]
                    sj, dj = idxA[j]
                    xi = xB[si:si+di]
                    xj = xA[sj:sj+dj]
                    left_start, right_start = nA + si, sj
                x_loc = np.concatenate([xi, xj])
                Jloc = f.jac_fn(x_loc)

                row = np.zeros((Jloc.shape[0], nA + nB))
                row[:, left_start:left_start+di]   = Jloc[:, :di] 
                row[:, right_start:right_start+dj] = Jloc[:, di:di+dj] 

                Jrows.append(row)
            return np.vstack(Jrows) 

        z_list = [f.measurement for f in cross]
        z_lambda_list = [f.measurement_lambda for f in cross]
        z_super = np.concatenate(z_list) 
        z_super_lambda = block_diag(*z_lambda_list)

        return meas_fn_super_between, jac_fn_super_between, z_super, z_super_lambda


    for e in super_edges:
        u, v = e["data"]["source"], e["data"]["target"]

        if v == "prior":
            meas_fn, jac_fn, z, z_lambda = make_super_prior_factor(u, base_graph.factors)
            f = Factor(len(fg.factors), [super_var_nodes[u]], z, z_lambda, meas_fn, jac_fn)
            f.adj_beliefs = [vn.belief for vn in f.adj_var_nodes]
            f.type = "super_prior"
            lin0 = make_linpoint_for_group(u)
            f.compute_factor(linpoint=lin0, update_self=True)
            fg.factors.append(f)
            super_var_nodes[u].adj_factors.append(f)
            
        else:
            meas_fn, jac_fn, z, z_lambda = make_super_between_factor(u, v, base_graph.factors)
            f = Factor(len(fg.factors), [super_var_nodes[u], super_var_nodes[v]], z, z_lambda, meas_fn, jac_fn)
            f.adj_beliefs = [vn.belief for vn in f.adj_var_nodes]
            f.type = "super_between"
            lin0 = np.concatenate([make_linpoint_for_group(u), make_linpoint_for_group(v)])
            f.compute_factor(linpoint=lin0, update_self=True)
            fg.factors.append(f)
            super_var_nodes[u].adj_factors.append(f)
            super_var_nodes[v].adj_factors.append(f)


    fg.n_factor_nodes = len(fg.factors)
    return fg



def build_abs_graph(
    layers,
    r_reduced = 2):

    abs_var_nodes = {}
    Bs = {}
    ks = {}
    k2s = {}
    r = 2

    # === 1. Build Abstraction Variables ===
    abs_fg = FactorGraph(nonlinear_factors=False, eta_damping=0)
    sup_fg = layers[-2]["graph"]

    for sn in sup_fg.var_nodes:
        if sn.dofs <= r:
            r = sn.dofs  # No reduction if dofs already <= r
        else:
            r = r_reduced

        sid = sn.variableID
        varis_sup_mu = sn.mu
        varis_sup_sigma = sn.Sigma
        
        # Step 1: Eigen decomposition of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(varis_sup_sigma)

        # Step 2: Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = np.argsort(eigvals)[::-1]      # Get indices of sorted eigenvalues (largest first)
        eigvals = eigvals[idx]               # Reorder eigenvalues
        eigvecs = eigvecs[:, idx]            # Reorder corresponding eigenvectors

        # Step 3: Select the top-k eigenvectors to form the projection matrix (principal subspace)
        B_reduced = eigvecs[:, :r]                 # B_reduced: shape (sup_dof, r), projects r to sup_dof
        Bs[sid] = B_reduced                        # Store the projection matrix for this variable

        # Step 4: Project eta and Lam onto the reduced 2D subspace
        varis_abs_mu = B_reduced.T @ varis_sup_mu          # Projected natural mean: shape (2,)
        varis_abs_sigma = B_reduced.T @ varis_sup_sigma @ B_reduced  # Projected covariance: shape (2, 2)
        ks[sid] = varis_sup_mu - B_reduced @ varis_abs_mu  # Store the mean offset for this variable
        #k2s[sid] = varis_sup_sigma - B_reduced @ varis_abs_sigma @ B_reduced.T  # Residual covariance

        varis_abs_lam = np.linalg.inv(varis_abs_sigma)  # Inverse covariance (precision matrix): shape (2, 2)
        varis_abs_eta = varis_abs_lam @ varis_abs_mu  # Natural parameters: shape (2,)

        v = VariableNode(sid, dofs=r)
        v.GT = sn.GT
        v.prior.lam = 1e-10 * np.eye(r, dtype=float)
        v.prior.eta = np.zeros(r, dtype=float)
        v.mu = varis_abs_mu
        v.Sigma = varis_abs_sigma
        v.belief = NdimGaussian(r, varis_abs_eta, varis_abs_lam)

        abs_var_nodes[sid] = v
        abs_fg.var_nodes.append(v)
    abs_fg.n_var_nodes = len(abs_fg.var_nodes)


    # === 2. Abstract Prior ===
    def make_abs_prior_factor(sup_factor):
        abs_id = sup_factor.adj_var_nodes[0].variableID
        B = Bs[abs_id]
        k = ks[abs_id]

        def meas_fn_abs_prior(x_abs, *args):
            return sup_factor.meas_fn(B @ x_abs + k)
        
        def jac_fn_abs_prior(x_abs, *args):
            return sup_factor.jac_fn(B @ x_abs + k) @ B

        return meas_fn_abs_prior, jac_fn_abs_prior, sup_factor.measurement, sup_factor.measurement_lambda
    


    # === 3. Abstract Between ===
    def make_abs_between_factor(sup_factor):
        vids = [v.variableID for v in sup_factor.adj_var_nodes]
        i, j = vids # two variable IDs
        ni = abs_var_nodes[i].dofs
        Bi, Bj = Bs[i], Bs[j]
        ki, kj = ks[i], ks[j]                       
    

        def meas_fn_super_between(xij, *args):
            xi, xj = xij[:ni], xij[ni:]
            return sup_factor.meas_fn(np.concatenate([Bi @ xi + ki, Bj @ xj + kj]))

        def jac_fn_super_between(xij, *args):
            xi, xj = xij[:ni], xij[ni:]
            J_sup = sup_factor.jac_fn(np.concatenate([Bi @ xi + ki, Bj @ xj + kj]))
            J_abs = np.zeros((J_sup.shape[0], ni + xj.shape[0]))
            J_abs[:, :ni] = J_sup[:, :Bi.shape[0]] @ Bi
            J_abs[:, ni:] = J_sup[:, Bi.shape[0]:] @ Bj
            return J_abs
        
        return meas_fn_super_between, jac_fn_super_between, sup_factor.measurement, sup_factor.measurement_lambda
    

    for f in sup_fg.factors:
        if len(f.adj_var_nodes) == 1:
            meas_fn, jac_fn, z, z_lambda = make_abs_prior_factor(f)
            v = abs_var_nodes[f.adj_var_nodes[0].variableID]
            abs_f = Factor(f.factorID, [v], z, z_lambda, meas_fn, jac_fn)
            abs_f.type = "abs_prior"
            abs_f.adj_beliefs = [v.belief]

            lin0 = v.mu
            abs_f.compute_factor(linpoint=lin0, update_self=True)

            abs_fg.factors.append(abs_f)
            v.adj_factors.append(abs_f)

        elif len(f.adj_var_nodes) == 2:
            meas_fn, jac_fn, z, z_lambda = make_abs_between_factor(f)
            i, j = [v.variableID for v in f.adj_var_nodes]
            vi, vj = abs_var_nodes[i], abs_var_nodes[j]
            abs_f = Factor(f.factorID, [vi, vj], z, z_lambda, meas_fn, jac_fn)
            abs_f.type = "abs_between"
            abs_f.adj_beliefs = [vi.belief, vj.belief]

            lin0 = np.concatenate([vi.mu, vj.mu])
            abs_f.compute_factor(linpoint=lin0, update_self=True)

            abs_fg.factors.append(abs_f)
            vi.adj_factors.append(abs_f)
            vj.adj_factors.append(abs_f)

    abs_fg.n_factor_nodes = len(abs_fg.factors)


    return abs_fg, Bs, ks, k2s


def bottom_up_modify_super_graph(layers):
    """
    Update super-node means (mu) from base nodes,
    and simultaneously adjust variable beliefs and adjacent messages.
    """

    base_graph = layers[-2]["graph"]
    super_graph = layers[-1]["graph"]
    node_map = layers[-1]["node_map"]

    id2var = {vn.variableID: vn for vn in base_graph.var_nodes}

    super_groups = {}
    for b_str, s_id in node_map.items():
        b_int = int(b_str)
        super_groups.setdefault(s_id, []).append(b_int)

    sid2idx = {sn["data"]["id"]: i for i, sn in enumerate(layers[-1]["nodes"])}

    for sid, group in super_groups.items():
        mu_blocks = [id2var[bid].mu for bid in group]
        mu_super = np.concatenate(mu_blocks) if mu_blocks else np.zeros(0)

        if sid in sid2idx:
            idx = sid2idx[sid]
            v = super_graph.var_nodes[idx]

            # Old belief
            old_belief = v.belief

            # 1. Update mu
            #v.mu = mu_super

            # 2. New belief (use old Sigma + new mu)
            #lam = np.linalg.inv(v.Sigma)
            #eta = lam @ v.mu
            #new_belief = NdimGaussian(v.dofs, eta, lam)
            #v.belief = new_belief
            #v.prior = NdimGaussian(v.dofs)


        """
            # 3. update adj_beliefs and messages
            if v.adj_factors:
                n_adj = len(v.adj_factors)
                d_eta = new_belief.eta - old_belief.eta
                d_lam = new_belief.lam - old_belief.lam
                for f in v.adj_factors:
                    if v in f.adj_var_nodes:
                        idx_in_factor = f.adj_var_nodes.index(v)
                        # update factor's adj_belief
                        f.adj_beliefs[idx_in_factor] = new_belief
                        # update corresponding messages
                        msg = f.messages[idx_in_factor]
                        msg.eta += d_eta / n_adj
                        msg.lam += d_lam / n_adj
                        f.messages[idx_in_factor] = msg
        """


def top_down_modify_base_and_abs_graph(layers):
    """
    From the super graph downward, split μ to the base graph,
    and simultaneously update the base variables' beliefs and the adjacent factors'
    adj_beliefs / messages.

    Assume layers[-1] is the super layer and layers[-2] is the base layer.
    """
    super_graph = layers[-1]["graph"]
    base_graph = layers[-2]["graph"]
    node_map   = layers[-1]["node_map"]  # { base_id(str) -> super_id(str) }


    # super_id -> [base_id(int)]
    super_groups = {}
    for b_str, s_id in node_map.items():
        b_int = int(b_str)
        super_groups.setdefault(s_id, []).append(b_int)

    # child lookup
    id2var_base = {vn.variableID: vn for vn in base_graph.var_nodes}

    a = 0
    for s_var in super_graph.var_nodes:
        sid = str(s_var.variableID)
        if sid not in super_groups:
            continue
        base_ids = super_groups[sid]

        # === split super.mu to base ===
        mu_super = s_var.mu
        off = 0
        for bid in base_ids:
            v = id2var_base[bid]
            d = v.dofs
            mu_child = mu_super[off:off+d]
            off += d

            old_belief = v.belief

            # 1. update mu
            v.mu = mu_child

            # 2. new belief（keep Σ unchanged，use new mu）
            eta = v.belief.lam @ v.mu
            new_belief = NdimGaussian(v.dofs, eta, v.belief.lam)
            v.belief = new_belief
            v.prior = NdimGaussian(v.dofs, 0.1*v.mu, 0.1*np.eye(v.dofs))


            # 3. Sync to adjacent factors (this step is important)
            if v.adj_factors:
                n_adj = len(v.adj_factors)
                d_eta = new_belief.eta - old_belief.eta
                d_lam = new_belief.lam - old_belief.lam

                for f in v.adj_factors:
                    if v in f.adj_var_nodes:
                        idx = f.adj_var_nodes.index(v)
                        # update adj_beliefs
                        f.adj_beliefs[idx] = new_belief
                        # correct coresponding message
                        msg = f.messages[idx]
                        msg.eta += 0.1*d_eta / n_adj
                        msg.lam += 0.1*d_lam / n_adj
                        f.messages[idx] = msg

    return base_graph


def top_down_modify_super_graph(layers):
    """
    From the abs graph downward, project mu / Sigma back to the super graph,
    and simultaneously update the super variables' beliefs and the adjacent
    factors' adj_beliefs / messages.

    Requirements:
      - layers[-1] is abs, layers[-2] is super
      - Factors at the abs level and the super level share the same factorID (one-to-one)
      - The columns of B are orthonormal (from covariance eigenvectors; eigenvectors from np.linalg.eigh are orthogonal)
    """

    abs_graph   = layers[-1]["graph"]
    super_graph = layers[-2]["graph"]
    Bs  = layers[-1]["Bs"]   # { super_id(int) -> B (d_super × r) }
    ks  = layers[-1]["ks"]   # { super_id(int) -> k (d_super,) }
    #k2s = layers[-1]["k2s"]  # { super_id(int) -> residual covariance (d_super × d_super) }

    # Prebuild abs factor index: factorID -> Factor
    #abs_f_by_id = {f.factorID: f for f in getattr(abs_graph, "factors", [])}

    # ---- First project variables' mu / Sigma and update beliefs ----
    for sn in super_graph.var_nodes:
        sid = sn.variableID
        if sid not in Bs or sid not in ks:
            continue
        B  = Bs[sid]    # (d_s × r)
        k  = ks[sid]    # (d_s,)
        #k2 = k2s[sid]   # (d_s × d_s)

        # x_s = B x_a + k; Σ_s = B Σ_a Bᵀ + k2
        mu_a    = abs_graph.var_nodes[sid].mu
        mu_s    = B @ mu_a + k
        sn.mu   = mu_s

        # Refresh super belief (natural parameters) with the new μ and Σ
        eta = sn.belief.lam @ sn.mu
        new_belief = NdimGaussian(sn.dofs, eta, sn.belief.lam)
        sn.belief  = new_belief
        #sn.prior = NdimGaussian(sn.dofs, 0.01*sn.mu, 0.01*np.eye(sn.dofs))

    # ---- Then push abs messages back to super, preserving the original super messages on the orthogonal complement ----
    # Idea: for the side of the super factor f_sup connected to variable sid:
    #   η_s_new = B η_a + (I - B Bᵀ) η_s_old
    #   Λ_s_new = B Λ_a Bᵀ + (I - B Bᵀ) Λ_s_old (I - B Bᵀ)
    # This ensures the subspace is governed by the abs message, while the orthogonal complement keeps the old super message.
    for sn in super_graph.var_nodes:
        # Iterate over super factors adjacent to this super variable
        for f_sup in sn.adj_factors:
            idx_side = f_sup.adj_var_nodes.index(sn)
            # update the factor's recorded adjacent belief on that side (optional; usually refreshed in the next iteration)
            f_sup.adj_beliefs[idx_side] = sn.belief

    return



def refresh_gbp_results(layers):
    """
    Precompute an affine map to the base plane for each layer:
      base:   A_i = I2, b_i = 0
      super:  A_s = (1/m) [A_c1, A_c2, ..., A_cm], b_s = (1/m) Σ b_cj
      abs:    A_a = A_super(s) @ B_s,             b_a = A_super(s) @ k_s + b_super(s)
    Then refresh gbp_result via pos = A @ mu + b.
    Convention: use string keys everywhere (aligned with Cytoscape ids).
    """
    if not layers:
        return

    # ---------- 1) Bottom-up: compute A, b for each layer ----------
    for li, L in enumerate(layers):
        g = L.get("graph")
        if g is None:
            L.pop("A", None); L.pop("b", None); L.pop("gbp_result", None)
            continue

        name = L["name"]
        # ---- base ----
        if name.startswith("base"):
            L["A"], L["b"] = {}, {}
            for v in g.var_nodes:
                key = str(v.variableID)
                L["A"][key] = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]], dtype=float)  # 2x3: 只取 x,y
                L["b"][key] = np.zeros(2, dtype=float)

        # ---- super ----
        elif name.startswith("super"):
            parent = layers[li - 1]
            node_map = L["node_map"]  # { prev_id(str) -> super_id(str) }

            # Grouping (preserve insertion order to match the concatenation order in build_super_graph)
            groups = {}
            for prev_id, s_id in node_map.items():
                prev_id = str(prev_id); s_id = str(s_id)
                groups.setdefault(s_id, []).append(prev_id)

            L["A"], L["b"] = {}, {}
            for s_id, children in groups.items():
                m = len(children)
                # Horizontal concatenation [A_c1, A_c2, ...]
                A_blocks = [parent["A"][cid] for cid in children]  # each block has shape 2×d_c
                A_concat = np.hstack(A_blocks) if A_blocks else np.zeros((2, 0))
                b_sum = sum((parent["b"][cid] for cid in children), start=np.zeros(2, dtype=float))
                L["A"][s_id] = (1.0 / m) * A_concat
                L["b"][s_id] = (1.0 / m) * b_sum

        # ---- abs ----
        elif name.startswith("abs"):
            parent = layers[li - 1]  # the corresponding super layer
            Bs, ks = L["Bs"], L["ks"]  # Note: keys are the super variableIDs (int)

            # Build a mapping between super variableID (int) and the super string id (follow node list order)
            # The order of nodes in the parent (super) and this (abs) layer is consistent (copy_to_abs preserves order)
            int2sid = {i: str(parent["nodes"][i]["data"]["id"]) for i in range(len(parent["nodes"]))}

            L["A"], L["b"] = {}, {}
            for av in g.var_nodes:
                sid_int = av.variableID              # super variableID (int)
                s_id = int2sid.get(sid_int, str(sid_int))  # super string id (also the abs node id)
                B = Bs[sid_int]                       # (sum d_c) × r
                k = ks[sid_int]                       # (sum d_c,)

                A_sup = parent["A"][s_id]             # shape 2 × (sum d_c)
                b_sup = parent["b"][s_id]             # shape (2,)

                L["A"][s_id] = A_sup @ B              # 2 × r
                L["b"][s_id] = A_sup @ k + b_sup      # 2,

        else:
            # Unknown layer type
            L["A"], L["b"] = {}, {}

    # ---------- 2) Compute gbp_result ----------
    for li, L in enumerate(layers):
        g = L.get("graph")
        if g is None:
            L.pop("gbp_result", None)
            continue

        name = L["name"]
        res = {}

        if name.startswith("base"):
            for v in g.var_nodes:
                vid = str(v.variableID)
                res[vid] = v.mu[:2].tolist()

        elif name.startswith("super"):
            # Directly use A_super, b_super mapping
            # nodes order is consistent with var_nodes order
            for i, v in enumerate(g.var_nodes):
                s_id = str(L["nodes"][i]["data"]["id"])
                A, b = L["A"][s_id], L["b"][s_id]   # A: 2×(sum d_c)
                res[s_id] = (A @ v.mu + b).tolist()

        elif name.startswith("abs"):
            parent = layers[li - 1]
            # Also align via string ids
            for i, v in enumerate(g.var_nodes):
                a_id = str(L["nodes"][i]["data"]["id"])  # same text as the super s_id
                A, b = L["A"][a_id], L["b"][a_id]        # A: 2×r
                res[a_id] = (A @ v.mu + b).tolist()

        L["gbp_result"] = res



def vloop(layers):
    """
    Simplified V-cycle:
    1) bottom-up: rebuild and iterate once for base / super / abs in order
    2) top-down: propagate mu from super -> base
    3) refresh gbp_result on each layer for UI use
    """

    # ---- bottom-up ----
    if layers and "graph" in layers[0]:
        layers[0]["graph"].synchronous_iteration()
        
    for i in range(1, len(layers)):
        # After build, one iteration per layer
        if "graph" in layers[i]:
            layers[i]["graph"].synchronous_iteration()

        name = layers[i]["name"]
        if name.startswith("super1"):
            # Update super using the previous layer's graph
            # layers[i]["graph"] = build_super_graph(layers[:i+1])
            bottom_up_modify_super_graph(layers[:i+1])

        elif name.startswith("super"):
            # Update super using the previous layer's graph
            layers[i]["graph"] = build_super_graph(layers[:i+1])

        elif name.startswith("abs"):
            # Rebuild abs using the previous super
            abs_graph, Bs, ks, k2s = build_abs_graph(layers[:i+1], r_reduced=2)
            layers[i]["graph"] = abs_graph
            layers[i]["Bs"], layers[i]["ks"], layers[i]["k2s"] = Bs, ks, k2s


    # ---- top-down (pass mu) ----
    for i in range(len(layers) - 1, 0, -1):
        # After two iterations per layer, reproject
        if "graph" in layers[i]:
            layers[i]["graph"].synchronous_iteration()
        # this is very important, but dont know why yet
        # so abs layer need more iterations
        if name.startswith("abs"):
            layers[i]["graph"].synchronous_iteration()  

        name = layers[i]["name"]
        if name.startswith("super"):
            # Split super.mu back to base/abs
            top_down_modify_base_and_abs_graph(layers[:i+1])
        elif name.startswith("abs"):
            # Project abs.mu back to super
            top_down_modify_super_graph(layers[:i+1])


    # ---- refresh gbp_result for UI ----
    refresh_gbp_results(layers)


# -----------------------
# Layout
# -----------------------
app.layout = html.Div([
    # ===== Top three rows =====
    # Row 1: Base graph parameters + New Graph
    html.Div([
        html.Div([
            html.Span("N:", style={"marginRight":"6px"}),
            dcc.Input(id="param-N", type="number", value=512, min=2, step=1,
                      style={"width":"100px", "marginRight":"12px"}),

            html.Span("step:", style={"marginRight":"6px"}),
            dcc.Input(id="param-step", type="number", value=25, step=1,
                      style={"width":"100px", "marginRight":"12px"}),

            html.Span("loop prob:", style={"marginRight":"6px"}),
            dcc.Input(id="param-prob", type="number", value=0.05, min=0, max=1, step=0.01,
                      style={"width":"100px", "marginRight":"12px"}),

            html.Span("loop radius:", style={"marginRight":"6px"}),
            dcc.Input(id="param-radius", type="number", value=50, step=1,
                      style={"width":"100px", "marginRight":"12px"}),

            html.Span("prior prop:", style={"marginRight":"6px"}),
            dcc.Input(id="prior-prop", type="number", value=0.02, step=0.01, min=0, max=1,
                      style={"width":"100px", "marginRight":"12px"}),

            html.Span("show number:", style={"marginRight":"6px"}),
            dcc.Checklist(
                id="show-number",
                options=[{"label": "", "value": "on"}],
                value=["on"],
                style={"display": "inline-block", "marginRight":"12px"}
            ),
        ], style={"flex":"1"}),

        html.Div([
            html.Button("New Graph", id="new-graph", n_clicks=0,
                        style={"display":"block", "width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"space-between", "alignItems":"center", "margin":"6px 10px"}),

    # Row 2: Aggregation parameters + Add Layer
    html.Div([
        html.Div([
            html.Span("Mode:", style={"marginRight":"6px"}),
            dcc.Dropdown(
                id="agg-mode",
                options=[
                    {"label":"Grid","value":"grid"},
                    {"label":"K-Means","value":"kmeans"},
                    {"label":"Order","value":"order"}
                ],
                value="order", clearable=False, className="mode-dd",
                style={"width":"120px","height":"26px","display":"inline-block","marginRight":"12px"}
            ),
            html.Span("Gx:", style={"marginRight":"6px"}),
            dcc.Input(id="grid-gx", type="number", value=2, min=1, step=1,
                      style={"width":"100px", "marginRight":"12px"}),
            html.Span("Gy:", style={"marginRight":"6px"}),
            dcc.Input(id="grid-gy", type="number", value=2, min=1, step=1,
                      style={"width":"100px", "marginRight":"12px"}),
            html.Span("K:", style={"marginRight":"6px"}),
            dcc.Input(id="kmeans-k", type="number", value=128, min=1, step=1,
                      style={"width":"100px", "marginRight":"12px"}),
            html.Span("seed:", style={"marginRight":"6px"}),
            dcc.Input(id="rand-seed", type="number", value=2001, step=1,
                      style={"width":"100px"})
        ], style={"flex":"1"}),

        html.Div([
            html.Button("Add Layer", id="add-layer", n_clicks=0,
                        style={"display":"block", "width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"space-between", "alignItems":"center", "margin":"6px 10px"}),

    # Row 3: GBP parameters + Project Layer
    html.Div([
        html.Div([
            html.Span("prior σ:", style={"marginRight":"6px"}),
            dcc.Input(id="prior-noise", type="number", value=1.0, step=0.01,
                    style={"width":"100px", "marginRight":"12px"}),

            html.Span("odom σ:", style={"marginRight":"6px"}),
            dcc.Input(id="odom-noise", type="number", value=1.0, step=0.01,
                    style={"width":"100px", "marginRight":"12px"}),

            html.Span("iters:", style={"marginRight":"6px"}),
            dcc.Input(id="param-iters", type="number", value=5, step=1,
                    style={"width":"100px", "marginRight":"12px"}),

            html.Span("snap:", style={"marginRight":"6px"}),
            dcc.Input(id="param-snap", type="number", value=0.1, step=0.1,
                    style={"width":"100px"})
        ], style={"flex":"1"}),

        html.Div([
            # ✅ Correct button id
            html.Button("Project Layer", id="project-layer", n_clicks=0,
                        style={"display":"block", "width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"space-between", "alignItems":"center", "margin":"6px 10px"}),


    # Row 4: Layer toolbar + V Cycle (combined in one line)
    html.Div([
        # Left side: Layer selection + status
        html.Div([
            html.Span("Layer:", style={"marginRight":"6px"}),
            dcc.Dropdown(
                id="layer-select",
                options=[{"label": "base", "value": "base"}],
                value="base",
                clearable=False,
                style={"width": "240px", "margin": "0"}
            ),
            html.Div(id="gbp-status",
                    style={"margin":"0 0 0 16px", "fontStyle":"italic", "color":"#444", "whiteSpace":"nowrap"}),
            html.Div(id="vcycle-status",
                    style={"margin":"0 0 0 12px", "fontStyle":"italic", "color":"#444", "whiteSpace":"nowrap"}),
        ], style={
            "display":"flex",
            "alignItems":"center",
            "gap":"8px",
            "flex":"1"
        }),

        # Right side: V Cycle button
        html.Div([
            html.Button("GBP Solver", id="gbp-run", n_clicks=0,
            
                        style={"display":"block", "background":"#111","color":"#fff","width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"space-between", "alignItems":"center", "margin":"6px 10px"}),


    # Row 5: V Cycle controls    
    html.Div([
        html.Div([
            html.Button("V Cycle", id="vcycle-run", n_clicks=0,
                        style={"display":"block", "background":"#FFA500","color":"#fff","width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"flex-end", "alignItems":"center", "margin":"6px 10px"}),

    # Row 6: Layer selection + status (compact toolbar)
    # Stores / Intervals / Cytoscape
    dcc.Store(id="gbp-state", data={"running": False, "iters_done": 0, "iters_total": 0, "snap_int": 5}),
    dcc.Store(id="gbp-poses", data=None),
    dcc.Interval(id="gbp-interval", interval=200, n_intervals=0, disabled=True),

    dcc.Store(id="vcycle-state", data={"running": False, "iters_done": 0, "iters_total": 0, "snap_int": 5}),
    dcc.Interval(id="vcycle-interval", interval=500, n_intervals=0, disabled=True),

    cyto.Cytoscape(
        id="cytoscape",
        style={"width": f"{VIEW_W}px", "height": f"{VIEW_H}px"},
        layout={"name": "preset"},
        elements=[]
    )
    ])



# -----------------------
# Manage Add / New Graph
# -----------------------
@app.callback(
    Output("layer-select","options"),
    Output("layer-select","value"),
    Input("add-layer","n_clicks"),
    Input("new-graph","n_clicks"),
    State("agg-mode","value"),
    State("grid-gx","value"),
    State("grid-gy","value"),
    State("kmeans-k","value"),
    State("layer-select","value"),
    State("param-N","value"),
    State("param-step","value"),
    State("param-prob","value"),
    State("param-radius","value"),
    State("prior-noise","value"),
    State("odom-noise","value"),
    State("prior-prop","value"),
    State("rand-seed","value"),

    prevent_initial_call=True
)
def manage_layers(add_clicks, new_clicks, mode, gx, gy, kk, current_value,
                  pN, pStep, pProb, pRadius, pPrior, pOdom, pPriorProp, seed):
    global layers, pair_idx, gbp_graph
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered == "new-graph":
        if seed is None or seed == 0:
            rng = np.random.default_rng()        # Random initialization
        else:
            rng = np.random.default_rng(seed)    # Fixed seed
        N = int(pN or 100)
        step = float(pStep or 25)
        prob = float(pProb or 0.05)
        radius = float(pRadius or 50)
        prior_prop=float(pPriorProp or 0.00)
        layers = init_layers(N, step, prob, radius, prior_prop, rng=rng)
        pair_idx = 0
        reset_global_bounds(layers[0]["nodes"])
        # Build the GBP graph (rendering still shows GT at this point)
        gbp_graph = build_noisy_pose_graph(layers[0]["nodes"], layers[0]["edges"],
                                           prior_sigma=float(pPrior or 1.0),
                                           odom_sigma=float(pOdom or 1.0)
                                           )
        layers[0]["graph"] = gbp_graph
        opts=[{"label":"base","value":"base"}]
        return opts, "base"

    if triggered == "add-layer":
        idx = next(i for i,L in enumerate(layers) if L["name"] == current_value)
        layers = layers[:idx+1]
        last = layers[-1]
        kind, k = parse_layer_name(last["name"])

        
        pair_idx = highest_pair_idx([L["name"] for L in layers])
        if kind == "super":
            abs_layer_idx = k*2
            abs_nodes, abs_edges = copy_to_abs(last["nodes"], last["edges"], abs_layer_idx)

            # Ensure super graph has run at least once
            layers[-1]["graph"].synchronous_iteration() 

            layers.append({"name":f"abs{k}", "nodes":abs_nodes, "edges":abs_edges})
            layers[abs_layer_idx]["graph"], layers[abs_layer_idx]["Bs"], layers[abs_layer_idx]["ks"], layers[abs_layer_idx]["k2s"] = build_abs_graph(layers, r_reduced=2)
        else:
            k_next = pair_idx + 1
            super_layer_idx = k_next*2 - 1
            if mode == "grid":
                super_nodes, super_edges, node_map = fuse_to_super_grid(last["nodes"], last["edges"], int(gx or 2), int(gy or 2), super_layer_idx)
            elif mode == "kmeans":
                super_nodes, super_edges, node_map = fuse_to_super_kmeans(last["nodes"], last["edges"], int(kk or 8), super_layer_idx)
            else:
                super_nodes, super_edges, node_map = fuse_to_super_order(last["nodes"], last["edges"], int(kk or 8), super_layer_idx, tail_heavy=True)
            # Ensure super graph has run at least once
            layers[-1]["graph"].synchronous_iteration() 
            layers.append({"name":f"super{k_next}", "nodes":super_nodes, "edges":super_edges, "node_map":node_map})
            layers[super_layer_idx]["graph"] = build_super_graph(layers)

            pair_idx = k_next

    opts=[{"label":L["name"],"value":L["name"]} for L in layers]
    return opts, layers[-1]["name"]


# -----------------------
# Render Cytoscape (including real-time overlay of base-layer poses)
# -----------------------
@app.callback(
    Output("cytoscape","elements"),
    Output("cytoscape","stylesheet"),
    Output("cytoscape","layout"),
    Input("layer-select","value"),
    Input("layer-select","options"),
    Input("gbp-poses","data"),
    Input("show-number","value"),
    State("param-N","value"),
)
def update_layer(layer_name, _options, gbp_poses, show_number, param_N):
    # Find the current layer
    layer = next((l for l in layers if l["name"] == layer_name), None)
    if layer is None:
        return [], [], {"name": "preset"}

    orig_nodes, edges = layer["nodes"], layer["edges"]
    edges = [e for e in edges if e["data"].get("target") != "prior"]

    # GBP solution for the current layer (if available)
    result = layer.get("gbp_result", None)

    nodes = []
    base_count = len(layers[0]["nodes"]) if layers else 1
    alpha = 0.3  # <1 → Faster in the early stage, slower in the later stage

    for n in orig_nodes:
        new_n = {
            "data": dict(n["data"]),
            "position": dict(n["position"])
        }

        # Apply GBP-updated pose
        if result and new_n["data"]["id"] in result:
            mu = result[new_n["data"]["id"]]
            new_n["position"]["x"] = float(mu[0])
            new_n["position"]["y"] = float(mu[1])

        nb = int(new_n["data"].get("num_base", 1))

        # ==== Color value (power scaling) ====
        color_val = float(((nb / base_count) ** alpha) * base_count)
        new_n["data"]["color_val"] = color_val

        # ==== Linear radius ====
        size_linear = 4

        # ==== Log-scaled radius ====
        size_val = float(size_linear * (np.log(1 + nb*500/param_N) / np.log(4)))
        new_n["data"]["size_val"] = size_val

        nodes.append(new_n)

    # Axes
    axis_nodes = [
        {"data": {"id": "x_axis_start"}, "position": {"x": float(GLOBAL_XMIN_ADJ - AXIS_PAD), "y": 0}, "classes": "axis-node"},
        {"data": {"id": "x_axis_end"},   "position": {"x": float(GLOBAL_XMAX_ADJ + AXIS_PAD), "y": 0}, "classes": "axis-node"},
        {"data": {"id": "y_axis_start"}, "position": {"x": 0, "y": float(GLOBAL_YMIN_ADJ - AXIS_PAD)}, "classes": "axis-node"},
        {"data": {"id": "y_axis_end"},   "position": {"x": 0, "y": float(GLOBAL_YMAX_ADJ + AXIS_PAD)}, "classes": "axis-node"},
    ]
    axis_edges = [
        {"data": {"source": "x_axis_start", "target": "x_axis_end"}, "classes": "axis"},
        {"data": {"source": "y_axis_start", "target": "y_axis_end"}, "classes": "axis"},
    ]

    label_style = "data(num_base)" if ("on" in show_number) else ""
    elements = nodes + edges + axis_nodes + axis_edges

    stylesheet = [
        {"selector": "node", "style": {
            "shape": "ellipse",
            "width": "data(size_val)",
            "height": "data(size_val)",
            "background-color": f"mapData(color_val,1,{base_count},hsl(120,100%,40%),hsl(0,100%,50%))",
            "label": label_style,
            "font-size": 8,
            "text-valign": "top",
            "border-width": f"mapData(size_val,1,20,0.2,1.0)",   # size_val small → 0.2px, large → 1px
            "border-color": "rgba(0,0,0,0)"                   # soft gray border
        }},

        {"selector": "edge", "style": {
            "line-color": "rgba(136,136,136,0.1)",  # #888 ≈ (136,136,136), alpha 0.1 ≈ 90% transparent
            "width": 0.9
        }},
        {"selector": ".axis", "style": {
            "line-color": "black", "width": 1,
            "target-arrow-shape": "triangle",
            "arrow-scale": 1, "curve-style": "straight"
        }},
        {"selector": ".axis-node", "style": {
            "width": 1, "height": 1, "background-color": "white",
            "border-width": 0, "opacity": 0.0
        }},
    ]

    layout = {"name": "preset"}
    return elements, stylesheet, layout




# -----------------------
# Merged GBP callback (button + interval + new-graph emergency stop)
# -----------------------
from dash import no_update

@app.callback(
    Output("gbp-poses", "data"),
    Output("gbp-status", "children"),
    Output("gbp-state", "data"),
    Output("gbp-interval", "disabled"),
    Output("gbp-interval", "n_intervals"),
    Output("vcycle-status", "children"),
    Output("vcycle-state", "data"),
    Output("vcycle-interval", "disabled"),
    Output("vcycle-interval", "n_intervals"),
    Input("gbp-run", "n_clicks"),
    Input("gbp-interval", "n_intervals"),
    Input("vcycle-run", "n_clicks"),
    Input("vcycle-interval", "n_intervals"),
    Input("project-layer", "n_clicks"),       # ✅ New: project layer
    Input("new-graph", "n_clicks"),           # ✅ Unified emergency stop remains
    State("gbp-state", "data"),
    State("vcycle-state", "data"),
    State("param-iters","value"),
    State("param-snap","value"),
    State("layer-select","value"),
    prevent_initial_call=True
)
def unified_solver(gbp_click, gbp_ticks,
                   vcycle_click, vcycle_ticks,
                   project_click,                    # ✅ Adds an argument
                   new_graph_click,
                   gbp_state, vcycle_state,
                   iters, snap_int, current_layer):

    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, gbp_state, True, no_update, \
               no_update, vcycle_state, True, no_update

    trig = ctx.triggered[0]["prop_id"]

    # ==== Reset when new graph ====
    if trig.startswith("new-graph"):
        reset_state = {"running": False, "iters_done": 0, "iters_total": 0, "snap_int": 5}
        return None, "Ready. New graph created.", reset_state, True, 0, \
               "", reset_state, True, 0

    # ==== Project Layer ====
    if trig.startswith("project-layer"):
        # locate the current layer
        try:
            idx = next(i for i, L in enumerate(layers) if L["name"] == current_layer)
        except StopIteration:
            return no_update, f"Layer '{current_layer}' not found.", gbp_state, True, no_update, \
                   no_update, vcycle_state, True, no_update

        kind, k = parse_layer_name(current_layer)

        if kind == "super":
            # super -> base
            if idx - 1 < 0 or "graph" not in layers[idx] or "graph" not in layers[idx-1]:
                return no_update, f"Cannot project: graphs not ready for {current_layer}.", gbp_state, True, no_update, \
                       no_update, vcycle_state, True, no_update
            top_down_modify_base_and_abs_graph(layers[:idx+1])
            msg = f"Projected {current_layer} → base."
        elif kind == "abs":
            # abs -> super
            if "graph" not in layers[idx] or "graph" not in layers[idx-1]:
                return no_update, f"Cannot project: graphs not ready for {current_layer}.", gbp_state, True, no_update, \
                       no_update, vcycle_state, True, no_update
            top_down_modify_super_graph(layers[:idx+1])
            msg = f"Projected {current_layer} → super."
        else:
            # base has no lower layer
            msg = "Base layer has no lower layer to project to."

        # Refresh and return the current layer poses (triggers Cytoscape redraw)
        refresh_gbp_results(layers)
        latest_positions = layers[idx].get("gbp_result", None)
        return latest_positions, msg, gbp_state, True, no_update, \
               no_update, vcycle_state, True, no_update

    # ==== GBP Solver ====
    if trig.startswith("gbp-run"):
        graph = next((L.get("graph") for L in layers if L["name"] == current_layer), None)
        if graph is None:
            return no_update, f"No factor graph in {current_layer}.", gbp_state, True, no_update, \
                   no_update, vcycle_state, True, no_update
        iters = int(iters or 50)
        snap_int = int(snap_int or 5)
        gbp_state = {"running": True, "iters_done": 0, "iters_total": iters, "snap_int": snap_int}
        return no_update, f"GBP running... 0/{iters}", gbp_state, False, 0, \
               no_update, vcycle_state, True, no_update

    if trig.startswith("gbp-interval"):
        if not gbp_state or not gbp_state.get("running"):
            return no_update, no_update, gbp_state, True, no_update, \
                   no_update, vcycle_state, True, no_update

        graph = next((L.get("graph") for L in layers if L["name"] == current_layer), None)
        if graph is None:
            return no_update, no_update, gbp_state, True, no_update, \
                   no_update, vcycle_state, True, no_update

        iters_done = gbp_state["iters_done"]
        iters_total = gbp_state["iters_total"]
        snap_int = gbp_state["snap_int"]
        batch = max(1, min(snap_int, iters_total - iters_done))

        for _ in range(batch):
            graph.synchronous_iteration()

        refresh_gbp_results(layers)
        latest_positions = layers[[L["name"] for L in layers].index(current_layer)]["gbp_result"]

        iters_done += batch
        gbp_state["iters_done"] = iters_done
        finished = iters_done >= iters_total
        gbp_state["running"] = not finished

        status = (f"GBP running {iters_done}/{iters_total}"
                  if not finished else f"GBP finished {iters_total} iters.")
        return latest_positions, status, gbp_state, finished, no_update, \
               no_update, vcycle_state, True, no_update

    # ==== V Cycle ====
    if trig.startswith("vcycle-run"):
        iters = int(iters or 20)
        snap_int = int(snap_int or 5)
        vcycle_state = {"running": True, "iters_done": 0, "iters_total": iters, "snap_int": snap_int}
        return no_update, no_update, gbp_state, True, no_update, \
               f"V Cycle running... 0/{iters}", vcycle_state, False, 0

    if trig.startswith("vcycle-interval"):
        if not vcycle_state or not vcycle_state.get("running"):
            return no_update, no_update, gbp_state, True, no_update, \
                   no_update, vcycle_state, True, no_update

        iters_done = vcycle_state["iters_done"]
        iters_total = vcycle_state["iters_total"]
        snap_int = vcycle_state["snap_int"]
        batch = max(1, min(snap_int, iters_total - iters_done))

        for _ in range(batch):
            vloop(layers)

        refresh_gbp_results(layers)
        latest_positions = layers[[L["name"] for L in layers].index(current_layer)]["gbp_result"]

        iters_done += batch
        vcycle_state["iters_done"] = iters_done
        finished = iters_done >= iters_total
        vcycle_state["running"] = not finished

        status = (f"V Cycle running {iters_done}/{iters_total}"
                  if not finished else f"V Cycle finished {iters_total} iters.")
        return latest_positions, no_update, gbp_state, True, no_update, \
               status, vcycle_state, finished, no_update

    return no_update, no_update, gbp_state, True, no_update, \
           no_update, vcycle_state, True, no_update


# -----------------------
if __name__=="__main__":
    app.run(debug=True, port=8050)
