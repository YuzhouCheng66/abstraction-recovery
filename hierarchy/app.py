import re
import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_cytoscape as cyto
import numpy as np

# ==== GBP 引入 ====
from gbp.gbp import FactorGraph, VariableNode, Factor

app = dash.Dash(__name__)
app.title = "Factor Graph SVD Abs&Recovery"

# -----------------------
# SLAM-like base graph
# -----------------------
def make_slam_like_graph(N=100, step_size=25, loop_prob=0.05, loop_radius=50, prior_prop=0.0, rng=None,):
    nodes, edges = [], []
    positions = []
    x, y = 0.0, 0.0
    positions.append((x, y))
    for _ in range(1, int(N)):
        dx, dy = np.random.randn(2)
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        dx, dy = dx / norm * float(step_size), dy / norm * float(step_size)
        x, y = x + dx, y + dy
        positions.append((x, y))
    for i, (px, py) in enumerate(positions):
        nodes.append({
            "data": {"id": f"{i}", "layer": 0, "dim": 2},
            "position": {"x": float(px), "y": float(py)}
        })
    for i in range(int(N) - 1):
        edges.append({"data": {"source": f"{i}", "target": f"{i+1}"}})
    for i in range(int(N)):
        for j in range(i + 5, int(N)):
            if np.random.rand() < float(loop_prob):
                xi, yi = positions[i]; xj, yj = positions[j]
                if np.hypot(xi-xj, yi-yj) < float(loop_radius):
                    edges.append({"data": {"source": f"{i}", "target": f"{j}"}})

    # 确定强先验的节点集合
    if rng is None:
        rng = np.random.default_rng()

    if prior_prop <= 0.0:
        strong_ids = {0}
    elif prior_prop >= 1.0:
        strong_ids = set(range(N))
    else:
        k = max(1, int(np.floor(prior_prop * N)))
        strong_ids = set(rng.choice(N, size=k, replace=False).tolist())

    # 给 strong prior 节点加 edge
    for i in strong_ids:
        edges.append({
            "data": {"source": f"{i}", "target": "prior"}
        })


    return nodes, edges

# -----------------------
# Grid 聚合
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
        dim_val = int(max(1, sum(child_dims)))
        nid = str(len(super_nodes))
        super_nodes.append({
            "data": {"id": nid, "layer": layer_idx, "dim": dim_val},
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
# K-Means 聚合
# -----------------------
def fuse_to_super_kmeans(prev_nodes, prev_edges, k, layer_idx, max_iters=20, tol=1e-6, seed=0):
    positions = np.array([[n["position"]["x"], n["position"]["y"]] for n in prev_nodes], dtype=float)
    n = positions.shape[0]
    if k <= 0: 
        k = 1
    k = min(k, n)
    rng = np.random.default_rng(seed)

    # -------- 改进版初始化 --------
    # 随机无放回抽 k 个点，保证一开始每簇有独立的点
    init_idx = rng.choice(n, size=k, replace=False)
    centers = positions[init_idx]

    # Lloyd 迭代
    for _ in range(max_iters):
        d2 = ((positions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)

        # -------- 空簇修补 --------
        counts = np.bincount(assign, minlength=k)
        empty_clusters = np.where(counts == 0)[0]
        for ci in empty_clusters:
            # 找到最大簇
            big_cluster = np.argmax(counts)
            big_idxs = np.where(assign == big_cluster)[0]
            # 偷一个点过来
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

    # final assign (再做一次保证)
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

    # ---------- 构造 super graph ----------
    super_nodes, node_map = [], {}
    for ci in range(k):
        idxs = np.where(assign == ci)[0]
        pts = positions[idxs]
        mean_x, mean_y = pts.mean(axis=0)
        child_dims = [prev_nodes[i]["data"]["dim"] for i in idxs]
        dim_val = int(max(1, sum(child_dims)))
        nid = f"{ci}"
        super_nodes.append({
            "data": {"id": nid, "layer": layer_idx, "dim": dim_val},
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
            "data": {"id": nid, "layer": layer_idx, "dim": n["data"]["dim"]},
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
# 工具
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
# 初始化 & 边界
# -----------------------
def init_layers(N=100, step_size=25, loop_prob=0.05, loop_radius=50, prior_prop=0.0):
    base_nodes, base_edges = make_slam_like_graph(N, step_size, loop_prob, loop_radius, prior_prop)
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

# ==== 全局状态 ====
layers = init_layers()
pair_idx = 0
reset_global_bounds(layers[0]["nodes"])
gbp_graph = None

# -----------------------
# GBP Graph 构建
# -----------------------

def build_noisy_pose_graph(
    nodes,
    edges,
    prior_sigma: float = 10,
    odom_sigma: float = 10,
    tiny_prior: float = 1e-10,
    rng=None,
):
    
    """
    构造二维 pose-only 因子图（线性，高斯），并注入噪声。
    参数:
      prior_sigma : 强先验的标准差（小=强）
      odom_sigma  : 里程计测量噪声标准差
      prior_prop  : 0.0=仅 anchor；(0,1)=按比例随机选；>=1.0=全体
      tiny_prior  : 所有节点默认加的极小先验，防止奇异
      seed        : 随机种子（可复现）
    """

    fg = FactorGraph(nonlinear_factors=False, eta_damping=0)

    var_nodes = []
    I2 = np.eye(2, dtype=float)
    N = len(nodes)

    # ---- 预生成噪声 ----
    prior_noises = {}
    odom_noises = {}

    if rng is None:
        rng = np.random.default_rng()

    # 为所有边生成噪声
    for e in edges:
        src = e["data"]["source"]; dst = e["data"]["target"]
        # 二元边
        if dst != "prior":
            odom_noises[(int(src[:]), int(dst[:]))] = rng.normal(0.0, odom_sigma, size=2)
        # 一元边（强先验）
        elif dst == "prior":
            prior_noises[int(src[:])] = rng.normal(0.0, prior_sigma, size=2)


    # ---- variable nodes ----
    for i, n in enumerate(nodes):
        v = VariableNode(i, dofs=2)
        v.GT = np.array([n["position"]["x"], n["position"]["y"]], dtype=float)

        # 极小先验
        v.prior.lam = tiny_prior * I2
        v.prior.eta = np.zeros(2, dtype=float)

        var_nodes.append(v)

    fg.var_nodes = var_nodes
    fg.n_var_nodes = len(var_nodes)


    # ---- prior factors ----
    def meas_fn_unary(x, *args):
        return x
    def jac_fn_unary(x, *args):
        return np.eye(2)
    # ---- odometry factors ----
    def meas_fn(xy, *args):
        return xy[2:] - xy[:2]
    def jac_fn(xy, *args):
        return np.array([[-1, 0, 1, 0],
                         [ 0,-1, 0, 1]], dtype=float)
    
    factors = []
    fid = 0

    for e in edges:
        src = e["data"]["source"]; dst = e["data"]["target"]
        if dst != "prior":
            i, j = int(src[:]), int(dst[:])
            vi, vj = var_nodes[i], var_nodes[j]

            meas = (vj.GT - vi.GT) + odom_noises[(i, j)]

            meas_lambda = np.eye(len(meas))/ (odom_sigma**2)
            f = Factor(fid, [vi, vj], meas, meas_lambda, meas_fn, jac_fn)
            f.type = "base"
            linpoint = np.r_[vi.GT, vj.GT]
            f.compute_factor(linpoint=linpoint, update_self=True)

            factors.append(f)
            vi.adj_factors.append(f)
            vj.adj_factors.append(f)
            fid += 1

        else:
            i = int(src[:])
            vi = var_nodes[i]
            z = vi.GT + prior_noises[i]

            z_lambda = np.eye(len(meas))/ (prior_sigma**2)
            f = Factor(fid, [vi], z, z_lambda, meas_fn_unary, jac_fn_unary)
            f.type = "prior"
            f.compute_factor(linpoint=z, update_self=True)

            factors.append(f)
            vi.adj_factors.append(f)
            fid += 1

    fg.factors = factors
    fg.n_factor_nodes = len(factors)
    return fg



def build_super_graph(layers):
    """
    基于 layers[-2] 的 base graph, 和 layers[-1] 的 super 分组，构造 super graph。
    要求: layers[-2]["graph"] 已经是构建好的基图（含 unary/binary 因子）。
    layers[-1]["node_map"]: { base_node_id(str, 如 'b12') -> super_node_id(str) }
    """
    from scipy.linalg import block_diag
    # ---------- 取出 base & super ----------
    base_graph = layers[-2]["graph"]
    super_nodes = layers[-1]["nodes"]
    super_edges = layers[-1]["edges"]
    node_map    = layers[-1]["node_map"]   # 'bN' -> 'sX_...'

    # base: id(int)->VariableNode，方便查 dofs 和 mu
    id2var = {vn.variableID: vn for vn in base_graph.var_nodes}

    # ---------- super_id -> [base_id(int)] ----------
    super_groups = {}
    for b_str, s_id in node_map.items():
        b_int = int(b_str)
        super_groups.setdefault(s_id, []).append(b_int)


    # ---------- 为每个 super 组建立 (start, dofs) 表 ----------
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


    # ---------- 创建 super VariableNodes ----------
    fg = FactorGraph(nonlinear_factors=False, eta_damping=0)

    super_var_nodes = {}
    for i, sn in enumerate(super_nodes):
        sid = sn["data"]["id"]
        dofs = total_dofs.get(sid, 0)

        v = VariableNode(i, dofs=dofs)

        # === 叠加 base GT ===
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


    fg.n_var_nodes = len(fg.var_nodes)

    # ---------- 工具：拼接某组的 linpoint（用 base belief 均值） ----------
    def make_linpoint_for_group(sid):
        x = np.zeros(total_dofs[sid])
        for bid, (st, d) in local_idx[sid].items():
            mu = getattr(id2var[bid], "mu", None)
            if mu is None or len(mu) != d:
                mu = np.zeros(d)
            x[st:st+d] = mu
        return x

    # ---------- 3) super prior（in_group unary + in_group binary） ----------
    def make_super_prior_factor(sid, base_factors):
        group = super_groups[sid]
        idx_map = local_idx[sid]
        ncols = total_dofs[sid]

        # 选出：所有变量都在组内的因子（unary 或 binary）
        in_group = []
        for f in base_factors:
            vids = [v.variableID for v in f.adj_var_nodes]
            if all(vid in group for vid in vids):
                in_group.append(f)

        def meas_fn_super_prior(x_super, *args):
            meas_fn = []
            for f in in_group:
                vids = [v.variableID for v in f.adj_var_nodes]
                # 拼本因子的局部 x
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
                # 构造本因子的局部 x，用于（潜在）非线性雅可比
                x_loc_list = []
                dims = []
                for vid in vids:
                    st, d = idx_map[vid]
                    dims.append(d)
                    x_loc_list.append(x_super[st:st+d])
                x_loc = np.concatenate(x_loc_list) if x_loc_list else np.zeros(0)

                Jloc = f.jac_fn(x_loc)
                # 将 Jloc 列块映射回 super 变量的列
                row = np.zeros((Jloc.shape[0], ncols))
                c0 = 0
                for vid, d in zip(vids, dims):
                    st, _ = idx_map[vid]
                    row[:, st:st+d] = Jloc[:, c0:c0+d]
                    c0 += d

                Jrows.append(row)
            return np.vstack(Jrows) if Jrows else np.zeros((0, ncols))

        # z_super：拼各 base 因子的 z
        z_list = [f.measurement for f in in_group]
        z_lambda_list = [f.measurement_lambda for f in in_group]
        z_super = np.concatenate(z_list) 
        z_super_lambda = block_diag(*z_lambda_list)

        return meas_fn_super_prior, jac_fn_super_prior, z_super, z_super_lambda 

    # ---------- 4) super between（cross_group binary） ----------
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
            # on side in A，the other side in B
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
            f.type = "super_prior"
            lin0 = make_linpoint_for_group(u)
            f.compute_factor(linpoint=lin0, update_self=True)
            fg.factors.append(f)
            super_var_nodes[u].adj_factors.append(f)
            
        else:
            meas_fn, jac_fn, z, z_lambda = make_super_between_factor(u, v, base_graph.factors)
            f = Factor(len(fg.factors), [super_var_nodes[u], super_var_nodes[v]], z, z_lambda, meas_fn, jac_fn)
            f.type = "super_between"
            lin0 = np.concatenate([make_linpoint_for_group(u), make_linpoint_for_group(v)])
            f.compute_factor(linpoint=lin0, update_self=True)
            fg.factors.append(f)
            super_var_nodes[u].adj_factors.append(f)
            super_var_nodes[v].adj_factors.append(f)


    fg.n_factor_nodes = len(fg.factors)
    return fg


def build_abs_graph():
    pass    


# -----------------------
# Layout
# -----------------------
app.layout = html.Div([
    # ===== 顶部三行 =====
    # 行1：基础图参数 + New Graph
    html.Div([
        html.Div([
            html.Span("N:", style={"marginRight":"6px"}),
            dcc.Input(id="param-N", type="number", value=500, min=2, step=1,
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
            dcc.Input(id="prior-prop", type="number", value=0, step=0.01, min=0, max=1,
                      style={"width":"100px"})

        ], style={"flex":"1"}),

        html.Div([
            html.Button("New Graph", id="new-graph", n_clicks=0,
                        style={"display":"block", "width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"space-between", "alignItems":"center", "margin":"10px 10px 6px"}),

    # 行2：聚合参数 + Add Layer
    html.Div([
        html.Div([
            html.Span("Mode:", style={"marginRight":"6px"}),
            dcc.Dropdown(
                id="agg-mode",
                options=[{"label":"Grid","value":"grid"},{"label":"K-Means","value":"kmeans"}],
                value="kmeans", clearable=False, className="mode-dd",
                style={"width":"120px","height":"26px","display":"inline-block","marginRight":"12px"}
            ),
            html.Span("Gx:", style={"marginRight":"6px"}),
            dcc.Input(id="grid-gx", type="number", value=2, min=1, step=1,
                      style={"width":"100px", "marginRight":"12px"}),
            html.Span("Gy:", style={"marginRight":"6px"}),
            dcc.Input(id="grid-gy", type="number", value=2, min=1, step=1,
                      style={"width":"100px", "marginRight":"12px"}),
            html.Span("K:", style={"marginRight":"6px"}),
            dcc.Input(id="kmeans-k", type="number", value=200, min=1, step=1,
                      style={"width":"100px"})
        ], style={"flex":"1"}),

        html.Div([
            html.Button("Add Layer", id="add-layer", n_clicks=0,
                        style={"display":"block", "width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"space-between", "alignItems":"center", "margin":"6px 10px"}),

    # 行3：GBP 参数 + GBP Solver
    html.Div([
        html.Div([
            html.Span("prior σ:", style={"marginRight":"6px"}),
            dcc.Input(id="prior-noise", type="number", value=1.0, step=0.1,
                      style={"width":"100px", "marginRight":"12px"}),

            html.Span("odom σ:", style={"marginRight":"6px"}),
            dcc.Input(id="odom-noise", type="number", value=1.0, step=0.1,
                      style={"width":"100px", "marginRight":"12px"}),

            html.Span("iters:", style={"marginRight":"6px"}),
            dcc.Input(id="param-iters", type="number", value=500, step=1,
                      style={"width":"100px", "marginRight":"12px"}),

            html.Span("snap:", style={"marginRight":"6px"}),
            dcc.Input(id="param-snap", type="number", value=1, step=1,
                      style={"width":"100px"})
        ], style={"flex":"1"}),

        html.Div([
            html.Button("GBP Solver", id="gbp-run", n_clicks=0,
                        style={"display":"block", "background":"#111","color":"#fff","width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"space-between", "alignItems":"center", "margin":"6px 10px 10px"}),

    # 下面保持不变
    dcc.Dropdown(
        id="layer-select",
        options=[{"label": "base", "value": "base"}],
        value="base", clearable=False, style={"width": "300px", "margin":"6px 10px"}
    ),

    html.Div(id="gbp-status", style={"margin":"6px 10px", "fontStyle":"italic", "color":"#444"}),

    dcc.Store(id="gbp-state", data={"running": False, "iters_done": 0, "iters_total": 0, "snap_int": 5}),
    dcc.Store(id="gbp-poses", data=None),
    dcc.Interval(id="gbp-interval", interval=200, n_intervals=0, disabled=True),

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

    prevent_initial_call=True
)
def manage_layers(add_clicks, new_clicks, mode, gx, gy, kk, current_value,
                  pN, pStep, pProb, pRadius, pPrior, pOdom, pPriorProp):
    global layers, pair_idx, gbp_graph
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered == "new-graph":
        N = int(pN or 100)
        step = float(pStep or 25)
        prob = float(pProb or 0.05)
        radius = float(pRadius or 50)
        prior_prop=float(pPriorProp or 0.0)
        layers = init_layers(N, step, prob, radius, prior_prop)
        pair_idx = 0
        reset_global_bounds(layers[0]["nodes"])
        # 构建 GBP 图（此时渲染仍显示 GT）
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
            layers.append({"name":f"abs{k}", "nodes":abs_nodes, "edges":abs_edges})
        else:
            k_next = pair_idx + 1
            super_layer_idx = k_next*2 - 1
            if mode == "grid":
                super_nodes, super_edges, node_map = fuse_to_super_grid(last["nodes"], last["edges"], int(gx or 2), int(gy or 2), super_layer_idx)
            else:
                super_nodes, super_edges, node_map = fuse_to_super_knn(last["nodes"], last["edges"], int(kk or 8), super_layer_idx)
            layers.append({"name":f"super{k_next}", "nodes":super_nodes, "edges":super_edges, "node_map":node_map})
            layers[super_layer_idx]["graph"] = build_super_graph(layers)

            pair_idx = k_next

    opts=[{"label":L["name"],"value":L["name"]} for L in layers]
    return opts, layers[-1]["name"]

# -----------------------
# 渲染 Cytoscape（含实时覆盖 base 层位姿）
# -----------------------
@app.callback(
    Output("cytoscape","elements"),
    Output("cytoscape","stylesheet"),
    Output("cytoscape","layout"),
    Input("layer-select","value"),
    Input("layer-select","options"),
    Input("gbp-poses","data"),
)
def update_layer(layer_name, _options, gbp_poses):
    layer = next(l for l in layers if l["name"] == layer_name)
    nodes, edges = layer["nodes"], layer["edges"]
    edges = [e for e in edges if e["data"].get("target") != "prior"]

    # 如果是 base 层，并且有 GBP 结果，覆盖坐标
    if layer_name == "base" and gbp_poses:
        m = min(len(nodes), len(gbp_poses))
        for i in range(m):
            nodes[i]["position"]["x"] = float(gbp_poses[i][0])
            nodes[i]["position"]["y"] = float(gbp_poses[i][1])

    axis_nodes = [
        {"data":{"id":"x_axis_start"},"position":{"x":GLOBAL_XMIN_ADJ-AXIS_PAD,"y":0},"classes":"axis-node"},
        {"data":{"id":"x_axis_end"},"position":{"x":GLOBAL_XMAX_ADJ+AXIS_PAD,"y":0},"classes":"axis-node"},
        {"data":{"id":"y_axis_start"},"position":{"x":0,"y":GLOBAL_YMIN_ADJ-AXIS_PAD},"classes":"axis-node"},
        {"data":{"id":"y_axis_end"},"position":{"x":0,"y":GLOBAL_YMAX_ADJ+AXIS_PAD},"classes":"axis-node"},
    ]
    axis_edges = [
        {"data":{"source":"x_axis_start","target":"x_axis_end"},"classes":"axis"},
        {"data":{"source":"y_axis_start","target":"y_axis_end"},"classes":"axis"},
    ]
    elements = nodes + edges + axis_nodes + axis_edges

    min_dim, min_size = 2, 3
    base_count = len(layers[0]["nodes"]) if layers else 1
    max_dim, max_size = max(2*base_count, min_dim+1), 10
    max_layer_idx = max(n["data"]["layer"] for L in layers for n in L["nodes"]) or 1

    stylesheet = [
        {"selector":"node","style":{
            "shape":"ellipse",
            "width":f"mapData(dim,{min_dim},{max_dim},{min_size},{max_size})",
            "height":f"mapData(dim,{min_dim},{max_dim},{min_size},{max_size})",
            "background-color":f"mapData(layer,0,{max_layer_idx},white,black)",
            "label":"",
            "border-width":1,"border-color":"black"}},
        {"selector":"edge","style":{"line-color":"#888","width":1}},
        {"selector":".axis","style":{
            "line-color":"black","width":1,"target-arrow-shape":"triangle",
            "arrow-scale":1,"curve-style":"straight"}},
        {"selector":".axis-node","style":{
            "width":1,"height":1,"background-color":"white","border-width":0,"opacity":0.0}},
    ]
    layout = {"name":"preset"}
    return elements, stylesheet, layout

# -----------------------
# 合并的 GBP 回调（按钮 + interval）
# -----------------------
# -----------------------
# 合并的 GBP 回调（按钮 + interval + new-graph 急停）
# -----------------------
from dash import no_update

@app.callback(
    Output("gbp-poses", "data"),
    Output("gbp-status", "children"),
    Output("gbp-state", "data"),
    Output("gbp-interval", "disabled"),
    Output("gbp-interval", "n_intervals"),
    Input("gbp-run", "n_clicks"),
    Input("gbp-interval", "n_intervals"),
    Input("new-graph", "n_clicks"),           # ← 新增：监听 New Graph
    State("gbp-state", "data"),
    State("param-iters","value"),
    State("param-snap","value"),
    prevent_initial_call=True
)

def gbp_unified(run_clicks, interval_ticks, new_graph_clicks, state, iters, snap_int):
    global gbp_graph
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, True, no_update

    trig = ctx.triggered[0]["prop_id"]

    # 1) New Graph 触发：立刻急停并复位可视化相关状态
    if trig.startswith("new-graph"):
        # 可选：如果你有 gbp_snapshots 全局，就清一下
        # global gbp_snapshots
        # gbp_snapshots = []

        # 复位 state；保留 snap_int（用户下次点 GBP 时还能用）
        snap_keep = (state or {}).get("snap_int", 5)
        reset_state = {"running": False, "iters_done": 0, "iters_total": 0, "snap_int": snap_keep}

        # 关闭 interval，清空 gbp-poses，这样 Cytoscape 会按 GT 显示
        return None, "Ready. New graph created and previous solver stopped.", reset_state, True, 0

    # 2) 点按钮：初始化并启动 interval
    if trig.startswith("gbp-run"):                                         
        if gbp_graph is None:
            return no_update, "No factor graph yet. Click New Graph first.", no_update, True, no_update
        iters = int(iters or 50)
        snap_int = int(snap_int or 5)
        state = {"running": True, "iters_done": 0, "iters_total": iters, "snap_int": snap_int}
        status = f"GBP running... 0 / {iters}"
        return no_update, status, state, False, 0  # 打开 interval & 重置 tick 计数

    # 3) interval 驱动：做一批迭代
    if not state or not state.get("running") or gbp_graph is None:
        # 没在跑，保险地把 interval 关掉
        return no_update, no_update, state, True, no_update

    iters_done  = int(state["iters_done"])
    iters_total = int(state["iters_total"])
    snap_int    = int(state["snap_int"])

    remaining = iters_total - iters_done
    batch = max(1, min(snap_int, remaining))

    for _ in range(batch):
        gbp_graph.synchronous_iteration()

    latest_positions = [v.mu.copy().tolist() for v in gbp_graph.var_nodes]

    iters_done += batch
    state["iters_done"] = iters_done
    finished = (iters_done >= iters_total)
    state["running"] = (not finished)

    # 已经在 gbp.py 里加了 energy_map，这里可以显示能量
    e = getattr(gbp_graph, "energy_map")(include_priors=True, include_factors=True)
    energy_txt = f", energy {e:.6f}"


    status = (f"GBP running... {iters_done} / {iters_total}{energy_txt}"
              if not finished
              else f"GBP finished {iters_total} iterations{energy_txt}.")

    # 完成就关 interval
    return latest_positions, status, state, finished, no_update


# -----------------------
if __name__=="__main__":
    app.run(debug=True, port=8050)
