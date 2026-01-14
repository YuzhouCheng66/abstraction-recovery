import re
import numpy as np
from scipy.linalg import block_diag
from collections import defaultdict
from copy import deepcopy

# ==== GBP import ====
from gbp.gbp import *

def make_slam_like_graph(N=100, step_size=25, loop_prob=0.05, loop_radius=50, prior_prop=0.0, seed=None):
    if seed is None :
        rng = np.random.default_rng()  # ✅ Ensure we have an RNG
    else:
        rng = np.random.default_rng(seed)
    nodes, edges = [], []
    positions = []
    x, y = 0.0, 0.0
    positions.append((x, y))

    # ✅ Deterministic-by-RNG: trajectory generation
    for _ in range(1, int(N)):
        dx, dy = rng.standard_normal(2)  # replace np.random.randn
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        dx, dy = dx / norm * float(step_size), dy / norm * float(step_size)
        x, y = x + dx, y + dy
        positions.append((x, y))

    # Sequential edges along the path
    for i, (px, py) in enumerate(positions):
        nodes.append({
            "data": {"id": f"{i}", "layer": 0, "dim": 2, "num_base": 1},
            "position": {"x": float(px), "y": float(py)}
        })

    for i in range(int(N) - 1):
        edges.append({"data": {"source": f"{i}", "target": f"{i+1}"}})

    # ✅ Deterministic-by-RNG: loop-closure edges
    for i in range(int(N)):
        for j in range(i + 5, int(N)):
            if rng.random() < float(loop_prob):  # replace np.random.rand
                xi, yi = positions[i]
                xj, yj = positions[j]
                if np.hypot(xi - xj, yi - yj) < float(loop_radius):
                    edges.append({"data": {"source": f"{i}", "target": f"{j}"}})

    # ✅ Sample priors using the same RNG
    if prior_prop <= 0.0:
        strong_ids = {0}
    elif prior_prop >= 1.0:
        strong_ids = set(range(N))
    else:
        k = max(1, int(np.floor(prior_prop * N)))
        strong_ids = set(rng.choice(N, size=k, replace=False).tolist())

    # Add edges for nodes with strong priors
    for i in strong_ids:
        edges.append({"data": {"source": f"{i}", "target": "prior"}})

    edges.append({"data": {"source": f"{0}", "target": "anchor"}}) 
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

        if (v != "prior") and (v != "anchor"):
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

        else:
            su = node_map[u]
            eid = tuple(sorted((su, "prior")))
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
        if (v != "prior") and (v != "anchor"):
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
    if rem > 0:
        sizes = [k]*(base) + [rem]     # Tail absorbs remainder: ..., last += rem
    else:
        sizes = [k]*(base)

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

        if (v != "prior") and (v != "anchor"):
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
def init_layers(N=100, step_size=25, loop_prob=0.05, loop_radius=50, prior_prop=0.0, seed=None):
    base_nodes, base_edges = make_slam_like_graph(N, step_size, loop_prob, loop_radius, prior_prop, seed)
    return [{"name": "base", "nodes": base_nodes, "edges": base_edges}]


# ==== Blobal Status ====
layers = init_layers()
pair_idx = 0
gbp_graph = None

# -----------------------
# GBP Graph Construction
# -----------------------
def build_noisy_pose_graph(
    nodes,
    edges,
    prior_sigma: float = 10,
    odom_sigma: float = 10,
    tiny_prior: float = 1e-12,
    seed=None,
):
    
    """
    Construct a 2D pose-only factor graph (linear, Gaussian) and inject noise.
    Parameters:
      prior_sigma : standard deviation of the strong prior (smaller = stronger)
      odom_sigma  : standard deviation of odometry measurement noise
      prior_prop  : 0.0 = anchor only; (0,1) = randomly select by proportion; >=1.0 = all
      tiny_prior  : a tiny prior added to all nodes to prevent singularity
      seed        : random seed (for reproducibility)
    """

    fg = FactorGraph(nonlinear_factors=False, eta_damping=0)

    var_nodes = []
    I2 = np.eye(2, dtype=float)
    N = len(nodes)

    # ---- Pre-generate noise ----
    prior_noises = {}
    odom_noises = {}

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    # Generate noise for all edges
    for e in edges:
        src = e["data"]["source"]; dst = e["data"]["target"]
        # Binary edge
        if (dst != "prior") and (dst != "anchor"):
            odom_noises[(int(src[:]), int(dst[:]))] = rng.normal(0.0, odom_sigma, size=2)
        # Unary edge (strong prior)
        elif dst == "prior":
            prior_noises[int(src[:])] = rng.normal(0.0, prior_sigma, size=2)


    # ---- variable nodes ----
    for i, n in enumerate(nodes):
        v = VariableNode(i, dofs=2)
        v.GT = np.array([n["position"]["x"], n["position"]["y"]], dtype=float)

        # Tiny prior
        v.prior.lam = tiny_prior * I2
        v.prior.eta = np.zeros(2, dtype=float)

        var_nodes.append(v)

    fg.var_nodes = var_nodes
    fg.n_var_nodes = len(var_nodes)


    # ---- prior factors ----
    def meas_fn_unary(x, *args):
        return [x]
    def jac_fn_unary(x, *args):
        return [np.eye(2)]
    # ---- odometry factors ----
    def meas_fn(xy, *args):
        return [xy[2:] - xy[:2]]
    def jac_fn(xy, *args):
        return [np.array([[-1, 0, 1, 0],
                         [ 0,-1, 0, 1]], dtype=float)]
    
    factors = []
    fid = 0

    for e in edges:
        src = e["data"]["source"]; dst = e["data"]["target"]
        if (dst != "prior") and (dst != "anchor"):
            i, j = int(src[:]), int(dst[:])
            vi, vj = var_nodes[i], var_nodes[j]

            meas = (vj.GT - vi.GT) + odom_noises[(i, j)]

            meas_lambda = np.eye(len(meas))/ (odom_sigma**2)
            f = Factor(fid, [vi, vj], [meas], [meas_lambda], meas_fn, jac_fn)
            f.type = "base"
            linpoint = np.r_[vi.GT, vj.GT]
            f.compute_factor(linpoint=linpoint, update_self=True)

            factors.append(f)
            vi.adj_factors.append(f)
            vj.adj_factors.append(f)
            fid += 1

        elif dst == "prior":
            i = int(src[:])
            vi = var_nodes[i]
            z = vi.GT + prior_noises[i]

            z_lambda = np.eye(len(meas))/ (prior_sigma**2)
            f = Factor(fid, [vi], [z], [z_lambda], meas_fn_unary, jac_fn_unary)
            f.type = "prior"
            f.compute_factor(linpoint=z, update_self=True)

            factors.append(f)
            vi.adj_factors.append(f)
            fid += 1

    # anchor for initial position
    v0 = var_nodes[0]
    z = v0.GT

    z_lambda = np.eye(len(meas))/ ((1e-4)**2)
    f = Factor(fid, [v0], [z], [z_lambda], meas_fn_unary, jac_fn_unary)
    f.type = "prior"
    f.compute_factor(linpoint=z, update_self=True)

    factors.append(f)
    v0.adj_factors.append(f)
    fid += 1

    fg.factors = factors
    fg.n_factor_nodes = len(factors)
    return fg


from collections import defaultdict
from scipy.linalg import block_diag

def build_super_graph(layers, eta_damping=0.4):
    """
    首次构建 super graph，固定结构 & closure。
    之后不要再重建，用 bottom_up_modify_super_graph 只更新数值。
    """
    # ---------- Extract base & super ----------
    base_graph  = layers[-2]["graph"]
    super_nodes = layers[-1]["nodes"]
    super_edges = layers[-1]["edges"]
    node_map    = layers[-1]["node_map"]   # 'bN' -> 'sX_...'

    # base: id(int) -> VariableNode
    id2var = {vn.variableID: vn for vn in base_graph.var_nodes}

    # ---------- super_id -> [base_id(int)] ----------
    super_groups = {}
    for b_str, s_id in node_map.items():
        b_int = int(b_str)
        super_groups.setdefault(s_id, []).append(b_int)

    # ---------- 对每个 super group 建 (start, dofs) 表 ----------
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

    # ============ 预计算：哪些 base factors 属于哪个 super group / pair ============

    def precompute_super_factor_maps(base_graph, node_map, super_groups):
        # in_group：super_id -> [fid, ...]
        group2factors_allin = { sid: [] for sid in super_groups }

        # cross： (sidA, sidB) (ordered) -> [fid, ...]
        pairgroups2factors  = defaultdict(list)

        for fid, f in enumerate(base_graph.factors):
            vids = [v.variableID for v in f.adj_var_nodes]

            # unary factor
            if len(vids) == 1:
                bid = vids[0]
                sid = node_map[str(bid)]
                group2factors_allin[sid].append(fid)
                continue

            # binary factor
            if len(vids) == 2:
                i, j = vids
                si, sj = node_map[str(i)], node_map[str(j)]
                if si == sj:
                    group2factors_allin[si].append(fid)
                else:
                    key = (si, sj) if si < sj else (sj, si)
                    pairgroups2factors[key].append(fid)
                continue

            # 更高阶 factor 需要时再扩展
        return group2factors_allin, pairgroups2factors

    group2factors_allin, pairgroups2factors = precompute_super_factor_maps(
        base_graph, node_map, super_groups
    )

    # 把这些结构性信息存到 layer 里，方便后续 in-place 更新使用
    layer = layers[-1]
    layer["super_groups"]          = super_groups
    layer["local_idx"]             = local_idx
    layer["total_dofs"]            = total_dofs
    layer["group2factors_allin"]   = group2factors_allin
    layer["pairgroups2factors"]    = pairgroups2factors

    # ---------- 创建 super VariableNodes（只做一次） ----------
    fg = FactorGraph(nonlinear_factors=False, eta_damping=eta_damping)

    super_var_nodes = {}   # sid(str) -> VariableNode
    for i, sn in enumerate(super_nodes):
        sid  = sn["data"]["id"]
        dofs = total_dofs.get(sid, 0)

        v = VariableNode(i, dofs=dofs)
        gt_vec = np.zeros(dofs)
        mu_blocks    = []
        Sigma_blocks = []

        for bid, (st, d) in local_idx[sid].items():
            # Stack base GT
            gt_base = getattr(id2var[bid], "GT", None)
            if gt_base is None or len(gt_base) != d:
                gt_base = np.zeros(d)
            gt_vec[st:st+d] = gt_base

            # Stack base belief
            vb = id2var[bid]
            mu_blocks.append(vb.mu)
            Sigma_blocks.append(vb.Sigma)

        super_var_nodes[sid] = v
        v.GT = gt_vec

        mu_super    = np.concatenate(mu_blocks) if mu_blocks else np.zeros(dofs)
        Sigma_super = block_diag(*Sigma_blocks) if Sigma_blocks else np.eye(dofs)
        lam = np.linalg.inv(Sigma_super)
        eta = lam @ mu_super
        v.mu     = mu_super
        v.Sigma  = Sigma_super
        v.belief = NdimGaussian(dofs, eta, lam)
        v.prior.lam = 1e-12 * lam
        v.prior.eta = 1e-12 * eta

        fg.var_nodes.append(v)

    fg.n_var_nodes = len(fg.var_nodes)

    # ---------- Utility: 组装一个 super group 的 linpoint（用 base belief means） ----------
    def make_linpoint_for_group(sid):
        x = np.zeros(total_dofs[sid])
        for bid, (st, d) in local_idx[sid].items():
            mu = getattr(id2var[bid], "mu", None)
            if mu is None or len(mu) != d:
                mu = np.zeros(d)
            x[st:st+d] = mu
        return x

    # 把这些辅助对象也挂到 layer 上，后面可以复用
    layer["super_var_nodes"]         = super_var_nodes
    layer["make_linpoint_for_group"] = make_linpoint_for_group

    # ---------- 3) super prior（in-group unary + in-group binary） ----------
    def make_super_prior_factor(sid):
        idx_map = local_idx[sid]
        ncols   = total_dofs[sid]

        factor_ids = group2factors_allin[sid]
        in_group   = [base_graph.factors[fid] for fid in factor_ids]

        def meas_fn_super_prior(x_super, *args):
            meas_fn = []
            for f in in_group:
                vids = [v.variableID for v in f.adj_var_nodes]

                x_loc_list = []
                for vid in vids:
                    st, d = idx_map[vid]
                    x_loc_list.append(x_super[st:st+d])
                x_loc = np.concatenate(x_loc_list) if x_loc_list else np.zeros(0)
                meas_fn.extend(f.meas_fn(x_loc))
            return meas_fn if meas_fn else np.zeros(0)

        def jac_fn_super_prior(x_super, *args):
            Jrows = []
            for f in in_group:
                vids = [v.variableID for v in f.adj_var_nodes]

                x_loc_list = []
                dims = []
                for vid in vids:
                    st, d = idx_map[vid]
                    dims.append(d)
                    x_loc_list.append(x_super[st:st+d])
                x_loc = np.concatenate(x_loc_list) if x_loc_list else np.zeros(0)

                Jloc  = f.jac_fn(x_loc)
                lens  = [J.shape[0] for J in Jloc]
                Jlocs = np.vstack(Jloc)

                rows = np.zeros((Jlocs.shape[0], ncols))
                c0 = 0
                for vid, d in zip(vids, dims):
                    st, _ = idx_map[vid]
                    rows[:, st:st+d] = Jlocs[:, c0:c0+d]
                    c0 += d
                cuts = np.cumsum(lens)[:-1]
                rows = np.split(rows, cuts, axis=0)
                Jrows.extend(rows)
            return Jrows if Jrows else np.zeros((0, ncols))

        z_super, z_super_lambda = [], []
        ext_z, ext_l = z_super.extend, z_super_lambda.extend
        for f in in_group:
            ext_z(f.measurement)
            ext_l(f.measurement_lambda)

        return meas_fn_super_prior, jac_fn_super_prior, z_super, z_super_lambda

    # ---------- 4) super between（cross-group binary） ----------
    def make_super_between_factor(sidA, sidB):
        groupA, groupB = super_groups[sidA], super_groups[sidB]
        idxA, idxB     = local_idx[sidA], local_idx[sidB]
        nA, nB         = total_dofs[sidA], total_dofs[sidB]

        key = (sidA, sidB) if sidA < sidB else (sidB, sidA)
        factor_ids = pairgroups2factors.get(key, [])
        cross = [base_graph.factors[fid] for fid in factor_ids]

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
                meas_fn.extend(f.meas_fn(x_loc))
            return meas_fn

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

                lens  = [J.shape[0] for J in Jloc]
                cuts  = np.cumsum(lens)[:-1]
                Jlocs = np.vstack(Jloc)
                rows  = np.zeros((Jlocs.shape[0], nA + nB))
                rows[:, left_start:left_start+di]  = Jlocs[:, :di]
                rows[:, right_start:right_start+dj] = Jlocs[:, di:di+dj]
                rows = np.split(rows, cuts, axis=0)
                Jrows.extend(rows)
            return Jrows

        z_super, z_super_lambda = [], []
        ext_z, ext_l = z_super.extend, z_super_lambda.extend
        for f in cross:
            ext_z(f.measurement)
            ext_l(f.measurement_lambda)

        return meas_fn_super_between, jac_fn_super_between, z_super, z_super_lambda

    # ---------- 5) 按 super_edges 建所有 factors（只做一次） ----------
    for e in super_edges:
        u, v = e["data"]["source"], e["data"]["target"]

        if v == "prior":
            meas_fn, jac_fn, z, z_lambda = make_super_prior_factor(u)
            f = Factor(len(fg.factors), [super_var_nodes[u]], z, z_lambda, meas_fn, jac_fn)
            f.adj_beliefs = [vn.belief for vn in f.adj_var_nodes]
            f.type = "super_prior"
            lin0 = make_linpoint_for_group(u)
            f.compute_factor(linpoint=lin0, update_self=True)
            fg.factors.append(f)
            super_var_nodes[u].adj_factors.append(f)
        else:
            meas_fn, jac_fn, z, z_lambda = make_super_between_factor(u, v)
            f = Factor(len(fg.factors), [super_var_nodes[u], super_var_nodes[v]], z, z_lambda, meas_fn, jac_fn)
            f.adj_beliefs = [vn.belief for vn in f.adj_var_nodes]
            f.type = "super_between"
            lin0 = np.concatenate([make_linpoint_for_group(u), make_linpoint_for_group(v)])
            f.compute_factor(linpoint=lin0, update_self=True)
            fg.factors.append(f)
            super_var_nodes[u].adj_factors.append(f)
            super_var_nodes[v].adj_factors.append(f)

    fg.n_factor_nodes = len(fg.factors)

    # 记得在外面：layers[-1]["graph"] = fg
    return fg


def build_abs_graph(
    layers,
    r_reduced = 2,
    eta_damping=0.4):

    abs_var_nodes = {}
    Bs = {}
    ks = {}
    k2s = {}

    # === 1. Build Abstraction Variables ===
    abs_fg = FactorGraph(nonlinear_factors=False, eta_damping=eta_damping)
    sup_fg = layers[-2]["graph"]

    for sn in sup_fg.var_nodes:
        if sn.dofs <= r_reduced:
            r = sn.dofs
        else:
            r = r_reduced

        sid = sn.variableID
        varis_sup_mu    = sn.mu
        varis_sup_sigma = sn.Sigma

        # eig 分解
        eigvals, eigvecs = np.linalg.eigh(varis_sup_sigma)
        idx     = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # B_reduced: (d_sup, r)
        B_reduced = eigvecs[:, :r]
        Bs[sid]   = B_reduced

        # 投影
        varis_abs_mu    = B_reduced.T @ varis_sup_mu
        varis_abs_sigma = B_reduced.T @ varis_sup_sigma @ B_reduced
        ks[sid]         = varis_sup_mu - B_reduced @ varis_abs_mu

        varis_abs_lam = np.linalg.inv(varis_abs_sigma)
        varis_abs_eta = varis_abs_lam @ varis_abs_mu

        v = VariableNode(sid, dofs=r)
        v.GT     = sn.GT
        v.mu     = varis_abs_mu
        v.Sigma  = varis_abs_sigma
        v.belief = NdimGaussian(r, varis_abs_eta, varis_abs_lam)

        abs_var_nodes[sid] = v
        abs_fg.var_nodes.append(v)

    abs_fg.n_var_nodes = len(abs_fg.var_nodes)

    # === 2. Abstract Prior ===
    def make_abs_prior_factor(sup_factor):
        sid = sup_factor.adj_var_nodes[0].variableID  # super 变量 ID

        def meas_fn_abs_prior(x_abs, *args):
            B = Bs[sid]
            k = ks[sid]
            return sup_factor.meas_fn(B @ x_abs + k)
        
        def jac_fn_abs_prior(x_abs, *args):
            B = Bs[sid]
            k = ks[sid]
            Jloc = sup_factor.jac_fn(B @ x_abs + k)
            lens = [J.shape[0] for J in Jloc]
            cuts = np.cumsum(lens)[:-1]
            return np.split(np.vstack(Jloc) @ B, cuts, axis=0)

        return meas_fn_abs_prior, jac_fn_abs_prior, sup_factor.measurement, sup_factor.measurement_lambda

    # === 3. Abstract Between ===
    def make_abs_between_factor(sup_factor):
        vids = [v.variableID for v in sup_factor.adj_var_nodes]
        i, j = vids
        ni   = abs_var_nodes[i].dofs

        def meas_fn_abs_between(xij, *args):
            xi, xj = xij[:ni], xij[ni:]
            Bi, Bj = Bs[i], Bs[j]
            ki, kj = ks[i], ks[j]
            return sup_factor.meas_fn(np.concatenate([Bi @ xi + ki, Bj @ xj + kj]))

        def jac_fn_abs_between(xij, *args):
            xi, xj = xij[:ni], xij[ni:]
            Bi, Bj = Bs[i], Bs[j]
            ki, kj = ks[i], ks[j]

            J_sup = sup_factor.jac_fn(np.concatenate([Bi @ xi + ki, Bj @ xj + kj]))
            lens  = [J.shape[0] for J in J_sup]
            cuts  = np.cumsum(lens)[:-1]
            J_sup = np.vstack(J_sup)

            J_abs = np.zeros((J_sup.shape[0], ni + xj.shape[0]))
            J_abs[:, :ni] = J_sup[:, :Bi.shape[0]] @ Bi
            J_abs[:, ni:] = J_sup[:, Bi.shape[0]:] @ Bj
            return np.split(J_abs, cuts, axis=0)
        
        return meas_fn_abs_between, jac_fn_abs_between, sup_factor.measurement, sup_factor.measurement_lambda

    # === 4. 构建所有 abs factors（只做一次） ===
    for f in sup_fg.factors:
        if len(f.adj_var_nodes) == 1:
            meas_fn, jac_fn, z, z_lambda = make_abs_prior_factor(f)
            v_id = f.adj_var_nodes[0].variableID
            v    = abs_var_nodes[v_id]

            abs_f = Factor(f.factorID, [v], z, z_lambda, meas_fn, jac_fn)
            abs_f.type = "abs_prior"
            abs_f.adj_beliefs = [v.belief]

            lin0 = v.mu
            abs_f.compute_factor(linpoint=lin0, update_self=True)

            abs_fg.factors.append(abs_f)
            v.adj_factors.append(abs_f)

        elif len(f.adj_var_nodes) == 2:
            meas_fn, jac_fn, z, z_lambda = make_abs_between_factor(f)
            i, j   = [v.variableID for v in f.adj_var_nodes]
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


def bottom_up_modify_abs_graph(
    layers,
    r_reduced = 2,
    eta_damping=0.4):
    """
    In-place 更新 abs 层：
      - 更新 Bs, ks
      - 更新 abs_fg.var_nodes 的 mu, Sigma, belief
      - 如需要，对 abs_fg.factors 重新 compute_factor
    不再 new FactorGraph / new Factor。
    """

    abs_layer = layers[-1]
    abs_fg    = abs_layer["graph"]
    abs_fg.eta_damping = eta_damping
    sup_fg    = layers[-2]["graph"]

    # 复用原有的 Bs/ks（如果你也想更新 B，可以在这里重新算）
    Bs = abs_layer["Bs"]
    ks = abs_layer["ks"]

    # === 1. 更新每个 abs 变量的投影 ===
    for sn in sup_fg.var_nodes:
        if sn.dofs <= r_reduced:
            r = sn.dofs
        else:
            r = r_reduced

        sid = sn.variableID
        varis_sup_mu    = sn.mu
        varis_sup_sigma = sn.Sigma

        # 这里有两种选择：
        #  A) 子空间固定：使用旧的 B（推荐，便宜）
        B_reduced = Bs[sid]

        #  B) 子空间随 covariance 变化：每次重新 eig（贵很多）
        # eigvals, eigvecs = np.linalg.eigh(varis_sup_sigma)
        # idx     = np.argsort(eigvals)[::-1]
        # eigvecs = eigvecs[:, idx]
        # B_reduced = eigvecs[:, :r]
        # Bs[sid]   = B_reduced

        varis_abs_mu    = B_reduced.T @ varis_sup_mu
        varis_abs_sigma = B_reduced.T @ varis_sup_sigma @ B_reduced
        ks[sid]         = varis_sup_mu - B_reduced @ varis_abs_mu

        varis_abs_lam = np.linalg.inv(varis_abs_sigma)
        varis_abs_eta = varis_abs_lam @ varis_abs_mu

        v = abs_fg.var_nodes[sid]   # 结构不变，直接拿旧节点
        v.mu     = varis_abs_mu
        v.Sigma  = varis_abs_sigma
        v.belief = NdimGaussian(r, varis_abs_eta, varis_abs_lam)

    abs_fg.n_var_nodes = len(abs_fg.var_nodes)

    # === 2. 如需重新线性化 abs factors（可选但通常是需要的） ===
    for f in abs_fg.factors:
        lin0 = np.concatenate([v.mu for v in f.adj_var_nodes])
        f.compute_factor(linpoint=lin0, update_self=True)

    # Bs, ks 已在 dict 中更新，closure 会自动看到新值
    abs_layer["Bs"] = Bs
    abs_layer["ks"] = ks
    # abs_layer["k2s"] = k2s
    return 

def bottom_up_modify_super_graph(layers, eta_damping=0.4):
    """
    In-place 更新 super graph：
      - 从 base_graph 更新 super var 的 mu / Sigma / belief
      - 不再重建 FactorGraph / Factor / VariableNode。
      - 不再对 big Σ 做 np.linalg.inv，而是复用 base belief 的 (η, Λ) 做 block_diag。
    """
    base_graph = layers[-2]["graph"]
    layer      = layers[-1]

    super_nodes      = layer["nodes"]
    node_map         = layer["node_map"]
    super_fg         = layer["graph"]
    super_groups     = layer["super_groups"]
    local_idx        = layer["local_idx"]
    total_dofs       = layer["total_dofs"]
    super_var_nodes  = layer["super_var_nodes"]   # sid(str) -> VariableNode

    # ---------- 0) 构建 id2var ----------
    id2var = {vn.variableID: vn for vn in base_graph.var_nodes}

    # ---------- 1) 更新每个 super variable 的 mu / Sigma / belief ----------
    for sn in super_nodes:
        sid  = sn["data"]["id"]
        v    = super_var_nodes[sid]
        dofs = total_dofs[sid]

        gt_vec = np.zeros(dofs)

        mu_blocks    = []
        Sigma_blocks = []
        lam_blocks   = []
        eta_blocks   = []

        # ---- 1.1 收集 base 节点信息 ----
        for bid, (st, d) in local_idx[sid].items():
            vb = id2var[bid]

            # GT
            gt_base = getattr(vb, "GT", None)
            if gt_base is None or len(gt_base) != d:
                gt_base = np.zeros(d)
            gt_vec[st:st+d] = gt_base

            # belief / covariance
            mu_blocks.append(vb.mu)           # shape (d,)
            Sigma_blocks.append(vb.Sigma)     # shape (d,d)

            # 直接复用 base 的 η, Λ，避免在 super 层重新求逆
            lam_blocks.append(vb.belief.lam)  # shape (d,d)
            eta_blocks.append(vb.belief.eta)  # shape (d,)
        v.GT = gt_vec

        # ---- 1.2 block_diag + concat ----
        mu_super    = np.concatenate(mu_blocks)      if mu_blocks    else np.zeros(dofs)
        Sigma_super = block_diag(*Sigma_blocks)      if Sigma_blocks else np.eye(dofs)
        lam_super   = block_diag(*lam_blocks)        if lam_blocks   else np.eye(dofs)
        eta_super   = np.concatenate(eta_blocks)     if eta_blocks   else np.zeros(dofs)

        # ---- 1.3 更新 belief / prior（不再有 inv + Λμ）----
        v.mu     = mu_super
        v.Sigma  = Sigma_super
        v.belief = NdimGaussian(dofs, eta_super, lam_super)
        v.prior.lam = 1e-12 * lam_super
        v.prior.eta = 1e-12 * eta_super

    super_fg.n_var_nodes = len(super_fg.var_nodes)

    # ---------- 2) 重新线性化 super factors（目前还是关闭） ----------
    for f in super_fg.factors:
        lin0 = np.concatenate([v.mu for v in f.adj_var_nodes])
        f.compute_factor(linpoint=lin0, update_self=True)

    return super_fg



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
            v.prior = NdimGaussian(v.dofs, 1e-10*eta, 1e-10*v.belief.lam)

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
                        msg.eta += d_eta / n_adj
                        msg.lam += d_lam / n_adj
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

        old_belief = sn.belief
        # Refresh super belief (natural parameters) with the new μ and Σ
        eta = sn.belief.lam @ sn.mu
        new_belief = NdimGaussian(sn.dofs, eta, sn.belief.lam)
        sn.belief  = new_belief
        sn.prior = NdimGaussian(sn.dofs, 1e-10*eta, 1e-10*sn.belief.lam)


        # Iterate over super factors adjacent to this super variable
        if sn.adj_factors:
            n_adj = len(sn.adj_factors)
            d_eta = new_belief.eta - old_belief.eta
            d_lam = new_belief.lam - old_belief.lam

            for f in sn.adj_factors:
                if sn in f.adj_var_nodes:
                    idx = f.adj_var_nodes.index(sn)
                    # update adj_beliefs
                    f.adj_beliefs[idx] = new_belief
                    # correct coresponding message
                    msg = f.messages[idx]
                    msg.eta += d_eta / n_adj
                    msg.lam += d_lam / n_adj
                    f.messages[idx] = msg

    return

def energy_map(graph, include_priors: bool = True, include_factors: bool = True) -> float:
    """
    It is actually the sum of squares of distances.
    """
    total = 0.0

    for v in graph.var_nodes[:graph.n_var_nodes]:
        gt = np.asarray(v.GT[0:2], dtype=float)
        r = np.asarray(v.belief.mu()[0:2], dtype=float) - gt
        total += 0.5 * float(r.T @ r)

    return total

class VGraph:
    def __init__(self,
                 layers,
                 nonlinear_factors=True,
                 eta_damping=0.2,
                 r_reduced=2,
                 beta=0.0,
                 iters_since_relinear=0,
                 num_undamped_iters=0,
                 min_linear_iters=100,
                 wild_thresh=0):

        self.layers = layers
        self.iters_since_relinear = iters_since_relinear
        self.min_linear_iters = min_linear_iters
        self.nonlinear_factors = nonlinear_factors
        self.eta_damping = eta_damping
        self.r_reduced = r_reduced
        self.wild_thresh = wild_thresh

        #self.energy_history = []
        #self.error_history = []
        #self.nmsgs_history = []
        #self.mus = []


    def vloop(self):
        """
        Simplified V-cycle:
        1) bottom-up: rebuild and iterate once for base / super / abs in order
        2) top-down: propagate mu from super -> base
        3) refresh gbp_result on each layer for UI use
        """

        layers = self.layers

        # ---- bottom-up ----
        #if layers and "graph" in layers[0]:
        #    layers[0]["graph"].synchronous_iteration()
            
        for i in range(1, len(layers)):
            name = layers[i]["name"]

            if name.startswith("super1"):
                # Update super using the previous base graph's new linearization points
                pass

            elif name.startswith("super"):
                #a = time.time()
                # Update super using the previous layer's graph
                layers[i]["graph"] = bottom_up_modify_super_graph(layers[:i+1], eta_damping=self.eta_damping)
                #print(f"Bottom-up {name} build time: {time.time() - a:.4f} sec")

            elif name.startswith("abs"):
                # Rebuild abs using the previous super
                #a = time.time()
                bottom_up_modify_abs_graph(layers[:i+1], eta_damping=self.eta_damping, r_reduced=self.r_reduced)
                #print(f"Bottom-up {name} build time: {time.time() - a:.4f} sec")

            # After build, one iteration per layer
            if "graph" in layers[i]:
                #a = time.time()
                layers[i]["graph"].residual_iteration_var_heap()
                #layers[i]["graph"].synchronous_iteration()
                #print(f"Bottom-up {name} iteration time: {time.time() - a:.4f} sec")

        # ---- top-down (pass mu) ----
        for i in range(len(layers) - 1, 0, -1):
            # After one iterations per layer, reproject
            if "graph" in layers[i]:
                #a = time.time()
                layers[i]["graph"].residual_iteration_var_heap()
                #layers[i]["graph"].synchronous_iteration()
                #print(f"Top-down {layers[i]['name']} iteration time: {time.time() - a:.4f} sec")

            #if i == len(layers) - 1:
            # extra iteration for abs layer
            #    layers[i]["graph"].synchronous_iteration()
            # this is very important, but dont know why yet
            # so abs layer need more iterations
            #if name.startswith("abs"):
            #    layers[i]["graph"].synchronous_iteration()  

            name = layers[i]["name"]
            if name.startswith("super"):
                #a = time.time()
                # Split super.mu back to base/abs
                top_down_modify_base_and_abs_graph(layers[:i+1])
                #print(f"Top-down {name} to base/abs time: {time.time() - a:.4f} sec")

            elif name.startswith("abs"):
                # Project abs.mu back to super
                #a = time.time()
                top_down_modify_super_graph(layers[:i+1])
                #print(f"Top-down {name} to super time: {time.time() - a:.4f} sec")

        # ---- refresh gbp_result for UI ----
        #refresh_gbp_results(layers)
        return layers




"这里是主要的cpp demo代码部分"


N = 512
step=25
prob=0.05
radius=50 
prior_prop=0.02
prior_sigma=1
odom_sigma=1
layers = []


layers = init_layers(N=N, step_size=step, loop_prob=prob, loop_radius=radius, prior_prop=prior_prop, seed=2001)
pair_idx = 0



# Create GBP graph
gbp_graph = build_noisy_pose_graph(layers[0]["nodes"], layers[0]["edges"],
                                    prior_sigma=prior_sigma,
                                    odom_sigma=odom_sigma,
                                    seed=2001)
layers[0]["graph"] = gbp_graph
gbp_graph.num_undamped_iters = 0
gbp_graph.min_linear_iters = 2000
opts=[{"label":"base","value":"base"}]


kk = 10
k_next = 1
super_layer_idx = k_next*2 - 1
last = layers[-1]
super_nodes, super_edges, node_map = fuse_to_super_order(last["nodes"], last["edges"], int(kk or 8), super_layer_idx, tail_heavy=True)
# Ensure base graph has run at least once
layers[-1]["graph"].synchronous_iteration() 
layers.append({"name":f"super{k_next}", "nodes":super_nodes, "edges":super_edges, "node_map":node_map})
if super_layer_idx > 1:
    layers[super_layer_idx]["graph"] = build_super_graph(layers)
else:
    layers[super_layer_idx]["graph"] = build_super_graph(layers)



abs_layer_idx = 2
k = 1
last = layers[-1]
abs_nodes, abs_edges = copy_to_abs(last["nodes"], last["edges"], abs_layer_idx)
# Ensure super graph has run at least once
layers[-1]["graph"].synchronous_iteration() 
layers.append({"name":f"abs{k}", "nodes":abs_nodes, "edges":abs_edges})
layers[abs_layer_idx]["graph"], layers[abs_layer_idx]["Bs"], layers[abs_layer_idx]["ks"], layers[abs_layer_idx]["k2s"] = build_abs_graph(
    layers, r_reduced=2)


k_next = 2
super_layer_idx = k_next*2 - 1
last = layers[-1]
super_nodes, super_edges, node_map = fuse_to_super_order(last["nodes"], last["edges"], int(kk or 8), super_layer_idx, tail_heavy=True)
# Ensure super graph has run at least once
layers[-1]["graph"].synchronous_iteration() 
layers.append({"name":f"super{k_next}", "nodes":super_nodes, "edges":super_edges, "node_map":node_map})
if super_layer_idx > 1:
    layers[super_layer_idx]["graph"] = build_super_graph(layers)
else:
    layers[super_layer_idx]["graph"] = build_super_graph(layers)



abs_layer_idx = 4
k = 2
last = layers[-1]
abs_nodes, abs_edges = copy_to_abs(last["nodes"], last["edges"], abs_layer_idx)
# Ensure super graph has run at least once
layers[-1]["graph"].synchronous_iteration() 
layers.append({"name":f"abs{k}", "nodes":abs_nodes, "edges":abs_edges})
layers[abs_layer_idx]["graph"], layers[abs_layer_idx]["Bs"], layers[abs_layer_idx]["ks"], layers[abs_layer_idx]["k2s"] = build_abs_graph(
    layers, r_reduced=2)



k_next = 3
super_layer_idx = k_next*2 - 1
last = layers[-1]
super_nodes, super_edges, node_map = fuse_to_super_order(last["nodes"], last["edges"], int(kk or 8), super_layer_idx, tail_heavy=True)
# Ensure super graph has run at least once
layers[-1]["graph"].synchronous_iteration() 
layers.append({"name":f"super{k_next}", "nodes":super_nodes, "edges":super_edges, "node_map":node_map})
if super_layer_idx > 1:
    layers[super_layer_idx]["graph"] = build_super_graph(layers)
else:
    layers[super_layer_idx]["graph"] = build_super_graph(layers)


abs_layer_idx = 6
k = 3
last = layers[-1]
abs_nodes, abs_edges = copy_to_abs(last["nodes"], last["edges"], abs_layer_idx)
# Ensure super graph has run at least once
layers[-1]["graph"].synchronous_iteration() 
layers.append({"name":f"abs{k}", "nodes":abs_nodes, "edges":abs_edges})
layers[abs_layer_idx]["graph"], layers[abs_layer_idx]["Bs"], layers[abs_layer_idx]["ks"], layers[abs_layer_idx]["k2s"] = build_abs_graph(layers)



vg = VGraph(layers)
energy_prev = 0
counter = 0
for _ in range(100):
    vg.layers = layers
    vg.r_reduced=2
    vg.eta_damping = 0
    vg.layers = vg.vloop()
    energy = energy_map(layers[0]["graph"], include_priors=True, include_factors=True)
    if np.abs(energy_prev-energy) < 1e-2:
        counter += 1
        if counter >= 2:
            break
    print(f"Iter {_+1:03d} | Energy = {energy:.6f}")
    #energy_prev = energy