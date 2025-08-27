import re
import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
import numpy as np

app = dash.Dash(__name__)
app.title = "Factor Graph SVD Abs&Recovery"

# -----------------------
# SLAM-like base graph
# -----------------------
def make_slam_like_graph(N=100, step_size=25, loop_prob=0.05, loop_radius=50):
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
            "data": {"id": f"b{i}", "layer": 0, "dim": 2},
            "position": {"x": float(px), "y": float(py)}
        })
    for i in range(int(N) - 1):
        edges.append({"data": {"source": f"b{i}", "target": f"b{i+1}"}})
    for i in range(int(N)):
        for j in range(i + 5, int(N)):
            if np.random.rand() < float(loop_prob):
                xi, yi = positions[i]; xj, yj = positions[j]
                if np.hypot(xi-xj, yi-yj) < float(loop_radius):
                    edges.append({"data": {"source": f"b{i}", "target": f"b{j}"}})
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
        nid = f"s{layer_idx}_{cid}"
        super_nodes.append({
            "data": {"id": nid, "layer": layer_idx, "dim": dim_val},
            "position": {"x": float(mean_x), "y": float(mean_y)}
        })
        for i in indices:
            node_map[prev_nodes[i]["data"]["id"]] = nid
    super_edges, seen = [], set()
    for e in prev_edges:
        u, v = e["data"]["source"], e["data"]["target"]
        su, sv = node_map[u], node_map[v]
        if su != sv:
            eid = tuple(sorted((su, sv)))
            if eid not in seen:
                super_edges.append({"data": {"source": su, "target": sv}})
                seen.add(eid)
    return super_nodes, super_edges

# -----------------------
# K-Means（完全 1-NN 驱动）
# -----------------------
def fuse_to_super_knn(prev_nodes, prev_edges, k, layer_idx, max_iters=20, tol=1e-6, seed=0):
    positions = np.array([[n["position"]["x"], n["position"]["y"]] for n in prev_nodes], dtype=float)
    n = positions.shape[0]
    if k <= 0: k = 1
    k = min(k, n)

    rng = np.random.default_rng(seed)

    # k-means++ init
    centers = np.empty((k, 2), dtype=float)
    first = rng.integers(0, n)
    centers[0] = positions[first]
    closest_d2 = np.sum((positions - centers[0])**2, axis=1)
    for i in range(1, k):
        denom = float(closest_d2.sum()) + 1e-12
        probs = closest_d2 / denom
        idx = rng.choice(n, p=probs)
        centers[i] = positions[idx]
        d2_new = np.sum((positions - centers[i])**2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2_new)

    # Lloyd iterations
    for _ in range(max_iters):
        d2 = ((positions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        moved = 0.0
        for ci in range(k):
            idxs = np.where(assign == ci)[0]
            if idxs.size == 0:
                far_idx = int(np.argmax(np.min(d2, axis=1)))
                new_c = positions[far_idx]
            else:
                new_c = positions[idxs].mean(axis=0)
            moved = max(moved, float(np.linalg.norm(new_c - centers[ci])))
            centers[ci] = new_c
        if moved < tol:
            break

    # final assign
    d2 = ((positions[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = np.argmin(d2, axis=1)

    super_nodes, node_map = [], {}
    for ci in range(k):
        idxs = np.where(assign == ci)[0]
        if idxs.size == 0:
            continue
        pts = positions[idxs]
        mean_x, mean_y = pts.mean(axis=0)
        child_dims = [prev_nodes[i]["data"]["dim"] for i in idxs]
        dim_val = int(max(1, sum(child_dims)))
        nid = f"s{layer_idx}_k{ci}"
        super_nodes.append({
            "data": {"id": nid, "layer": layer_idx, "dim": dim_val},
            "position": {"x": float(mean_x), "y": float(mean_y)}
        })
        for i in idxs:
            node_map[prev_nodes[i]["data"]["id"]] = nid

    super_edges, seen = [], set()
    for e in prev_edges:
        u, v = e["data"]["source"], e["data"]["target"]
        su, sv = node_map[u], node_map[v]
        if su != sv:
            key = tuple(sorted((su, sv)))
            if key not in seen:
                super_edges.append({"data": {"source": su, "target": sv}})
                seen.add(key)

    return super_nodes, super_edges

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
# 初始化 & 边界更新
# -----------------------
def init_layers(N=100, step_size=25, loop_prob=0.05, loop_radius=50):
    base_nodes, base_edges = make_slam_like_graph(N, step_size, loop_prob, loop_radius)
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

# 初始
layers = init_layers()
pair_idx = 0
reset_global_bounds(layers[0]["nodes"])

# -----------------------
# Layout
# -----------------------
app.layout = html.Div([
    html.Div([
        # 左边输入区
        html.Div([
            # 第一行：Mode / Gx / Gy / K
            html.Div([
                html.Span("Mode:", style={"marginRight":"6px"}),
                dcc.Dropdown(
                    id="agg-mode",
                    options=[{"label":"Grid","value":"grid"},{"label":"K-Means","value":"kmeans"}],
                    value="kmeans",
                    clearable=False,
                    className="mode-dd",   # 👈 给 mode 下拉框一个专属 class
                    style={
                        "width":"120px",
                        "height":"26px",
                        "display":"inline-block",
                        "marginRight":"12px"
                    }
                )

                ,
                html.Span("Gx:", style={"marginRight":"6px"}),
                dcc.Input(id="grid-gx", type="number", value=2, min=1, step=1,
                          style={"width":"100px", "marginRight":"12px"}),
                html.Span("Gy:", style={"marginRight":"6px"}),
                dcc.Input(id="grid-gy", type="number", value=2, min=1, step=1,
                          style={"width":"100px", "marginRight":"12px"}),
                html.Span("K:", style={"marginRight":"6px"}),
                dcc.Input(id="kmeans-k", type="number", value=8, min=1, step=1,
                          style={"width":"100px"})
            ], style={"marginBottom":"8px"}),

            # 第二行：N / step_size / loop_prob / loop_radius
            html.Div([
                html.Span("N:", style={"marginRight":"6px"}),
                dcc.Input(id="param-N", type="number", value=500, min=2, step=1,
                          style={"width":"100px", "marginRight":"12px"}),
                html.Span("step:", style={"marginRight":"6px"}),
                dcc.Input(id="param-step", type="number", value=25, step=1,
                          style={"width":"100px", "marginRight":"12px"}),
                html.Span("loop_prob:", style={"marginRight":"6px"}),
                dcc.Input(id="param-prob", type="number", value=0.05, min=0, max=1, step=0.01,
                          style={"width":"100px", "marginRight":"12px"}),
                html.Span("loop_radius:", style={"marginRight":"6px"}),
                dcc.Input(id="param-radius", type="number", value=25, step=1,
                          style={"width":"100px"})
            ])
        ], style={"flex":"1"}),

        # 右边按钮区（竖排，占两行高度）
        html.Div([
            html.Button("Add Layer", id="add-layer", n_clicks=0,
                        style={"display":"block", "marginBottom":"6px", "width":"120px"}),
            html.Button("New Graph", id="new-graph", n_clicks=0,
                        style={"display":"block", "marginBottom":"6px", "width":"120px"}),
            html.Button("GBP Solver", id="gbp-run", n_clicks=0,
                        style={"display":"block", "background":"#111","color":"#fff","width":"120px"})
        ], style={"marginLeft":"20px", "flex":"0 0 auto"})
    ], style={"display":"flex", "justifyContent":"space-between", "alignItems":"flex-start", "margin":"10px"}),

    dcc.Dropdown(
        id="layer-select",
        options=[{"label": "base", "value": "base"}],
        value="base", clearable=False, style={"width": "300px", "margin":"6px 10px"}
    ),

    html.Div(id="gbp-status", style={"margin":"6px 10px", "fontStyle":"italic", "color":"#444"}),

    cyto.Cytoscape(
        id="cytoscape",
        style={"width": f"{VIEW_W}px", "height": f"{VIEW_H}px"},
        layout={"name": "preset"},
        elements=[]
    )
])


# -----------------------
# 管理 Add / New
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
    # 新增：基图参数
    State("param-N","value"),
    State("param-step","value"),
    State("param-prob","value"),
    State("param-radius","value"),
    prevent_initial_call=True
)
def manage_layers(add_clicks, new_clicks, mode, gx, gy, kk, current_value, pN, pStep, pProb, pRadius):
    global layers, pair_idx
    ctx=dash.callback_context
    triggered=ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if triggered=="new-graph":
        # 用输入参数重建 base
        N = int(pN if pN is not None else 100)
        step = float(pStep if pStep is not None else 25)
        prob = float(pProb if pProb is not None else 0.05)
        radius = float(pRadius if pRadius is not None else 50)

        layers = init_layers(N, step, prob, radius)
        pair_idx = 0
        reset_global_bounds(layers[0]["nodes"])
        opts=[{"label":"base","value":"base"}]
        return opts,"base"

    if triggered=="add-layer":
        # 截断当前 layer 之后的所有层
        idx = next(i for i,L in enumerate(layers) if L["name"] == current_value)
        layers = layers[:idx+1]

        last=layers[-1]
        kind,k=parse_layer_name(last["name"])
        pair_idx=highest_pair_idx([L["name"] for L in layers])

        if kind=="super":
            abs_layer_idx=k*2
            abs_nodes,abs_edges=copy_to_abs(last["nodes"],last["edges"],abs_layer_idx)
            layers.append({"name":f"abs{k}","nodes":abs_nodes,"edges":abs_edges})
        else:
            k_next=pair_idx+1
            super_layer_idx=k_next*2-1
            if mode == "grid":
                super_nodes,super_edges = fuse_to_super_grid(
                    last["nodes"], last["edges"], int(gx), int(gy), super_layer_idx
                )
            else:
                super_nodes,super_edges = fuse_to_super_knn(
                    last["nodes"], last["edges"], int(kk), super_layer_idx
                )
            layers.append({"name":f"super{k_next}","nodes":super_nodes,"edges":super_edges})
            pair_idx=k_next

    opts=[{"label":L["name"],"value":L["name"]} for L in layers]
    return opts,layers[-1]["name"]

# -----------------------
# 渲染 Cytoscape
# -----------------------
@app.callback(
    Output("cytoscape","elements"),
    Output("cytoscape","stylesheet"),
    Output("cytoscape","layout"),
    Input("layer-select","value"),
    Input("layer-select","options")
)
def update_layer(layer_name,_options):
    layer=next(l for l in layers if l["name"]==layer_name)
    nodes,edges=layer["nodes"],layer["edges"]

    axis_nodes=[
        {"data":{"id":"x_axis_start"},"position":{"x":GLOBAL_XMIN_ADJ-AXIS_PAD,"y":0},"classes":"axis-node"},
        {"data":{"id":"x_axis_end"},"position":{"x":GLOBAL_XMAX_ADJ+AXIS_PAD,"y":0},"classes":"axis-node"},
        {"data":{"id":"y_axis_start"},"position":{"x":0,"y":GLOBAL_YMIN_ADJ-AXIS_PAD},"classes":"axis-node"},
        {"data":{"id":"y_axis_end"},"position":{"x":0,"y":GLOBAL_YMAX_ADJ+AXIS_PAD},"classes":"axis-node"},
    ]
    axis_edges=[
        {"data":{"source":"x_axis_start","target":"x_axis_end"},"classes":"axis"},
        {"data":{"source":"y_axis_start","target":"y_axis_end"},"classes":"axis"},
    ]
    elements=nodes+edges+axis_nodes+axis_edges

    min_dim,min_size=2,3
    base_count=len(layers[0]["nodes"]) if layers else 1
    max_dim,max_size=max(2*base_count,min_dim+1),10
    max_layer_idx=max(n["data"]["layer"] for L in layers for n in L["nodes"]) or 1

    stylesheet=[
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
    layout={"name":"preset"}
    return elements,stylesheet,layout

# -----------------------
# GBP Solver（占位回调）
# -----------------------
@app.callback(
    Output("gbp-status", "children"),
    Input("gbp-run", "n_clicks"),
    prevent_initial_call=True
)
def gbp_solver_placeholder(n_clicks):
    return f"GBP Solver: Coming soon… (clicked {n_clicks} time{'s' if n_clicks!=1 else ''})"

# -----------------------
if __name__=="__main__":
    app.run(debug=True,port=8050)
