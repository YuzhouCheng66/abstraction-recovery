import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_javascript import st_javascript
from gbp.gbp import *
from gbp.factor import *
from gbp.grid import *



st.set_page_config(layout="wide")  



#width = st_javascript("window.innerWidth")
# 获取浏览器窗口高度
#height = st_javascript("window.innerHeight")



# --- GBP Pipeline ---
seed = 0
prior_noise_std = 1.0
odom_noise_std = 0.01

positions, prior_meas, between_meas = generate_grid_slam_data(H=16, W=16,
                                prior_noise_std=prior_noise_std, odom_noise_std=odom_noise_std, seed=seed)

varis, prior_facs, between_facs = build_pose_slam_graph(N=256, prior_meas=prior_meas, between_meas=between_meas,
                            prior_std=prior_noise_std, odom_std=odom_noise_std,
                            Ni_v=10, D=2)

varis, prior_facs, between_facs, energy_log, linpoints_log = gbp_solve(
    varis, prior_facs, between_facs, num_iters=30, visualize=True)

varis_coarse, prior_facs_coarse, horizontal_facs_coarse, vertical_facs_coarse = build_coarse_slam_graph(
    varis_fine=varis,
    prior_facs_fine=prior_facs,
    between_facs_fine=between_facs,
    H=16, W=16, stride=2,
    prior_std=1.0, between_std=0.1)

varis_coarse, prior_facs_coarse, horizontal_facs_coarse, vertical_facs_coarse, energy_log_coarse, linpoints_log_coarse = gbp_solve_coarse(
    varis_coarse, prior_facs_coarse, horizontal_facs_coarse, vertical_facs_coarse,
    num_iters=30, visualize=True)

varis_coarser, prior_facs_coarser, horizontal_facs_coarser, vertical_facs_coarser = build_coarser_slam_graph(
    varis_coarse=varis_coarse,
    prior_facs_coarse=prior_facs_coarse,
    horizontal_between_facs=horizontal_facs_coarse,
    vertical_between_facs=vertical_facs_coarse,
    H=16, W=16, stride=2)

varis_coarser, prior_facs_coarser, horizontal_facs_coarser, vertical_facs_coarser, energy_log_coarser, linpoints_log_coarser = gbp_solve_coarse(
    varis_coarser, prior_facs_coarser, horizontal_facs_coarser, vertical_facs_coarser,
    num_iters=30, visualize=True, prior_h=h6_fn, between_h=[h7_fn, h8_fn])








# Function to ensure only point visibility is toggled by buttons,
# while energy curves always stay visible (last 3 traces)
def make_visibility(show_fine, show_coarse, show_coarser):
    return [show_fine, show_coarse, show_coarser, True, True, True]

# Compute overall axis bounds for equal scaling
all_points = np.concatenate([
    linpoints_log.reshape(-1, 2),
    linpoints_log_coarse.reshape(-1, 2),
    linpoints_log_coarser.reshape(-1, 2)
], axis=0)

x_min, y_min = all_points.min(axis=0)
x_max, y_max = all_points.max(axis=0)

x_range = x_max - x_min
y_range = y_max - y_min
max_range = max(x_range, y_range) / 2

x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2

x_min_equal = x_center - max_range - 0.001 * max_range
x_max_equal = x_center + max_range + 0.001 * max_range
y_min_equal = y_center - max_range - 0.001 * max_range
y_max_equal = y_center + max_range + 0.001 * max_range

# Dimensions
num_iters = linpoints_log.shape[0]
fine_points = linpoints_log.shape[1]
coarse_points = linpoints_log_coarse.shape[1]
coarser_points = linpoints_log_coarser.shape[1]

# Assign colors to each point group
fine_base = px.colors.qualitative.Set3
fine_colors = (fine_base * ((fine_points // len(fine_base)) + 1))[:fine_points]
coarse_colors = px.colors.qualitative.Plotly * 10
coarser_colors = px.colors.qualitative.D3 * 10

# Precompute color assignments per point
coarse_color_list = [coarse_colors[i] for i in range(coarse_points) for _ in range(4)]
coarser_color_list = [coarser_colors[i] for i in range(coarser_points) for _ in range(16)]

# Create animation frames
frames = []
for t in range(num_iters):
    # Fine-level points (right side)
    fine_trace = go.Scatter(
        x=linpoints_log[t, :, 0],
        y=linpoints_log[t, :, 1],
        mode='markers',
        marker=dict(size=4, color=fine_colors),
        name='fine',
        showlegend=False,
        xaxis='x1',
        yaxis='y1'
    )

    # Coarse-level points (right side)
    coarse_pts = linpoints_log_coarse[t].reshape(-1, 2)
    coarse_trace = go.Scatter(
        x=coarse_pts[:, 0],
        y=coarse_pts[:, 1],
        mode='markers',
        marker=dict(size=4, color=coarse_color_list),
        name='coarse',
        showlegend=False,
        xaxis='x1',
        yaxis='y1'
    )

    # Coarser-level points (right side)
    coarser_pts = linpoints_log_coarser[t].reshape(-1, 2)
    coarser_trace = go.Scatter(
        x=coarser_pts[:, 0],
        y=coarser_pts[:, 1],
        mode='markers',
        marker=dict(size=4, color=coarser_color_list),
        name='coarser',
        showlegend=False,
        xaxis='x1',
        yaxis='y1'
    )

    # Sliding window for energy plot (left bottom)
    start_idx = max(0, t - 4)
    x_vals = np.arange(start_idx, t + 1)

    # Energy curves: fine, coarse, coarser
    energy_trace_fine = go.Scatter(
        x=x_vals,
        y=energy_log[start_idx:t + 1],
        mode='lines+markers',
        line=dict(color='pink'),
        marker=dict(size=3),
        name='Energy Fine',
        xaxis='x2',
        yaxis='y2',
        showlegend=False
    )

    energy_trace_coarse = go.Scatter(
        x=x_vals,
        y=energy_log_coarse[start_idx:t + 1],
        mode='lines+markers',
        line=dict(color='lightblue'),
        marker=dict(size=3),
        name='Energy Coarse',
        xaxis='x2',
        yaxis='y2',
        showlegend=False
    )

    energy_trace_coarser = go.Scatter(
        x=x_vals,
        y=energy_log_coarser[start_idx:t + 1],
        mode='lines+markers',
        line=dict(color='lightgreen'),
        marker=dict(size=3),
        name='Energy Coarser',
        xaxis='x2',
        yaxis='y2',
        showlegend=False
    )

    # Dynamic y-range for zoomed-in energy chart
    dynamic_ymax = max(
        energy_log[start_idx:t + 1].max(),
        energy_log_coarse[start_idx:t + 1].max(),
        energy_log_coarser[start_idx:t + 1].max()
    )
    dynamic_ymin = min(
        energy_log[start_idx:t + 1].min(),
        energy_log_coarse[start_idx:t + 1].min(),
        energy_log_coarser[start_idx:t + 1].min()
    )


    # Create and append this frame
    frames.append(go.Frame(
        name=str(t),
        data=[
            fine_trace, coarse_trace, coarser_trace,
            energy_trace_fine, energy_trace_coarse, energy_trace_coarser
        ],
        layout=go.Layout(
            xaxis2=dict(range=[start_idx, t], fixedrange=True),
            yaxis2=dict(range=[dynamic_ymin, dynamic_ymax], fixedrange=True)
        )
    ))


layout = go.Layout(
    autosize=True,
    margin=dict(l=60, r=20, t=60, b=60), 
    # Annotation (右上角图例标签)
    annotations=[
        dict(
            x=0.4, y=0.3,
            xref='paper', yref='paper',
            text="<span style='color:pink; font-size:12px;'>Fine</span><br>"
                 "<span style='color:lightblue; font-size:12px;'>Coarse</span><br>"
                 "<span style='color:lightgreen; font-size:12px;'>Coarser</span>",
            showarrow=False,
            align='right',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='lightgray',
            borderwidth=0.5,
            borderpad=0,
        )
    ],

    # 主图（右侧大图）
    xaxis1=dict(
        domain=[0.4, 1.0],
        anchor='y1',
        constrain="domain",
        range=[x_min_equal, x_max_equal]  # 💡 显式加上
    ),
    yaxis1=dict(
        domain=[0.0, 1],
        anchor='x1',
        scaleanchor='x1',
        scaleratio=1,

        range=[y_min_equal, y_max_equal]  # 💡 显式加上
    ),

    # 能量图（左下角）
    #xaxis2=dict(domain=[0.0, 0.35], anchor='y2', title='Iteration'),

    xaxis2 = dict(
        domain=[0.1, 0.4],
        anchor='y2',
        title='Iteration',
        tickmode='linear',
        dtick=1,                              # 强制 tick 步长
        tickfont=dict(family='Arial', size=10),
        fixedrange=True,
        automargin=False
    ),

    
    #yaxis2=dict(domain=[0.0, 0.45], anchor='x2', title='Energy'),

    yaxis2 = dict(
        domain=[0, 1],
        anchor='x2',
        title='Energy',
        tickformat='.0f',                     # 所有 tick 统一格式
        tickfont=dict(family='Arial', size=10),  # 字体固定
        fixedrange=True,                      # 禁止用户缩放，防止 layout 改变
        automargin=False                      # 关闭自动 margin（否则 tick 会影响 layout）
    ),

    # 动画控制按钮（靠右，靠上）
    updatemenus=[
        {
            "type": "buttons",
            "direction": "down",
            "x": 0,  # 稍微偏右，避免和 Streamlit 控件重叠
            "xanchor": "right",
            "y": 0.8,
            "yanchor": "top",
            "pad": {"r": 0, "t": 0},
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {
                    "frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {
                    "mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
                {"label": "Stop", "method": "animate", "args": [["0"], {
                    "mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
            ]
        },
        {
            "type": "buttons",
            "direction": "right",
            "x": 0.6,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "top",
            "pad": {"r": 0, "t": 0},
            "buttons": [
                {"label": "All", "method": "update", "args": [{"visible": make_visibility(True, True, True)}]},
                {"label": "Fine", "method": "update", "args": [{"visible": make_visibility(True, False, False)}]},
                {"label": "Coarse", "method": "update", "args": [{"visible": make_visibility(False, True, False)}]},
                {"label": "Coarser", "method": "update", "args": [{"visible": make_visibility(False, False, True)}]}
            ]
        }
    ]
)




# 👇 控制面板区域：左下图的正上方


st.markdown("<h1 style='text-align: center;'>Multi-Scale GBP Animation</h3>", unsafe_allow_html=True)


# 第一行：三个输入框
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
with col1:
    prior_std = st.number_input("Prior Std", min_value=0.1, max_value=5.0, value=1.0, step=0.0001)
with col2:
    between_std = st.number_input("Between Std", min_value=0.001, max_value=1.0, value=0.01, step=0.0001)
with col3:
    H = st.number_input("Grid Height (H)", min_value=2, max_value=64, value=16, step=1)
with col4:
    W = st.number_input("Grid Width (W)", min_value=2, max_value=64, value=16, step=1)
with col5:
    iters = st.number_input("Iterations", min_value=1, max_value=200, value=30, step=1)
with col6:
    seed = st.number_input("Seed", min_value=0, max_value=1000, value=0, step=1)



# 第二行：两个按钮居中
col_space1, col6, col7, col_space2 = st.columns([2, 1, 1, 2])
with col6:
    gen_button = st.button("Generate Data")
with col7:
    solve_button = st.button("GBP Solve")




fig = go.Figure(data=frames[0].data, frames=frames, layout=layout)
st.plotly_chart(fig, use_container_width=True)


