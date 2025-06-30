from gbp.gbp import *
from gbp.factor import *
from gbp.grid import *

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# generate measurements on a fine-level grid
positions, prior_meas, between_meas = generate_grid_slam_data(H=16, W=16, 
                                    prior_noise_std=1, odom_noise_std=0.01, seed=0)


# build the fine-level pose SLAM grid
varis, prior_facs, between_facs = build_pose_slam_graph(N=256, prior_meas=prior_meas, between_meas=between_meas, 
                                                        prior_std=1, odom_std=0.01,
                                                        Ni_v=10, D=2)


# Solve the fine-level grid with GBP
varis, prior_facs, between_facs, energy_log, linpoints_log = gbp_solve(
    varis, prior_facs, between_facs, num_iters=30, visualize=True
)


# build the coarse-level pose SLAM grid
varis_coarse, prior_facs_coarse, horizontal_facs_coarse, vertical_facs_coarse  = build_coarse_slam_graph(
    varis_fine=varis,
    prior_facs_fine=prior_facs,
    between_facs_fine=between_facs,
    H=16, W=16,
    stride = 2,
    prior_std = 1.0,
    between_std = 0.1,
)


# Solve the coarse-level grid with GBP
varis_coarse, prior_facs_coarse, horizontal_facs_coarse, vertical_facs_coarse, energy_log_coarse, linpoints_log_coarse = gbp_solve_coarse(
    varis_coarse, prior_facs_coarse, horizontal_facs_coarse, vertical_facs_coarse, 
    num_iters=30, visualize=True
)


# build a coarser-level pose SLAM grid
varis_coarser, prior_facs_coarser, horizontal_facs_coarser, vertical_facs_coarser = build_coarser_slam_graph(
    varis_coarse=varis_coarse,
    prior_facs_coarse=prior_facs_coarse,
    horizontal_between_facs=horizontal_facs_coarse,
    vertical_between_facs=vertical_facs_coarse,
    H=16, W=16,
    stride=2
)


# Solve the coarser-level grid with GBP
varis_coarser, prior_facs_coarser, horizontal_facs_coarser, vertical_facs_coarser, energy_log_coarser, linpoints_log_coarser = gbp_solve_coarse(
    varis_coarser, prior_facs_coarser, horizontal_facs_coarser, vertical_facs_coarser, 
    num_iters=30, visualize=True, prior_h=h6_fn, between_h=[h7_fn,h8_fn]
)




import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
            xaxis2=dict(range=[start_idx, t], domain=[0.1, 0.35], anchor='y2'),
            yaxis2=dict(range=[0, dynamic_ymax], domain=[0, 0.45], anchor='x2')
        )
    ))

# Static layout (position and energy plots, animation buttons)
layout = go.Layout(
    width=1000,
    height=650,
    title=dict(
        text='Multi-Scale GBP Animation',
        x=0.5,
        xanchor='center',
        y=0.95,
        yanchor='top',
        font=dict(family='Times New Roman, serif', size=20, color='black')
    ),
    margin=dict(l=0, r=0, t=0.2 * 650, b=0.1 * 650),


    annotations=[
        dict(
            x=0.98, y=0.95,
            xref='x2 domain', yref='y2 domain',
            text="<span style='color:pink; font-size:10px;'>Fine</span><br>"
                "<span style='color:lightblue; font-size:10px;'>Coarse</span><br>"
                "<span style='color:lightgreen; font-size:10px;'>Coarser</span>",
            showarrow=False,
            align='right',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='lightgray',
            borderwidth=0.5,
            borderpad=2,
        )
    ],


    # Main position plot (right)
    xaxis1=dict(range=[x_min_equal, x_max_equal], domain=[0.4, 1], constrain='domain', anchor='y1'),
    yaxis1=dict(range=[y_min_equal, y_max_equal], domain=[0, 1], scaleanchor='x1', scaleratio=1, anchor='x1'),

    # Energy plot (left-bottom)
    xaxis2=dict(domain=[0.1, 0.35], anchor="y2", title="Iteration", automargin=False),
    yaxis2=dict(domain=[0, 0.45], anchor="x2", title="Energy", automargin=False),

    # Button menus
    updatemenus=[
        {
            "type": "buttons",
            "direction": "down",
            "x": 0.44,
            "xanchor": "right",
            "y": 0.6,
            "yanchor": "top",
            "pad": {"r": 0, "t": 0},
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
                {"label": "Stop", "method": "animate", "args": [["0"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
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
                {"label": "Fine", "method": "update", "args": [{"visible": make_visibility(True, False, False)}]},
                {"label": "Coarse", "method": "update", "args": [{"visible": make_visibility(False, True, False)}]},
                {"label": "Coarser", "method": "update", "args": [{"visible": make_visibility(False, False, True)}]}
            ]
        }
    ]


    
)

# Create and display figure
fig = go.Figure(data=frames[0].data, frames=frames, layout=layout)
fig.show()




