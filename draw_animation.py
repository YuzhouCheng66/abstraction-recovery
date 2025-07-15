import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def make_visibility(show_fine, show_coarse, show_coarser):
    return [show_fine, show_coarse, show_coarser, True, True, True]


def draw_gbp_animation(linpoints_log, linpoints_log_coarse, linpoints_log_coarser,
                        energy_log, energy_log_coarse, energy_log_coarser, frame_duration=1000):
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
                xaxis2=dict(
                    domain=[0.1, 0.4],
                    anchor='y2',
                    title='Iteration',
                    tickmode='linear',
                    dtick=1,
                    tickfont=dict(family='Arial', size=10),
                    fixedrange=True,
                    automargin=False,
                    range=[start_idx, t]
                ),
                yaxis2=dict(
                    domain=[0, 1],
                    anchor='x2',
                    title='Energy',
                    tickformat='.0f',
                    tickfont=dict(family='Arial', size=10),
                    fixedrange=True,
                    automargin=False,
                    range=[dynamic_ymin, dynamic_ymax]
                ),
                title=f"Iters {t}" 
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
                    {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": True},  # redraw ✅
                            "fromcurrent": True,
                            "mode": "immediate"  # mode 默认值，写上更稳妥
                        }
                    ]
                    },
                    {"label": "Pause", "method": "animate", "args": [[None], {
                        "mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]},
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
                "active": 3,  # 👈 指定默认选中的按钮为第4个（即 All）
                "buttons": [
                    {"label": "Fine", "method": "update", "args": [{"visible": make_visibility(True, False, False)}]},
                    {"label": "Coarse", "method": "update", "args": [{"visible": make_visibility(False, True, False)}]},
                    {"label": "Coarser", "method": "update", "args": [{"visible": make_visibility(False, False, True)}]},
                    {"label": "All", "method": "update", "args": [{"visible": make_visibility(True, True, True)}]}
                ]
            }
        ]
    )


    fig = go.Figure(data=frames[0].data, frames=frames, layout=layout)
    return fig
