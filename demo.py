import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gbp.gbp import *
from gbp.factor import *
from gbp.grid import *
from draw_animation import draw_gbp_animation

st.set_page_config(layout="wide")

# ----------------------
# 初始化参数状态
# ----------------------
defaults = {
    "prior_std": 1.0,
    "between_std": 0.01,
    "H": 16,
    "W": 16,
    "iters": 30,
    "seed": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------
# 页面标题
# ----------------------
st.markdown("<h3 style='text-align: center;'>Multi-Scale GBP Animation</h3>", unsafe_allow_html=True)

# ----------------------
# 参数输入区
# ----------------------
col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

with col1:
    prior_std_str = st.text_input("Prior Std", value=str(st.session_state["prior_std"]), key="prior_std_input")
    try:
        val = float(prior_std_str)
        if val > 0:
            st.session_state["prior_std"] = val
        else:
            st.warning("Prior Std must be greater than 0.")
    except ValueError:
        st.warning("Please enter a valid number for Prior Std.")

with col2:
    between_std_str = st.text_input("Between Std", value=str(st.session_state["between_std"]), key="between_std_input")
    try:
        val = float(between_std_str)
        if val > 0:
            st.session_state["between_std"] = val
        else:
            st.warning("Between Std must be greater than 0.")
    except ValueError:
        st.warning("Please enter a valid number for Between Std.")


with col3:
    st.number_input("Grid Height (H)", min_value=2, max_value=64,
                    step=1, key="H")

with col4:
    st.number_input("Grid Width (W)", min_value=2, max_value=64,
                    step=1, key="W")

with col5:
    st.number_input("Iterations", min_value=1, max_value=200,
                    step=1, key="iters")

with col6:
    st.number_input("Seed", min_value=0, max_value=1000,
                    step=1, key="seed")

with col7:
    speed_choice = st.selectbox("Animation Speed", [
        "0.5x", "1x (1fps)", "2x", "4x"
    ], index=1, key="speed_choice")

# 存储动画速度（毫秒）到 session_state
speed_map = {
    "0.5x": 2000,
    "1x (1fps)": 1000,
    "2x": 500,
    "4x": 250
}
st.session_state["frame_duration"] = speed_map[speed_choice]


col_space1, col6, col7, col_space2 = st.columns([2, 1, 1, 2])
with col6:
    gen_button = st.button("Generate Data")
with col7:
    solve_button = st.button("GBP Solve")


# ----------------------
# 数据生成函数（不缓存）
# ----------------------
def generate_all(H, W, prior_std, between_std, seed):
    positions, prior_meas, between_meas = generate_grid_slam_data(
        H=H, W=W, prior_noise_std=prior_std, odom_noise_std=between_std, seed=seed
    )
    varis, prior_facs, between_facs = build_pose_slam_graph(
        N=H * W, prior_meas=prior_meas, between_meas=between_meas,
        prior_std=prior_std, odom_std=between_std, Ni_v=10, D=2
    )
    varis_c, pf_c, hf_c, vf_c = build_coarse_slam_graph(
        prior_facs_fine=prior_facs, between_facs_fine=between_facs,
        H=H, W=W, stride=2
    )
    varis_cc, pf_cc, hf_cc, vf_cc = build_coarser_slam_graph(
        prior_facs_coarse=pf_c, horizontal_between_facs=hf_c, vertical_between_facs=vf_c,
        H=H, W=W, stride=2
    )
    return (
        positions, prior_meas, between_meas,
        varis, prior_facs, between_facs,
        varis_c, pf_c, hf_c, vf_c,
        varis_cc, pf_cc, hf_cc, vf_cc
    )

# ----------------------
# 解算函数（不缓存）
# ----------------------
def solve_all(varis, prior_facs, between_facs,
              varis_c, pf_c, hf_c, vf_c,
              varis_cc, pf_cc, hf_cc, vf_cc, iters):
    v_f, pf_f, bf_f, e_f, lp_f = gbp_solve(varis, prior_facs, between_facs, num_iters=iters, visualize=True)
    v_c, pf_c, hf_c, vf_c, e_c, lp_c = gbp_solve_coarse(varis_c, pf_c, hf_c, vf_c, num_iters=iters, visualize=True)
    v_cc, pf_cc, hf_cc, vf_cc, e_cc, lp_cc = gbp_solve_coarse(
        varis_cc, pf_cc, hf_cc, vf_cc, num_iters=iters, visualize=True,
        prior_h=h6_fn, between_h=[h7_fn, h8_fn]
    )
    return e_f, lp_f, e_c, lp_c, e_cc, lp_cc

# ----------------------
# Generate 按钮逻辑
# ----------------------
if gen_button:
    with st.spinner("Generating and building graphs... (first time may take a while)"):
        results = generate_all(
            st.session_state["H"], st.session_state["W"],
            st.session_state["prior_std"], st.session_state["between_std"],
            st.session_state["seed"]
        )
        st.session_state["gen_data"] = results
        st.session_state["solved"] = False  # 标记为未解算

# ----------------------
# Solve 按钮逻辑
# ----------------------
if solve_button and "gen_data" in st.session_state:
    with st.spinner("Solving GBP on all levels... (first time may take a while)"):
        (positions, prior_meas, between_meas,
         varis, prior_facs, between_facs,
         varis_c, pf_c, hf_c, vf_c,
         varis_cc, pf_cc, hf_cc, vf_cc) = st.session_state["gen_data"]

        ef, lpf, ec, lpc, ecc, lpcc = solve_all(
            varis, prior_facs, between_facs,
            varis_c, pf_c, hf_c, vf_c,
            varis_cc, pf_cc, hf_cc, vf_cc,
            st.session_state["iters"]
        )

        st.session_state.update({
            "solved": True,
            "energy_log": ef,
            "linpoints_log": lpf,
            "energy_log_coarse": ec,
            "linpoints_log_coarse": lpc,
            "energy_log_coarser": ecc,
            "linpoints_log_coarser": lpcc,
        })

# ----------------------
# 绘图
# ----------------------
if st.session_state.get("solved", False):
    fig = draw_gbp_animation(
        st.session_state["linpoints_log"],
        st.session_state["linpoints_log_coarse"],
        st.session_state["linpoints_log_coarser"],
        st.session_state["energy_log"],
        st.session_state["energy_log_coarse"],
        st.session_state["energy_log_coarser"],
        frame_duration=st.session_state["frame_duration"]
    )
    st.plotly_chart(fig, use_container_width=True)
