
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="New Kind of Chaos 3D", layout="wide")
st.title("🧠 New Kind of Chaos: 3D Symbolic Identity Simulation")

# Sidebar controls
timesteps = st.sidebar.slider("Timesteps", 100, 1000, 300, step=50)
alpha = st.sidebar.slider("Identity Rigidity (α)", 0.0, 1.0, 0.4, step=0.05)
beta = st.sidebar.slider("Memory Reinforcement (β)", 0.0, 1.0, 0.6, step=0.05)
delta = st.sidebar.slider("Symbolic Deformation (δ)", 0.0, 1.0, 0.3, step=0.05)
num_shocks = st.sidebar.slider("Number of Symbolic Shocks", 1, 20, 10)

# Initialize arrays
H = np.zeros(timesteps)
M = np.zeros(timesteps)
S = np.zeros(timesteps)

# Symbolic shock input
shock_times = np.random.choice(range(20, timesteps - 20), size=num_shocks, replace=False)
shock_magnitudes = np.random.uniform(-2, 2, size=num_shocks)
for i, t in enumerate(shock_times):
    S[t:t+5] += shock_magnitudes[i]

# Initial state
H[0] = 0.2
M[0] = 0.2

# Run simulation
for t in range(1, timesteps):
    noise = np.random.normal(0, 0.05)
    H[t] = (1 - alpha) * H[t - 1] + beta * M[t - 1] + delta * S[t] + noise
    M[t] = (1 - delta) * M[t - 1] + delta * H[t - 1]

# 3D Plot with Plotly
fig = go.Figure(data=[
    go.Scatter3d(
        x=list(range(timesteps)),
        y=H,
        z=M,
        mode='lines',
        line=dict(color='royalblue', width=4),
        name='H-M Trajectory'
    ),
    go.Scatter3d(
        x=list(range(timesteps)),
        y=S,
        z=[0]*timesteps,
        mode='lines',
        line=dict(color='indianred', dash='dot', width=2),
        name='Symbolic Input'
    )
])

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis_title='Time',
        yaxis_title='H(t) / S(t)',
        zaxis_title='M(t)',
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
    ),
    showlegend=True,
    template='plotly_white',
    height=700,
    annotations=[
        dict(
            showarrow=False,
            text="<b>H(t)</b> = Identity • <b>M(t)</b> = Memory • <b>S(t)</b> = Symbolic Shock",
            xref="paper", yref="paper",
            x=0, y=1.08,
            font=dict(size=12),
            align="left",
            bgcolor="white",
            opacity=0.8
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)
