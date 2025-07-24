
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="New Kind of Chaos", layout="wide")
st.title("🧠 New Kind of Chaos: Symbolic Identity Simulation")

# Sidebar parameters
timesteps = st.sidebar.slider("Timesteps", 100, 1000, 300, step=50)
alpha = st.sidebar.slider("Identity Rigidity (α)", 0.0, 1.0, 0.4, step=0.05)
beta = st.sidebar.slider("Memory Reinforcement (β)", 0.0, 1.0, 0.6, step=0.05)
delta = st.sidebar.slider("Symbolic Deformation (δ)", 0.0, 1.0, 0.3, step=0.05)
num_shocks = st.sidebar.slider("Number of Symbolic Shocks", 1, 20, 10)

# Initialize arrays
H = np.zeros(timesteps)
M = np.zeros(timesteps)
symbolic_input = np.zeros(timesteps)

# Random symbolic shocks
shock_times = np.random.choice(range(20, timesteps - 20), size=num_shocks, replace=False)
shock_magnitudes = np.random.uniform(-2, 2, size=num_shocks)
for i, t in enumerate(shock_times):
    symbolic_input[t:t+5] += shock_magnitudes[i]

# Initial state
H[0] = 0.2
M[0] = 0.2

# Recursive simulation
for t in range(1, timesteps):
    noise = np.random.normal(0, 0.05)
    H[t] = (1 - alpha) * H[t - 1] + beta * M[t - 1] + delta * symbolic_input[t] + noise
    M[t] = (1 - delta) * M[t - 1] + delta * H[t - 1]

# Create DataFrame
df = pd.DataFrame({'Timestep': range(timesteps), 'H(t)': H, 'M(t)': M, 'Symbolic Input': symbolic_input})

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Timestep'], df['H(t)'], label='H(t): Identity', linewidth=2)
ax.plot(df['Timestep'], df['M(t)'], label='M(t): Memory', linewidth=2)
ax.plot(df['Timestep'], df['Symbolic Input'], label='Symbolic Shock Input', linestyle='--', alpha=0.6)
ax.set_title("Symbolic Identity Under Chaotic Shock", fontsize=16)
ax.set_xlabel("Time")
ax.set_ylabel("State Value")
ax.grid(True)
ax.legend()
st.pyplot(fig)
