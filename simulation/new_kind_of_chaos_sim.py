
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
timesteps = 300
alpha = 0.4  # identity rigidity
beta = 0.6   # memory reinforcement
delta = 0.3  # deformation influence

# Initialize arrays
H = np.zeros(timesteps)
M = np.zeros(timesteps)
symbolic_input = np.zeros(timesteps)

# Random symbolic shocks at irregular intervals
shock_times = np.random.choice(range(20, timesteps - 20), size=10, replace=False)
shock_magnitudes = np.random.uniform(-2, 2, size=10)
for i, t in enumerate(shock_times):
    symbolic_input[t:t+5] += shock_magnitudes[i]  # pulse effect lasts a few steps

# Initial state
H[0] = 0.2
M[0] = 0.2

# Recursive dynamics with chaotic symbolic influence
for t in range(1, timesteps):
    noise = np.random.normal(0, 0.05)  # small randomness for chaos
    H[t] = (1 - alpha) * H[t - 1] + beta * M[t - 1] + delta * symbolic_input[t] + noise
    M[t] = (1 - delta) * M[t - 1] + delta * H[t - 1]

# Create DataFrame
df = pd.DataFrame({'Timestep': range(timesteps), 'H(t)': H, 'M(t)': M, 'Symbolic Input': symbolic_input})

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['Timestep'], df['H(t)'], label='H(t): Identity', linewidth=2)
plt.plot(df['Timestep'], df['M(t)'], label='M(t): Memory', linewidth=2)
plt.plot(df['Timestep'], df['Symbolic Input'], label='Symbolic Shock Input', linestyle='--', alpha=0.6)
plt.title("New Kind of Chaos: Identity under Symbolic Shock")
plt.xlabel("Time")
plt.ylabel("State Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
