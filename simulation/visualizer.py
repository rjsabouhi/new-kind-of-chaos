
import matplotlib.pyplot as plt
import numpy as np

class SymbolicVisualizer:
    def __init__(self, figsize=(6, 6)):
        self.figsize = figsize

    def plot_symbolic_states(self, states, title='Symbolic Phase Space', t=0):
        """
        Visualizes symbolic states in a 2D projection: x vs memory magnitude.
        """
        x_vals = [state.x for state in states]
        m_mags = [np.linalg.norm(state.m) for state in states]

        plt.figure(figsize=self.figsize)
        plt.scatter(x_vals, m_mags, c='blue', alpha=0.7)
        plt.xlabel('Symbol x')
        plt.ylabel('||Memory||')
        plt.title(f'{title} at t={t}')
        plt.grid(True)
        plt.show()

    def plot_collapse(self, collapse_flags, title='Collapse Events Over Time'):
        """
        Plots binary collapse events across simulation time steps.
        """
        plt.figure(figsize=self.figsize)
        plt.plot(collapse_flags, 'ro-', label='Collapse Detected')
        plt.xlabel('Time Step')
        plt.ylabel('Collapse (1=True, 0=False)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
