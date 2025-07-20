
import matplotlib.pyplot as plt
import numpy as np

class SymbolicVisualizer:
    def __init__(self):
        self.history = []

    def record(self, states):
        # Store x and ||m|| for each symbolic state
        snapshot = [(state.x, np.linalg.norm(state.m)) for state in states]
        self.history.append(snapshot)

    def plot(self, collapsed_flags=None, step=None, save=False):
        plt.figure(figsize=(8, 6))
        colors = []

        # Plot all previous steps as faint traces
        for t, snapshot in enumerate(self.history[:-1]):
            xs, ms = zip(*snapshot)
            plt.scatter(xs, ms, color='lightgray', alpha=0.3, s=20)

        # Plot current state with color
        if self.history:
            xs, ms = zip(*self.history[-1])
            if collapsed_flags is None:
                colors = ['blue'] * len(xs)
            else:
                colors = ['red' if c else 'blue' for c in collapsed_flags]
            plt.scatter(xs, ms, c=colors, s=60, edgecolor='black', linewidths=0.5)

        plt.xlabel('Symbolic Identity (x)')
        plt.ylabel('Memory Magnitude (||m||)')
        plt.title(f'Symbolic State Space at Step {step if step is not None else len(self.history)}')
        plt.grid(True)

        if save:
            plt.savefig(f"symbolic_step_{step:03d}.png", dpi=150)
        else:
            plt.show()
