
import matplotlib.pyplot as plt
import numpy as np
import os

class SymbolicAnimator:
    def __init__(self, history):
        self.history = history

    def frame(self, step, collapsed_flags=None, show=True, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 6))

        for t in range(step):
            if t < len(self.history):
                xs, ms = zip(*self.history[t])
                ax.scatter(xs, ms, color='lightgray', alpha=0.3, s=20)

        if step < len(self.history):
            xs, ms = zip(*self.history[step])
            if collapsed_flags:
                colors = ['red' if flag else 'blue' for flag in collapsed_flags]
            else:
                colors = ['blue'] * len(xs)
            ax.scatter(xs, ms, c=colors, s=60, edgecolor='black', linewidths=0.5)

        ax.set_xlabel("Symbolic Identity (x)")
        ax.set_ylabel("Memory Magnitude (||m||)")
        ax.set_title(f"Step {step} / {len(self.history)}")
        ax.grid(True)

        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        elif show:
            plt.show()
        else:
            return fig

    def save_gif(self, output_filename="symbolic_evolution.gif", duration=200):
        import imageio
        tmp_dir = "_frames"
        os.makedirs(tmp_dir, exist_ok=True)

        frames = []
        for t in range(len(self.history)):
            path = os.path.join(tmp_dir, f"frame_{t:03d}.png")
            self.frame(t, save_path=path)
            frames.append(imageio.v2.imread(path))

        imageio.mimsave(output_filename, frames, duration=duration / 1000)
