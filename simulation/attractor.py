
import numpy as np

class Attractor:
    def __init__(self, center_x, memory_pattern, identity_function):
        self.center_x = center_x                # Symbolic basin center
        self.memory_pattern = memory_pattern    # Attractor's stable memory vector
        self.identity_function = identity_function  # Recursive identity loop

    def influence(self, state, t):
        """
        Determines the stabilizing force the attractor exerts on a symbolic state.
        Returns a vector pointing toward the attractor basin center.
        """
        force_x = self.center_x - state.x
        force_m = self.memory_pattern - state.m
        force_r = self.identity_function(t) - state.r(t)

        return np.array([force_x, force_m, force_r], dtype=object)

    def stabilize(self, state, t, alpha=0.05):
        """
        Gently pulls a symbolic state toward the attractor basin.
        Alpha is the attractor's influence rate.
        """
        force = self.influence(state, t)
        new_x = state.x + alpha * force[0]
        new_m = state.m + alpha * force[1]
        new_r = lambda τ: state.r(τ) + alpha * (self.identity_function(τ) - state.r(τ))

        return new_x, new_m, new_r
