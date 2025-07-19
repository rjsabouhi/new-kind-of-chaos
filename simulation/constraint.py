
import numpy as np

class KarmicConstraintField:
    def __init__(self, decay_rate=0.01, coupling_strength=0.1):
        self.decay_rate = decay_rate
        self.coupling_strength = coupling_strength

    def K(self, state, t=0, neighbor_states=None):
        """
        Computes the symbolic constraint energy 𝒦(s) at a given state.
        Includes impermanence (decay) and interdependence (coupling).
        """
        decay_term = self.decay_rate * np.exp(-t) * (np.linalg.norm(state.m) + np.linalg.norm(state.x))

        coupling_term = 0
        if neighbor_states:
            for neighbor in neighbor_states:
                coupling_term += self.coupling_strength * np.dot(state.m, neighbor.m)

        energy = decay_term + coupling_term
        return energy

    def gradient(self, state, t=0, neighbor_states=None):
        """
        Computes ∇𝒦(s): gradient of the karmic constraint field.
        Returns a 3D vector over [x, m, r].
        """
        # Impermanence gradient acts against current stability
        grad_x = self.decay_rate * np.exp(-t) * state.x
        grad_m = self.decay_rate * np.exp(-t) * state.m

        # Interdependence: influenced by other memory vectors
        if neighbor_states:
            for neighbor in neighbor_states:
                grad_m += self.coupling_strength * neighbor.m

        # Recursive term influence is symbolic (simple decay placeholder)
        grad_r = self.decay_rate * t  # treated as scalar influence

        return np.array([grad_x, grad_m, grad_r], dtype=object)
