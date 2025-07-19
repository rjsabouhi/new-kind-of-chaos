
import numpy as np

class Observer:
    def __init__(self, identity_function, memory_vector):
        self.identity_function = identity_function  # R_obs(t)
        self.memory_vector = memory_vector          # m_obs(t)

    def boundary_deformation(self, state, t):
        """
        Computes δ𝒦_obs(s) = f(R_obs(t), m_obs(t))
        Symbolic observer-induced deformation of the KCE field.
        """
        r_obs = self.identity_function(t)
        m_obs = self.memory_vector

        # Influence is proportional to difference in memory and identity
        delta_k = np.dot(state.m - m_obs, m_obs) + abs(state.r(t) - r_obs)
        return delta_k
