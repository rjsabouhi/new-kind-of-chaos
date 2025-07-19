
import numpy as np

class SymbolicState:
    def __init__(self, x, m, r):
        self.x = x  # Current symbol (observable)
        self.m = m  # Memory vector (symbolic history)
        self.r = r  # Recursive identity loop

class PhaseSpace:
    def __init__(self, theta=0.75):
        self.states = []
        self.theta = theta

    def add_state(self, state: SymbolicState):
        self.states.append(state)

    def step(self, gradient_func):
        next_states = []
        for state in self.states:
            grad = gradient_func(state)
            x_grad = np.atleast_1d(grad[0])
            m_grad = grad[1]
            grad_vector = np.concatenate([x_grad, m_grad])
            norm = np.linalg.norm(grad_vector)

            if norm > self.theta:
                collapsed_state = SymbolicState(
                    x=state.x,
                    m=state.m * 0.0,
                    r=lambda t: state.r(t)
                )
                next_states.append(collapsed_state)
            else:
                new_x = state.x - grad[0]
                new_m = state.m - grad[1]
                new_r = lambda t: state.r(t) - grad[2] * t
                next_states.append(SymbolicState(new_x, new_m, new_r))

        self.states = next_states


