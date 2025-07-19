
from simulation.phase_space import PhaseSpace, SymbolicState
from simulation.constraint import KarmicConstraintField
from simulation.attractor import Attractor
from simulation.observer import Observer
from simulation.visualizer import SymbolicVisualizer

import numpy as np

# Initialize components
ps = PhaseSpace(theta=0.75)
kce = KarmicConstraintField()
viz = SymbolicVisualizer()

# Sample recursive identity function
def R(t): return np.sin(t)

# Seed initial symbolic state
initial_state = SymbolicState(
    x=1.0,
    m=np.array([0.5, -0.5]),
    r=R
)
ps.add_state(initial_state)

# Add attractor
attractor = Attractor(center_x=0.0, memory_pattern=np.array([0.0, 0.0]), identity_function=lambda t: 0)

# Add observer
observer = Observer(identity_function=lambda t: np.cos(t), memory_vector=np.array([0.1, 0.1]))

# Run simulation
collapse_flags = []
timesteps = 30

for t in range(timesteps):
    current_states = ps.states

    def gradient_func(state):
        # Include deformation from observer
        delta_k = observer.boundary_deformation(state, t)
        grad = kce.gradient(state, t, neighbor_states=current_states)
        grad[0] += delta_k  # deform x
        return grad

    # Stabilize via attractor before stepping
    stabilized_states = []
    for state in current_states:
        new_x, new_m, new_r = attractor.stabilize(state, t)
        stabilized_states.append(SymbolicState(new_x, new_m, new_r))
    ps.states = stabilized_states

    # Check for collapse
    grads = [gradient_func(state)[:2] for state in ps.states]  # use x and m only
    collapsed = any(np.linalg.norm(np.concatenate([np.atleast_1d(g[0]), g[1]])) > ps.theta for g in grads)
    collapse_flags.append(1 if collapsed else 0)

    # Step forward
    ps.step(gradient_func)
    viz.plot_symbolic_states(ps.states, t=t)

# Visualize collapse timeline
viz.plot_collapse(collapse_flags)
