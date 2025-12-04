# run/example_birth_death.py

from src.models.base_model import ReactionNetwork
from src.variational.control_functions import ConstantControl
from src.moments.moment_ode import compute_moment_rhs
from src.solvers.ode_euler import euler_integrate


def run_example():
    # Define birth–death network
    species = ["X"]
    rates = [1.0, 0.5]        # birth, death
    reactants = [
        [],                  # birth: no reactant
        [0]                  # death: 1 reactant X
    ]
    stoich = [
        [1],                 # birth: X -> X+1
        [-1]                 # death: X -> X-1
    ]
    classes = [0, 1]

    net = ReactionNetwork(species, rates, reactants, stoich, classes)

    # Constant variational parameters (baseline)
    control = ConstantControl({0: 1.0, 1: 1.0})

    def rhs(t, state):
        return compute_moment_rhs(state, net, control.eval(t))

    # Initial condition
    y0 = [10.0]
    t0, t1, dt = 0.0, 10.0, 0.1

    times, traj = euler_integrate(rhs, y0, t0, t1, dt)

    for t, m in zip(times, traj):
        print(f"{t:.2f}   {m}")


if __name__ == "__main__":
    run_example()
