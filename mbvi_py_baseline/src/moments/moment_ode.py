# src/moments/moment_ode.py

from src.moments.moment_closure import mean_field_closure

def compute_moment_rhs(moment_vector, network, eta):
    """
    Compute RHS of ODE under moment closure.
    moment_vector is mean state only.
    """
    state = moment_vector
    lam = mean_field_closure(state, network.rates, network.reactants)

    dx = [0.0 for _ in range(network.n_species)]

    for r in range(network.R):
        class_id = network.classes[r]
        scale = eta.get(class_id, 1.0)

        for i in range(network.n_species):
            dx[i] += scale * lam[r] * network.stoich[r][i]

    return dx
