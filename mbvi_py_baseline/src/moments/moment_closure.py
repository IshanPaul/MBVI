# src/moments/moment_closure.py

import src.propensities as prop

def mean_field_closure(mean_state, rates, reactants):
    """
    E[lambda_r(X)] ≈ lambda_r(E[X]).
    """
    return prop.mass_action_propensity(mean_state, rates, reactants)
