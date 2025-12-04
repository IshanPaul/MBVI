def mass_action_propensity(state, rates, reactants):
    """
    Compute hazard λ_r(x) for each reaction r:
    λ_r = k_r * Π_i state[i]
    """
    out = []
    for r, k in enumerate(rates):
        prod = k
        for idx in reactants[r]:
            prod *= state[idx]
        out.append(prod)
    return out
