"""
Compute posterior trajectories given optimized variational parameters
"""

import numpy as np
from scipy.integrate import solve_ivp


def compute_posterior_trajectory(c1, c2, y0, T, lambda_birth, lambda_death, 
                                 t_grid, t_eval=None):
    """
    Compute posterior mean trajectory with optimized variational parameters
    
    Parameters
    ----------
    c1 : float
        Birth rate
    c2 : float
        Death rate
    y0 : float
        Initial mean
    T : float
        Time horizon
    lambda_birth : ndarray
        Optimized birth scaling
    lambda_death : ndarray
        Optimized death scaling
    t_grid : ndarray
        Time points for lambda values
    t_eval : ndarray, optional
        Evaluation times (default: use t_grid)
    
    Returns
    -------
    t_eval : ndarray
        Evaluation times
    posterior_mean : ndarray
        Posterior mean trajectory
    """
    from src.moments.moment_ode_extended import forward_ode_birth_death
    
    if t_eval is None:
        t_eval = t_grid
    
    # Create lambda interpolator
    lambda_func = lambda t: (
        np.interp(t, t_grid, lambda_birth),
        np.interp(t, t_grid, lambda_death)
    )
    
    # Solve ODE
    sol = solve_ivp(
        lambda t, y: forward_ode_birth_death(t, y, c1, c2, lambda_func),
        [0, T],
        [y0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6
    )
    
    return sol.t, sol.y[0]


def compute_prior_trajectory(c1, c2, y0, T, t_eval):
    """
    Compute prior mean trajectory (lambda = 1, unmodified dynamics)
    
    Parameters
    ----------
    c1 : float
        Birth rate
    c2 : float
        Death rate
    y0 : float
        Initial mean
    T : float
        Time horizon
    t_eval : ndarray
        Evaluation times
    
    Returns
    -------
    prior_mean : ndarray
        Prior mean trajectory
    """
    # Analytical solution for birth-death process
    steady_state = c1 / c2
    
    if y0 == 0:
        # Starting from zero
        prior_mean = steady_state * (1 - np.exp(-c2 * t_eval))
    else:
        # General case
        prior_mean = steady_state + (y0 - steady_state) * np.exp(-c2 * t_eval)
    
    return prior_mean

