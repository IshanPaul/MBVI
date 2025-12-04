"""
Extended moment ODE functions for variational inference
"""

import numpy as np


def forward_ode_birth_death(t, y, c1, c2, lambda_func):
    """
    Forward ODE for mean dynamics with variational control
    
    For birth-death process:
        dy/dt = lambda_1(t) * c1 - lambda_2(t) * c2 * y
    
    Parameters
    ----------
    t : float
        Current time
    y : array-like
        Current state [mean]
    c1 : float
        Birth rate
    c2 : float
        Death rate constant
    lambda_func : callable
        Function returning (lambda_birth, lambda_death) at time t
    
    Returns
    -------
    dydt : list
        Time derivative
    """
    lam_birth, lam_death = lambda_func(t)
    dy_dt = lam_birth * c1 - lam_death * c2 * y[0]
    return [dy_dt]


def compute_moment_functions(mean_traj, c1, c2):
    """
    Compute natural moment functions φ_i(t) for birth-death process
    
    Parameters
    ----------
    mean_traj : ndarray
        Mean trajectory over time
    c1 : float
        Birth rate
    c2 : float
        Death rate
    
    Returns
    -------
    phi_birth : ndarray
        Birth moment function φ_1(t) = c1
    phi_death : ndarray
        Death moment function φ_2(t) = c2 * E[X(t)]
    """
    phi_birth = c1 * np.ones_like(mean_traj)
    phi_death = c2 * mean_traj
    
    return phi_birth, phi_death