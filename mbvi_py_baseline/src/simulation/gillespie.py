"""
Gillespie algorithm for exact stochastic simulation of Markov jump processes
"""

import numpy as np


def gillespie_birth_death(c1, c2, x0, T, seed=None):
    """
    Simulate exact birth-death trajectory using Gillespie algorithm
    
    Parameters
    ----------
    c1 : float
        Birth rate constant
    c2 : float
        Death rate constant (per capita)
    x0 : int
        Initial population
    T : float
        Time horizon
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    times : ndarray
        Event times
    states : ndarray
        Population at each event
    """
    if seed is not None:
        np.random.seed(seed)
    
    times = [0.0]
    states = [x0]
    
    t = 0.0
    x = x0
    
    while t < T:
        # Compute propensities
        lam_birth = c1
        lam_death = c2 * x
        lam_total = lam_birth + lam_death
        
        if lam_total == 0:
            break
        
        # Sample waiting time
        tau = np.random.exponential(1.0 / lam_total)
        
        if t + tau > T:
            break
        
        t += tau
        
        # Sample reaction
        if np.random.rand() < lam_birth / lam_total:
            x += 1  # Birth
        else:
            x = max(0, x - 1)  # Death
        
        times.append(t)
        states.append(x)
    
    # Extend to T
    times.append(T)
    states.append(x)
    
    return np.array(times), np.array(states)


def generate_observations(true_times, true_states, obs_times, obs_std, seed=None):
    """
    Generate noisy observations from true trajectory
    
    Parameters
    ----------
    true_times : ndarray
        True event times
    true_states : ndarray
        True population values
    obs_times : ndarray
        Times at which to observe
    obs_std : float
        Observation noise standard deviation
    seed : int, optional
        Random seed
    
    Returns
    -------
    obs_values : ndarray
        Observed values (true + Gaussian noise)
    true_values : ndarray
        True values at observation times
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Interpolate true trajectory at observation times
    true_at_obs = np.interp(obs_times, true_times, true_states)
    
    # Add Gaussian noise
    obs_values = true_at_obs + np.random.normal(0, obs_std, size=len(obs_times))
    
    return obs_values, true_at_obs