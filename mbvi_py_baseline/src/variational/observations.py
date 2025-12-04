"""
Observation model and likelihood functions
"""

import numpy as np


class GaussianObservationModel:
    """
    Gaussian observation model: Y_k ~ N(X(t_k), sigma^2)
    """
    
    def __init__(self, obs_times, obs_values, obs_std):
        """
        Parameters
        ----------
        obs_times : ndarray
            Observation times
        obs_values : ndarray
            Observed values
        obs_std : float
            Observation noise standard deviation
        """
        self.obs_times = obs_times
        self.obs_values = obs_values
        self.obs_std = obs_std
        self.n_obs = len(obs_times)
    
    def log_likelihood(self, mean_traj_func):
        """
        Compute log-likelihood of observations given mean trajectory
        
        Parameters
        ----------
        mean_traj_func : callable
            Function that returns mean at any time t
        
        Returns
        -------
        log_lik : float
            Total log-likelihood
        """
        log_lik = 0.0
        
        for t_obs, y_obs in zip(self.obs_times, self.obs_values):
            mean_at_obs = mean_traj_func(t_obs)
            log_lik -= 0.5 * ((mean_at_obs - y_obs) / self.obs_std) ** 2
        
        return log_lik
    
    def gradient_at_time(self, t, mean, influence_width=1.0):
        """
        Compute gradient contribution from observations at time t
        
        Parameters
        ----------
        t : float
            Current time
        mean : float
            Current mean value
        influence_width : float
            Width of temporal influence window
        
        Returns
        -------
        grad : float
            Gradient contribution
        """
        grad = 0.0
        
        for t_obs, y_obs in zip(self.obs_times, self.obs_values):
            # Temporal influence (Gaussian kernel)
            time_dist = abs(t - t_obs)
            influence = np.exp(-0.5 * (time_dist / influence_width) ** 2)
            
            # Observation error
            obs_error = mean - y_obs
            
            # Gradient contribution
            grad += influence * obs_error / (self.obs_std ** 2)
        
        return grad