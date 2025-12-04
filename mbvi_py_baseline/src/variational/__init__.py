"""Variational inference module"""
from .observations import GaussianObservationModel
from .optimization import VariationalOptimizer
from .posterior import compute_posterior_trajectory, compute_prior_trajectory

__all__ = [
    'GaussianObservationModel',
    'VariationalOptimizer', 
    'compute_posterior_trajectory',
    'compute_prior_trajectory'
]
