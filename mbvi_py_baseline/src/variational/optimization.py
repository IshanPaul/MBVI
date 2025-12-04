"""
Variational parameter optimization using natural gradient descent
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class VariationalOptimizer:
    """
    Optimize variational scaling parameters lambda(t) for MBVI
    """
    
    def __init__(self, c1, c2, T, n_time_points=100):
        """
        Parameters
        ----------
        c1 : float
            Birth rate
        c2 : float
            Death rate
        T : float
            Time horizon
        n_time_points : int
            Number of time discretization points
        """
        self.c1 = c1
        self.c2 = c2
        self.T = T
        self.n_time_points = n_time_points
        self.t_grid = np.linspace(0, T, n_time_points)
    
    def _create_lambda_interpolator(self, lambda_birth, lambda_death):
        """Create interpolation function for lambda values"""
        return lambda t: (
            np.interp(t, self.t_grid, lambda_birth),
            np.interp(t, self.t_grid, lambda_death)
        )
    
    def _forward_solve(self, y0, lambda_func):
        """Solve forward ODE for mean trajectory"""
        from src.moments.moment_ode_extended import forward_ode_birth_death
        
        sol = solve_ivp(
            lambda t, y: forward_ode_birth_death(t, y, self.c1, self.c2, lambda_func),
            [0, self.T],
            [y0],
            t_eval=self.t_grid,
            method='RK45',
            rtol=1e-6
        )
        
        return sol.y[0]
    
    def _compute_kl_divergence(self, lambda_birth, lambda_death, phi_birth, phi_death):
        """Compute KL divergence term D[Q || P]"""
        kl = np.trapz(
            phi_birth * (1 - lambda_birth + lambda_birth * np.log(lambda_birth + 1e-10)) +
            phi_death * (1 - lambda_death + lambda_death * np.log(lambda_death + 1e-10)),
            self.t_grid
        )
        return kl
    
    def _compute_gradients(self, lambda_birth, lambda_death, phi_birth, phi_death,
                          mean_traj, obs_model):
        """Compute gradients for natural gradient descent"""
        n = len(self.t_grid)
        grad_birth = np.zeros(n)
        grad_death = np.zeros(n)
        
        for i, t in enumerate(self.t_grid):
            # KL gradient
            grad_birth[i] = phi_birth[i] * np.log(lambda_birth[i] + 1e-10)
            grad_death[i] = phi_death[i] * np.log(lambda_death[i] + 1e-10)
            
            # Observation gradient
            obs_grad = obs_model.gradient_at_time(t, mean_traj[i], influence_width=1.5)
            
            if obs_grad != 0:
                # Adjust lambda to reduce observation error
                if obs_grad > 0:  # Mean too high
                    grad_birth[i] += obs_grad
                    grad_death[i] -= obs_grad
                else:  # Mean too low
                    grad_birth[i] += obs_grad
                    grad_death[i] -= obs_grad
        
        return grad_birth, grad_death
    
    def optimize(self, y0, obs_model, n_iter=30, learning_rate=0.05, verbose=True):
        """
        Optimize variational parameters using natural gradient descent
        
        Parameters
        ----------
        y0 : float
            Initial mean
        obs_model : GaussianObservationModel
            Observation model
        n_iter : int
            Number of optimization iterations
        learning_rate : float
            Learning rate for gradient descent
        verbose : bool
            Print progress
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - t_grid: time points
            - lambda_birth: optimized birth scaling
            - lambda_death: optimized death scaling
            - objective_history: objective value at each iteration
        """
        # Initialize lambda in log space (ensures positivity)
        log_lambda_birth = np.zeros(self.n_time_points)
        log_lambda_death = np.zeros(self.n_time_points)
        
        objective_history = []
        
        if verbose:
            print(f"\nOptimizing variational parameters:")
            print(f"  Time points: {self.n_time_points}")
            print(f"  Iterations: {n_iter}")
            print(f"  Learning rate: {learning_rate}")
        
        for iteration in range(n_iter):
            # Current lambda values
            lambda_birth = np.exp(log_lambda_birth)
            lambda_death = np.exp(log_lambda_death)
            
            # Create interpolator
            lambda_func = self._create_lambda_interpolator(lambda_birth, lambda_death)
            
            # Forward solve for mean trajectory
            mean_traj = self._forward_solve(y0, lambda_func)
            
            # Compute moment functions
            from src.moments.moment_ode_extended import compute_moment_functions
            phi_birth, phi_death = compute_moment_functions(mean_traj, self.c1, self.c2)
            
            # Compute objective
            kl = self._compute_kl_divergence(lambda_birth, lambda_death, phi_birth, phi_death)
            
            mean_func = interp1d(self.t_grid, mean_traj, kind='cubic', fill_value='extrapolate')
            log_lik = obs_model.log_likelihood(mean_func)
            
            objective = kl - log_lik
            objective_history.append(objective)
            
            # Compute gradients
            grad_birth, grad_death = self._compute_gradients(
                lambda_birth, lambda_death, phi_birth, phi_death,
                mean_traj, obs_model
            )
            
            # Update (natural gradient: scaled by lambda)
            log_lambda_birth -= learning_rate * grad_birth
            log_lambda_death -= learning_rate * grad_death
            
            if verbose and iteration % 5 == 0:
                print(f"  Iter {iteration:3d}: Objective = {objective:10.3f}, "
                      f"KL = {kl:8.3f}, LogLik = {log_lik:8.3f}")
        
        if verbose:
            print(f"  Final objective: {objective_history[-1]:.3f}")
        
        return {
            't_grid': self.t_grid,
            'lambda_birth': np.exp(log_lambda_birth),
            'lambda_death': np.exp(log_lambda_death),
            'objective_history': objective_history
        }