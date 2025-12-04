"""
Main execution script for MBVI inference
Runs complete inference pipeline and saves results to file
"""

import numpy as np
import pickle
import json
from pathlib import Path

# Import from src modules
from src.simulation.gillespie import gillespie_birth_death, generate_observations
from src.variational.observations import GaussianObservationModel
from src.variational.optimization import VariationalOptimizer
from src.variational.posterior import compute_posterior_trajectory, compute_prior_trajectory


def run_mbvi_inference(config, output_file='results/mbvi_results.pkl'):
    """
    Run complete MBVI inference pipeline
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - c1: birth rate
        - c2: death rate
        - T: time horizon
        - x0: initial population
        - obs_times: observation times
        - obs_std: observation noise
        - seed: random seed
        - n_time_points: discretization points
        - n_iter: optimization iterations
        - learning_rate: optimization learning rate
    output_file : str
        Path to save results
    
    Returns
    -------
    results : dict
        Complete results dictionary
    """
    print("="*70)
    print("MBVI INFERENCE PIPELINE")
    print("="*70)
    
    # Extract configuration
    c1 = config['c1']
    c2 = config['c2']
    T = config['T']
    x0 = config['x0']
    obs_times = np.array(config['obs_times'])
    obs_std = config['obs_std']
    seed = config.get('seed', 42)
    n_time_points = config.get('n_time_points', 100)
    n_iter = config.get('n_iter', 30)
    learning_rate = config.get('learning_rate', 0.05)
    
    print(f"\nConfiguration:")
    print(f"  Birth rate (c1): {c1}")
    print(f"  Death rate (c2): {c2}")
    print(f"  Time horizon (T): {T}")
    print(f"  Initial population: {x0}")
    print(f"  Observations at: {obs_times}")
    print(f"  Observation noise: {obs_std}")
    
    # Step 1: Generate true trajectory
    print("\n[1/6] Simulating true trajectory (Gillespie)...")
    true_times, true_states = gillespie_birth_death(c1, c2, x0, T, seed=seed)
    print(f"  Generated {len(true_times)} events")
    
    # Step 2: Generate observations
    print("\n[2/6] Generating observations...")
    obs_values, true_at_obs = generate_observations(
        true_times, true_states, obs_times, obs_std, seed=seed+1
    )
    print(f"  True values at obs times: {true_at_obs}")
    print(f"  Observed values: {obs_values}")
    
    # Step 3: Create observation model
    print("\n[3/6] Setting up observation model...")
    obs_model = GaussianObservationModel(obs_times, obs_values, obs_std)
    
    # Step 4: Compute prior trajectory
    print("\n[4/6] Computing prior trajectory...")
    t_eval = np.linspace(0, T, 200)
    prior_mean = compute_prior_trajectory(c1, c2, y0=x0, T=T, t_eval=t_eval)
    
    # Step 5: Optimize variational parameters
    print("\n[5/6] Running variational optimization...")
    optimizer = VariationalOptimizer(c1, c2, T, n_time_points=n_time_points)
    opt_result = optimizer.optimize(
        y0=x0,
        obs_model=obs_model,
        n_iter=n_iter,
        learning_rate=learning_rate,
        verbose=True
    )
    
    # Step 6: Compute posterior trajectory
    print("\n[6/6] Computing posterior trajectory...")
    t_posterior, posterior_mean = compute_posterior_trajectory(
        c1, c2, y0=x0, T=T,
        lambda_birth=opt_result['lambda_birth'],
        lambda_death=opt_result['lambda_death'],
        t_grid=opt_result['t_grid'],
        t_eval=t_eval
    )
    
    # Compute statistics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Prior statistics
    prior_at_obs = np.interp(obs_times, t_eval, prior_mean)
    prior_residuals = obs_values - prior_at_obs
    prior_rmse = np.sqrt(np.mean(prior_residuals**2))
    
    # Posterior statistics
    posterior_at_obs = np.interp(obs_times, t_eval, posterior_mean)
    posterior_residuals = obs_values - posterior_at_obs
    posterior_rmse = np.sqrt(np.mean(posterior_residuals**2))
    
    improvement = (1 - posterior_rmse / prior_rmse) * 100
    
    print(f"\nPrior RMSE:        {prior_rmse:.3f}")
    print(f"Posterior RMSE:    {posterior_rmse:.3f}")
    print(f"Improvement:       {improvement:.1f}%")
    print(f"Noise level (σ):   {obs_std:.3f}")
    print(f"Posterior RMSE/σ:  {posterior_rmse/obs_std:.3f}")
    
    if posterior_rmse < obs_std:
        print("✓ Posterior residuals within noise level")
    
    # Package results
    results = {
        'config': config,
        'true_trajectory': {
            'times': true_times,
            'states': true_states
        },
        'observations': {
            'times': obs_times,
            'values': obs_values,
            'true_values': true_at_obs,
            'std': obs_std
        },
        'prior': {
            'times': t_eval,
            'mean': prior_mean,
            'at_obs': prior_at_obs,
            'residuals': prior_residuals,
            'rmse': prior_rmse
        },
        'posterior': {
            'times': t_posterior,
            'mean': posterior_mean,
            'at_obs': posterior_at_obs,
            'residuals': posterior_residuals,
            'rmse': posterior_rmse
        },
        'optimization': {
            't_grid': opt_result['t_grid'],
            'lambda_birth': opt_result['lambda_birth'],
            'lambda_death': opt_result['lambda_death'],
            'objective_history': opt_result['objective_history']
        },
        'statistics': {
            'prior_rmse': prior_rmse,
            'posterior_rmse': posterior_rmse,
            'improvement_percent': improvement
        }
    }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Also save config and summary as JSON
    json_file = output_path.with_suffix('.json')
    summary = {
        'config': config,
        'statistics': results['statistics']
    }
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {json_file}")
    
    return results


if __name__ == "__main__":
    # Configuration matching Wildner & Koeppl (2019)
    config = {
        'c1': 5.0,           # Birth rate
        'c2': 0.1,           # Death rate
        'T': 25.0,           # Time horizon
        'x0': 0,             # Initial population
        'obs_times': [5.0, 10.0, 15.0, 20.0, 25.0],
        'obs_std': 5.0,      # Observation noise
        'seed': 42,
        'n_time_points': 100,
        'n_iter': 30,
        'learning_rate': 0.05
    }
    
    results = run_mbvi_inference(config, output_file='results/mbvi_results.pkl')
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    print("\nNext step: Run visualization script")
    print("  python run/visualize_results.py")

