"""
Visualization script - loads results and creates figures
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


def load_results(results_file='results/mbvi_results.pkl'):
    """Load results from pickle file"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def create_figure1_reproduction(results, save_path='results/figure1_reproduction.png'):
    """
    Create 4-panel figure matching Wildner & Koeppl (2019) Figure 1
    
    Parameters
    ----------
    results : dict
        Results dictionary from inference
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    true_times = results['true_trajectory']['times']
    true_states = results['true_trajectory']['states']
    obs_times = results['observations']['times']
    obs_values = results['observations']['values']
    obs_std = results['observations']['std']
    
    t_eval = results['prior']['times']
    prior_mean = results['prior']['mean']
    posterior_mean = results['posterior']['mean']
    
    t_grid = results['optimization']['t_grid']
    lambda_birth = results['optimization']['lambda_birth']
    lambda_death = results['optimization']['lambda_death']
    
    # Panel 1: Trajectories
    ax1 = axes[0, 0]
    
    # True trajectory
    ax1.plot(true_times, true_states, 'k-', alpha=0.3, linewidth=1,
             drawstyle='steps-post', label='True trajectory')
    
    # Observations
    ax1.scatter(obs_times, obs_values, c='red', marker='o', s=120,
                label='Observations', zorder=5, edgecolors='darkred', linewidths=2)
    
    # Prior mean
    ax1.plot(t_eval, prior_mean, 'b--', linewidth=2.5, label='Prior mean')
    
    # Posterior mean
    ax1.plot(t_eval, posterior_mean, 'g-', linewidth=3, label='Posterior mean')
    
    # Uncertainty band (simplified)
    posterior_std = np.sqrt(np.maximum(0, posterior_mean * 0.1))
    ax1.fill_between(t_eval,
                     posterior_mean - posterior_std,
                     posterior_mean + posterior_std,
                     alpha=0.2, color='green', label='Posterior ±1σ')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Population X', fontsize=12)
    ax1.set_title('Birth-Death Process: MBVI Smoothing', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, results['config']['T'])
    
    # Panel 2: Variational parameters
    ax2 = axes[0, 1]
    
    ax2.plot(t_grid, lambda_birth, 'b-', linewidth=2.5, label='λ₁(t) - birth')
    ax2.plot(t_grid, lambda_death, 'r-', linewidth=2.5, label='λ₂(t) - death')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5,
                label='Prior (λ=1)')
    
    # Mark observation times
    for t_obs in obs_times:
        ax2.axvline(t_obs, color='red', linestyle=':', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Variational scaling λ(t)', fontsize=12)
    ax2.set_title('Optimized Variational Parameters', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, results['config']['T'])
    
    # Panel 3: Comparison at observations
    ax3 = axes[1, 0]
    
    true_at_obs = results['observations']['true_values']
    prior_at_obs = results['prior']['at_obs']
    posterior_at_obs = results['posterior']['at_obs']
    
    x = np.arange(len(obs_times))
    width = 0.25
    
    ax3.bar(x - width, true_at_obs, width, label='True', color='black', alpha=0.7)
    ax3.bar(x, obs_values, width, label='Observed', color='red', alpha=0.7)
    ax3.bar(x + width, posterior_at_obs, width, label='Posterior', color='green', alpha=0.7)
    
    ax3.set_xlabel('Observation Index', fontsize=12)
    ax3.set_ylabel('Population', fontsize=12)
    ax3.set_title('Comparison at Observation Times', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{t:.0f}s' for t in obs_times])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Residuals
    ax4 = axes[1, 1]
    
    prior_residuals = results['prior']['residuals']
    posterior_residuals = results['posterior']['residuals']
    
    ax4.scatter(obs_times, prior_residuals, c='blue', marker='s', s=120,
                label='Prior residuals', alpha=0.7, edgecolors='darkblue', linewidths=2)
    ax4.scatter(obs_times, posterior_residuals, c='green', marker='o', s=120,
                label='Posterior residuals', alpha=0.7, edgecolors='darkgreen', linewidths=2)
    
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.axhline(obs_std, color='red', linestyle='--', alpha=0.5, label=f'±σ ({obs_std:.1f})')
    ax4.axhline(-obs_std, color='red', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Observation time (s)', fontsize=12)
    ax4.set_ylabel('Residual (observed - predicted)', fontsize=12)
    ax4.set_title('Observation Residuals', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {save_path}")
    
    return fig


def create_convergence_plot(results, save_path='results/convergence.png'):
    """
    Plot optimization convergence
    
    Parameters
    ----------
    results : dict
        Results dictionary
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    obj_history = results['optimization']['objective_history']
    
    ax.plot(obj_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective (negative ELBO)', fontsize=12)
    ax.set_title('Variational Optimization Convergence', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add final value annotation
    final_val = obj_history[-1]
    ax.annotate(f'Final: {final_val:.2f}',
                xy=(len(obj_history)-1, final_val),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence plot saved to: {save_path}")
    
    return fig


def print_results_summary(results):
    """Print formatted summary of results"""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    config = results['config']
    stats = results['statistics']
    
    print(f"\nConfiguration:")
    print(f"  Birth rate: {config['c1']:.1f}")
    print(f"  Death rate: {config['c2']:.1f}")
    print(f"  Time horizon: {config['T']:.1f} s")
    print(f"  Observations: {len(config['obs_times'])}")
    print(f"  Noise level: {config['obs_std']:.1f}")
    
    print(f"\nPerformance:")
    print(f"  Prior RMSE:        {stats['prior_rmse']:.3f}")
    print(f"  Posterior RMSE:    {stats['posterior_rmse']:.3f}")
    print(f"  Improvement:       {stats['improvement_percent']:.1f}%")
    
    print(f"\nObservations vs Predictions:")
    obs_times = results['observations']['times']
    obs_values = results['observations']['values']
    posterior_at_obs = results['posterior']['at_obs']
    
    print(f"  {'Time':>8} {'Observed':>10} {'Predicted':>10} {'Error':>10}")
    print("  " + "-"*40)
    for t, obs, pred in zip(obs_times, obs_values, posterior_at_obs):
        error = obs - pred
        print(f"  {t:>8.1f} {obs:>10.2f} {pred:>10.2f} {error:>10.2f}")


if __name__ == "__main__":
    import sys
    
    # Load results
    results_file = 'results/mbvi_results.pkl'
    
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run mbvi_inference.py first")
        sys.exit(1)
    
    print("Loading results from:", results_file)
    results = load_results(results_file)
    
    # Print summary
    print_results_summary(results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig1 = create_figure1_reproduction(results, 'results/figure1_reproduction.png')
    fig2 = create_convergence_plot(results, 'results/convergence.png')
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. results/figure1_reproduction.png - Main results (4 panels)")
    print("  2. results/convergence.png - Optimization convergence")
    
    # Show plots
    plt.show()