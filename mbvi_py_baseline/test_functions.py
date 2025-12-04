# test_functions.py
#!/usr/bin/env python3
import time
import sys
sys.path.insert(0, '.')

print("Testing if functions actually work...")

# Test 1: Gillespie
print("\n[1] Testing Gillespie (should take ~0.5-1s)...")
from src.simulation.gillespie import gillespie_birth_death

start = time.time()
times, states = gillespie_birth_death(5.0, 0.1, 0, 25.0, seed=42)
elapsed = time.time() - start

print(f"   Generated {len(times)} events in {elapsed:.4f}s")
if elapsed < 0.01:
    print("   ⚠️  TOO FAST - Function might be a stub!")
else:
    print("   ✓ Seems real")

# Test 2: Optimization
print("\n[2] Testing VariationalOptimizer (should take ~10-20s)...")
from src.variational.optimization import VariationalOptimizer
from src.variational.observations import GaussianObservationModel
import numpy as np

obs_model = GaussianObservationModel(
    np.array([5.0, 10.0]),
    np.array([20.0, 35.0]),
    5.0
)

optimizer = VariationalOptimizer(5.0, 0.1, 10.0, n_time_points=20)

start = time.time()
result = optimizer.optimize(
    y0=0.0,
    obs_model=obs_model,
    n_iter=5,
    learning_rate=0.05,
    verbose=False
)
elapsed = time.time() - start

print(f"   Optimization took {elapsed:.4f}s")
if elapsed < 1.0:
    print("   ⚠️  TOO FAST - Function might be a stub!")
else:
    print("   ✓ Seems real")

# Test 3: ODE solving
print("\n[3] Testing forward ODE (should take ~0.1-0.5s)...")
from src.moments.moment_ode_extended import forward_ode_birth_death
from scipy.integrate import solve_ivp

lambda_func = lambda t: (1.0, 1.0)

start = time.time()
sol = solve_ivp(
    lambda t, y: forward_ode_birth_death(t, y, 5.0, 0.1, lambda_func),
    [0, 10],
    [0.0],
    t_eval=np.linspace(0, 10, 100),
    method='RK45'
)
elapsed = time.time() - start

print(f"   ODE solve took {elapsed:.4f}s")
if elapsed < 0.01:
    print("   ⚠️  TOO FAST - Function might be a stub!")
else:
    print("   ✓ Seems real")

print("\n" + "="*60)