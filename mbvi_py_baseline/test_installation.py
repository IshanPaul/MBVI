#!/usr/bin/env python3
"""
Quick test to verify installation and imports
"""

import sys

print("Testing MBVI installation...")
print("-" * 50)

# Test basic imports
try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import NumPy: {e}")
    sys.exit(1)

try:
    import scipy
    print("✓ SciPy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SciPy: {e}")
    sys.exit(1)

try:
    import matplotlib
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Matplotlib: {e}")
    sys.exit(1)

# Test original modules
try:
    from src.models.base_model import ReactionNetwork
    print("✓ Original ReactionNetwork imported")
except ImportError as e:
    print(f"⚠ Could not import ReactionNetwork: {e}")

try:
    from src.moments.moment_ode import compute_moment_rhs
    print("✓ Original moment_ode imported")
except ImportError as e:
    print(f"⚠ Could not import moment_ode: {e}")

# Test if new files exist
import os
new_files = [
    'src/simulation/gillespie.py',
    'src/moments/moment_ode_extended.py',
    'src/variational/observations.py',
    'src/variational/optimization.py',
    'src/variational/posterior.py',
    'run/mbvi_inference.py',
    'run/visualize_results.py'
]

print("\nChecking for new module files...")
for filepath in new_files:
    if os.path.exists(filepath):
        print(f"✓ Found: {filepath}")
    else:
        print(f"✗ Missing: {filepath} (needs to be created)")

print("-" * 50)
print("\nInstallation check complete!")
print("\nNext steps:")
print("1. Create the new module files listed above")
print("2. Run: python run/mbvi_inference.py")
print("3. Run: python run/visualize_results.py")
