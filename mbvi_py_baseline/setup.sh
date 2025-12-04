#!/bin/bash
# ============================================================================
# FILE: setup.sh
# Setup script for MBVI Python baseline
# ============================================================================

echo "=========================================="
echo "MBVI Setup Script"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Create directory structure
echo ""
echo "Creating directory structure..."

# Create src subdirectories if they don't exist
mkdir -p src/simulation
mkdir -p src/moments
mkdir -p src/variational

# Create run directory
mkdir -p run

# Create results directory
mkdir -p results

# Create __init__.py files
touch src/__init__.py
touch src/simulation/__init__.py
touch src/variational/__init__.py
touch run/__init__.py

echo "✓ Directory structure created"

# Check if original files exist
echo ""
echo "Checking existing files..."

if [ -f "src/models/base_model.py" ]; then
    echo "✓ Found: src/models/base_model.py"
else
    echo "✗ Missing: src/models/base_model.py"
fi

if [ -f "src/moments/moment_ode.py" ]; then
    echo "✓ Found: src/moments/moment_ode.py"
else
    echo "✗ Missing: src/moments/moment_ode.py"
fi

# Create placeholder __init__.py files with imports
echo ""
echo "Setting up package imports..."

cat > src/simulation/__init__.py << 'EOF'
"""Simulation module for exact stochastic simulation"""
from .gillespie import gillespie_birth_death, generate_observations

__all__ = ['gillespie_birth_death', 'generate_observations']
EOF

cat > src/variational/__init__.py << 'EOF'
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
EOF

echo "✓ Package imports configured"

# Create a test script
echo ""
echo "Creating test script..."

cat > test_installation.py << 'EOF'
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
EOF

chmod +x test_installation.py

echo "✓ Test script created: test_installation.py"

# Run the test
echo ""
echo "Running installation test..."
python3 test_installation.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy the new module code into the files listed above"
echo "2. Run: python run/mbvi_inference.py"
echo "3. Run: python run/visualize_results.py"
echo ""
echo "For help, see MODULAR_STRUCTURE_GUIDE.md"
echo ""