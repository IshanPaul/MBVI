#!/bin/bash

# Create top-level repo directory
mkdir -p mbvi_py_baseline

# Create src directory
mkdir -p mbvi_py_baseline/src

# Create subdirectories (mirroring MATLAB structure but Pythonic)
mkdir -p mbvi_py_baseline/src/models
mkdir -p mbvi_py_baseline/src/moments
mkdir -p mbvi_py_baseline/src/variational
mkdir -p mbvi_py_baseline/src/solvers
mkdir -p mbvi_py_baseline/src/utils

# Create run directory
mkdir -p mbvi_py_baseline/run

# Create tests directory (optional but helpful later)
mkdir -p mbvi_py_baseline/tests

# Create README
touch mbvi_py_baseline/README.md

# Create placeholder __init__ files for Python packages
touch mbvi_py_baseline/src/__init__.py
touch mbvi_py_baseline/src/models/__init__.py
touch mbvi_py_baseline/src/moments/__init__.py
touch mbvi_py_baseline/src/variational/__init__.py
touch mbvi_py_baseline/src/solvers/__init__.py
touch mbvi_py_baseline/src/utils/__init__.py
touch mbvi_py_baseline/run/__init__.py
touch mbvi_py_baseline/tests/__init__.py

echo "Directory structure for mbvi_py_baseline created successfully."
