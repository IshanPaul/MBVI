# MBVI Python Baseline (Non-Optimized)

This repository contains a **baseline** Python translation of the original
MATLAB code for Moment-Based Variational Inference (MBVI) for MJPs.

## Features

- Structure mirrors original MBVI repo
- No vectorization (loops preserved)
- Simple Euler ODE solver for profiling
- Moment closure = mean-field
- Variational control = constant scaling
- Pure Python (no PyTorch/JAX yet)

## Goal

This code is intended for **profiling** to create a baseline performance reference.
After profiling, we will add:
- vectorization
- numerical stability enhancements
- PyTorch/JAX autodiff
- parallelization
- optimized ODE solvers

## Run example

