# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rW6Ucg5vBQ3zR052jg5Zp5mLWuXUNJWf
"""

# Part (e)
import numpy as np
from scipy.integrate import simps
from math import factorial, sqrt, pi
import os

# Parameters
L = 4
xspan = np.linspace(-L, L, 81)
n_vals = 5  # Number of eigenvalues/eigenfunctions to compute
norm_errors_a = np.zeros(n_vals)  # To store errors for part (a)
norm_errors_b = np.zeros(n_vals)  # To store errors for part (b)
relative_errors_a = np.zeros(n_vals)
relative_errors_b = np.zeros(n_vals)

# Generate exact Gauss-Hermite eigenfunctions for comparison
h = np.array([
    np.ones_like(xspan),
    2 * xspan,
    4 * xspan**2 - 2,
    8 * xspan**3 - 12 * xspan,
    16 * xspan**4 - 48 * xspan**2 + 12
])

phi_exact = np.zeros((len(xspan), n_vals))
for j in range(n_vals):
    prefactor = np.exp(-xspan**2 / 2) / sqrt(factorial(j) * 2**j * sqrt(pi))
    phi_exact[:, j] = prefactor * h[j, :]

# Calculate or load 'wave_function_matrix' and 'eigenvalues' here.
wave_functions_matrix = []
eigenvalues = []

# Load A1 and A2 from part (a) and A3 and A4 from part (b)
A1 = np.array(wave_functions_matrix)
A2 = np.array(eigenvalues)
A3 = np.array(wave_functions_matrix)
A4 = np.array(eigenvalues)
eigenfunctions_a = np.load('A1.npy')
eigenvalues_a = np.load('A2.npy')
eigenfunctions_b = np.load('A3.npy')
eigenvalues_b = np.load('A4.npy')

# Exact eigenvalues
exact_eigenvalues = np.array([(2 * j + 1) for j in range(n_vals)])

# Calculate errors for part (a) eigenfunctions and eigenvalues
for j in range(n_vals):
    # Eigenfunction error norm
    norm_errors_a[j] = simps((np.abs(A1[:, j]) - np.abs(phi_exact[:, j]))**2, xspan)
    # Relative percent error for eigenvalues
    relative_errors_a[j] = 100 * abs(A2[j] - exact_eigenvalues[j]) / exact_eigenvalues[j]

# Calculate errors for part (b) eigenfunctions and eigenvalues
for j in range(n_vals):
    # Eigenfunction error norm
    norm_errors_b[j] = simps((np.abs(A3[:, j]) - np.abs(phi_exact[:, j]))**2, xspan)
    # Relative percent error for eigenvalues
    relative_errors_b[j] = 100 * abs(A4[j] - exact_eigenvalues[j]) / exact_eigenvalues[j]

# Save the error vectors
A10 = norm_errors_a  # Eigenfunction errors for part (a)
A11 = relative_errors_a  # Eigenvalue errors for part (a)
A12 = norm_errors_b  # Eigenfunction errors for part (b)
A13 = relative_errors_b  # Eigenvalue errors for part (b)

# Print results
print("Eigenfunction errors for part (a):\n", A10)
print("Eigenvalue errors for part (a):\n", A11)
print("Eigenfunction errors for part (b):\n", A12)
print("Eigenvalue errors for part (b):\n", A13)