# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rW6Ucg5vBQ3zR052jg5Zp5mLWuXUNJWf
"""

# Part (c)
import numpy as np
from scipy.integrate import solve_ivp, simps
import matplotlib.pyplot as plt

# Parameters for the nonlinear Schrödinger equation
L = 2  # Interval length for this part
x = np.arange(-L, L + 0.1, 0.1)  # xspan
n = len(x)  # Number of x points
tol = 1e-6  # Tolerance for convergence
dA = 0.01  # Initial increment for amplitude adjustment

# Set gamma values for focusing and defocusing cases
gamma_values = [0.05, -0.05]
results = {}

# Define the RHS function for the ODE with nonlinearity
def hw3_rhs_c(x, y, E, gamma):
    return [y[1], (gamma * y[0]**2 + x**2 - E) * y[0]]

# Loop over both gamma cases
for gamma in gamma_values:
    eigenvalues = []
    wave_functions = []

    for mode in range(2):  # First two modes
        E = 0.2 + 0.5 * mode  # Starting energy estimate for mode
        dE = 0.2  # Initial energy step size
        A = 0.1  # Initial amplitude guess

        # Shooting method for eigenvalue and eigenfunction calculation
        for _ in range(10000):  # Maximum iterations
            # Initial conditions
            y0 = [A, np.sqrt(L**2 - E) * A]

            # Solve the ODE with the current E and gamma
            sol = solve_ivp(lambda x, y: hw3_rhs_c(x, y, E, gamma), [x[0], x[-1]], y0, t_eval=x)
            ys = sol.y[0]  # Solution for φ_n(x)
            bc = sol.y[1, -1] + np.sqrt(L**2 - E) * sol.y[0, -1]  # Boundary condition at x = L

            # Check if the boundary condition is met
            if abs(bc) < tol:
                eigenvalues.append(E)
                break

            # Adjust energy based on boundary condition
            if (-1)**(mode + 1) * bc > 0:
                E += dE
            else:
                E -= dE / 2
                dE /= 2

            # Normalize by adjusting amplitude
            area = simps(ys**2, x)
            if abs(area - 1) < tol:
                break
            if area < 1:
                A += dA
            else:
                A -= dA / 2
                dA /= 2

        # Normalize eigenfunction
        norm_factor = np.sqrt(simps(ys**2, x))
        wave_functions.append(np.abs(ys) / norm_factor)

    # Save results for the current gamma
    results[gamma] = {
        'eigenvalues': np.array(eigenvalues),
        'eigenfunctions': np.array(wave_functions).T
    }

    # Plot results for the current gamma
    plt.figure()
    for i in range(2):
        plt.plot(x, results[gamma]['eigenfunctions'][:, i], label=f'|φ_{i+1}| for γ={gamma}')
    plt.xlabel('x')
    plt.ylabel('Absolute value of eigenfunctions')
    plt.title(f'Normalized eigenfunctions for γ={gamma}')
    plt.legend()
    plt.grid()
    plt.show()

# Save results to files
A5 = results[0.05]['eigenfunctions']
A6 = results[0.05]['eigenvalues']
A7 = results[-0.05]['eigenfunctions']
A8 = results[-0.05]['eigenvalues']

np.save('A5.npy', A5)
np.save('A6.npy', A6)
np.save('A7.npy', A7)
np.save('A8.npy', A8)

print("Eigenvalues for γ=0.05:", A6)
print("Eigenvalues for γ=-0.05:", A8)