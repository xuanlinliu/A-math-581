# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rW6Ucg5vBQ3zR052jg5Zp5mLWuXUNJWf
"""

# Part (d)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define RHS of differential equation
def hw1_rhs_a(x, y, E):
    return [y[1], (x**2 - E) * y[0]]

# Parameters
L = 2
x_span = [-L, L]
E = 1
A = 1
y0 = [A, np.sqrt(L**2 - E) * A]
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# Initialize lists to store average step sizes
dt45, dt23, dtRadau, dtBDF = [], [], [], []

# Perform the study for each tolerance
for tol in tols:
    options = {'rtol': tol, 'atol': tol}

    # Solve with RK45
    sol45 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK45', args=(E,), **options)
    avg_step_size_RK45 = np.mean(np.diff(sol45.t))
    dt45.append(avg_step_size_RK45)

    # Solve with RK23
    sol23 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK23', args=(E,), **options)
    avg_step_size_RK23 = np.mean(np.diff(sol23.t))
    dt23.append(avg_step_size_RK23)

    # Solve with Radau
    solRadau = solve_ivp(hw1_rhs_a, x_span, y0, method='Radau', args=(E,), **options)
    avg_step_size_Radau = np.mean(np.diff(solRadau.t))
    dtRadau.append(avg_step_size_Radau)

    # Solve with BDF
    solBDF = solve_ivp(hw1_rhs_a, x_span, y0, method='BDF', args=(E,), **options)
    avg_step_size_BDF = np.mean(np.diff(solBDF.t))
    dtBDF.append(avg_step_size_BDF)

# Log-log plot and linear fit for each method
log_tols = np.log(tols)

# Compute slopes for each method
# RK45 slope
log_dt45 = np.log(dt45)
fit45 = np.polyfit(log_tols, log_dt45, 1)
slope_RK45 = fit45[0]

# RK23 slope
log_dt23 = np.log(dt23)
fit23 = np.polyfit(log_tols, log_dt23, 1)
slope_RK23 = fit23[0]

# Radau slope
log_dtRadau = np.log(dtRadau)
fitRadau = np.polyfit(log_tols, log_dtRadau, 1)
slope_Radau = fitRadau[0]

# BDF slope
log_dtBDF = np.log(dtBDF)
fitBDF = np.polyfit(log_tols, log_dtBDF, 1)
slope_BDF = fitBDF[0]

# Save slopes in A9
A9 = np.array([slope_RK45, slope_RK23, slope_Radau, slope_BDF])

# Print slopes
print("Slopes for convergence study (RK45, RK23, Radau, BDF):")
print(A9)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(log_tols, log_dt45, 'o-', label=f'RK45 (slope: {slope_RK45:.2f})')
plt.plot(log_tols, log_dt23, 'o-', label=f'RK23 (slope: {slope_RK23:.2f})')
plt.plot(log_tols, log_dtRadau, 'o-', label=f'Radau (slope: {slope_Radau:.2f})')
plt.plot(log_tols, log_dtBDF, 'o-', label=f'BDF (slope: {slope_BDF:.2f})')
plt.xlabel('log(Tolerance)')
plt.ylabel('log(Average Step Size)')
plt.title('Convergence Study of Different Solvers')
plt.legend()
plt.grid(True)
plt.show()