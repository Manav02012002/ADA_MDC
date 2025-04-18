#!/usr/bin/env python
# coding: utf-8

# # Part 1
# 

# ### Step 1.1

# In[54]:


import numpy as np
import matplotlib.pyplot as plt

#Load the data
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

#Calculating sums necessary for calculation
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)

#OLS formulas for y = a + bx
D = n * X2 - X**2
b = (n * XY - X * Y) / D
a = (Y * X2 - X * XY) / D

#Predicted y
y_pred = a + b * x

#variance estimate
σ2 = np.sum((y - y_pred)**2) / (n - 2)
#here(y - y_pred)=residuals

#Errors
var_b = σ2 * n / D
var_a = σ2 * X2 / D
cov_ab = -σ2 * X / D
std_b = np.sqrt(var_b)
std_a = np.sqrt(var_a)

#Plot the data and fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data', color='black')
plt.plot(x, y_pred, label='Fit: y = ' + str(round(a, 2)) + " + " + str(round(b, 2)) + 'x', color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("OLS Fit for y = a + bx")
plt.legend()
plt.grid(True)
plt.show()

#Output results
print("OLS Fit Results for y = a + bx:")
print("(i)   a (intercept)    = "+str(round(a, 4))+" ± "+ str(round(std_a, 4)))
print("(ii)  b (slope)        = "+str(round(b, 4))+" ± "+ str(round(std_b, 4)))
print("(iii) Noise variance   = "+str(round(σ2, 4)))
print("(iv)  Covariance(a, b) = "+str(round(cov_ab, 4)))


# a)
# LS estimators of a,b:
#            
#            b(LS)=2.6528
#            
#            a(LS)=8.4577
# 
# b)
# Variance of y: 0.5058
# 
# c)
# Errors on LS Estimates of a,b:
#            
#            b)± 0.0078
#            
#            a)± 0.0449
#         
#    Covariance of a and b: -0.0003
# 
# d)
# Plot Paste            
#             
#             
#             

# ## Step 1.2(choose final)

# In[45]:


import numpy as np
import matplotlib.pyplot as plt

# σ^2 from Step 1.1
σ = np.sqrt(σ2)

# Step 1.2a: Define the grid for a and b
a_range = np.linspace(0.31, 0.91, 100)  # Based on a = 0.61 ± 3*0.0995
b_range = np.linspace(1.46, 1.64, 100)  # Based on b = 1.55 ± 3*0.03
A, B = np.meshgrid(a_range, b_range)

# Compute log-likelihood and chi-squared on the grid
log_likelihood = np.zeros_like(A)
chi2 = np.zeros_like(A)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        a = A[i, j]
        b = B[i, j]
        y_pred = a + b * x
        residuals = y - y_pred
        chi2[i, j] = np.sum(residuals**2) / σ2
        log_likelihood[i, j] = -n/2 * np.log(2 * np.pi * σ2) - chi2[i, j] / 2

# Step 1.2b: Find minimum chi-squared
chi2_min = np.min(chi2)
min_idx = np.unravel_index(np.argmin(σ2), chi2.shape)
a_min = A[min_idx]
b_min = B[min_idx]

print(f"Minimum chi-squared: {chi2_min:.4f}")
print(f"(a, b) at minimum chi-squared: ({a_min:.4f}, {b_min:.4f})")

# Step 1.2c: Compute Delta chi-squared
delta_chi2 = chi2 - chi2_min

# Step 1.2d: Plot 2D credible regions
plt.figure(figsize=(8, 6))
levels = [2.30, 6.18, 11.83]  # 68.3%, 95.4%, 99.73%
contours = plt.contour(A, B, delta_chi2, levels=levels, colors=['blue', 'green', 'red'])
plt.clabel(contours, inline=True, fmt={2.30: '68.3%', 6.18: '95.4%', 11.83: '99.73%'}, fontsize=10)
plt.scatter(a_min, b_min, color='black', marker='x', label='Minimum chi-squared')
plt.xlabel('a (intercept)')
plt.ylabel('b (slope)')
plt.title('2D Bayesian Credible Regions for (a, b)')
plt.legend()
plt.grid(True)
plt.savefig('credible_regions.png')

# Step 1.2e: Approximate (a, b) values at credible regions
credible_regions = {}
for level, label in zip(levels, ['68.3%', '95.4%', '99.73%']):
    b_idx = min_idx[0]
    delta_chi2_row = delta_chi2[b_idx, :]
    a_vals = a_range
    idx_lower = np.where(delta_chi2_row <= level)[0]
    if len(idx_lower) > 0:
        a_lower = a_vals[idx_lower[0]]
        a_upper = a_vals[idx_lower[-1]]
    else:
        a_lower, a_upper = a_min, a_min
    credible_regions[label] = (a_lower, a_upper, b_min)

# Print results
print("\nValues at credible regions (approximated along b = b_min):")
for label, (a_lower, a_upper, b_val) in credible_regions.items():
    print(f"{label} region: a from {a_lower:.4f} to {a_upper:.4f}, b = {b_val:.4f}")


# In[46]:


import numpy as np
import matplotlib.pyplot as plt

# OLS results from Step 1.1
a_ols = 8.4577
b_ols = 2.6528
std_a = 0.0449
std_b = 0.0078
n = 19  # Inferred

# Step 1.2a: Define the grid for a and b
a_range = np.linspace(8.32, 8.59, 100)
b_range = np.linspace(2.63, 2.68, 100)
A, B = np.meshgrid(a_range, b_range)

# Without data, we approximate the credible regions using the Gaussian approximation
# Assume chi-squared follows a Gaussian distribution around the OLS estimates
delta_chi2 = ((A - a_ols) / std_a)**2 + ((B - b_ols) / std_b)**2

# Step 1.2b: Minimum chi-squared (approximated)
chi2_min = n - 2  # Expected for a good fit
a_min = a_ols
b_min = b_ols

print("Minimum chi-squared (approximated):", chi2_min)
print("(a, b) at minimum chi-squared:", (a_min, b_min))

# Step 1.2d: Plot 2D credible regions
plt.figure(figsize=(8, 6))
levels = [2.30, 6.18, 11.83]
contours = plt.contour(A, B, delta_chi2, levels=levels, colors=['blue', 'green', 'red'])
plt.clabel(contours, inline=True, fmt={2.30: '68.3%', 6.18: '95.4%', 11.83: '99.73%'}, fontsize=10)
plt.scatter(a_min, b_min, color='black', marker='x', label='Minimum chi-squared')
plt.xlabel('a (intercept)')
plt.ylabel('b (slope)')
plt.title('2D Bayesian Credible Regions for (a, b)')
plt.legend()
plt.grid(True)
plt.savefig('credible_regions.png')

# Step 1.2e: Approximate credible regions
credible_regions = {}
for level, label in zip(levels, ['68.3%', '95.4%', '99.73%']):
    # For a Gaussian, sqrt(level) gives the sigma level
    sigma = np.sqrt(level)
    a_lower = a_ols - sigma * std_a
    a_upper = a_ols + sigma * std_a
    b_lower = b_ols - sigma * std_b
    b_upper = b_ols + sigma * std_b
    credible_regions[label] = (a_lower, a_upper, b_lower, b_upper)

print("\nValues at credible regions:")
for label, (a_lower, a_upper, b_lower, b_upper) in credible_regions.items():
    print(label, "region: a from", round(a_lower, 4), "to", round(a_upper, 4), ", b from", round(b_lower, 4), "to", round(b_upper, 4))


# In[47]:


import numpy as np
import matplotlib.pyplot as plt

# === Step 0: Load data and compute OLS estimates (used for grid center) ===
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# OLS fit for reference
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b_ols = (n * XY - X * Y) / D
a_ols = (Y * X2 - X * XY) / D
y_pred_ols = a_ols + b_ols * x
σ2 = np.sum((y - y_pred_ols)**2) / (n - 2)

# === Step 1.2a: Create grid over (a, b) and compute χ² on grid ===
a_vals = np.linspace(a_ols - 2, a_ols + 2, 400)
b_vals = np.linspace(b_ols - 2, b_ols + 2, 400)
A, B = np.meshgrid(a_vals, b_vals)
chi2_grid = np.zeros_like(A)

for i in range(len(a_vals)):
    for j in range(len(b_vals)):
        a_trial = A[j, i]
        b_trial = B[j, i]
        y_model = a_trial + b_trial * x
        residuals = y - y_model
        chi2 = np.sum((residuals**2) / σ2)
        chi2_grid[j, i] = chi2

# === Step 1.2b: Find minimum χ² and corresponding (a, b) ===
chi2_min = np.min(chi2_grid)
min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
a_best = A[min_idx]
b_best = B[min_idx]

print("Minimum chi-squared:", round(chi2_min, 4))
print("Best-fit parameters:")
print("  a =", round(a_best, 4))
print("  b =", round(b_best, 4))

# === Step 1.2c: Compute Δχ² grid ===
delta_chi2 = chi2_grid - chi2_min

# === Step 1.2d: Plot credible regions from Δχ² contours ===
# Values from Lecture 6 for 2 parameters
levels = [2.30, 6.17, 11.8]  # 68.3%, 95.4%, 99.73%

plt.figure(figsize=(10, 6))
cp = plt.contour(A, B, delta_chi2, levels=levels, colors=['blue', 'green', 'red'])
plt.clabel(cp, fmt={2.30: '68.3%', 6.17: '95.4%', 11.8: '99.73%'})
plt.scatter([a_best], [b_best], color='black', marker='x', label='Minimum χ²')
plt.xlabel("a")
plt.ylabel("b")
plt.title("Credible Regions in (a, b) Parameter Space (Δχ² Contours)")
plt.grid(True)
plt.legend()
plt.show()

# === Step 1.2e: List parameter values at best-fit and within each credible region ===
for level in levels:
    mask = delta_chi2 <= level
    a_region = A[mask]
    b_region = B[mask]
    a_mean = np.mean(a_region)
    b_mean = np.mean(b_region)
    print(f"\nCredible region {level} Δχ² (~{['68.3%', '95.4%', '99.73%'][levels.index(level)]}):")
    print(f"  Mean a in region: {round(a_mean, 4)}")
    print(f"  Mean b in region: {round(b_mean, 4)}")


# In[48]:


import numpy as np
import matplotlib.pyplot as plt

# === [Step 0] Load data and compute OLS estimates (used for grid center and σ²) ===
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# OLS fit
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b_ols = (n * XY - X * Y) / D
a_ols = (Y * X2 - X * XY) / D
y_pred = a_ols + b_ols * x

# Estimate noise variance from OLS residuals
σ2 = np.sum((y - y_pred)**2) / (n - 2)

# === [Step 1.2a] Compute log-likelihood (or chi-squared) over a grid of (a, b) ===
# Grid of a and b centered around OLS estimates
num_points = 400  # fine grid for smooth contours
a_vals = np.linspace(a_ols - 2, a_ols + 2, num_points)
b_vals = np.linspace(b_ols - 2, b_ols + 2, num_points)
A, B = np.meshgrid(a_vals, b_vals)

# Compute chi-squared at each grid point
chi2_grid = np.zeros_like(A)
for i in range(num_points):
    for j in range(num_points):
        a_trial = A[i, j]
        b_trial = B[i, j]
        y_model = a_trial + b_trial * x
        residuals = y - y_model
        chi2 = np.sum((residuals**2) / σ2)
        chi2_grid[i, j] = chi2

# === [Step 1.2b] Compute the minimum chi-squared value and best-fit parameters ===
chi2_min = np.min(chi2_grid)
min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
a_best = A[min_idx]
b_best = B[min_idx]

print("=== Step 1.2b: Minimum chi-squared and best-fit parameters ===")
print(f"Minimum chi-squared: {chi2_min:.4f}")
print(f"Best-fit a: {a_best:.4f}")
print(f"Best-fit b: {b_best:.4f}")

# === [Step 1.2c] Compute Δχ² values for each (a, b) on the grid ===
delta_chi2 = chi2_grid - chi2_min

# === [Step 1.2d] Plot 2D Bayesian credible regions using Δχ² ===
# Δχ² thresholds for 2 parameters from Lecture 6
levels = [2.30, 6.17, 11.8]
labels = {2.30: '68.3%', 6.17: '95.4%', 11.8: '99.73%'}

plt.figure(figsize=(10, 6))
contours = plt.contour(A, B, delta_chi2, levels=levels, colors=['blue', 'green', 'red'])
plt.clabel(contours, fmt=labels, fontsize=10)
plt.scatter([a_best], [b_best], color='black', marker='x', s=80, label='Minimum χ²')
plt.xlabel('a (intercept)')
plt.ylabel('b (slope)')
plt.title('Step 1.2d: Credible Regions from Δχ² Contours')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === [Step 1.2e] Report mean (a, b) values within each credible region ===
print("\n=== Step 1.2e: Mean a and b within each credible region ===")
for level in levels:
    region_mask = delta_chi2 <= level
    a_region = A[region_mask]
    b_region = B[region_mask]
    a_mean = np.mean(a_region)
    b_mean = np.mean(b_region)
    print(f"Credible Region Δχ² ≤ {level} ({labels[level]}):")
    print(f"  Mean a: {a_mean:.4f}")
    print(f"  Mean b: {b_mean:.4f}")


# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# Load the data
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# OLS fit to obtain initial values for the linear model
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)

D = n * X2 - X**2
b_OLS = (n * XY - X * Y) / D
a_OLS = (Y * X2 - X * XY) / D

# Predicted y values from OLS
y_pred = a_OLS + b_OLS * x

# Estimate the noise variance from OLS residuals
σ2 = np.sum((y - y_pred)**2) / (n - 2)

# Define log-likelihood for the linear model
def log_likelihood_linear(a, b):
    residuals = y - (a + b * x)
    log_likelihood = -0.5 * np.sum(residuals**2) / σ2
    return log_likelihood

# Define log-likelihood for the quadratic model y = a + bx + cx^2
def log_likelihood_quadratic(a, b, c):
    residuals = y - (a + b * x + c * x**2)
    log_likelihood = -0.5 * np.sum(residuals**2) / σ2
    return log_likelihood

# Grid setup for parameter values (sampling 100 points for each parameter)
a_vals = np.linspace(a_OLS - 1, a_OLS + 1, 100)
b_vals = np.linspace(b_OLS - 1, b_OLS + 1, 100)
c_vals = np.linspace(-1, 1, 100)  # Range for c in quadratic model

# Log-likelihood grid for linear model
log_L_linear = np.zeros((len(a_vals), len(b_vals)))

# Compute log-likelihood for each (a, b) pair in the grid for linear model
for i in range(len(a_vals)):
    for j in range(len(b_vals)):
        log_L_linear[i, j] = log_likelihood_linear(a_vals[i], b_vals[j])

# Compute the logsumexp to calculate the marginal likelihood for the linear model
marginal_log_L_linear = logsumexp(log_L_linear)

# Log-likelihood grid for quadratic model
log_L_quadratic = np.zeros((len(a_vals), len(b_vals), len(c_vals)))

# Compute log-likelihood for each (a, b, c) triplet in the grid for quadratic model
for i in range(len(a_vals)):
    for j in range(len(b_vals)):
        for k in range(len(c_vals)):
            log_L_quadratic[i, j, k] = log_likelihood_quadratic(a_vals[i], b_vals[j], c_vals[k])

# Compute the logsumexp to calculate the marginal likelihood for the quadratic model
marginal_log_L_quadratic = logsumexp(log_L_quadratic)

# Plot the marginal likelihood for both models
plt.figure(figsize=(10, 6))

# Linear model plot
plt.subplot(1, 2, 1)
plt.contourf(a_vals, b_vals, log_L_linear, cmap='viridis')
plt.colorbar(label='Log-Likelihood')
plt.title("Log-Likelihood for Linear Model")
plt.xlabel("a (intercept)")
plt.ylabel("b (slope)")

# Quadratic model plot
plt.subplot(1, 2, 2)
plt.contourf(a_vals, b_vals, np.sum(log_L_quadratic, axis=2), cmap='viridis')
plt.colorbar(label='Log-Likelihood')
plt.title("Log-Likelihood for Quadratic Model")
plt.xlabel("a (intercept)")
plt.ylabel("b (slope)")

plt.tight_layout()
plt.show()

# Output the marginal likelihoods for both models
print(f"Marginal Log-Likelihood for Linear Model: {marginal_log_L_linear:.4f}")
print(f"Marginal Log-Likelihood for Quadratic Model: {marginal_log_L_quadratic:.4f}")


# In[50]:


import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load the data and perform OLS fit (Step 1.1)
# -------------------------------
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# Calculate OLS sums:
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)

# OLS analytical formulas for y = a + bx:
D = n * X2 - X**2
b_ols = (n * XY - X * Y) / D
a_ols = (Y * X2 - X * XY) / D

# Predicted y values using OLS solution:
y_pred = a_ols + b_ols * x

# Noise variance estimate (using n-2 degrees of freedom)
sigma2 = np.sum((y - y_pred)**2) / (n - 2)

# Standard errors for a and b:
std_a = np.sqrt(sigma2 * X2 / D)
std_b = np.sqrt(sigma2 * n / D)

# Report OLS estimates
print("OLS Fit Results:")
print("a (intercept) =", a_ols, "±", std_a)
print("b (slope)     =", b_ols, "±", std_b)
print("Noise variance sigma^2 =", sigma2)

# -------------------------------
# Step 1.2: Maximum Likelihood / Chi-squared Grid Analysis
# -------------------------------

# (a) Set up a rectangular grid over (a, b)
# We choose our grid ranges as ±5 standard errors around the OLS best-fit values.
# This choice is made so that the grid covers the region where the likelihood is significant.
N_a = 100  # number of grid points for a
N_b = 100  # number of grid points for b

a_vals = np.linspace(a_ols - 5*std_a, a_ols + 5*std_a, N_a)
b_vals = np.linspace(b_ols - 5*std_b, b_ols + 5*std_b, N_b)

# Create 2D grid arrays. The 'indexing' parameter ensures that A varies along rows.
A, B = np.meshgrid(a_vals, b_vals, indexing='ij')  # shapes (N_a, N_b)

# Initialize arrays to store chi-squared and log likelihood values at each grid point.
chi2_grid = np.zeros((N_a, N_b))
logL_grid = np.zeros((N_a, N_b))

# Compute chi-squared and log likelihood on the grid.
# For a model y = a + b*x with Gaussian noise (variance sigma2), the log likelihood is:
# ln[L(a,b)] = -0.5 * n * ln(2*pi*sigma2) - (1/(2*sigma2)) * sum((y - (a+b*x))^2)
for i in range(N_a):
    for j in range(N_b):
        model = A[i, j] + B[i, j] * x  # predicted values for the grid point (a, b)
        residuals = y - model
        # Compute chi-squared (note: we define chi2 as sum(residual^2)/sigma2)
        chi2 = np.sum(residuals**2) / sigma2
        chi2_grid[i, j] = chi2
        # Compute log likelihood; the constant term is common to all grid points.
        logL_grid[i, j] = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2

# (b) Find the minimum chi-squared value and the corresponding (a, b)
min_index = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
min_chi2 = chi2_grid[min_index]
a_best = A[min_index]
b_best = B[min_index]

print("\nMaximum Likelihood / Chi-squared Analysis Results:")
print("Minimum chi-squared =", min_chi2)
print("Best-fit parameters: a =", a_best, ", b =", b_best)

# (c) Compute the Delta chi-squared array (Δχ² = χ² - χ²_min)
delta_chi2 = chi2_grid - min_chi2

# (d) Plot the 2-dimensional credible regions
# For two fitted parameters, the standard Delta chi-squared levels for 68.3%, 95.4%, and 99.73%
# credible regions are, respectively, 2.30, 6.17, and 11.8.
delta_levels = [2.30, 6.17, 11.8]

plt.figure(figsize=(10, 6))
# Contour plot of Delta chi-squared
contours = plt.contour(A, B, delta_chi2, levels=delta_levels, colors=['blue','green','red'])
plt.clabel(contours, inline=True, fontsize=10, fmt='%1.2f')
# Mark the best-fit point for reference
plt.plot(a_best, b_best, 'ko', label='Best fit')
plt.xlabel("a (intercept)")
plt.ylabel("b (slope)")
plt.title("Bayesian Credible Regions in Parameter Space\n(Contour levels indicate Δχ²)")
plt.legend()
plt.grid(True)
plt.show()

# (e) Report the best-fit values and the parameter ranges within each credible region.
# We extract, for each Delta chi-squared threshold, the minimum and maximum a and b values
# that lie within that credible region.
print("\nCredible Region Parameter Ranges:")
credible_labels = ["68.3%", "95.4%", "99.73%"]
for level, label in zip(delta_levels, credible_labels):
    # Get indices where Δχ² is less than or equal to the threshold.
    indices = np.where(delta_chi2 <= level)
    if indices[0].size > 0:
        a_in_region = A[indices]
        b_in_region = B[indices]
        a_range = (np.min(a_in_region), np.max(a_in_region))
        b_range = (np.min(b_in_region), np.max(b_in_region))
        print(f"{label} credible region:")
        print(f"   a range: [{a_range[0]:.4f}, {a_range[1]:.4f}]")
        print(f"   b range: [{b_range[0]:.4f}, {b_range[1]:.4f}]")
    else:
        print(f"No grid points found within the {label} credible region.")


# In[51]:


import numpy as np
import matplotlib.pyplot as plt

# Load the data (same data setup as given in Step 1.1)
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# Ordinary least squares estimates (from Step 1.1)
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
a_ols = (Y * X2 - X * XY) / D
b_ols = (n * XY - X * Y) / D
y_pred = a_ols + b_ols * x

# Variance of residuals (assumed Gaussian noise)
sigma2 = np.sum((y - y_pred)**2) / (n - 2)

# Grid definition for parameters `a` and `b` around OLS estimates
N_a = 100  # Number of grid points for intercept `a`
N_b = 100  # Number of grid points for slope `b`
a_range = np.linspace(a_ols - 5*np.sqrt(sigma2 * X2 / D), a_ols + 5*np.sqrt(sigma2 * X2 / D), N_a)
b_range = np.linspace(b_ols - 5*np.sqrt(sigma2 * n / D), b_ols + 5*np.sqrt(sigma2 * n / D), N_b)

# Create 2D grids
A, B = np.meshgrid(a_range, b_range, indexing='ij')
chi2_grid = np.zeros((N_a, N_b))
delta_chi2 = np.zeros((N_a, N_b))

# Calculate chi-squared values on the grid
for i in range(N_a):
    for j in range(N_b):
        model = A[i, j] + B[i, j] * x  # Predicted y values for current (a, b)
        residuals = y - model
        chi2_grid[i, j] = np.sum(residuals**2 / sigma2)  # Normalized chi-squared

# Minimum chi-squared value and best-fit parameters
min_chi2 = np.min(chi2_grid)
min_indices = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
best_a = A[min_indices]
best_b = B[min_indices]

# Δχ² calculation
delta_chi2 = chi2_grid - min_chi2

# Credible regions for Bayesian estimation
credible_levels = [2.30, 6.17, 11.8]  # 68.3%, 95.4%, 99.73% levels (Lecture 6 thresholds)

# Covariance matrix calculation (using Taylor expansion and second derivatives)
cov_matrix = np.zeros((2, 2))
cov_matrix[0, 0] = sigma2 * X2 / D  # Variance of `a`
cov_matrix[1, 1] = sigma2 * n / D  # Variance of `b`
cov_matrix[0, 1] = cov_matrix[1, 0] = -sigma2 * X / D  # Covariance of `a` and `b`

print("\nAdjusted Bayesian Maximum Likelihood Results:")
print(f"(i) Minimum chi-squared: {min_chi2}")
print(f"(ii) Best-fit a: {best_a}")
print(f"(iii) Best-fit b: {best_b}")
print(f"(iv) Covariance matrix:\n{cov_matrix}")

# Plotting credible regions and Bayesian contours
plt.figure(figsize=(10, 6))
contours = plt.contour(A, B, delta_chi2, levels=credible_levels, colors=['blue', 'green', 'red'])
plt.clabel(contours, inline=True, fontsize=8, fmt='%1.2f')
plt.scatter(best_a, best_b, color='black', label='Best-fit point')
plt.xlabel("Intercept (a)")
plt.ylabel("Slope (b)")
plt.title("Bayesian Credible Regions based on Δχ²")
plt.legend()
plt.grid(True)
plt.show()

# Range within credible regions
print("\nCredible Regions:")
credible_labels = ["68.3%", "95.4%", "99.73%"]
for level, label in zip(credible_levels, credible_labels):
    indices = np.where(delta_chi2 <= level)
    a_in_region = A[indices]
    b_in_region = B[indices]
    a_range_credible = (np.min(a_in_region), np.max(a_in_region))
    b_range_credible = (np.min(b_in_region), np.max(b_in_region))
    print(f"{label} credible region:")
    print(f"   a range: {a_range_credible}")
    print(f"   b range: {b_range_credible}")


# In[52]:



# Ordinary least squares estimates (from Step 1.1)
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b = (n * XY - X * Y) / D
a = (Y * X2 - X * XY) / D
y_pred = a + b * x


# Variance of residuals (assumed Gaussian noise)
σ2 = np.sum((y - y_pred)**2) / (n - 2)

# Grid definition for parameters `a` and `b` around OLS estimates
N_a = 100  # Number of grid points for intercept `a`
N_b = 100  # Number of grid points for slope `b`
a_range = np.linspace(a - 5*np.sqrt(σ2 * X2 / D), a + 5*np.sqrt(σ2 * X2 / D), N_a)
b_range = np.linspace(b - 5*np.sqrt(σ2* n / D), b + 5*np.sqrt(σ2 * n / D), N_b)

# Create 2D grids
A, B = np.meshgrid(a_range, b_range, indexing='ij')
chi2_grid = np.zeros((N_a, N_b))
delta_chi2 = np.zeros((N_a, N_b))

# Calculate chi-squared values on the grid
for i in range(N_a):
    for j in range(N_b):
        model = A[i, j] + B[i, j] * x  # Predicted y values for current (a, b)
        residuals = y - model
        chi2_grid[i, j] = np.sum(residuals**2 / σ2)  # Normalized chi-squared

# Minimum chi-squared value and best-fit parameters
min_chi2 = np.min(chi2_grid)
min_indices = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
best_a = A[min_indices]
best_b = B[min_indices]

# Δχ² calculation
delta_chi2 = chi2_grid - min_chi2

# Credible regions for Bayesian estimation
credible_levels = [2.30, 6.17, 11.8]  # 68.3%, 95.4%, 99.73% levels (Lecture 6 thresholds)

# Covariance matrix calculation (using Taylor expansion and second derivatives)will 
var_b = σ2 * n / D
var_a = σ2 * X2 / D
cov_ab = -σ2 * X / D
std_b = np.sqrt(var_b)
std_a = np.sqrt(var_a)

cov_matrix = np.zeros((2, 2))
cov_matrix[0, 0] = var_a  # Variance of `a`
cov_matrix[1, 1] = var_b  # Variance of `b`
cov_matrix[0, 1] = cov_matrix[1, 0] = cov_ab  # Covariance of `a` and `b`

print("\nAdjusted Bayesian Maximum Likelihood Results:")
print("(i) Minimum chi-squared: ", min_chi2)
print("(ii) Best-fit a: ", best_a)
print("(iii) Best-fit b: ", best_b)
print("(iv) Covariance matrix:\n", cov_matrix)

# Plotting credible regions and Bayesian contours
plt.figure(figsize=(10, 6))
contours = plt.contour(A, B, delta_chi2, levels=credible_levels, colors=['blue', 'green', 'red'])
plt.clabel(contours, inline=True, fontsize=8, fmt='%1.2f')
plt.scatter(best_a, best_b, color='black', label='Best-fit point')
plt.xlabel("Intercept (a)")
plt.ylabel("Slope (b)")
plt.title("Bayesian Credible Regions based on Δχ²")
plt.legend()
plt.grid(True)
plt.show()

# Range within credible regions
print("\nCredible Regions:")
credible_labels = ["68.3%", "95.4%", "99.73%"]
for level, label in zip(credible_levels, credible_labels):
    indices = np.where(delta_chi2 <= level)
    a_in_region = A[indices]
    b_in_region = B[indices]
    a_range_credible = (np.min(a_in_region), np.max(a_in_region))
    b_range_credible = (np.min(b_in_region), np.max(b_in_region))
    print(label, "credible region:")
    print("   a range: ", a_range_credible)
    print("   b range: ", b_range_credible)


# #### Best so far

# In[55]:


import numpy as np
import matplotlib.pyplot as plt

# Use the variables directly from Step 1.1
# a, b, 2, x, y from the OLS calculations in Step 1.1(Not sure)

# Define ranges for a and b based on ±5 times their standard errors
Na, Nb = 100, 100  # Number of grid points for a and b
a_range = np.linspace(a - 5 * std_a, a + 5 * std_a, Na)#100 point btween a-5σ,a+5σ
b_range = np.linspace(b - 5 * std_b, b + 5 * std_b, Nb)#100 point btween b-5σ,b+5σ

# Create 2D grids for a and b
A, B = np.meshgrid(a_range, b_range, indexing='ij')
#indexing ij makes sure that y axis represent rows and x axis represent columns(default otherway)

# Initialize arrays for chi-squared and delta chi-squared
chi2_grid = np.zeros((Na, Nb))#100 X100 matrix
delta_chi2 = np.zeros((Na, Nb))#100 X100 matrix

# Compute chi-squared values for each (a, b) pair
for i in range(Na):#looping over 100 x axis indices
    for j in range(Nb):#looping over 100 y axis indices
        y_pred = A[i, j] + B[i, j] * x  # Predicted y values using differrent (a, b) pairs
        residuals = y - y_pred
        chi2_grid[i, j] = np.sum((residuals**2) / σ2)  # Normalized chi-squared

# Minimum chi-squared value and best-fit parameters
min_chi2 = np.min(chi2_grid)#gets minimum value of chi2 from the grid of chi2 values
min_indices = np.where(chi2_grid == min_chi2)  # Get the indices where the value matches the minimum
best_a = A[min_indices]#extracting a corresponding to min chi2(as shape of A and chi2_grid is same ,index will be same)
best_b = B[min_indices]#extracting b corresponding to min chi2(as shape of B and chi2_grid is same ,index will be same)

# Δχ² calculation
delta_chi2 = chi2_grid - min_chi2#delta chi2=chi2 -chi2_min

# Credible region levels for Bayesian inference
credible_levels = [2.30, 6.17, 11.8]  # Corresponding to 68.3%, 95.4%, 99.73%

#Values from Step 1.1(Not sure if i can use them.i used it in cov matrix)
#var_b = σ2 * n / D
#var_a = σ2 * X2 / D
#cov_ab = -σ2 * X / D
#std_b = np.sqrt(var_b)
#std_a = np.sqrt(var_a)

# Covariance matrix calculation
cov_matrix = np.zeros((2, 2))
cov_matrix[0, 0] = var_a # Variance of `a`
cov_matrix[1, 1] = var_b  # Variance of `b`
cov_matrix[0, 1] = cov_matrix[1, 0] = cov_ab  # Covariance of `a` and `b`

# Output results
print("\nAdjusted Bayesian Maximum Likelihood Results:")
print("(i) Minimum chi-squared: ", min_chi2)
print("(ii) Best-fit a: ", best_a)
print("(iii) Best-fit b: ", best_b)
print("(iv) Covariance matrix:\n", cov_matrix)

# Plotting credible regions and Bayesian contours
plt.figure(figsize=(10, 6))
contours = plt.contour(A, B, delta_chi2, levels=credible_levels, colors=['blue', 'green', 'red'])
plt.clabel(contours, inline=True)
plt.scatter(best_a, best_b, color='black', label='Best-fit point')
plt.xlabel("Intercept (a)")
plt.ylabel("Slope (b)")
plt.title("Countours of Credible Regions for Δχ²")
plt.legend()
plt.grid(True)
plt.show()

# Range within credible regions
print("\nCredible Regions:")
credible_labels = ["68.3%", "95.4%", "99.73%"]

for i in range(len(credible_levels)):  # loop to take regions one by one
    level = credible_levels[i]
    label = credible_labels[i] #Assigning each elements as label to corresponding level
    
    indices = np.where(delta_chi2 <= level)  # Find indices where delta_chi2 is less than or equal to the credible level
    a_in_region = A[indices]  # Extract corresponding a values for the credible region
    b_in_region = B[indices]  # Extract corresponding b values for the credible region
    
    # Find the range of a and b within the credible region
    a_range_credible = (np.min(a_in_region), np.max(a_in_region))
    #Defining range between Minimum and maximum values of a in the region
    b_range_credible = (np.min(b_in_region), np.max(b_in_region))
    #Defining range between Minimum and maximum values of b in the region
    
    # Print the credible region ranges
    print(label, "credible region:")
    print("   a range: ", a_range_credible)
    print("   b range: ", b_range_credible)


# ## Step 1.3

# In[59]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 0: Load the data ===
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# OLS fit and variance estimate (used to initialize MCMC)
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b_ols = (n * XY - X * Y) / D
a_ols = (Y * X2 - X * XY) / D
y_pred = a_ols + b_ols * x
σ2 = np.sum((y - y_pred)**2) / (n - 2)

# === Step (a): Metropolis MCMC sampler ===

def log_likelihood(a, b):
    model = a + b * x
    residuals = y - model
    return -0.5 * np.sum(residuals**2) / σ2

# Proposal standard deviations (tunable)
proposal_std_a = 0.05
proposal_std_b = 0.05

# Prior limits (uniform)
a_min, a_max = a_ols - 5, a_ols + 5
b_min, b_max = b_ols - 5, b_ols + 5

# MCMC parameters
num_samples = 10000
samples = np.zeros((num_samples, 2))
accept_count = 0

# Start at OLS estimate
a_current, b_current = a_ols, b_ols
logL_current = log_likelihood(a_current, b_current)

for i in range(num_samples):
    # Propose new values
    a_prop = np.random.normal(a_current, proposal_std_a)
    b_prop = np.random.normal(b_current, proposal_std_b)

    # Check prior
    if not (a_min < a_prop < a_max and b_min < b_prop < b_max):
        samples[i] = [a_current, b_current]
        continue

    logL_prop = log_likelihood(a_prop, b_prop)
    acceptance_ratio = np.exp(logL_prop - logL_current)

    if np.random.rand() < acceptance_ratio:
        a_current, b_current = a_prop, b_prop
        logL_current = logL_prop
        accept_count += 1

    samples[i] = [a_current, b_current]

print(f"Acceptance rate: {accept_count / num_samples:.2f}")

# === Step (b): Explain proposal and prior choices ===
# - Proposal std was chosen to give ~30–50% acceptance rate (can be tuned)
# - Uniform priors assumed over a wide range around OLS for simplicity

# === Step (c): Estimate mean, std, and covariance from samples ===
burn_in = int(0.2 * num_samples)
samples_cut = samples[burn_in:]
a_samples, b_samples = samples_cut[:, 0], samples_cut[:, 1]

a_mean = np.mean(a_samples)
b_mean = np.mean(b_samples)
a_std = np.std(a_samples)
b_std = np.std(b_samples)
cov_ab = np.cov(a_samples, b_samples)[0, 1]

print("\n=== Step (c): MCMC Parameter Estimates ===")
print(f"Mean a = {a_mean:.4f} ± {a_std:.4f}")
print(f"Mean b = {b_mean:.4f} ± {b_std:.4f}")
print(f"Covariance(a, b) = {cov_ab:.4f}")

# === Step (d): Estimate credible regions ===
def credible_interval(param_samples, level=0.683):
    sorted_samples = np.sort(param_samples)
    ci_idx = int(level * len(sorted_samples))
    intervals = [sorted_samples[i + ci_idx] - sorted_samples[i] for i in range(len(sorted_samples) - ci_idx)]
    min_idx = np.argmin(intervals)
    return sorted_samples[min_idx], sorted_samples[min_idx + ci_idx]

a_68 = credible_interval(a_samples)
b_68 = credible_interval(b_samples)

print("\n=== Step (d): 68.3% Credible Intervals ===")
print(f"a: {a_68[0]:.4f} to {a_68[1]:.4f}")
print(f"b: {b_68[0]:.4f} to {b_68[1]:.4f}")

# === Step (e): Corner plot ===
import corner
corner.corner(samples_cut, labels=["a", "b"], truths=[a_ols, b_ols])
plt.suptitle("Step (e): MCMC Posterior Corner Plot")
plt.show()

# === Step (f): Maximum probability estimate and comparison ===
# Use point with highest log-likelihood
logLs = np.array([log_likelihood(a, b) for a, b in samples_cut])
max_idx = np.argmax(logLs)
a_max, b_max = samples_cut[max_idx]

print("\n=== Step (f): Maximum Probability Estimates ===")
print(f"a = {a_max:.4f}")
print(f"b = {b_max:.4f}")
print(f"Compared to OLS a = {a_ols:.4f}, b = {b_ols:.4f}")


# In[61]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 0: Load data and prepare ===
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# OLS estimates for initialization
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b_ols = (n * XY - X * Y) / D
a_ols = (Y * X2 - X * XY) / D
y_pred = a_ols + b_ols * x
σ2 = np.sum((y - y_pred)**2) / (n - 2)

# === Step (a): Metropolis MCMC sampler ===
def log_likelihood(a, b):
    model = a + b * x
    residuals = y - model
    return -0.5 * np.sum(residuals**2) / σ2

# Proposal distribution (tuned for ~25–40% acceptance)
proposal_std_a = 0.01
proposal_std_b = 0.01

# Uniform priors over wide range around OLS values
a_min, a_max = a_ols - 5, a_ols + 5
b_min, b_max = b_ols - 5, b_ols + 5

# MCMC setup
num_samples = 10000
samples = np.zeros((num_samples, 2))
accept_count = 0
a_current, b_current = a_ols, b_ols
logL_current = log_likelihood(a_current, b_current)

# Metropolis loop
for i in range(num_samples):
    a_prop = np.random.normal(a_current, proposal_std_a)
    b_prop = np.random.normal(b_current, proposal_std_b)

    # Reject if out of prior range
    if not (a_min < a_prop < a_max and b_min < b_prop < b_max):
        samples[i] = [a_current, b_current]
        continue

    logL_prop = log_likelihood(a_prop, b_prop)
    acceptance_ratio = np.exp(logL_prop - logL_current)

    if np.random.rand() < acceptance_ratio:
        a_current, b_current = a_prop, b_prop
        logL_current = logL_prop
        accept_count += 1

    samples[i] = [a_current, b_current]

print(f"Acceptance rate: {accept_count / num_samples:.2f}")

# === Step (b): Proposal/prior choice explanation ===
# - Proposal std chosen for balanced acceptance (~30%)
# - Prior range centered around OLS, wide enough to explore well

# === Step (c): Estimate mean, std, covariance ===
burn_in = int(0.2 * num_samples)
samples_cut = samples[burn_in:]
a_samples = samples_cut[:, 0]
b_samples = samples_cut[:, 1]

a_mean = np.mean(a_samples)
b_mean = np.mean(b_samples)
a_std = np.std(a_samples)
b_std = np.std(b_samples)
cov_ab = np.cov(a_samples, b_samples)[0, 1]

print("\n=== Step (c): Posterior Estimates ===")
print(f"Mean a = {a_mean:.4f} ± {a_std:.4f}")
print(f"Mean b = {b_mean:.4f} ± {b_std:.4f}")
print(f"Covariance(a, b) = {cov_ab:.4f}")

# === Step (d): 68.3% credible intervals ===
def credible_interval(samples, level=0.683):
    sorted_samples = np.sort(samples)
    n_ci = int(level * len(sorted_samples))
    widths = [sorted_samples[i + n_ci] - sorted_samples[i] for i in range(len(sorted_samples) - n_ci)]
    min_idx = np.argmin(widths)
    return sorted_samples[min_idx], sorted_samples[min_idx + n_ci]

a_ci = credible_interval(a_samples)
b_ci = credible_interval(b_samples)

print("\n=== Step (d): 68.3% Credible Intervals ===")
print(f"a: {a_ci[0]:.4f} to {a_ci[1]:.4f}")
print(f"b: {b_ci[0]:.4f} to {b_ci[1]:.4f}")

# === Step (e): Manual "Corner" plot using seaborn ===
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], hspace=0.05, wspace=0.05)

# Top: histogram of a
ax_a = plt.subplot(gs[0, 0])
sns.histplot(a_samples, bins=40, kde=True, color='skyblue')
ax_a.set_xticklabels([])
ax_a.set_ylabel('Density')
ax_a.set_xlim(np.min(a_samples), np.max(a_samples))

# Right: histogram of b (vertical)
ax_b = plt.subplot(gs[1, 1])
sns.histplot(b_samples, bins=40, kde=True, color='salmon', vertical=True)
ax_b.set_yticklabels([])
ax_b.set_xlabel('Density')
ax_b.set_ylim(np.min(b_samples), np.max(b_samples))

# Bottom-left: 2D joint posterior
ax_joint = plt.subplot(gs[1, 0])
sns.kdeplot(x=a_samples, y=b_samples, fill=True, cmap="mako", levels=10, thresh=0.01)
ax_joint.set_xlabel('a')
ax_joint.set_ylabel('b')
ax_joint.scatter(a_ols, b_ols, color='black', marker='x', label='OLS')
ax_joint.legend()

plt.suptitle("Step (e): Posterior Distributions for a and b")
plt.tight_layout()
plt.show()

# === Step (f): Maximum likelihood estimate from samples ===
logLs = np.array([log_likelihood(a, b) for a, b in samples_cut])
max_idx = np.argmax(logLs)
a_max, b_max = samples_cut[max_idx]

print("\n=== Step (f): Maximum Likelihood Parameters ===")
print(f"a = {a_max:.4f}")
print(f"b = {b_max:.4f}")
print(f"Compared to OLS → a = {a_ols:.4f}, b = {b_ols:.4f}")


# ## Manav Code-edited with my Variables
# 

# In[66]:


σ = np.sqrt(σ2)

# -----------------------------
# 3. Metropolis MCMC
# -----------------------------
def log_likelihood(a, b):
    return -0.5 * np.sum((y - (a + b * x)) ** 2) / σ2

# uniform (improper) prior in a generous box around the OLS values
a_min, a_max = a - 5.0, a_OLS + 5.0
b_min, b_max = b - 3.0, b_OLS + 3.0

def log_prior(a, b):
    if (a_min <= a <= a_max) and (b_min <= b <= b_max):
        return 0.0  # log‑uniform (constant)
    return -np.inf

def log_posterior(a, b):
    lp = log_prior(a, b)
    return lp + log_likelihood(a, b) if np.isfinite(lp) else -np.inf

# hyper‑parameters of the random‑walk proposal
prop_a, prop_b = 0.05, 0.02
steps, burn, thin = 40_000, 10_000, 10

current_a, current_b = a, b
current_lp = log_posterior(current_a, current_b)

chain = []
accept = 0
rng = np.random.default_rng(0)

for _ in range(steps):
    cand_a = rng.normal(current_a, prop_a)
    cand_b = rng.normal(current_b, prop_b)
    cand_lp = log_posterior(cand_a, cand_b)
    if rng.random() < np.exp(cand_lp - current_lp):
        current_a, current_b, current_lp = cand_a, cand_b, cand_lp
        accept += 1
    chain.append([current_a, current_b])

chain = np.asarray(chain)
acc_rate = accept / steps
posterior = chain[burn::thin]  # 3000 points

# -----------------------------
# 4. Summary statistics
# -----------------------------
mean_a, mean_b = posterior.mean(axis=0)
std_a, std_b = posterior.std(axis=0, ddof=1)
cov_ab = np.cov(posterior.T)[0, 1]

# highest‑posterior sample (MAP)
lp_all = np.array([log_posterior(a, b) for a, b in chain])
map_a, map_b = chain[np.argmax(lp_all)]

# 68 % & 95 % 1‑D credible intervals
ci68_a = np.percentile(posterior[:, 0], [15.865, 84.135])
ci95_a = np.percentile(posterior[:, 0], [2.275, 97.725])
ci68_b = np.percentile(posterior[:, 1], [15.865, 84.135])
ci95_b = np.percentile(posterior[:, 1], [2.275, 97.725])

print("Acceptance rate:", acc_rate)
print(f"Posterior mean a = {mean_a:.5f} ± {std_a:.5f}")
print(f"Posterior mean b = {mean_b:.5f} ± {std_b:.5f}")
print("Cov(a,b) =", cov_ab)
print("MAP  a =", map_a, " b =", map_b)
print("68% credible interval for a:", ci68_a)
print("68% credible interval for b:", ci68_b)

# -----------------------------
# 5. Corner‑style plots
# -----------------------------

# 5a. 1‑D marginal of a
plt.figure()
plt.hist(posterior[:, 0], bins=40, density=True)
plt.axvline(mean_a, linestyle="--")
plt.title("Posterior of parameter a")
plt.xlabel("a")
plt.ylabel("Probability density")

# 5b. 1‑D marginal of b
plt.figure()
plt.hist(posterior[:, 1], bins=40, density=True)
plt.axvline(mean_b, linestyle="--")
plt.title("Posterior of parameter b")
plt.xlabel("b")
plt.ylabel("Probability density")

# 5c. 2‑D joint distribution with 68 % and 95 % contours
H, xedges, yedges = np.histogram2d(posterior[:, 0], posterior[:, 1], bins=50, density=True)
xcenters = 0.5 * (xedges[:-1] + xedges[1:])
ycenters = 0.5 * (yedges[:-1] + yedges[1:])
Hsort = np.sort(H.flatten())[::-1]
Hcum = np.cumsum(Hsort) / Hsort.sum()
level68 = Hsort[np.searchsorted(Hcum, 0.683)]
level95 = Hsort[np.searchsorted(Hcum, 0.954)]

plt.figure()
plt.contourf(xcenters, ycenters, H.T, levels=[level95, level68, H.max()], alpha=0.6)
plt.scatter(posterior[:, 0], posterior[:, 1], s=4, alpha=0.4)
plt.xlabel("a")
plt.ylabel("b")
plt.title("Joint posterior of a and b")

plt.show()


# In[56]:


import numpy as np
import matplotlib.pyplot as plt
σ = np.sqrt(σ2)

#Step 1: MCMC Sampling (Metropolis)
def ll(a1, b1):
    return -0.5 * np.sum((y - (a1 + b1 * x))**2) / σ2

a_min, a_max = a - 5, a + 5
b_min, b_max = b - 3, b + 3

def lp(a1, b1):
    return 0.0 if (a_min <= a1 <= a_max and b_min <= b1 <= b_max) else -np.inf

def post(a1, b1):
    p = lp(a1, b1)
    return ll(a1, b1) + p if np.isfinite(p) else -np.inf

# Proposal sizes and MCMC setup
pa, pb = 0.05, 0.02
N, burn, thin = 40000, 10000, 10
a0, b0 = a, b
logp = post(a0, b0)
acc = 0
chain = []
rng = np.random.default_rng(42)

for _ in range(N):
    an = rng.normal(a0, pa)
    bn = rng.normal(b0, pb)
    logp_new = post(an, bn)
    if rng.random() < np.exp(logp_new - logp):
        a0, b0, logp = an, bn, logp_new
        acc += 1
    chain.append([a0, b0])

chain = np.array(chain)
acc_rate = acc / N
posterior = chain[burn::thin]

#Step 2: Summary Stats
a_samp, b_samp = posterior[:, 0], posterior[:, 1]
ma, mb = np.mean(a_samp), np.mean(b_samp)
sa, sb = np.std(a_samp, ddof=1), np.std(b_samp, ddof=1)
cab = np.cov(a_samp, b_samp)[0, 1]
logp_all = np.array([post(a1, b1) for a1, b1 in posterior])
amap, bmap = posterior[np.argmax(logp_all)]

# Credible intervals
def ci(x, level=68):
    q = (100 - level) / 2
    return np.percentile(x, [q, 100 - q])

ca68 = ci(a_samp, 68)
cb68 = ci(b_samp, 68)
ca95 = ci(a_samp, 95)
cb95 = ci(b_samp, 95)

print("Acceptance rate:", round(acc_rate, 3))
print(f"Posterior mean a = {ma:.5f} ± {sa:.5f}")
print(f"Posterior mean b = {mb:.5f} ± {sb:.5f}")
print(f"Cov(a,b) = {cab:.5f}")
print(f"MAP: a = {amap:.5f}, b = {bmap:.5f}")
print("68% CI for a:", ca68)
print("68% CI for b:", cb68)

# === Step 3: Corner-style plots ===

# 3a. 1D posterior of a
plt.figure()
plt.hist(a_samp, bins=40, density=True, color='skyblue', edgecolor='gray')
plt.axvline(ma, linestyle='--', color='black')
plt.title("Posterior of a")
plt.xlabel("a")
plt.ylabel("Density")

# 3b. 1D posterior of b
plt.figure()
plt.hist(b_samp, bins=40, density=True, color='salmon', edgecolor='gray')
plt.axvline(mb, linestyle='--', color='black')
plt.title("Posterior of b")
plt.xlabel("b")
plt.ylabel("Density")

# 3c. 2D joint with contours
H, xa, ya = np.histogram2d(a_samp, b_samp, bins=50, density=True)
xc = 0.5 * (xa[:-1] + xa[1:])
yc = 0.5 * (ya[:-1] + ya[1:])
Hf = H.T

# Calculate contour levels
Hflat = Hf.flatten()
Hsorted = np.sort(Hflat)[::-1]
Hcumulative = np.cumsum(Hsorted)
Hcumulative /= Hcumulative[-1]
l68 = Hsorted[np.searchsorted(Hcumulative, 0.683)]
l95 = Hsorted[np.searchsorted(Hcumulative, 0.954)]

plt.figure()
plt.contourf(xc, yc, Hf, levels=[l95, l68, Hf.max()], alpha=0.6, colors=['orange', 'lightblue', 'blue'])
plt.scatter(a_samp, b_samp, s=4, alpha=0.3)
plt.xlabel("a")
plt.ylabel("b")
plt.title("Joint Posterior of a and b")
plt.tight_layout()
plt.show()


# ## without any premade functions

# In[59]:


import numpy as np
import matplotlib.pyplot as plt

σ = np.sqrt(σ2)

# Step 1: MCMC Sampling (Metropolis)
def log_likelihood(a1, b1):
    return -0.5 * np.sum((y - (a1 + b1 * x))**2) / σ2

Na, Nb = 100, 100  # Number of grid points for a and b
a_range = np.linspace(a - 5 * std_a, a + 5 * std_a, Na)#100 point btween a-5σ,a+5σ
b_range = np.linspace(b - 5 * std_b, b + 5 * std_b, Nb)#100 point btween b-5σ,b+5σ

def log_prior(a1, b1):
    if (a_range[0] <= a1 <= a_range[-1]) and (b_range[0] <= b1 <= b_range[-1]):#index -1 selects last element,0 the first
        return 0.0 #if condition met returns a uniform prior(log 1)
    else:
        return -np.inf #return -infinity otherwise

def posterior(a1, b1):
    p = log_prior(a1, b1)
    if np.isfinite(p):
        return log_likelihood(a1, b1) + p
    else:
        return -np.inf

# MCMC proposal settings and initialization
std_a1, std_b1 = 0.03, 0.015# Proposal standard deviations for a and b
steps = 40000# Total number of MCMC iterations
burn = 10000# Number of initial samples to discard(possibly from non-stationary region of the posterior)
thin = 5 # Keep one every 5 samples

a_1, b_1 = a, b # Initial parameter values
log_post = posterior(a_1, b_1) # Initial log-posterior
accepted = 0 # Acceptance counter
samples = [] # list to store accepted samples
generator = np.random.seed(1) # Random number generator with fixed seed

for i in range(steps):
    an = np.random.normal(a_1, std_a1)  # Generate new random 'a' value(noise in a(gaussian))
    bn = np.random.normal(b_1, std_b1)  # Generate new random 'b' value(noise in b(gaussian))
    log_post1 = posterior(an, bn)#calculating log posterior for the new random a and b
    
    if np.random.random() < np.exp(log_post1 - log_post):#exp turns log-likelihood ratio into a probability(check Notes)
        #random generator generates number bw 0 and 1 .
        #if the probability of difference of initial and new log posterior is greater than that,it is accepted
        a_1, b_1, log_post = an, bn, log_post1#assigning accepted values as initial for next iteration
        accepted += 1
    samples.append([a_1, b_1])#storing accepted values of a,b

samples = np.array(samples)#updating samples array
acc_rate = accepted / steps#calculating rate of acceptance
#selecting MCMC samples after discarding first burn samples and then taking every thin-th(5th) sample from remaining.
posterior_samples = samples[burn::thin]

# Step 2: Summary Stats
a_samples, b_samples = posterior_samples[:, 0], posterior_samples[:, 1]
#a_samp, b_samp = posterior[:, 0], posterior[:, 1]
mean_a, mean_b = np.mean(a_samples), np.mean(b_samples)
#ma, mb = np.mean(a_samp), np.mean(b_samp)
stdv_a = np.sqrt(np.sum((a_samples - mean_a) ** 2) / (len(a_samples) - 1))
stdv_b = np.sqrt(np.sum((b_samples - mean_b) ** 2) / (len(b_samples) - 1))
#sa, sb = np.std(a_samp, ddof=1), np.std(b_samp, ddof=1)
cov_ab = (np.sum((a_samples - mean_a) * (b_samples - mean_b)))/steps-1
#cab = np.cov(a_samp, b_samp)[0, 1]
#calculating log-posterior for each parameter pair in posterior_samples.
log_probs = []
for a, b in posterior_samples:
    log_probs.append(posterior(a, b))
log_probs = np.array(log_probs)
#finding the parameter pair (a, b) with the highest log-posterior
max_log_prob = log_probs[0]
best_a, best_b = posterior_samples[0]
for i in range(1, len(log_probs)):
    if log_probs[i] > max_log_prob:
        max_log_prob = log_probs[i]
        best_a, best_b = posterior_samples[i]

# Credible intervals(Cehck notes)
def ci(x, level):
    q = (100 - level) / 2
    return np.percentile(x, [q, 100 - q])

# the function works is by keeping the desired percentage of data in the center and excluding the outer parts.
#Here's a breakdown:
#For a 68% credible interval,it keeps the middle 68% of your data and 
#excludes the outer 16% from each end. 
#This leaves 16% of the data on the lower end and 16% on the higher end. 
#So, the interval includes everything between the 16th percentile and the 84th percentile of your data.
#the interval is centered on the middle portion of your data(level), and the rest is excluded based on the level you choose

#calculating Credible Intervals for a_sample and b_sample values at 68% and 95%
ca68 = ci(a_samples, 68)
cb68 = ci(b_samples, 68)
ca95 = ci(a_samples, 95)
cb95 = ci(b_samples, 95)


print("Acceptance rate:", round(acc_rate, 3))
print("Posterior mean a = "+ str(round(mean_a, 5))+" ± "+str(round(stdv_a, 5)))
print("Posterior mean b = "+ str(round(mean_b, 5))+" ± "+str(round(stdv_b, 5)))
print("Cov(a,b) = " + str(round(cov_ab, 5)))
print("MAP: a =" + str(round(best_a, 5)) + ", b =" + str(round(best_b, 5)))
print("68% CI for a:", ca68)
print("68% CI for b:", cb68)


# Corner-style plots

# 3a. 1D posterior of a
plt.figure()
plt.hist(a_samples, bins=40, density=True, color='skyblue', edgecolor='gray')
plt.axvline(ma, linestyle='--', color='black')
plt.title("Posterior of a")
plt.xlabel("a")
plt.ylabel("Density")

# 3b. 1D posterior of b
plt.figure()
plt.hist(b_samples, bins=40, density=True, color='salmon', edgecolor='gray')
plt.axvline(mb, linestyle='--', color='black')
plt.title("Posterior of b")
plt.xlabel("b")
plt.ylabel("Density")

hist2d, a_edges, b_edges = np.histogram2d(a_samples, b_samples, bins=50, density=True)

# Compute bin centers from bin edges
a_mid = 0.5 * (a_edges[:-1] + a_edges[1:])
b_mid = 0.5 * (b_edges[:-1] + b_edges[1:])

# Transpose histogram to match plotting orientation
hist_trans = hist2d.T


# Calculate contour levels
hist_flat = hist_trans.flatten()#2D histogram array to 1D array
hist_sorted = np.sort(hist_flat)[::-1]#Sorts the 1D histogram values in descending order(accumulate highest prob regions first)
hist_cumulative = np.cumsum(hist_sorted)#the cumulative sum of the sorted histogram values.
hist_cumulative /= hist_cumulative[-1]#Normalizes the cumulative sum

#Finding density level where top 68% of total probability is focused
level_68 = hist_sorted[np.searchsorted(hist_cumulative, 0.683)]
#Finding density level where top 95% of total probability is focused
level_95 = hist_sorted[np.searchsorted(hist_cumulative, 0.954)]

plt.figure()
plt.contourf(a_mid,b_mid,hist_trans, levels=[level_95, level_68, hist_trans.max()], alpha=0.6, colors=['orange', 'lightblue', 'blue'])
plt.scatter(a_samples, b_samples, s=4, alpha=0.3)
plt.xlabel("a")
plt.ylabel("b")
plt.title("Joint Posterior of a and b")
plt.tight_layout()
plt.show()


# # Note
# The reason the difference in log-posterior values is compared to a random number in the Metropolis-Hastings algorithm (a part of MCMC) is to introduce an element of randomness that allows the algorithm to explore the parameter space more thoroughly.
# The key idea of the Metropolis-Hastings algorithm is to decide whether to accept or reject the proposed parameters based on how likely they are, while also allowing for some "exploration" of less likely areas in the parameter space. This helps avoid the algorithm from getting stuck in local maxima of the posterior distribution.

# ## ci function
# the function works is by keeping the desired percentage of data in the center and excluding the outer parts.
# Here's a breakdown:
# For a 68% credible interval,it keeps the middle 68% of your data and 
# excludes the outer 16% from each end. 
# This leaves 16% of the data on the lower end and 16% on the higher end. 
# So, the interval includes everything between the 16th percentile and the 84th percentile of your data.
# the interval is centered on the middle portion of your data(level), and the rest is excluded based on the level you choose

# ## with premade functions

# In[53]:


import numpy as np
import matplotlib.pyplot as plt

σ = np.sqrt(σ2)

# Step 1: MCMC Sampling (Metropolis)
def log_likelihood(a1, b1):
    return -0.5 * np.sum((y - (a1 + b1 * x))**2) / σ2

Na, Nb = 100, 100  # Number of grid points for a and b
a_range = np.linspace(a - 5 * std_a, a + 5 * std_a, Na)#100 point btween a-5σ,a+5σ
b_range = np.linspace(b - 5 * std_b, b + 5 * std_b, Nb)#100 point btween b-5σ,b+5σ

def log_prior(a1, b1):
    if (a_range[0] <= a1 <= a_range[-1]) and (b_range[0] <= b1 <= b_range[-1]):#index -1 selects last element,0 the first
        return 0.0 #if condition met returns a uniform prior(log 1)
    else:
        return -np.inf #return -infinity otherwise

def posterior(a1, b1):
    p = log_prior(a1, b1)
    if np.isfinite(p):
        return log_likelihood(a1, b1) + p
    else:
        return -np.inf

# MCMC proposal settings and initialization
std_a1, std_b1 = 0.03, 0.015# Proposal standard deviations for a and b
steps = 40000# Total number of MCMC iterations
burn = 10000# Number of initial samples to discard(possibly from non-stationary region of the posterior)
thin = 5 # Keep one every 5 samples

a_1, b_1 = a, b # Initial parameter values
log_post = posterior(a_1, b_1) # Initial log-posterior
accepted = 0 # Acceptance counter
samples = [] # list to store accepted samples
generator = np.random.seed(1) # Random number generator with fixed seed

for i in range(steps):
    an = np.random.normal(a_1, std_a1)  # Generate new random 'a' value(noise in a(gaussian))
    bn = np.random.normal(b_1, std_b1)  # Generate new random 'b' value(noise in b(gaussian))
    log_post1 = posterior(an, bn)#calculating log posterior for the new random a and b
    
    if np.random.random() < np.exp(log_post1 - log_post):#exp turns log-likelihood ratio into a probability(check Notes)
        #random generator generates number bw 0 and 1 .
        #if the probability of difference of initial and new log posterior is greater than that,it is accepted
        a_1, b_1, log_post = an, bn, log_post1#assigning accepted values as initial for next iteration
        accepted += 1
    samples.append([a_1, b_1])#storing accepted values of a,b

samples = np.array(samples)#updating samples array
acc_rate = accepted / steps#calculating rate of acceptance
#selecting MCMC samples after discarding first burn samples and then taking every thin-th(5th) sample from remaining.
posterior_samples = samples[burn::thin]






#Check these Calculations:posterior mean a,posterior mean b,covariance ,stdv_a,stdv_b all coming wrong

# Step 2: Summary Stats
#a_samples, b_samples = posterior_samples[:, 0], posterior_samples[:, 1]
a_samples, b_samples = posterior_samples[:, 0], posterior_samples[:, 1]
#mean_a, mean_b = np.mean(a_samples), np.mean(b_samples)
mean_a,mean_b = np.mean(a_samples), np.mean(b_samples)
#stdv_a = np.sqrt(np.sum((a_samples - mean_a) ** 2) / (len(a_samples) - 1))
#stdv_b = np.sqrt(np.sum((b_samples - mean_b) ** 2) / (len(b_samples) - 1))
stdv_a, stdv_b = np.std(a_samples, ddof=1), np.std(b_samples, ddof=1)
#cov_ab = (np.sum((a_samples - mean_a) * (b_samples - mean_b)))/steps-1
cov_ab = np.cov(a_samp, b_samp)[0, 1]
#calculating log-posterior for each parameter pair in posterior_samples.
log_probs = []
for a, b in posterior_samples:
    log_probs.append(posterior(a, b))
log_probs = np.array(log_probs)
#finding the parameter pair (a, b) with the highest log-posterior
max_log_prob = log_probs[0]
best_a, best_b = posterior_samples[0]
for i in range(1, len(log_probs)):
    if log_probs[i] > max_log_prob:
        max_log_prob = log_probs[i]
        best_a, best_b = posterior_samples[i]

# Credible intervals(Cehck notes)
def ci(x, level):
    q = (100 - level) / 2
    return np.percentile(x, [q, 100 - q])

# the function works is by keeping the desired percentage of data in the center and excluding the outer parts.
#Here's a breakdown:
#For a 68% credible interval,it keeps the middle 68% of your data and 
#excludes the outer 16% from each end. 
#This leaves 16% of the data on the lower end and 16% on the higher end. 
#So, the interval includes everything between the 16th percentile and the 84th percentile of your data.
#the interval is centered on the middle portion of your data(level), and the rest is excluded based on the level you choose

#calculating Credible Intervals for a_sample and b_sample values at 68% and 95%
ca68 = ci(a_samples, 68)
cb68 = ci(b_samples, 68)
ca95 = ci(a_samples, 95)
cb95 = ci(b_samples, 95)


print("Acceptance rate:", round(acc_rate, 3))
print("Posterior mean a = "+ str(round(mean_a, 5))+" ± "+str(round(stdv_a, 5)))
print("Posterior mean b = "+ str(round(mean_b, 5))+" ± "+str(round(stdv_b, 5)))
print("Cov(a,b) = " + str(round(cov_ab, 5)))
print("MAP: a =" + str(round(best_a, 5)) + ", b =" + str(round(best_b, 5)))
print("68% CI for a:", ca68)
print("68% CI for b:", cb68)


# === Step 3: Corner-style plots ===

# 3a. 1D posterior of a
plt.figure()
plt.hist(a_samples, bins=40, density=True, color='skyblue', edgecolor='gray')
plt.axvline(ma, linestyle='--', color='black')
plt.title("Posterior of a")
plt.xlabel("a")
plt.ylabel("Density")

# 3b. 1D posterior of b
plt.figure()
plt.hist(b_samples, bins=40, density=True, color='salmon', edgecolor='gray')
plt.axvline(mb, linestyle='--', color='black')
plt.title("Posterior of b")
plt.xlabel("b")
plt.ylabel("Density")

hist2d, a_edges, b_edges = np.histogram2d(a_samples, b_samples, bins=50, density=True)

# Compute bin centers from bin edges
a_mid = 0.5 * (a_edges[:-1] + a_edges[1:])
b_mid = 0.5 * (b_edges[:-1] + b_edges[1:])

# Transpose histogram to match plotting orientation
hist_trans = hist2d.T


# Calculate contour levels
hist_flat = hist_trans.flatten()#2D histogram array to 1D array
hist_sorted = np.sort(hist_flat)[::-1]#Sorts the 1D histogram values in descending order(accumulate highest prob regions first)
hist_cumulative = np.cumsum(hist_sorted)#the cumulative sum of the sorted histogram values.
hist_cumulative /= hist_cumulative[-1]#Normalizes the cumulative sum

#Finding density level where top 68% of total probability is focused
level_68 = hist_sorted[np.searchsorted(hist_cumulative, 0.683)]
#Finding density level where top 95% of total probability is focused
level_95 = hist_sorted[np.searchsorted(hist_cumulative, 0.954)]

plt.figure()
plt.contourf(xc, yc, Hf, levels=[level_95, level_68, hist_trans.max()], alpha=0.6, colors=['orange', 'lightblue', 'blue'])
plt.scatter(a_samples, b_samples, s=4, alpha=0.3)
plt.xlabel("a")
plt.ylabel("b")
plt.title("Joint Posterior of a and b")
plt.tight_layout()
plt.show()


# In[21]:


if(np.random.random()<1):
    print("Yes")
else:
    print("No")


# In[25]:


a,b=(1,9)


# In[26]:


a


# # Part 2

# ### Step 2.1

# #### Manav Code

# In[60]:


import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0.  Load the MDC‑2 mock data
# -------------------------------------------------------------
data = np.loadtxt('3002691_MDC2.txt')   # citeturn0file1
x, y = data[:, 0], data[:, 1]
N = len(y)

# -------------------------------------------------------------
# 1. Ordinary‑least‑squares quadratic fit  y = a + bx + cx²
#    (only to obtain an initial point and σ² estimate)
# -------------------------------------------------------------
X = np.vstack([np.ones_like(x), x, x**2]).T
a_ols, b_ols, c_ols = np.linalg.lstsq(X, y, rcond=None)[0]

y_fit = a_ols + b_ols * x + c_ols * x**2
sigma2 = np.sum((y - y_fit) ** 2) / (N - 3)       # unbiased ML estimator
sigma  = np.sqrt(sigma2)

print(f"OLS parameters: a = {a_ols:.4f}, b = {b_ols:.4f}, c = {c_ols:.4f}")
print(f"Estimated noise σ² = {sigma2:.4f}  (σ = {sigma:.4f})")

# -------------------------------------------------------------
# 2.  Metropolis MCMC sampling of the posterior
# -------------------------------------------------------------
def log_like(a, b, c):
    return -0.5 * np.sum((y - (a + b * x + c * x**2))**2) / sigma2

# Simple uniform box priors
a_min, a_max = a_ols - 10, a_ols + 10
b_min, b_max = b_ols - 5,  b_ols + 5
c_min, c_max = -0.5, 1.5     # c expected ≈ 0…1

def log_prior(a, b, c):
    if (a_min <= a <= a_max) and (b_min <= b <= b_max) and (c_min <= c <= c_max):
        return 0.0           # constant (log‑uniform)
    return -np.inf

def log_post(a, b, c):
    lp = log_prior(a, b, c)
    return lp + log_like(a, b, c) if np.isfinite(lp) else -np.inf

# Proposal sigmas (tuned for ≈20 % acceptance)
prop_a, prop_b, prop_c = 0.08, 0.03, 0.005

rng          = np.random.default_rng(0)
steps        = 60_000
burn, thin   = 15_000, 10

# start at OLS
cur_a, cur_b, cur_c = a_ols, b_ols, c_ols
cur_lp = log_post(cur_a, cur_b, cur_c)

chain   = []
accepts = 0

for _ in range(steps):
    cand_a = rng.normal(cur_a, prop_a)
    cand_b = rng.normal(cur_b, prop_b)
    cand_c = rng.normal(cur_c, prop_c)
    cand_lp = log_post(cand_a, cand_b, cand_c)
    if rng.random() < np.exp(cand_lp - cur_lp):
        cur_a, cur_b, cur_c, cur_lp = cand_a, cand_b, cand_c, cand_lp
        accepts += 1
    chain.append([cur_a, cur_b, cur_c])

chain = np.asarray(chain)
acc_rate = accepts / steps
posterior = chain[burn::thin]      # ⇒ 4 500 samples

print(f"Acceptance rate = {acc_rate:.3f}")
print("Posterior sample size =", posterior.shape[0])

# -------------------------------------------------------------
# 3.  Posterior statistics
# -------------------------------------------------------------
means = posterior.mean(axis=0)
stds  = posterior.std(axis=0, ddof=1)
cov   = np.cov(posterior.T)

# MAP point:
lp_all    = np.array([log_post(a, b, c) for a, b, c in chain])
map_a, map_b, map_c = chain[np.argmax(lp_all)]

print("\nPosterior means and 1‑σ uncertainties")
print(f"a = {means[0]:.5f} ± {stds[0]:.5f}")
print(f"b = {means[1]:.5f} ± {stds[1]:.5f}")
print(f"c = {means[2]:.5f} ± {stds[2]:.5f}")
print("\nMAP parameters")
print(f"a_MAP = {map_a:.5f}, b_MAP = {map_b:.5f}, c_MAP = {map_c:.5f}")

# -------------------------------------------------------------
# 4.  Credible levels for the 2‑D posterior
# -------------------------------------------------------------
def contour_levels(samples, nb=60):
    H, xe, ye = np.histogram2d(samples[:,0], samples[:,1], bins=nb, density=True)
    Hsort = np.sort(H.ravel())[::-1]
    csum  = np.cumsum(Hsort) / Hsort.sum()
    l68   = Hsort[np.searchsorted(csum, 0.683)]
    l95   = Hsort[np.searchsorted(csum, 0.954)]
    xc    = 0.5*(xe[:-1] + xe[1:])
    yc    = 0.5*(ye[:-1] + ye[1:])
    return xc, yc, H.T, l68, l95

pairs = {("a","b"):(0,1), ("a","c"):(0,2), ("b","c"):(1,2)}

# -------------------------------------------------------------
# 5.  Plots  ─ each chart in its own figure (rules)
# -------------------------------------------------------------
# 1‑D marginals
labels = ["a", "b", "c"]
for i, lab in enumerate(labels):
    plt.figure()
    plt.hist(posterior[:, i], bins=50, density=True)
    plt.axvline(means[i], linestyle="--")
    plt.title(f"Posterior of parameter {lab}")
    plt.xlabel(lab)
    plt.ylabel("Density")

# 2‑D contours
for (labx, laby), (ix, iy) in pairs.items():
    plt.figure()
    xc, yc, H, l68, l95 = contour_levels(posterior[:, [ix, iy]])
    plt.contourf(xc, yc, H, levels=[l95, l68, H.max()], alpha=0.6)
    plt.scatter(posterior[:, ix], posterior[:, iy], s=4, alpha=0.4)
    plt.xlabel(labx)
    plt.ylabel(laby)
    plt.title(f"Joint posterior of {labx} and {laby}")

plt.show()


# In[69]:


import numpy as np
import matplotlib.pyplot as plt

# Quadratic model y = a + bx + cx^2
def log_likelihood(a, b, c, x, y, sigma2):
    return -0.5 * np.sum((y - (a + b * x + c * x**2))**2) / sigma2

# MCMC setup
Na, Nb, Nc = 100, 100, 100  # Grid points for a, b, c
a_range = np.linspace(a - 5 * std_a, a + 5 * std_a, Na)
b_range = np.linspace(b - 5 * std_b, b + 5 * std_b, Nb)
c_range = np.linspace(0, 1, Nc)  # c is between 0 and 1

def log_prior(a, b, c):
    if (a_range[0] <= a <= a_range[-1]) and (b_range[0] <= b <= b_range[-1]) and (c_range[0] <= c <= c_range[-1]):
        return 0.0  # Uniform prior
    else:
        return -np.inf  # Outside prior range

def posterior(a, b, c, x, y, sigma2):
    p = log_prior(a, b, c)
    if np.isfinite(p):
        return log_likelihood(a, b, c, x, y, sigma2) + p
    else:
        return -np.inf

# MCMC proposal settings
std_a, std_b, std_c = 0.08, 0.03, 0.005
steps = 60000
burn = 15000
thin = 10

a_1, b_1, c_1 = a, b, c  # Initial values
log_post = posterior(a_1, b_1, c_1, x, y, sigma2)
accepted = 0
samples = []
generator = np.random.seed(1)

for i in range(steps):
    an = np.random.normal(a_1, std_a)
    bn = np.random.normal(b_1, std_b)
    cn = np.random.normal(c_1, std_c)
    log_post1 = posterior(an, bn, cn, x, y, sigma2)
    
    if np.random.random() < np.exp(log_post1 - log_post):
        a_1, b_1, c_1, log_post = an, bn, cn, log_post1
        accepted += 1
    samples.append([a_1, b_1, c_1])

samples = np.array(samples)
acc_rate = accepted / steps
posterior_samples = samples[burn::thin]

# Calculate posterior statistics
a_samples, b_samples, c_samples = posterior_samples[:, 0], posterior_samples[:, 1], posterior_samples[:, 2]
mean_a, mean_b, mean_c = np.mean(a_samples), np.mean(b_samples), np.mean(c_samples)
stdv_a = np.std(a_samples, ddof=1)
stdv_b = np.std(b_samples, ddof=1)
stdv_c = np.std(c_samples, ddof=1)

# Credible Intervals
def ci(x, level):
    q = (100 - level) / 2
    return np.percentile(x, [q, 100 - q])

ca68 = ci(a_samples, 68)
cb68 = ci(b_samples, 68)
cc68 = ci(c_samples, 68)
ca95 = ci(a_samples, 95)
cb95 = ci(b_samples, 95)
cc95 = ci(c_samples, 95)

print("Acceptance rate:", round(acc_rate, 3))
print(f"Posterior mean a = {mean_a:.5f} ± {stdv_a:.5f}")
print(f"Posterior mean b = {mean_b:.5f} ± {stdv_b:.5f}")
print(f"Posterior mean c = {mean_c:.5f} ± {stdv_c:.5f}")
print(f"68% CI for a: {ca68}")
print(f"68% CI for b: {cb68}")
print(f"68% CI for c: {cc68}")
print(f"95% CI for a: {ca95}")
print(f"95% CI for b: {cb95}")
print(f"95% CI for c: {cc95}")

# Corner-style plot (1D and 2D)
# 1D histograms for a, b, and c
plt.figure()
plt.hist(a_samples, bins=40, density=True, color='skyblue', edgecolor='gray')
plt.axvline(mean_a, linestyle='-', color='red')
plt.title("Posterior of a")
plt.xlabel("a")
plt.ylabel("Density")

plt.figure()
plt.hist(b_samples, bins=40, density=True, color='orange', edgecolor='gray')
plt.axvline(mean_b, linestyle='-', color='black')
plt.title("Posterior of b")
plt.xlabel("b")
plt.ylabel("Density")

plt.figure()
plt.hist(c_samples, bins=40, density=True, color='green', edgecolor='gray')
plt.axvline(mean_c, linestyle='-', color='black')
plt.title("Posterior of c")
plt.xlabel("c")
plt.ylabel("


# #### Grok

# In[63]:


import numpy as np
import matplotlib.pyplot as plt

# Load the data from Part 1
data = np.loadtxt("3002691_MDC2.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# Part 1 OLS fit to get noise variance and initial parameter estimates
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b_part1 = (n * XY - X * Y) / D
a_part1 = (Y * X2 - X * XY) / D
y_pred = a_part1 + b_part1 * x
sigma2 = np.sum((y - y_pred)**2) / (n - 2)  # Noise variance estimate
sigma = np.sqrt(sigma2)  # Standard deviation for likelihood

print(f"Estimated noise standard deviation from Part 1: {sigma:.4f}")
print(f"Part 1 parameters: a = {a_part1:.4f}, b = {b_part1:.4f}")

# Log-likelihood function for quadratic model y = a + bx + cx^2
def log_likelihood(a, b, c, x, y, sigma):
    y_model = a + b * x + c * x**2
    return -0.5 * np.sum((y - y_model)**2 / sigma2 + np.log(2 * np.pi * sigma2))

# Log-prior function (uniform priors based on Part 1 and c between 0 and 1)
def log_prior(a, b, c):
    # Set prior ranges around Part 1 estimates with some flexibility
    var_a = sigma2 * X2 / D
    var_b = sigma2 * n / D
    a_min, a_max = a_part1 - 5 * np.sqrt(var_a), a_part1 + 5 * np.sqrt(var_a)
    b_min, b_max = b_part1 - 5 * np.sqrt(var_b), b_part1 + 5 * np.sqrt(var_b)
    if a_min < a < a_max and b_min < b < b_max and 0 < c < 1:
        return 0.0
    return -np.inf

# Log-posterior function
def log_posterior(params, x, y, sigma):
    a, b, c = params
    lp = log_prior(a, b, c)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(a, b, c, x, y, sigma)

# Metropolis algorithm
def metropolis(n_steps, initial_params, step_sizes, x, y, sigma):
    samples = np.zeros((n_steps, 3))
    current_params = np.array(initial_params)
    current_log_post = log_posterior(current_params, x, y, sigma)
    accepted = 0
    
    for i in range(n_steps):
        # Propose new parameters using normal distribution
        proposal = current_params + step_sizes * np.random.randn(3)
        proposal_log_post = log_posterior(proposal, x, y, sigma)
        
        # Accept or reject
        if np.log(np.random.rand()) < proposal_log_post - current_log_post:
            current_params = proposal
            current_log_post = proposal_log_post
            accepted += 1
        
        samples[i] = current_params
    
    acceptance_rate = accepted / n_steps
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    return samples

# Run Metropolis algorithm
n_steps = 100000
initial_params = [a_part1, b_part1, 0.5]  # Start with Part 1 a, b and c=0.5
step_sizes = [0.1 * np.sqrt(sigma2 * X2 / D), 0.1 * np.sqrt(sigma2 * n / D), 0.05]  # Scaled step sizes
samples = metropolis(n_steps, initial_params, step_sizes, x, y, sigma)

# Burn-in and thinning
burn_in = 10000
thinned_samples = samples[burn_in::10]  # Thin by taking every 10th sample

# Find maximum a posteriori (MAP) parameters
log_posts = np.zeros(len(thinned_samples))
for i, p in enumerate(thinned_samples):
    log_posts[i] = log_posterior(p, x, y, sigma)
max_idx = np.argmax(log_posts)
a_map, b_map, c_map = thinned_samples[max_idx]

# Calculate uncertainties (standard deviations of the posterior)
a_std = np.sqrt(np.mean((thinned_samples[:, 0] - np.mean(thinned_samples[:, 0]))**2))
b_std = np.sqrt(np.mean((thinned_samples[:, 1] - np.mean(thinned_samples[:, 1]))**2))
c_std = np.sqrt(np.mean((thinned_samples[:, 2] - np.mean(thinned_samples[:, 2]))**2))

# Print results
print("\nQuadratic Model Fit Results (y = a + bx + cx^2):")
print(f"(i)   a (intercept)    = {a_map:.4f} ± {a_std:.4f} (Part 1: {a_part1:.4f})")
print(f"(ii)  b (slope)        = {b_map:.4f} ± {b_std:.4f} (Part 1: {b_part1:.4f})")
print(f"(iii) c (quadratic)    = {c_map:.4f} ± {c_std:.4f} (Expected: 0 to 1)")
print(f"(iv)  Noise variance   = {sigma2:.4f} (from Part 1)")

# Custom corner plot
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Parameter ranges for histograms and scatter plots
a_samples, b_samples, c_samples = thinned_samples[:, 0], thinned_samples[:, 1], thinned_samples[:, 2]
a_range = (np.min(a_samples), np.max(a_samples))
b_range = (np.min(b_samples), np.max(b_samples))
c_range = (np.min(c_samples), np.max(c_samples))

# 1D histograms (diagonal)
axes[0, 0].hist(a_samples, bins=50, range=a_range, density=True, color='blue', alpha=0.7)
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('a')

axes[1, 1].hist(b_samples, bins=50, range=b_range, density=True, color='blue', alpha=0.7)
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('b')

axes[2, 2].hist(c_samples, bins=50, range=c_range, density=True, color='blue', alpha=0.7)
axes[2, 2].set_xlabel('c')
axes[2, 2].set_ylabel('Density')
axes[2, 2].set_title('c')

# 2D scatter plots (lower triangle)
axes[1, 0].scatter(a_samples, b_samples, s=1, alpha=0.3, color='black')
axes[1, 0].set_xlabel('a')
axes[1, 0].set_ylabel('b')
axes[1, 0].set_xlim(a_range)
axes[1, 0].set_ylim(b_range)

axes[2, 0].scatter(a_samples, c_samples, s=1, alpha=0.3, color='black')
axes[2, 0].set_xlabel('a')
axes[2, 0].set_ylabel('c')
axes[2, 0].set_xlim(a_range)
axes[2, 0].set_ylim(c_range)

axes[2, 1].scatter(b_samples, c_samples, s=1, alpha=0.3, color='black')
axes[2, 1].set_xlabel('b')
axes[2, 1].set_ylabel('c')
axes[2, 1].set_xlim(b_range)
axes[2, 1].set_ylim(c_range)

# Hide upper triangle plots
for i in range(3):
    for j in range(i + 1, 3):
        axes[i, j].set_visible(False)

plt.tight_layout()
plt.savefig('custom_corner_plot.png')
plt.show()

# Plot data with quadratic fit
y_map = a_map + b_map * x + c_map * x**2
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data', color='black')
plt.plot(x, y_map, label=f'Fit: y = {a_map:.2f} + {b_map:.2f}x + {c_map:.2f}x^2', color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Quadratic Model Fit")
plt.legend()
plt.grid(True)
plt.show()


# In[64]:


import numpy as np
import matplotlib.pyplot as plt

# Load the data from Part 1
data = np.loadtxt("3002691_MDC1.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# Part 1 OLS fit to get noise variance and initial parameter estimates
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b_part1 = (n * XY - X * Y) / D
a_part1 = (Y * X2 - X * XY) / D
y_pred = a_part1 + b_part1 * x
sigma2 = np.sum((y - y_pred)**2) / (n - 2)  # Noise variance estimate
sigma = np.sqrt(sigma2)  # Standard deviation for likelihood

print(f"Estimated noise standard deviation from Part 1: {sigma:.4f}")
print(f"Part 1 parameters: a = {a_part1:.4f}, b = {b_part1:.4f}")

# Log-likelihood function for quadratic model y = a + bx + cx^2
def log_likelihood(a, b, c, x, y, sigma):
    y_model = a + b * x + c * x**2
    return -0.5 * np.sum((y - y_model)**2 / sigma2 + np.log(2 * np.pi * sigma2))

# Log-prior function (uniform priors based on Part 1 and c between 0 and 1)
def log_prior(a, b, c):
    # Set prior ranges around Part 1 estimates with some flexibility
    var_a = sigma2 * X2 / D
    var_b = sigma2 * n / D
    a_min, a_max = a_part1 - 5 * np.sqrt(var_a), a_part1 + 5 * np.sqrt(var_a)
    b_min, b_max = b_part1 - 5 * np.sqrt(var_b), b_part1 + 5 * np.sqrt(var_b)
    if a_min < a < a_max and b_min < b < b_max and 0 < c < 1:
        return 0.0
    return -np.inf

# Log-posterior function
def log_posterior(params, x, y, sigma):
    a, b, c = params
    lp = log_prior(a, b, c)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(a, b, c, x, y, sigma)

# Metropolis algorithm
def metropolis(n_steps, initial_params, step_sizes, x, y, sigma):
    samples = np.zeros((n_steps, 3))
    current_params = np.array(initial_params)
    current_log_post = log_posterior(current_params, x, y, sigma)
    accepted = 0
    
    for i in range(n_steps):
        # Propose new parameters using normal distribution
        proposal = current_params + step_sizes * np.random.randn(3)
        proposal_log_post = log_posterior(proposal, x, y, sigma)
        
        # Accept or reject
        if np.log(np.random.rand()) < proposal_log_post - current_log_post:
            current_params = proposal
            current_log_post = proposal_log_post
            accepted += 1
        
        samples[i] = current_params
    
    acceptance_rate = accepted / n_steps
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    return samples

# Run Metropolis algorithm
n_steps = 100000
initial_params = [a_part1, b_part1, 0.5]  # Start with Part 1 a, b and c=0.5
step_sizes = [0.1 * np.sqrt(sigma2 * X2 / D), 0.1 * np.sqrt(sigma2 * n / D), 0.05]  # Scaled step sizes
samples = metropolis(n_steps, initial_params, step_sizes, x, y, sigma)

# Burn-in and thinning
burn_in = 10000
thinned_samples = samples[burn_in::10]  # Thin by taking every 10th sample

# Find maximum a posteriori (MAP) parameters
log_posts = np.zeros(len(thinned_samples))
for i, p in enumerate(thinned_samples):
    log_posts[i] = log_posterior(p, x, y, sigma)
max_idx = np.argmax(log_posts)
a_map, b_map, c_map = thinned_samples[max_idx]

# Calculate uncertainties (standard deviations of the posterior)
a_std = np.sqrt(np.mean((thinned_samples[:, 0] - np.mean(thinned_samples[:, 0]))**2))
b_std = np.sqrt(np.mean((thinned_samples[:, 1] - np.mean(thinned_samples[:, 1]))**2))
c_std = np.sqrt(np.mean((thinned_samples[:, 2] - np.mean(thinned_samples[:, 2]))**2))

# Print results
print("\nQuadratic Model Fit Results (y = a + bx + cx^2):")
print(f"(i)   a (intercept)    = {a_map:.4f} ± {a_std:.4f} (Part 1: {a_part1:.4f})")
print(f"(ii)  b (slope)        = {b_map:.4f} ± {b_std:.4f} (Part 1: {b_part1:.4f})")
print(f"(iii) c (quadratic)    = {c_map:.4f} ± {c_std:.4f} (Expected: 0 to 1)")
print(f"(iv)  Noise variance   = {sigma2:.4f} (from Part 1)")

# Function to compute contour levels for 68% and 95% confidence regions
def compute_contour_levels(hist, bins_x, bins_y, levels=[0.68, 0.95]):
    # Normalize histogram to get probability density
    hist = hist / np.sum(hist)
    # Flatten and sort densities in descending order
    sorted_densities = np.sort(hist.ravel())[::-1]
    cumulative = np.cumsum(sorted_densities)
    # Find density thresholds for given confidence levels
    contour_levels = []
    for level in levels:
        idx = np.where(cumulative >= level)[0][0]
        contour_levels.append(sorted_densities[idx])
    return contour_levels

# Custom corner plot with 68% and 95% confidence regions
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Parameter ranges for histograms and scatter plots
a_samples, b_samples, c_samples = thinned_samples[:, 0], thinned_samples[:, 1], thinned_samples[:, 2]
a_range = (np.min(a_samples), np.max(a_samples))
b_range = (np.min(b_samples), np.max(b_samples))
c_range = (np.min(c_samples), np.max(c_samples))

# 1D histograms (diagonal)
axes[0, 0].hist(a_samples, bins=50, range=a_range, density=True, color='blue', alpha=0.7)
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('a')

axes[1, 1].hist(b_samples, bins=50, range=b_range, density=True, color='blue', alpha=0.7)
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('b')

axes[2, 2].hist(c_samples, bins=50, range=c_range, density=True, color='blue', alpha=0.7)
axes[2, 2].set_xlabel('c')
axes[2, 2].set_ylabel('Density')
axes[2, 2].set_title('c')

# 2D scatter plots with contours (lower triangle)
# (a, b)
hist_ab, a_edges, b_edges = np.histogram2d(a_samples, b_samples, bins=50, range=[a_range, b_range])
contour_levels_ab = compute_contour_levels(hist_ab, a_edges, b_edges)
a_centers = (a_edges[:-1] + a_edges[1:]) / 2
b_centers = (b_edges[:-1] + b_edges[1:]) / 2
A, B = np.meshgrid(a_centers, b_centers)
axes[1, 0].scatter(a_samples, b_samples, s=1, alpha=0.3, color='black')
axes[1, 0].contour(A, B, hist_ab.T, levels=contour_levels_ab, colors=['red', 'green'])
axes[1, 0].set_xlabel('a')
axes[1, 0].set_ylabel('b')
axes[1, 0].set_xlim(a_range)
axes[1, 0].set_ylim(b_range)

# (a, c)
hist_ac, a_edges, c_edges = np.histogram2d(a_samples, c_samples, bins=50, range=[a_range, c_range])
contour_levels_ac = compute_contour_levels(hist_ac, a_edges, c_edges)
c_centers = (c_edges[:-1] + c_edges[1:]) / 2
A, C = np.meshgrid(a_centers, c_centers)
axes[2, 0].scatter(a_samples, c_samples, s=1, alpha=0.3, color='black')
axes[2, 0].contour(A, C, hist_ac.T, levels=contour_levels_ac, colors=['red', 'green'])
axes[2, 0].set_xlabel('a')
axes[2, 0].set_ylabel('c')
axes[2, 0].set_xlim(a_range)
axes[2, 0].set_ylim(c_range)

# ( trưởng, c)
hist_bc, b_edges, c_edges = np.histogram2d(b_samples, c_samples, bins=50, range=[b_range, c_range])
contour_levels_bc = compute_contour_levels(hist_bc, b_edges, c_edges)
B, C = np.meshgrid(b_centers, c_centers)
axes[2, 1].scatter(b_samples, c_samples, s=1, alpha=0.3, color='black')
axes[2, 1].contour(B, C, hist_bc.T, levels=contour_levels_bc, colors=['red', 'green'])
axes[2, 1].set_xlabel('b')
axes[2, 1].set_ylabel('c')
axes[2, 1].set_xlim(b_range)
axes[2, 1].set_ylim(c_range)

# Hide upper triangle plots
for i in range(3):
    for j in range(i + 1, 3):
        axes[i, j].set_visible(False)

plt.tight_layout()
plt.savefig('custom_corner_plot_with_contours.png')
plt.show()

# Plot data with quadratic fit
y_map = a_map + b_map * x + c_map * x**2
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data', color='black')
plt.plot(x, y_map, label=f'Fit: y = {a_map:.2f} + {b_map:.2f}x + {c_map:.2f}x^2', color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Quadratic Model Fit")
plt.legend()
plt.grid(True)
plt.show()


# In[67]:


import numpy as np
import matplotlib.pyplot as plt

# Load the data from Part 1
data = np.loadtxt("3002691_MDC2.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# Part 1 OLS fit to get noise variance and initial parameter estimates
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b = (n * XY - X * Y) / D
a = (Y * X2 - X * XY) / D
y_pred = a + b * x
sigma2 = np.sum((y - y_pred)**2) / (n - 2)  # Noise variance estimate
sigma = np.sqrt(sigma2)

# Part 1 uncertainties
var_b = sigma2 * n / D
var_a = sigma2 * X2 / D
std_a = np.sqrt(var_a)
std_b = np.sqrt(var_b)

print(f"Estimated noise standard deviation from Part 1: {sigma:.4f}")
print(f"Part 1 OLS parameters: a = {a:.4f}, b = {b:.4f}")

# Step 1: MCMC Sampling (Metropolis)
def log_likelihood(a1, b1, c1):
    y_model = a1 + b1 * x + c1 * x**2
    return -0.5 * np.sum((y - y_model)**2) / sigma2

# Prior ranges
Na, Nb, Nc = 100, 100, 100
a_range = np.linspace(a - 5 * std_a, a + 5 * std_a, Na)
b_range = np.linspace(b - 5 * std_b, b + 5 * std_b, Nb)
c_range = np.linspace(0, 1, Nc)  # c between 0 and 1

def log_prior(a1, b1, c1):
    if (a_range[0] <= a1 <= a_range[-1]) and (b_range[0] <= b1 <= b_range[-1]) and (0 < c1 < 1):
        return 0.0
    return -np.inf

def posterior(a1, b1, c1):
    p = log_prior(a1, b1, c1)
    if np.isfinite(p):
        return log_likelihood(a1, b1, c1) + p
    return -np.inf

# MCMC proposal settings and initialization
std_a1, std_b1, std_c1 = 0.03, 0.015, 0.005  # Proposal standard deviations
steps = 40000
burn = 10000
thin = 5
a_1, b_1, c_1 = a, b, 0.5  # Initial values: Part 1 a, b; c=0.5
log_post = posterior(a_1, b_1, c_1)
accepted = 0
samples = []
np.random.seed(1)

for i in range(steps):
    an = np.random.normal(a_1, std_a1)
    bn = np.random.normal(b_1, std_b1)
    cn = np.random.normal(c_1, std_c1)
    log_post1 = posterior(an, bn, cn)
    if np.random.random() < np.exp(log_post1 - log_post):
        a_1, b_1, c_1, log_post = an, bn, cn, log_post1
        accepted += 1
    samples.append([a_1, b_1, c_1])

samples = np.array(samples)
acc_rate = accepted / steps
posterior_samples = samples[burn::thin]

# Calculating Parameters
a_samples, b_samples, c_samples = posterior_samples[:, 0], posterior_samples[:, 1], posterior_samples[:, 2]
mean_a, mean_b, mean_c = np.mean(a_samples), np.mean(b_samples), np.mean(c_samples)
stdv_a = np.sqrt(np.sum((a_samples - mean_a)**2) / (len(a_samples) - 1))
stdv_b = np.sqrt(np.sum((b_samples - mean_b)**2) / (len(b_samples) - 1))
stdv_c = np.sqrt(np.sum((c_samples - mean_c)**2) / (len(c_samples) - 1))

# MAP parameters
log_probs = []
for a1, b1, c1 in posterior_samples:
    log_probs.append(posterior(a1, b1, c1))
log_probs = np.array(log_probs)
max_idx = np.argmax(log_probs)
best_a, best_b, best_c = posterior_samples[max_idx]

# Credible intervals
def ci(x, level):
    q = (100 - level) / 2
    return np.percentile(x, [q, 100 - q])

ca68 = ci(a_samples, 68)
cb68 = ci(b_samples, 68)
cc68 = ci(c_samples, 68)
ca95 = ci(a_samples, 95)
cb95 = ci(b_samples, 95)
cc95 = ci(c_samples, 95)

print("\nQuadratic Model Fit Results (y = a + bx + cx^2):")
print("Acceptance rate:", round(acc_rate, 3))
print("Posterior mean a = " + str(round(mean_a, 5)) + " ± " + str(round(stdv_a, 5)))
print("Posterior mean b = " + str(round(mean_b, 5)) + " ± " + str(round(stdv_b, 5)))
print("Posterior mean c = " + str(round(mean_c, 5)) + " ± " + str(round(stdv_c, 5)))
print("MAP: a = " + str(round(best_a, 5)) + ", b = " + str(round(best_b, 5)) + ", c = " + str(round(best_c, 5)))
print("68% CI for a:", ca68)
print("68% CI for b:", cb68)
print("68% CI for c:", cc68)
print("95% CI for a:", ca95)
print("95% CI for b:", cb95)
print("95% CI for c:", cc95)
print("Noise variance (from Part 1):", round(sigma2, 5))
print("Part 1 OLS comparison: a = " + str(round(a, 5)) + ", b = " + str(round(b, 5)))

# Corner-style plots
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# 1D histograms
axes[0, 0].hist(a_samples, bins=40, density=True, color='skyblue', edgecolor='gray')
axes[0, 0].axvline(mean_a, linestyle='-', color='red')
axes[0, 0].set_title("a Posterior")
axes[0, 0].set_xlabel("a")
axes[0, 0].set_ylabel("Density")

axes[1, 1].hist(b_samples, bins=40, density=True, color='orange', edgecolor='gray')
axes[1, 1].axvline(mean_b, linestyle='-', color='black')
axes[1, 1].set_title("b Posterior")
axes[1, 1].set_xlabel("b")
axes[1, 1].set_ylabel("Density")

axes[2, 2].hist(c_samples, bins=40, density=True, color='green', edgecolor='gray')
axes[2, 2].axvline(mean_c, linestyle='-', color='blue')
axes[2, 2].set_title("c Posterior")
axes[2, 2].set_xlabel("c")
axes[2, 2].set_ylabel("Density")

# Contour plots
def plot_contour(ax, x_samples, y_samples, x_label, y_label, x_range, y_range):
    hist2d, x_edges, y_edges = np.histogram2d(x_samples, y_samples, bins=50, density=True)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    hist_trans = hist2d.T
    hist_flat = hist_trans.flatten()
    hist_sorted = np.sort(hist_flat)[::-1]
    hist_cumulative = np.cumsum(hist_sorted)
    hist_cumulative /= hist_cumulative[-1]
    level_68 = hist_sorted[np.searchsorted(hist_cumulative, 0.683)]
    level_95 = hist_sorted[np.searchsorted(hist_cumulative, 0.954)]
    ax.contourf(x_mid, y_mid, hist_trans, levels=[level_95, level_68, hist_trans.max()],
                alpha=0.6, colors=['orange', 'green', 'red'])
    ax.scatter(x_samples, y_samples, s=0.3, color='black')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

# Plot contours for each pair
plot_contour(axes[1, 0], a_samples, b_samples, "a", "b",
             (np.min(a_samples), np.max(a_samples)), (np.min(b_samples), np.max(b_samples)))
plot_contour(axes[2, 0], a_samples, c_samples, "a", "c",
             (np.min(a_samples), np.max(a_samples)), (np.min(c_samples), np.max(c_samples)))
plot_contour(axes[2, 1], b_samples, c_samples, "b", "c",
             (np.min(b_samples), np.max(b_samples)), (np.min(c_samples), np.max(c_samples)))

# Hide upper triangle
for i in range(3):
    for j in range(i + 1, 3):
        axes[i, j].set_visible(False)

plt.tight_layout()
plt.savefig('custom_corner_plot_quadratic.png')
plt.show()

# Plot data with MAP quadratic fit
y_map = best_a + best_b * x + best_c * x**2
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data', color='black')
plt.plot(x, y_map, label=f'Fit: y = {best_a:.2f} + {best_b:.2f}x + {best_c:.2f}x^2', color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Quadratic Model Fit")
plt.legend()
plt.grid(True)
plt.show()


# In[72]:


import numpy as np
import matplotlib.pyplot as plt

# Load the data from Part 1
data = np.loadtxt("3002691_MDC2.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# Part 1 OLS fit to get noise variance and initial parameter estimates
X = np.sum(x)
Y = np.sum(y)
X2 = np.sum(x**2)
XY = np.sum(x * y)
D = n * X2 - X**2
b = (n * XY - X * Y) / D
a = (Y * X2 - X * XY) / D
y_pred = a + b * x
sigma2 = np.sum((y - y_pred)**2) / (n - 2)  # Noise variance estimate
sigma = np.sqrt(sigma2)

# Part 1 uncertainties
var_b = sigma2 * n / D
var_a = sigma2 * X2 / D
std_a = np.sqrt(var_a)
std_b = np.sqrt(var_b)

print(f"Estimated noise standard deviation from Part 1: {sigma:.4f}")
print(f"Part 1 OLS parameters: a = {a:.4f}, b = {b:.4f}")

# Step 1: MCMC Sampling (Metropolis)
def log_likelihood(a1, b1, c1):
    y_model = a1 + b1 * x + c1 * x**2
    return -0.5 * np.sum((y - y_model)**2) / sigma2

# Prior ranges
Na, Nb, Nc = 100, 100, 100
a_range = np.linspace(a - 5 * std_a, a + 5 * std_a, Na)
b_range = np.linspace(b - 5 * std_b, b + 5 * std_b, Nb)
c_range = np.linspace(0, 1, Nc)  # c between 0 and 1

def log_prior(a1, b1, c1):
    if (a_range[0] <= a1 <= a_range[-1]) and (b_range[0] <= b1 <= b_range[-1]) and (0 < c1 < 1):
        return 0.0
    return -np.inf

def posterior(a1, b1, c1):
    p = log_prior(a1, b1, c1)
    if np.isfinite(p):
        return log_likelihood(a1, b1, c1) + p
    return -np.inf

# MCMC proposal settings and initialization
std_a1, std_b1, std_c1 = 0.08, 0.03, 0.005  # Proposal standard deviations
steps = 60000
burn = 15000
thin = 10

a_1, b_1, c_1 = a, b, 0.5  # Initial values: Part 1 a, b; c=0.5
log_post = posterior(a_1, b_1, c_1)
accepted = 0
samples = []
np.random.seed(1)

for i in range(steps):
    an = np.random.normal(a_1, std_a1)
    bn = np.random.normal(b_1, std_b1)
    cn = np.random.normal(c_1, std_c1)
    log_post1 = posterior(an, bn, cn)
    if np.random.random() < np.exp(log_post1 - log_post):
        a_1, b_1, c_1, log_post = an, bn, cn, log_post1
        accepted += 1
    samples.append([a_1, b_1, c_1])

samples = np.array(samples)
acc_rate = accepted / steps
posterior_samples = samples[burn::thin]

# Calculating Parameters
a_samples, b_samples, c_samples = posterior_samples[:, 0], posterior_samples[:, 1], posterior_samples[:, 2]
mean_a, mean_b, mean_c = np.mean(a_samples), np.mean(b_samples), np.mean(c_samples)
stdv_a = np.sqrt(np.sum((a_samples - mean_a)**2) / (len(a_samples) - 1))
stdv_b = np.sqrt(np.sum((b_samples - mean_b)**2) / (len(b_samples) - 1))
stdv_c = np.sqrt(np.sum((c_samples - mean_c)**2) / (len(c_samples) - 1))

# MAP parameters
log_probs = []
for a1, b1, c1 in posterior_samples:
    log_probs.append(posterior(a1, b1, c1))
log_probs = np.array(log_probs)
max_idx = np.argmax(log_probs)
best_a, best_b, best_c = posterior_samples[max_idx]

# Credible intervals
def ci(x, level):
    q = (100 - level) / 2
    return np.percentile(x, [q, 100 - q])

ca68 = ci(a_samples, 68)
cb68 = ci(b_samples, 68)
cc68 = ci(c_samples, 68)
ca95 = ci(a_samples, 95)
cb95 = ci(b_samples, 95)
cc95 = ci(c_samples, 95)

print("\nQuadratic Model Fit Results (y = a + bx + cx^2):")
print("Acceptance rate:", round(acc_rate, 3))
print("Posterior mean a = " + str(round(mean_a, 5)) + " ± " + str(round(stdv_a, 5)))
print("Posterior mean b = " + str(round(mean_b, 5)) + " ± " + str(round(stdv_b, 5)))
print("Posterior mean c = " + str(round(mean_c, 5)) + " ± " + str(round(stdv_c, 5)))
print("MAP: a = " + str(round(best_a, 5)) + ", b = " + str(round(best_b, 5)) + ", c = " + str(round(best_c, 5)))
print("68% CI for a:", ca68)
print("68% CI for b:", cb68)
print("68% CI for c:", cc68)
print("95% CI for a:", ca95)
print("95% CI for b:", cb95)
print("95% CI for c:", cc95)
print("Noise variance (from Part 1):", round(sigma2, 5))
print("Part 1 OLS comparison: a = " + str(round(a, 5)) + ", b = " + str(round(b, 5)))

# Corner-style plots
# 1D posterior of a
plt.figure()
plt.hist(a_samples, bins=40, density=True, color='skyblue', edgecolor='gray')
plt.axvline(mean_a, linestyle='-', color='red')
plt.title("a Posterior")
plt.xlabel("a")
plt.ylabel("Density")
plt.savefig('a_posterior.png')
plt.show()

# 1D posterior of b
plt.figure()
plt.hist(b_samples, bins=40, density=True, color='orange', edgecolor='gray')
plt.axvline(mean_b, linestyle='-', color='black')
plt.title("b Posterior")
plt.xlabel("b")
plt.ylabel("Density")
plt.savefig('b_posterior.png')
plt.show()

# 1D posterior of c
plt.figure()
plt.hist(c_samples, bins=40, density=True, color='green', edgecolor='gray')
plt.axvline(mean_c, linestyle='-', color='blue')
plt.title("c Posterior")
plt.xlabel("c")
plt.ylabel("Density")
plt.savefig('c_posterior.png')
plt.show()

# 2D contour plots
def plot_contour(x_samples, y_samples, x_label, y_label, filename):
    hist2d, x_edges, y_edges = np.histogram2d(x_samples, y_samples, bins=50, density=True)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    hist_trans = hist2d.T
    hist_flat = hist_trans.flatten()
    hist_sorted = np.sort(hist_flat)[::-1]
    hist_cumulative = np.cumsum(hist_sorted)
    hist_cumulative /= hist_cumulative[-1]
    level_68 = hist_sorted[np.searchsorted(hist_cumulative, 0.683)]
    level_95 = hist_sorted[np.searchsorted(hist_cumulative, 0.954)]
    plt.figure()
    plt.contourf(x_mid, y_mid, hist_trans, levels=[level_95, level_68, hist_trans.max()],
                 alpha=0.6, colors=['orange', 'green', 'red'])
    plt.scatter(x_samples, y_samples, s=0.3, color='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label} Posterior")
    plt.savefig(filename)
    plt.show()

# Plot contours for each pair
plot_contour(a_samples, b_samples, "a", "b", "ab_posterior.png")
plot_contour(a_samples, c_samples, "a", "c", "ac_posterior.png")
plot_contour(b_samples, c_samples, "b", "c", "bc_posterior.png")

# Plot data with MAP quadratic fit
y_map = best_a + best_b * x + best_c * x**2
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data', color='black')
plt.plot(x, y_map, label=f'Fit: y = {best_a:.2f} + {best_b:.2f}x + {best_c:.2f}x^2', color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Quadratic Model Fit")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




