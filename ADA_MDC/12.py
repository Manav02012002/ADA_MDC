import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
data = np.loadtxt('3060309_MDC1.txt')
x = data[:, 0]
y = data[:, 1]
# assume uniform σ_i = 1
sigma = np.ones_like(y)
N = len(x)

# 2. Manual least‐squares estimates for a0, b0
x_mean = np.mean(x)
y_mean = np.mean(y)
Sxx = np.sum((x - x_mean)**2)
Sxy = np.sum((x - x_mean) * (y - y_mean))
a0 = Sxy / Sxx
b0 = y_mean - a0 * x_mean

# residual variance & parameter uncertainties
resid = y - (a0 * x + b0)
chi2_ls = np.sum((resid / sigma)**2)
sigma2_resid = chi2_ls / (N - 2)
sigma_a = np.sqrt(sigma2_resid / Sxx)
sigma_b = np.sqrt(sigma2_resid * (1.0/N + x_mean**2 / Sxx))

# 3. Build (a,b) grid ±3σ around LS solution
n_a, n_b = 200, 200
a_vals = np.linspace(a0 - 3*sigma_a, a0 + 3*sigma_a, n_a)
b_vals = np.linspace(b0 - 3*sigma_b, b0 + 3*sigma_b, n_b)
A, B = np.meshgrid(a_vals, b_vals)

# 4. Compute χ² on the grid
# residuals shaped (N, n_b, n_a)
res = y[:, None, None] - (A[None,:,:] * x[:, None, None] + B[None,:,:])
Chi2 = np.sum((res / sigma[:, None, None])**2, axis=0)

# 5. Find best‐fit and Δχ²
min_idx = np.unravel_index(np.argmin(Chi2), Chi2.shape)
a_best = A[min_idx]
b_best = B[min_idx]
chi2_min = Chi2[min_idx]
delta_chi2 = Chi2 - chi2_min

# 6. Compute credible‐region bounds
levels = [2.30, 6.17, 11.8]  # for 68.3%, 95.4%, 99.73%
region_ranges = []
for lvl in levels:
    mask = delta_chi2 <= lvl
    a_min, a_max = A[mask].min(), A[mask].max()
    b_min, b_max = B[mask].min(), B[mask].max()
    region_ranges.append((a_min, a_max, b_min, b_max))

# 7. Print results
print("Best‐fit parameters:")
print(f"  a = {a_best:.5f}")
print(f"  b = {b_best:.5f}")
print(f"  χ²_min = {chi2_min:.3f}\n")

for pct, (a_min, a_max, b_min, b_max) in zip([68.3, 95.4, 99.73], region_ranges):
    print(f"{pct:.1f}% credible region:")
    print(f"  a ∈ [{a_min:.5f}, {a_max:.5f}]")
    print(f"  b ∈ [{b_min:.5f}, {b_max:.5f}]\n")

# 8. Plot Δχ² contours
plt.figure(figsize=(8, 6))
# draw full red, green, blue contour lines
cs = plt.contour(
    A, B, delta_chi2,
    levels=levels,
    colors=['red', 'green', 'blue'],
    linewidths=[1.5, 1.5, 1.5]
)
plt.plot(a_best, b_best, 'k+', label='Best‐fit')

plt.xlabel('a')
plt.ylabel('b')
plt.title(r'$\Delta\chi^2$ Contours for $(a,b)$')
plt.clabel(cs, inline=True, fmt={2.30:'68.3%', 6.17:'95.4%', 11.8:'99.73%'})

# 9. Increase axis range by 30% margin to show full green & blue contours
x_min, x_max = a_vals.min(), a_vals.max()
y_min, y_max = b_vals.min(), b_vals.max()
x_margin = 0.1 * (x_max - x_min)
y_margin = 0.1 * (y_max - y_min)
plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)

plt.legend()
plt.show()
