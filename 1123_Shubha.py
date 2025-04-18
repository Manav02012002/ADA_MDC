"""
Restructured version of the original script.
Algorithms are identical; only names and layout have been modern‑ised.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------#
# Helper utilities
# -----------------------------------------------------------------------------#
def ones_template(arr: np.ndarray) -> np.ndarray:
    """Return an array of ones with the same shape/dtype as *arr*."""
    return np.ones(arr.shape, dtype=arr.dtype)


def unravel_index_custom(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Convert *flat_idx* to a multi‑dimensional index for a given *shape*.
    A stripped‑down replacement for ``np.unravel_index``.
    """
    idx: list[int] = []
    for dim in reversed(shape):
        idx.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(idx))


# -----------------------------------------------------------------------------#
# Ordinary Least Squares
# -----------------------------------------------------------------------------#
data_path = "3060309_MDC1.txt"

raw_data = np.loadtxt(data_path)
x_data, y_data = raw_data[:, 0], raw_data[:, 1]
num_points = x_data.size

x_mean, y_mean = np.mean(x_data), np.mean(y_data)

sum_xx = np.sum((x_data - x_mean) ** 2)          # Σ (x‑x̄)²
sum_xy = np.sum((x_data - x_mean) * (y_data - y_mean))  # Σ (x‑x̄)(y‑ȳ)

slope_ols = sum_xy / sum_xx
intercept_ols = y_mean - slope_ols * x_mean

residuals_ols = y_data - (intercept_ols + slope_ols * x_data)
ssr_ols = np.sum(residuals_ols ** 2)
noise_var_hat = ssr_ols / (num_points - 2)       # σ̂²

var_slope = noise_var_hat / sum_xx
var_intercept = noise_var_hat * (1 / num_points + x_mean ** 2 / sum_xx)
cov_intercept_slope = -x_mean * noise_var_hat / sum_xx

se_slope = np.sqrt(var_slope)
se_intercept = np.sqrt(var_intercept)

print("\n--- Ordinary Least Squares ---")
print(f"n                     = {num_points}")
print(f"slope (β̂)            = {slope_ols:.5f}")
print(f"intercept (α̂)        = {intercept_ols:.5f}")
print(f"noise variance (σ̂²)  = {noise_var_hat:.5f}")
print(f"SE[β̂]                = {se_slope:.5f}")
print(f"SE[α̂]                = {se_intercept:.5f}")
print(f"cov(α̂, β̂)           = {cov_intercept_slope:.5f}")

# -- plot the fit
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, s=60, alpha=0.7, color="maroon", label="Data")
x_line = np.array([x_data.min(), x_data.max()])
y_line = intercept_ols + slope_ols * x_line
plt.plot(
    x_line,
    y_line,
    "--",
    color="#FFA700",
    linewidth=2,
    label=f"Fit: y = {intercept_ols:.3f} + {slope_ols:.3f}x",
)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.title("Ordinary Least Squares Fit", fontsize=16)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------#
# χ² landscape around the least‑squares solution
# -----------------------------------------------------------------------------#
sigma_vec = ones_template(y_data)                # unit uncertainties
slope_ls = slope_ols                             # identical to OLS
intercept_ls = intercept_ols                     # identical to OLS

residual_var = ssr_ols / (num_points - 2)
slope_err = np.sqrt(residual_var / sum_xx)
intercept_err = np.sqrt(residual_var * (1 / num_points + x_mean ** 2 / sum_xx))

grid_size = 200
slope_grid = np.linspace(slope_ls - 3 * slope_err,
                         slope_ls + 3 * slope_err, grid_size)
intc_grid = np.linspace(intercept_ls - 3 * intercept_err,
                        intercept_ls + 3 * intercept_err, grid_size)
SLOPE, INTC = np.meshgrid(slope_grid, intc_grid)

# build χ² across the grid (broadcast magic)
grid_res = y_data[:, None, None] - (
    SLOPE[None, :, :] * x_data[:, None, None] + INTC[None, :, :]
)
chi2_map = np.sum((grid_res / sigma_vec[:, None, None]) ** 2, axis=0)

flat_min = chi2_map.argmin()
row_min, col_min = unravel_index_custom(flat_min, chi2_map.shape)
best_slope = SLOPE[row_min, col_min]
best_intercept = INTC[row_min, col_min]
chi2_min = chi2_map[row_min, col_min]

delta_chi2 = chi2_map - chi2_min
cred_levels = [2.30, 6.17, 11.8]                 # 68.3%, 95.4%, 99.73%

print("\n--- χ² grid search ---")
print(f"best slope      = {best_slope:.5f}")
print(f"best intercept  = {best_intercept:.5f}")
print(f"χ²_min          = {chi2_min:.3f}")

# print bounds of the credible regions
for pct, lv in zip([68.3, 95.4, 99.73], cred_levels):
    mask = delta_chi2 <= lv
    slope_bounds = (SLOPE[mask].min(), SLOPE[mask].max())
    intc_bounds = (INTC[mask].min(), INTC[mask].max())
    print(f"{pct:5.2f}% region: slope ∈ [{slope_bounds[0]:.5f}, {slope_bounds[1]:.5f}], "
          f"intercept ∈ [{intc_bounds[0]:.5f}, {intc_bounds[1]:.5f}]")

# contour plot
plt.figure(figsize=(8, 6))
cs = plt.contour(
    SLOPE, INTC, delta_chi2,
    levels=cred_levels,
    colors=["maroon", "orange", "olive"],
    linewidths=1.5,
)
plt.plot(best_slope, best_intercept, "k+", ms=10, label="Best‑fit")
plt.clabel(cs, fmt={lv: lbl for lv, lbl in zip(cred_levels, ["68.3%", "95.4%", "99.7%"])})
plt.xlabel("slope")
plt.ylabel("intercept")
plt.title(r"Δχ² contours")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------#
# Metropolis MCMC
# -----------------------------------------------------------------------------#
def log_likelihood(slope: float, intercept: float) -> float:
    """Gaussian log‑likelihood with known noise variance."""
    return -0.5 * np.sum((y_data - (intercept + slope * x_data)) ** 2) / noise_var_hat


# flat top‑hat priors centred on the OLS values
slope_prior_bounds = (slope_ols - 3.0, slope_ols + 3.0)
intc_prior_bounds = (intercept_ols - 5.0, intercept_ols + 5.0)


def log_prior(slope: float, intercept: float) -> float:
    """Uniform prior inside the rectangular window, −∞ outside."""
    if (slope_prior_bounds[0] <= slope <= slope_prior_bounds[1]
            and intc_prior_bounds[0] <= intercept <= intc_prior_bounds[1]):
        return 0.0
    return -np.inf


def log_posterior(slope: float, intercept: float) -> float:
    lp = log_prior(slope, intercept)
    return lp + log_likelihood(slope, intercept) if np.isfinite(lp) else -np.inf


# MCMC parameters
prop_sigma_slope, prop_sigma_intc = 0.05, 0.02   # proposal widths
n_steps, burn_in, thinning = 40_000, 10_000, 10

rng = np.random.default_rng(seed=0)
current_slope, current_intc = slope_ols, intercept_ols
current_lp = log_posterior(current_slope, current_intc)

samples: list[list[float]] = []
accepted = 0

for _ in range(n_steps):
    cand_slope = rng.normal(current_slope, prop_sigma_slope)
    cand_intc = rng.normal(current_intc, prop_sigma_intc)
    cand_lp = log_posterior(cand_slope, cand_intc)

    if rng.random() < np.exp(cand_lp - current_lp):
        current_slope, current_intc, current_lp = cand_slope, cand_intc, cand_lp
        accepted += 1

    samples.append([current_slope, current_intc])

chain = np.asarray(samples)
posterior = chain[burn_in::thinning]

mean_slope, mean_intc = posterior.mean(axis=0)
std_slope, std_intc = posterior.std(axis=0, ddof=1)

print("\n--- Metropolis MCMC ---")
print(f"acceptance rate = {accepted / n_steps:.3f}")
print(f"slope  = {mean_slope:.5f} ± {std_slope:.5f}")
print(f"intcpt = {mean_intc:.5f} ± {std_intc:.5f}")

# 1‑D histograms
plt.figure()
plt.hist(posterior[:, 0], bins=40, density=True,
         color="orange", edgecolor="black", linewidth=1.2)
plt.axvline(mean_slope, ls="--", color="black")
plt.title("Posterior of slope")
plt.xlabel("slope")
plt.ylabel("probability density")

plt.figure()
plt.hist(posterior[:, 1], bins=40, density=True,
         color="orange", edgecolor="black", linewidth=1.2)
plt.axvline(mean_intc, ls="--", color="black")
plt.title("Posterior of intercept")
plt.xlabel("intercept")
plt.ylabel("probability density")

# 2‑D posterior density with credibility contours
H, xedges, yedges = np.histogram2d(
    posterior[:, 0], posterior[:, 1], bins=50, density=True
)
xcent = 0.5 * (xedges[:-1] + xedges[1:])
ycent = 0.5 * (yedges[:-1] + yedges[1:])
H_flat = np.sort(H.ravel())[::-1]
H_cum = np.cumsum(H_flat) / H_flat.sum()
lvl_68 = H_flat[np.searchsorted(H_cum, 0.683)]
lvl_95 = H_flat[np.searchsorted(H_cum, 0.954)]

plt.figure()
plt.contourf(xcent, ycent, H.T,
             levels=[lvl_95, lvl_68, H.max()], alpha=0.6)
plt.scatter(posterior[:, 0], posterior[:, 1], s=4, alpha=0.4, color="k")
plt.title("Joint posterior of slope & intercept")
plt.xlabel("slope")
plt.ylabel("intercept")
plt.tight_layout()
plt.show()
