"""
Restructured version of the quadratic‑fit / Metropolis‑MCMC script.
All numerical procedures are unchanged; only names and formatting differ.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────
def ones_mimic(arr: np.ndarray) -> np.ndarray:
    """Return an array of ones matching *arr* in shape and dtype."""
    return np.ones(arr.shape, dtype=arr.dtype)


def solve_normal_eq(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Solve linear‑least‑squares via the normal equations
    (designᵀ design) β = designᵀ target.
    """
    xt = design.T
    normal_mat = xt @ design
    rhs = xt @ target
    return np.linalg.solve(normal_mat, rhs)


# ────────────────────────────────────────────────────────────────────────────
# 1. Load data & ordinary least squares (quadratic model)
# ────────────────────────────────────────────────────────────────────────────
data_file = "3060309_MDC2.txt"
raw = np.loadtxt(data_file)

x_vals, y_vals = raw[:, 0], raw[:, 1]
n_pts = y_vals.size

# design matrix for y = α + β x + γ x²
design_mat = np.column_stack((ones_mimic(x_vals), x_vals, x_vals**2))
alpha_ols, beta_ols, gamma_ols = solve_normal_eq(design_mat, y_vals)

y_hat = alpha_ols + beta_ols * x_vals + gamma_ols * x_vals**2
res_var = np.sum((y_vals - y_hat) ** 2) / (n_pts - 3)
res_sigma = np.sqrt(res_var)

print("--- Quadratic OLS fit ------------------------------------------------")
print(f"α = {alpha_ols:.4f}  β = {beta_ols:.4f}  γ = {gamma_ols:.4f}")
print(f"Estimated noise: σ² = {res_var:.4f}   σ = {res_sigma:.4f}")

# ────────────────────────────────────────────────────────────────────────────
# 2. Log‑posterior definitions
# ────────────────────────────────────────────────────────────────────────────
def ln_likelihood(a: float, b: float, c: float) -> float:
    """Gaussian log‑likelihood (up to an additive constant)."""
    return -0.5 * np.sum((y_vals - (a + b * x_vals + c * x_vals**2)) ** 2) / res_var


# Uniform (“box”) priors centred on the OLS coefficients
alpha_bounds = (alpha_ols - 10.0, alpha_ols + 10.0)
beta_bounds = (beta_ols - 5.0,  beta_ols + 5.0)
gamma_bounds = (-0.5, 1.5)


def ln_prior(a: float, b: float, c: float) -> float:
    if (alpha_bounds[0] <= a <= alpha_bounds[1]
            and beta_bounds[0] <= b <= beta_bounds[1]
            and gamma_bounds[0] <= c <= gamma_bounds[1]):
        return 0.0          # constant inside the box
    return -np.inf          # zero probability outside


def ln_posterior(a: float, b: float, c: float) -> float:
    lp = ln_prior(a, b, c)
    return lp + ln_likelihood(a, b, c) if np.isfinite(lp) else -np.inf


# ────────────────────────────────────────────────────────────────────────────
# 3. Metropolis MCMC sampler
# ────────────────────────────────────────────────────────────────────────────
prop_width = dict(a=0.08, b=0.03, c=0.005)      # proposal std‑devs

rng = np.random.default_rng(seed=0)
n_iter = 60_000
burn_in = 15_000
thin_step = 10

cur_a, cur_b, cur_c = alpha_ols, beta_ols, gamma_ols
cur_lnpost = ln_posterior(cur_a, cur_b, cur_c)

samples: list[list[float]] = []
n_accept = 0

for _ in range(n_iter):
    cand_a = rng.normal(cur_a, prop_width["a"])
    cand_b = rng.normal(cur_b, prop_width["b"])
    cand_c = rng.normal(cur_c, prop_width["c"])

    cand_lnpost = ln_posterior(cand_a, cand_b, cand_c)
    if rng.random() < np.exp(cand_lnpost - cur_lnpost):
        cur_a, cur_b, cur_c, cur_lnpost = cand_a, cand_b, cand_c, cand_lnpost
        n_accept += 1
    samples.append([cur_a, cur_b, cur_c])

samples_arr = np.asarray(samples)
posterior = samples_arr[burn_in::thin_step]
accept_rate = n_accept / n_iter

print("\n--- Metropolis MCMC stats -------------------------------------------")
print(f"acceptance rate     = {accept_rate:.3f}")
print(f"posterior draw size = {posterior.shape[0]}")

# posterior summary
post_mean = posterior.mean(axis=0)
post_std = posterior.std(axis=0, ddof=1)

lnpost_all = np.array([ln_posterior(a, b, c) for a, b, c in samples_arr])
map_idx = lnpost_all.argmax()
alpha_map, beta_map, gamma_map = samples_arr[map_idx]

print("\nPosterior means ±1σ")
print(f"α = {post_mean[0]:.5f} ± {post_std[0]:.5f}")
print(f"β = {post_mean[1]:.5f} ± {post_std[1]:.5f}")
print(f"γ = {post_mean[2]:.5f} ± {post_std[2]:.5f}")
print("\nMaximum‑a‑posteriori (MAP) parameters")
print(f"α_MAP = {alpha_map:.5f}  β_MAP = {beta_map:.5f}  γ_MAP = {gamma_map:.5f}")

# ────────────────────────────────────────────────────────────────────────────
# 4. Credible‑region helper for 2‑D marginals
# ────────────────────────────────────────────────────────────────────────────
def credibility_levels(two_cols: np.ndarray, nbins: int = 60):
    h, xedges, yedges = np.histogram2d(
        two_cols[:, 0], two_cols[:, 1], bins=nbins, density=True
    )
    h_sorted = np.sort(h.ravel())[::-1]
    cdf = np.cumsum(h_sorted) / h_sorted.sum()
    lvl_68 = h_sorted[np.searchsorted(cdf, 0.683)]
    lvl_95 = h_sorted[np.searchsorted(cdf, 0.954)]
    x_cent = 0.5 * (xedges[:-1] + xedges[1:])
    y_cent = 0.5 * (yedges[:-1] + yedges[1:])
    return x_cent, y_cent, h.T, lvl_68, lvl_95


# ────────────────────────────────────────────────────────────────────────────
# 5. Plotting – each graphic in its own figure
# ────────────────────────────────────────────────────────────────────────────
param_labels = ("α", "β", "γ")

# 1‑D marginals
for idx, lbl in enumerate(param_labels):
    plt.figure()
    plt.hist(
        posterior[:, idx],
        bins=50,
        density=True,
        color="orange",
        edgecolor="black",
        linewidth=1.2,
    )
    plt.axvline(post_mean[idx], ls="--", color="black")
    plt.title(f"Posterior of {lbl}")
    plt.xlabel(lbl)
    plt.ylabel("Density")

# 2‑D joint distributions
pairs = {("α", "β"): (0, 1), ("α", "γ"): (0, 2), ("β", "γ"): (1, 2)}
for (lx, ly), (ix, iy) in pairs.items():
    plt.figure()
    xcent, ycent, hist2d, lvl68, lvl95 = credibility_levels(posterior[:, [ix, iy]])
    plt.contourf(xcent, ycent, hist2d, levels=[lvl95, lvl68, hist2d.max()], alpha=0.6)
    plt.scatter(posterior[:, ix], posterior[:, iy], s=4, alpha=0.4)
    plt.xlabel(lx)
    plt.ylabel(ly)
    plt.title(f"Joint posterior of {lx} and {ly}")

plt.show()
