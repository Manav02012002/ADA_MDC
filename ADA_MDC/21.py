import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0.  Load the MDC‑2 mock data
# -------------------------------------------------------------
data = np.loadtxt('3060309_MDC2.txt')   # citeturn0file1
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
