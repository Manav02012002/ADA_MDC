import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load data
# -----------------------------
data = np.loadtxt('3060309_MDC1.txt')
x, y = data[:, 0], data[:, 1]
N = len(y)

# -----------------------------
# 2. Ordinary‑Least‑Squares (for σ²)
# -----------------------------
A = np.vstack([np.ones_like(x), x]).T
a_OLS, b_OLS = np.linalg.lstsq(A, y, rcond=None)[0]
sigma2 = np.sum((y - (a_OLS + b_OLS * x)) ** 2) / (N - 2)
sigma = np.sqrt(sigma2)

# -----------------------------
# 3. Metropolis MCMC
# -----------------------------
def log_likelihood(a, b):
    return -0.5 * np.sum((y - (a + b * x)) ** 2) / sigma2

# uniform (improper) prior in a generous box around the OLS values
a_min, a_max = a_OLS - 5.0, a_OLS + 5.0
b_min, b_max = b_OLS - 3.0, b_OLS + 3.0

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

current_a, current_b = a_OLS, b_OLS
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
