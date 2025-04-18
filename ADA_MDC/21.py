import numpy as np                            # import NumPy for numerical operations
import matplotlib.pyplot as plt               # import pyplot for plotting

# custom ol: mimic np.ones_like
def ol(arr):
    return np.ones(arr.shape, dtype=arr.dtype)  # return array of ones matching shape and dtype of arr

# custom least-squares solver: solve (X^T X) β = X^T y
def cl(X, y):
    XT = X.T                                  # compute transpose of design matrix
    ATA = XT @ X                              # form normal matrix X^T X
    ATy = XT @ y                              # form right-hand side X^T y
    return np.linalg.solve(ATA, ATy)          # solve linear system for β

# -------------------------------------------------------------
# 0.  Load the MDC‑2 mock data
# -------------------------------------------------------------
data = np.loadtxt('3060309_MDC2.txt')          # read two-column mock data from text file
x, y = data[:, 0], data[:, 1]                  # unpack into x and y arrays
N = len(y)                                     # number of observations

# -------------------------------------------------------------
# 1. Ordinary‑least‑squares quadratic fit  y = a + b x + c x²
#    (only to obtain an initial point and σ² estimate)
# -------------------------------------------------------------
# build design matrix [1, x, x²] without np.ones_like
X = np.column_stack((ol(x), x, x**2))   
# solve for [a_ols, b_ols, c_ols] using custom_lstsq
a_ols, b_ols, c_ols = cl(X, y)

y_fit = a_ols + b_ols*x + c_ols*x**2            # compute fitted values from quadratic model
sigma2 = np.sum((y - y_fit)**2) / (N - 3)       # estimate residual variance σ² = SSR/(N-3)
sigma  = np.sqrt(sigma2)                       # compute standard deviation of residuals

print(f"OLS parameters: a = {a_ols:.4f}, b = {b_ols:.4f}, c = {c_ols:.4f}")  
                                               # display OLS coefficient estimates
print(f"Estimated noise σ² = {sigma2:.4f}  (σ = {sigma:.4f})")  
                                               # display estimated noise variance and σ

# -------------------------------------------------------------
# 2.  Metropolis MCMC sampling of the posterior
# -------------------------------------------------------------
def log_like(a, b, c):
    return -0.5 * np.sum((y - (a + b*x + c*x**2))**2) / sigma2  
                                               # Gaussian log-likelihood (up to additive constant)

# uniform box priors around the OLS estimates
a_min, a_max = a_ols - 10, a_ols + 10          # prior bounds for a
b_min, b_max = b_ols - 5,  b_ols + 5           # prior bounds for b
c_min, c_max = -0.5,     1.5                  # prior bounds for c

def log_prior(a, b, c):
    if (a_min <= a <= a_max) and (b_min <= b <= b_max) and (c_min <= c <= c_max):
        return 0.0                             # flat (uniform) log-prior inside the box
    return -np.inf                             # log-prior = -∞ outside the box

def log_post(a, b, c):
    lp = log_prior(a, b, c)                    # compute log-prior
    return lp + log_like(a, b, c) if np.isfinite(lp) else -np.inf  
                                               # sum log-prior and log-likelihood if finite

# proposal distribution widths for random-walk Metropolis
prop_a, prop_b, prop_c = 0.08, 0.03, 0.005     # std dev for proposals in (a, b, c)

rng        = np.random.default_rng(0)         # random number generator with fixed seed
steps      = 60000                            # total MCMC iterations
burn, thin = 15000, 10                         # burn-in and thinning parameters

# initialize the chain at the OLS estimates
cur_a, cur_b, cur_c = a_ols, b_ols, c_ols       # starting parameter values
cur_lp = log_post(cur_a, cur_b, cur_c)          # log-posterior at start

chain   = []                                   # list to store samples
accepts = 0                                    # counter for accepted proposals

for _ in range(steps):                         # loop over MCMC iterations
    cand_a = rng.normal(cur_a, prop_a)         # propose new a
    cand_b = rng.normal(cur_b, prop_b)         # propose new b
    cand_c = rng.normal(cur_c, prop_c)         # propose new c
    cand_lp = log_post(cand_a, cand_b, cand_c) # compute log-posterior at candidate
    if rng.random() < np.exp(cand_lp - cur_lp):# accept with Metropolis criterion
        cur_a, cur_b, cur_c, cur_lp = cand_a, cand_b, cand_c, cand_lp  # update current state
        accepts += 1                           # increment accept count
    chain.append([cur_a, cur_b, cur_c])        # record current state in chain

chain     = np.asarray(chain)                  # convert list to NumPy array
acc_rate  = accepts / steps                    # compute acceptance rate
posterior = chain[burn::thin]                  # discard burn-in and thin the chain

print(f"\nAcceptance rate = {acc_rate:.3f}")   # output acceptance rate
print("Posterior sample size =", posterior.shape[0])  # output number of samples

# -------------------------------------------------------------
# 3.  Posterior statistics
# -------------------------------------------------------------
means = posterior.mean(axis=0)                 # compute posterior means for a, b, c
stds  = posterior.std(axis=0, ddof=1)          # compute posterior standard deviations

# find MAP estimate in the full chain
lp_all             = np.array([log_post(a, b, c) for a, b, c in chain])  
                                               # compute log-posterior for each sample
map_idx            = np.argmax(lp_all)          # index of maximum log-posterior
map_a, map_b, map_c = chain[map_idx]            # extract MAP parameter values

print("\nPosterior means and 1‑σ uncertainties")  
print(f"a = {means[0]:.5f} ± {stds[0]:.5f}")     # report mean ± std for a
print(f"b = {means[1]:.5f} ± {stds[1]:.5f}")     # report mean ± std for b
print(f"c = {means[2]:.5f} ± {stds[2]:.5f}")     # report mean ± std for c
print("\nMAP parameters")                      
print(f"a_MAP = {map_a:.5f}, b_MAP = {map_b:.5f}, c_MAP = {map_c:.5f}")  
                                               # display MAP estimates

# -------------------------------------------------------------
# 4.  Credible levels for the 2‑D posterior
# -------------------------------------------------------------
def clvl(samples, nb=60):
    H, xe, ye = np.histogram2d(samples[:,0], samples[:,1],
                               bins=nb, density=True)  
                                               # compute 2D density histogram
    Hsort = np.sort(H.ravel())[::-1]            # sort densities descending
    csum  = np.cumsum(Hsort) / Hsort.sum()      # cumulative distribution
    l68   = Hsort[np.searchsorted(csum, 0.683)] # threshold for 68.3%
    l95   = Hsort[np.searchsorted(csum, 0.954)] # threshold for 95.4%
    xc    = 0.5*(xe[:-1] + xe[1:])              # compute x bin centers
    yc    = 0.5*(ye[:-1] + ye[1:])              # compute y bin centers
    return xc, yc, H.T, l68, l95                # return grid, density, and levels

pairs = {("a","b"):(0,1), ("a","c"):(0,2), ("b","c"):(1,2)}  # parameter pairs for joint plots

# -------------------------------------------------------------
# 5.  Plots  ─ each chart in its own figure
# -------------------------------------------------------------
labels = ["a","b","c"]                         # parameter labels

# 1‑D marginal histograms
for i, lab in enumerate(labels):
    plt.figure()                              # new figure
    plt.hist(
        posterior[:, i],                      # samples for parameter
        bins=50,                              # number of bins
        density=True,                         # normalize to probability density
        color='orange',                       # bar fill color
        edgecolor='black',                    # bar border color
        linewidth=1.2                         # border line width
    )
    plt.axvline(means[i], linestyle="--", color='black')  
                                               # vertical line at posterior mean
    plt.title(f"Posterior of parameter {lab}")# set title
    plt.xlabel(lab)                           # x-label
    plt.ylabel("Density")                     # y-label

# 2‑D joint posterior contours
for (labx, laby), (ix, iy) in pairs.items():
    plt.figure()                              # new figure
    xc, yc, H, l68, l95 = clvl(posterior[:, [ix, iy]])
    plt.contourf(                             # filled contour plot
        xc, yc, H, levels=[l95, l68, H.max()], alpha=0.6
    )
    plt.scatter(posterior[:, ix], posterior[:, iy], s=4, alpha=0.4)  
                                               # overlay raw samples
    plt.xlabel(labx)                          # x-label
    plt.ylabel(laby)                          # y-label
    plt.title(f"Joint posterior of {labx} and {laby}")  # set title

plt.show()                                    # display all plots
