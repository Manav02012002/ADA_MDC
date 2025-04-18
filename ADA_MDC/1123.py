import numpy as np                            # import NumPy for numerical operations
import matplotlib.pyplot as plt               # import pyplot for plotting

# ------------------------------------------------------------
# Part 1: Ordinary Least Squares (unchanged)
# ------------------------------------------------------------

# Load the data from the text file into a NumPy array
data = np.loadtxt('3060309_MDC1.txt')         # read two-column data (x,y) from file
# Separate into x (first column) and y (second column)
x = data[:, 0]                                # extract x values
y = data[:, 1]                                # extract y values
# Determine the number of observations
n = len(x)                                    # count data points

# Compute the sample means of x and y
x_bar = np.mean(x)                            # mean of x
y_bar = np.mean(y)                            # mean of y
# Compute the sum of squared deviations of x
S_xx = np.sum((x - x_bar) ** 2)               # Σ(x - x̄)²
# Compute the sum of cross‑deviations between x and y
S_xy = np.sum((x - x_bar) * (y - y_bar))       # Σ(x - x̄)(y - ȳ)
# Calculate the OLS slope estimator b̂
b_hat = S_xy / S_xx                           # slope = S_xy / S_xx
# Calculate the OLS intercept estimator â
a_hat = y_bar - b_hat * x_bar                 # intercept = ȳ - b̂ x̄

# Compute residuals (difference between observed and fitted y)
res = y - (a_hat + b_hat * x)                 # residuals y - (a + bx)
# Sum of squared residuals
SSR = np.sum(res ** 2)                        # Σ residual²
# Estimate of the noise variance σ²
s2h = SSR / (n - 2)                    # SSR/(n-2)

# Variance of the slope estimator
var_b = s2h / S_xx                     # σ²/S_xx
# Variance of the intercept estimator
var_a = s2h * (1 / n + x_bar ** 2 / S_xx)  # σ²(1/n + x̄²/S_xx)
# Covariance between intercept and slope estimators
cov_ab = -x_bar * s2h / S_xx           # -x̄ σ²/S_xx
# Standard error of the intercept
se_a = np.sqrt(var_a)                         # √var_a
# Standard error of the slope
se_b = np.sqrt(var_b)                         # √var_b

# Print the key OLS results
print(f"Number of data points: {n}")           # display n
print(f"Estimated slope (b): {b_hat:.5f}")     # display b̂
print(f"Estimated intercept (a): {a_hat:.5f}") # display â
print(f"Estimated noise variance (σ²): {s2h:.5f}")  # display σ² estimate
print(f"Standard error of a: {se_a:.5f}")      # display SE of intercept
print(f"Standard error of b: {se_b:.5f}")      # display SE of slope
print(f"Covariance of (a, b): {cov_ab:.5f}")   # display covariance

# Create a new figure with specified dimensions
plt.figure(figsize=(10, 6))                    # new figure 10x6 inches
# Scatter plot of the data points in maroon, with size and transparency
plt.scatter(x, y, s=60, alpha=0.7, color='maroon', label='Data Points')  # plot points
# Prepare two x‑values for drawing the fitted line across the range
x_line = np.array([x.min(), x.max()])          # endpoints for line
# Compute corresponding y‑values on the fitted line
y_line = a_hat + b_hat * x_line                # y = a + b x_line
# Plot the fitted line as a dashed chrome‑yellow line
plt.plot(
    x_line, y_line, '--', linewidth=2, color='#FFA700',
    label=f'Fit: y = {a_hat:.3f} + {b_hat:.3f}x'
)  # plot fit line

# Label the x-axis
plt.xlabel('x', fontsize=14)                   # set x label
# Label the y-axis
plt.ylabel('y', fontsize=14)                   # set y label
# Add a title to the plot
plt.title('Ordinary Least Squares Fit', fontsize=16)  # set title
# Show the legend in the plot
plt.legend(fontsize=12)                        # display legend
# Add a subtle dashed grid with major and minor ticks
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # enable grid
plt.minorticks_on()                            # enable minor ticks
plt.tight_layout()                             # adjust spacing
plt.show()                                     # render the plot


def ol(arr):                                
    # return an array of ones with the same shape and data type as arr
    return np.ones(arr.shape, dtype=arr.dtype)  # mimic np.ones_like

# custom ui for arbitrary-dimensional arrays
def ui(flat_index, shape):
    # convert a flat index into a tuple of coordinate indices for an array of given shape
    idxs = []                                    # list to collect indices
    for dim in reversed(shape):                  # iterate dims last-first
        idxs.append(flat_index % dim)            # compute index in this dim
        flat_index //= dim                       # reduce flat_index
    return tuple(reversed(idxs))                 # reverse to normal order

# 1. Load data
data = np.loadtxt('3060309_MDC1.txt')           # read data again
x = data[:, 0]                                  # extract x
y = data[:, 1]                                  # extract y
sigma = ol(y)                            # constant errors array
N = len(x)                                      # count points

# 2. Manual least‑squares estimates for a0, b0
x_mean = np.mean(x)                             # mean of x
y_mean = np.mean(y)                             # mean of y
Sxx = np.sum((x - x_mean) ** 2)                 # Σ(x - x̄)²
Sxy = np.sum((x - x_mean) * (y - y_mean))       # Σ(x - x̄)(y - ȳ)
a0 = Sxy / Sxx                                  # slope a₀
b0 = y_mean - a0 * x_mean                       # intercept b₀

# residual variance & parameter uncertainties
resid = y - (a0 * x + b0)                       # residuals
chi2_ls = np.sum((resid / sigma) ** 2)          # χ² of LS
s2r = chi2_ls / (N - 2)                # variance estimate
sigma_a = np.sqrt(s2r / Sxx)           # SE of slope
sigma_b = np.sqrt(s2r * (1.0/N + x_mean**2 / Sxx))  
                                                # SE of intercept

# 3. Build (a,b) grid ±3σ around LS solution
n_a, n_b = 200, 200                             # grid sizes
a_vals = np.linspace(a0 - 3*sigma_a, a0 + 3*sigma_a, n_a)  # a grid
b_vals = np.linspace(b0 - 3*sigma_b, b0 + 3*sigma_b, n_b)  # b grid
A, B = np.meshgrid(a_vals, b_vals)              # meshgrid arrays

# 4. Compute χ² on the grid
# broadcast x and y across grid dims
res = y[:, None, None] - (A[None, :, :] * x[:, None, None] + B[None, :, :])  # residuals grid
Chi2 = np.sum((res / sigma[:, None, None]) ** 2, axis=0)  # sum over data pts

# 5. Find best‑fit and Δχ² using custom ui
flat_idx = np.argmin(Chi2)                      # index of min χ²
i_min, j_min = ui(flat_idx, Chi2.shape)  # row,col of min
a_best = A[i_min, j_min]                        # best-fit a
b_best = B[i_min, j_min]                        # best-fit b
chi2_min = Chi2[i_min, j_min]                   # minimal χ²
delta_chi2 = Chi2 - chi2_min                    # Δχ² field

# 6. Compute credible‑region bounds
levels = [2.30, 6.17, 11.8]                     # Δχ² thresholds
region_ranges = []                              # list of bounds
for lvl in levels:
    mask = delta_chi2 <= lvl                    # mask region
    a_min, a_max = A[mask].min(), A[mask].max()  # a bounds
    b_min, b_max = B[mask].min(), B[mask].max()  # b bounds
    region_ranges.append((a_min, a_max, b_min, b_max))  # store

# 7. Print results
print("Best‑fit parameters:")                    # header
print(f"  a = {a_best:.5f}")                     # print a_best
print(f"  b = {b_best:.5f}")                     # print b_best
print(f"  χ²_min = {chi2_min:.3f}\n")            # print χ²_min

for pct, (a_min, a_max, b_min, b_max) in zip([68.3, 95.4, 99.73], region_ranges):
    print(f"{pct:.1f}% credible region:")       # label
    print(f"  a ∈ [{a_min:.5f}, {a_max:.5f}]")    # a interval
    print(f"  b ∈ [{b_min:.5f}, {b_max:.5f}]\n")  # b interval

# 8. Plot Δχ² contours
plt.figure(figsize=(8, 6))                      # new figure
cs = plt.contour(                               # contour lines
    A, B, delta_chi2,
    levels=levels,
    colors=['maroon', 'orange', 'olive'],
    linewidths=[1.5, 1.5, 1.5]
)
plt.plot(a_best, b_best, 'k+', label='Best‑fit') # mark best-fit

plt.xlabel('a')                                 # x-label
plt.ylabel('b')                                 # y-label
plt.title(r'$\Delta\chi^2$ Contours for $(a,b)$')  # title
plt.clabel(cs, inline=True, fmt={2.30:'68.3%', 6.17:'95.4%', 11.8:'99.73%'})  # annotations

# 9. Increase axis range by 5% margin
x_min, x_max = a_vals.min(), a_vals.max()       # a limits
y_min, y_max = b_vals.min(), b_vals.max()       # b limits
x_margin = 0.05 * (x_max - x_min)                # margin a
y_margin = 0.05 * (y_max - y_min)                # margin b
plt.xlim(x_min - x_margin, x_max + x_margin)     # set xlim
plt.ylim(y_min - y_margin, y_max + y_margin)     # set ylim

plt.legend()                                    # show legend
plt.show()                                      # render contour plot

# ------------------------------------------------------------
# Part 2: Metropolis MCMC using those OLS parameters
# ------------------------------------------------------------

# Log‑likelihood function using σ² from the OLS stage
def log_likelihood(a, b):
    return -0.5 * np.sum((y - (a + b * x))**2) / s2h  # return log L

# Define uniform priors ±5 around a_hat and ±3 around b_hat
a_min, a_max = a_hat - 5.0, a_hat + 5.0          # a prior bounds
b_min, b_max = b_hat - 3.0, b_hat + 3.0          # b prior bounds

def log_prior(a, b):
    if a_min <= a <= a_max and b_min <= b <= b_max:
        return 0.0                                # flat prior inside
    return -np.inf                                # zero outside

def log_posterior(a, b):
    lp = log_prior(a, b)                         # compute lp
    return lp + log_likelihood(a, b) if np.isfinite(lp) else -np.inf

# MCMC hyperparameters
prop_a, prop_b = 0.05, 0.02                     # proposal std devs
steps, burn, thin = 40000, 10000, 10            # steps, burn-in, thinning

# Initialize chain at the OLS estimates
current_a, current_b = a_hat, b_hat             # start values
current_lp = log_posterior(current_a, current_b)  # initial log-posterior

chain = []                                      # to store samples
accept = 0                                      # acceptance counter
rng = np.random.default_rng(0)                  # seeded RNG

for _ in range(steps):                          # MCMC loop
    cand_a = rng.normal(current_a, prop_a)       # propose new a
    cand_b = rng.normal(current_b, prop_b)       # propose new b
    cand_lp = log_posterior(cand_a, cand_b)      # compute new log-posterior
    if rng.random() < np.exp(cand_lp - current_lp):
        current_a, current_b, current_lp = cand_a, cand_b, cand_lp  # accept
        accept += 1
    chain.append([current_a, current_b])         # record

chain = np.asarray(chain)                       # to array
posterior = chain[burn::thin]                   # apply burn-in and thinning

# Summary statistics of the posterior
mean_a, mean_b = posterior.mean(axis=0)         # posterior means
std_a, std_b = posterior.std(axis=0, ddof=1)    # posterior std devs

print("\nMCMC results:")                         # header
print("Acceptance rate:", accept / steps)        # print rate
print(f"Posterior mean a = {mean_a:.5f} ± {std_a:.5f}")  # print a ±
print(f"Posterior mean b = {mean_b:.5f} ± {std_b:.5f}")  # print b ±

# Corner‑style plots of the posterior
plt.figure()                                    # new fig for a
plt.hist(posterior[:, 0], bins=40, density=True,
         color='orange', edgecolor='black', linewidth=1.2)  # hist of a
plt.axvline(mean_a, linestyle='--', color='black')         # mean line
plt.title("Posterior of parameter a")          # title
plt.xlabel("a")                                # x-label
plt.ylabel("Probability density")              # y-label

plt.figure()                                    # new fig for b
plt.hist(posterior[:, 1], bins=40, density=True,
         color='orange', edgecolor='black', linewidth=1.2)  # hist of b
plt.axvline(mean_b, linestyle='--', color='black')         # mean line
plt.title("Posterior of parameter b")          # title
plt.xlabel("b")                                # x-label
plt.ylabel("Probability density")              # y-label

H, xedges, yedges = np.histogram2d(             # 2D hist of (a,b)
    posterior[:, 0], posterior[:, 1], bins=50, density=True
)
xcenters = 0.5 * (xedges[:-1] + xedges[1:])      # x bin centers
ycenters = 0.5 * (yedges[:-1] + yedges[1:])      # y bin centers
Hsort = np.sort(H.flatten())[::-1]               # sorted densities
Hcum = np.cumsum(Hsort) / Hsort.sum()            # cumulative
level68 = Hsort[np.searchsorted(Hcum, 0.683)]    # 68% level
level95 = Hsort[np.searchsorted(Hcum, 0.954)]    # 95% level

plt.figure()                                    # new fig for contour
plt.contourf(xcenters, ycenters, H.T,
             levels=[level95, level68, H.max()], alpha=0.6)  # filled contours
plt.scatter(posterior[:, 0], posterior[:, 1], s=4, alpha=0.4)  # scatter samples
plt.xlabel("a")                                # x-label
plt.ylabel("b")                                # y-label
plt.title("Joint posterior of a and b")        # title

plt.show()                                     # render final plot
