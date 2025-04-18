import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt("3002691_MDC2.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# OLS Fit for Quadratic Model
Sx = np.sum(x)
Sx2 = np.sum(x**2)
Sx3 = np.sum(x**3)
Sx4 = np.sum(x**4)
Sy = np.sum(y)
Sxy = np.sum(x * y)
Sx2y = np.sum(x**2 * y)

# Build normal equations matrix and vector
A = np.array([[n, Sx, Sx2],[Sx, Sx2, Sx3],[Sx2, Sx3, Sx4]])
B = np.array([Sy, Sxy, Sx2y])

# Solve for coefficients [a, b, c]
abc_ols = np.linalg.solve(A, B)
a, b, c = abc_ols

# Predicted values and noise variance
y_fit = a + b * x + c * x**2
residuals = y - y_fit
sigma2 = np.sum(residuals**2) / (n - 3)
sigma = np.sqrt(sigma2)

print("Manual OLS fit: a = " + str(round(a, 5)) + ", b = " + str(round(b, 5)) + ", c = " + str(round(c, 5)))
print("Estimated noise std dev: " + str(round(sigma, 5)))

def log_likelihood(a1, b1, c1):
    y_model = a1 + b1 * x + c1 * x**2
    return -0.5 * np.sum((y - y_model)**2) / sigma2

# Prior ranges around OLS results
a_min, a_max = a - 10, a + 10
b_min, b_max = b - 5, b + 5
c_min, c_max = -0.5, 1.5

def log_prior(a1, b1, c1):
    if a_min <= a1 <= a_max and b_min <= b1 <= b_max and c_min <= c1 <= c_max:
        return 0.0
    return -np.inf

def log_posterior(a1, b1, c1):
    lp = log_prior(a1, b1, c1)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(a1, b1, c1)

# Proposal settings
std_a1, std_b1, std_c1 = 0.08, 0.03, 0.005
steps = 60000
burn = 15000
thin = 10

# Initialize chain
a_1, b_1, c_1 = a, b, c
log_post = log_posterior(a_1, b_1, c_1)
samples = []
accepted = 0
generator = np.random.seed(1)

for i in range(steps):
    an = np.random.normal(a_1, std_a1)
    bn = np.random.normal(b_1, std_b1)
    cn = np.random.normal(c_1, std_c1)
    log_post1 = log_posterior(a_1, b_1, c_1)
    
    if np.random.random() < np.exp(log_post1 - log_post):#exp turns log-likelihood ratio into a probability(check Notes)
        #random generator generates number bw 0 and 1 .
        #if the probability of difference of initial and new log posterior is greater than that,it is accepted
        a_1, b_1,c_1,log_post = an, bn,cn,log_post1#assigning accepted values as initial for next iteration
        accepted += 1
    samples.append([a_1, b_1,c_1])#storing accepted values of a,b,c

samples = np.array(samples)#updating samples array
acc_rate = accepted / steps#calculating rate of acceptance
#selecting MCMC samples after discarding first burn samples and then taking every thin-th(5th) sample from remaining.
posterior_samples = samples[burn::thin]

# Posterior stats
a_samples, b_samples, c_samples = posterior[:, 0], posterior[:, 1], posterior[:, 2]
mean_a, mean_b, mean_c = np.mean(a_samples), np.mean(b_samples), np.mean(c_samples)
stdv_a = np.sqrt(np.sum((a_samples - mean_a) ** 2) / (len(a_samples) - 1))
stdv_b = np.sqrt(np.sum((b_samples - mean_b) ** 2) / (len(b_samples) - 1))
stdv_c = np.sqrt(np.sum((c_samples - mean_c) ** 2) / (len(c_samples) - 1))



#calculating log-posterior for each parameter pair in posterior_samples.
log_probs = []
for a, b, c in posterior:
    log_probs.append(log_posterior(a, b, c))
log_probs = np.array(log_probs)

# Finding the parameter triplet (a, b, c) with the highest log-posterior
max_log_prob = log_probs[0]
best_a, best_b, best_c = posterior[0]
for i in range(1, len(log_probs)):
    if log_probs[i] > max_log_prob:
        max_log_prob = log_probs[i]
        best_a, best_b, best_c = posterior[i]

# The values of best_a, best_b, best_c are now the MAP estimates


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

#Print results
print("\nQuadratic Model Fit Results (y = a + bx + cx^2):")
print("Acceptance rate:", round(acc_rate, 3))
print("Posterior mean a = " + str(round(mean_a, 5)) + " ± " + str(round(std_a, 5)))
print("Posterior mean b = " + str(round(mean_b, 5)) + " ± " + str(round(std_b, 5)))
print("Posterior mean c = " + str(round(mean_c, 5)) + " ± " + str(round(std_c, 5)))
print("MAP: a = " + str(round(best_a, 5)) + ", b = " + str(round(best_b, 5)) + ", c = " + str(round(best_c, 5)))
print("68% CI for a:", ca68)
print("68% CI for b:", cb68)
print("68% CI for c:", cc68)
print("95% CI for a:", ca95)
print("95% CI for b:", cb95)
print("95% CI for c:", cc95)

#Plotting
plt.figure()
plt.hist(a_samples, bins=40, density=True, color='skyblue', edgecolor='gray')
plt.axvline(mean_a, linestyle='-', color='red')
plt.title("a Posterior")
plt.xlabel("a")
plt.ylabel("Density")
plt.show()

plt.figure()
plt.hist(b_samples, bins=40, density=True, color='orange', edgecolor='gray')
plt.axvline(mean_b, linestyle='-', color='black')
plt.title("b Posterior")
plt.xlabel("b")
plt.ylabel("Density")
plt.show()

plt.figure()
plt.hist(c_samples, bins=40, density=True, color='green', edgecolor='gray')
plt.axvline(mean_c, linestyle='-', color='blue')
plt.title("c Posterior")
plt.xlabel("c")
plt.ylabel("Density")
plt.show()

# 2D contour plots 
def plot_contour(x_samples, y_samples, x_label, y_label):
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
    
    # Plot the contour manually
    plt.figure()
    plt.contourf(x_mid, y_mid, hist_trans, levels=[level_95, level_68, hist_trans.max()], 
                 alpha=0.6, colors=['orange', 'green', 'red'])
    plt.scatter(x_samples, y_samples, s=0.3, color='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label} Posterior")
    plt.show()

plot_contour(a_samples, b_samples, "a", "b")
plot_contour(a_samples, c_samples, "a", "c")
plot_contour(b_samples, c_samples, "b", "c")

