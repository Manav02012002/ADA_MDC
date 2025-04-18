#!/usr/bin/env python
# coding: utf-8

# ## Part 1

# ### Step 1.1
# 

# In[26]:


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
stdv_b = np.sqrt(var_b)
stdv_a = np.sqrt(var_a)

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
print("(i)   a (intercept)    = "+str(round(a, 4))+" ± "+ str(round(stdv_a, 4)))
print("(ii)  b (slope)        = "+str(round(b, 4))+" ± "+ str(round(stdv_b, 4)))
print("(iii) Noise variance   = "+str(round(σ2, 4)))
print("(iv)  Covariance(a, b) = "+str(round(cov_ab, 4)))


# ### Step 1.2

# In[27]:


import numpy as np
import matplotlib.pyplot as plt

# Use the variables and calculated parameters from Step 1.1

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

#Values from 1.1 used to construct cov matrix)

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


# ## Section 1.3

# In[28]:


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
    if np.isfinite(p):#if log prior is finite,adds the log likelihood to the log prior
        return log_likelihood(a1, b1) + p
    else:
        return -np.inf#else returns -infinity

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

# Calculating Parameters
a_samples, b_samples = posterior_samples[:, 0], posterior_samples[:, 1]
mean_a, mean_b = np.mean(a_samples), np.mean(b_samples)
stdv_a = np.sqrt(np.sum((a_samples - mean_a) ** 2) / (len(a_samples) - 1))
stdv_b = np.sqrt(np.sum((b_samples - mean_b) ** 2) / (len(b_samples) - 1))
cov_ab = (np.sum((a_samples - mean_a) * (b_samples - mean_b)))/steps-1

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
print("95% CI for a:", ca95)
print("95% CI for b:", cb95)


# Corner-style plots

# 3a. 1D posterior of a
plt.figure()
plt.hist(a_samples, bins=40, density=True, color='skyblue', edgecolor='gray')
plt.axvline(mean_a, linestyle='-', color='red')
plt.title("a Posterior")
plt.xlabel("a")
plt.ylabel("Density")

# 3b. 1D posterior of b
plt.figure()
plt.hist(b_samples, bins=40, density=True, color='orange', edgecolor='gray')
plt.axvline(mean_b, linestyle='-', color='black')
plt.title("b Posterior")
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
plt.contourf(a_mid,b_mid, hist_trans, levels=[level_95, level_68, hist_trans.max()], alpha=0.6, colors=['orange', 'green', 'red'])
plt.scatter(a_samples, b_samples,s=0.3)
plt.xlabel("a")
plt.ylabel("b")
plt.title("Combined Posterior")
plt.show()


# ## Note
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

# In[ ]:





# ## Part 2

# ### 2.1

# In[29]:


import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt("3002691_MDC2.txt")
x = data[:, 0]
y = data[:, 1]
n = len(x)

# OLS Fit for Quadratic Model
X = np.sum(x)
X2 = np.sum(x**2)
X3 = np.sum(x**3)
X4 = np.sum(x**4)
Y = np.sum(y)
XY = np.sum(x * y)
X2Y = np.sum(x**2 * y)

# Build normal equations matrix and vector
A = np.array([[n, X, X2],[X, X2, X3],[X2, X3, X4]])
B = np.array([Y, XY, X2Y])

# Solve for coefficients [a, b, c]
abc = np.linalg.solve(A, B)
a, b, c = abc

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
    log_post1 = log_posterior(an, bn, cn)
    
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
a_samples, b_samples, c_samples = posterior_samples[:, 0], posterior_samples[:, 1],posterior_samples[:, 2]
mean_a, mean_b, mean_c = np.mean(a_samples), np.mean(b_samples), np.mean(c_samples)
stdv_a = np.sqrt(np.sum((a_samples - mean_a) ** 2) / (len(a_samples) - 1))
stdv_b = np.sqrt(np.sum((b_samples - mean_b) ** 2) / (len(b_samples) - 1))
stdv_c = np.sqrt(np.sum((c_samples - mean_c) ** 2) / (len(c_samples) - 1))



#calculating log-posterior for each parameter pair in posterior_samples.
log_probs = []
for a, b, c in posterior_samples:
    log_probs.append(log_posterior(a, b, c))
log_probs = np.array(log_probs)

# Finding the parameter triplet (a, b, c) with the highest log-posterior
max_log_prob = log_probs[0]
best_a, best_b, best_c = posterior_samples[0]
for i in range(1, len(log_probs)):
    if log_probs[i] > max_log_prob:
        max_log_prob = log_probs[i]
        best_a, best_b, best_c = posterior_samples[i]

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
    
    plt.figure()
    levels = sorted([level_95, level_68, hist_trans.max()])
    plt.contourf(x_mid, y_mid, hist_trans, levels=levels, 
             alpha=0.6, colors=['orange', 'green', 'black'])
    plt.scatter(x_samples, y_samples, s=0.3, color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label} Posterior")
    plt.show()


plot_contour(a_samples, b_samples, "a", "b")
plot_contour(a_samples, c_samples, "a", "c")
plot_contour(b_samples, c_samples, "b", "c")


# ### Step 2.2

# In[5]:


import numpy as np
from scipy.special import logsumexp

# Load data
data = np.loadtxt("3002691_MDC2.txt")
x, y = data[:, 0], data[:, 1]
n = len(x)

# Precompute shared moments
X   = np.sum(x)
X2  = np.sum(x**2)
X3  = np.sum(x**3)
X4  = np.sum(x**4)
Y   = np.sum(y)
XY  = np.sum(x * y)
X2Y = np.sum(x**2 * y)
Y2  = np.sum(y**2)

# Manual OLS for Linear Model
def ols_linear(x, y):
    D = n * X2 - X**2
    b = (n * XY - X * Y) / D
    a = (Y * X2 - X * XY) / D
    y_pred = a + b * x
    residual = y - y_pred
    σ2 = np.sum(residual**2) / (n - 2)
    var_b = σ2 * n / D
    var_a = σ2 * X2 / D
    cov_ab = -σ2 * X / D
    stdv_b = np.sqrt(var_b)
    stdv_a = np.sqrt(var_a)
    
    return a, b, σ2, stdv_a, stdv_b

# Manual OLS for Quadratic Model
def ols_quadratic(x, y):
    # Build normal equations matrix and vector
    A = np.array([[n, X, X2], [X, X2, X3], [X2, X3, X4]])
    B = np.array([Y, XY, X2Y])

    # Solve for coefficients [a, b, c]
    abc = np.linalg.solve(A, B)
    a, b, c = abc

    # Predicted values and residuals
    y_fit = a + b * x + c * x**2
    residuals = y - y_fit

    # Compute variance (sigma^2)
    sig2 = np.sum(residuals**2) / (n - 3)
    sig = np.sqrt(sig2)

    return a, b, c, sig2, sig

# Linear model grid setup
al, bl, σ2l, stdv_al, stdv_bl = ols_linear(x, y)
nal, nbl = 70, 70
a_vals = np.linspace(al - 3*stdv_al, al + 3*stdv_al, nal)
b_vals = np.linspace(bl - 3*stdv_bl, bl + 3*stdv_bl, nbl)
Al, Bl = np.meshgrid(a_vals, b_vals, indexing='ij')

# χ² for linear model
chi2_l=(Y2 - 2*Al*Y - 2*Bl*XY+ Al**2 * n+ Bl**2 * X2+ 2*Al*Bl * X)

logL_l = -0.5 * chi2_l / σ2l
dAl, dBl   = a_vals[1] - a_vals[0], b_vals[1] - b_vals[0]
logZ_l = logsumexp(logL_l) + np.log(dAl) + np.log(dBl)

# Quadratic model grid setup
aq, bq, cq, σ2q, σq = ols_quadratic(x, y)  # Unpack the correct number of values
naq, nbq,ncq = 50, 50, 50
a_vals_q = np.linspace(aq - 3*σq, aq + 3*σq, naq)
b_vals_q = np.linspace(bq - 3*σq, bq + 3*σq, nbq)
c_vals = np.linspace(-0.5, 1.5, ncq)
Aq, Bq, Cq = np.meshgrid(a_vals_q, b_vals_q, c_vals, indexing='ij')

# χ² for quadratic model
chi2_q = (Y2- 2*Aq*Y - 2*Bq*XY - 2*Cq*X2Y + Aq**2 * n + Bq**2 * X2 + Cq**2 * X4 + 2*Aq*Bq * X + 2*Aq*Cq * X2 + 2*Bq*Cq * X3)

logL_q = -0.5 * chi2_q / σ2q
dAq, dBq, dCq = a_vals_q[1] - a_vals_q[0], b_vals_q[1] - b_vals_q[0], c_vals[1] - c_vals[0]
logZ_q = logsumexp(logL_q) + np.log(dAq) + np.log(dBq) + np.log(dCq)

# Bayes factor
logK = logZ_q - logZ_l
K = np.exp(logK)

# Results
print("Log-evidence (linear)    ln Z_L =", logZ_l)
print("Log-evidence (quadratic) ln Z_Q =", logZ_q)
print("log Bayes factor ln K    =", logK)
print("Bayes factor      K      =", K)


# In[ ]:




