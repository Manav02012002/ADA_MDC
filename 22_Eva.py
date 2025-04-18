#!/usr/bin/env python3
import numpy as np                             # import NumPy for numerical operations
from scipy.special import logsumexp            # import logsumexp for stable log‑sum‑exp computations

# 0. Load data
d = np.loadtxt("3060309_MDC2.txt")             # load two‐column text file into array d
x, y = d[:, 0], d[:, 1]                        # unpack first column into x and second into y
N = len(y)                                     # number of data points

# 1. Precompute moments
m0 = N                                         # zeroth moment (count)
m1 = x.sum()                                   # first moment of x
m2 = (x**2).sum()                              # second moment of x
m3 = (x**3).sum()                              # third moment of x
m4 = (x**4).sum()                              # fourth moment of x
s0 = y.sum()                                   # sum of y values
s1 = (y * x).sum()                             # sum of y·x
s2 = (y * x**2).sum()                          # sum of y·x²
s3 = (y**2).sum()                              # sum of y²

# 2. OLS solver via normal equations
def ols(A, b):                                 # define function ols taking design matrix A and observations b
    At = A.T                                  # compute transpose of A
    AtA = At @ A                              # compute AᵀA
    Atb = At @ b                              # compute Aᵀb
    coeff, var = np.linalg.solve(AtA, Atb), None  # solve for coefficients, placeholder for variance
    r = b - A @ coeff                         # compute residual vector
    var = (r**2).sum() / (len(b) - A.shape[1])# estimate residual variance σ² = sum(resid²)/(N–#params)
    return coeff, var                         # return fitted coefficients and variance

# Linear model grid evidence
A1 = np.column_stack((np.ones(N), x))         # build design matrix [1, x]
(c1, v1) = ols(A1, y)                          # perform OLS to get coeffs c1=[a1,b1] and variance v1
a1, b1 = c1                                   # unpack intercept a1 and slope b1
cov1 = np.linalg.inv(A1.T @ A1) * v1          # Fisher‑matrix covariance = (AᵀA)⁻¹·σ²
e1, e2 = np.sqrt(cov1[0, 0]), np.sqrt(cov1[1, 1])  # standard deviations for a and b

# α,β grid ±3σ around OLS estimates
n1a, n1b = 70, 70                             # number of grid points in a and b
ga = np.linspace(a1 - 3*e1, a1 + 3*e1, n1a)    # array of a values spanning ±3σ
gb = np.linspace(b1 - 3*e2, b1 + 3*e2, n1b)    # array of b values spanning ±3σ
A_g, B_g = np.meshgrid(ga, gb, indexing='ij')  # create 2D grid for (a,b)

# χ²(α,β) calculation using moments
chi1 = (                                     
    s3                                       
    - 2*A_g*s0 - 2*B_g*s1                     # cross terms
    + A_g**2 * m0                             # a²·N
    + B_g**2 * m2                             # b²·∑x²
    + 2*A_g*B_g * m1                          # 2ab·∑x
)
L1 = -0.5 * chi1 / v1                         # log‑likelihood grid (up to constant)
dga, dgb = ga[1] - ga[0], gb[1] - gb[0]       # grid spacing in a and b
Z1 = logsumexp(L1) + np.log(dga) + np.log(dgb)# log‑evidence for linear model

# Quadratic model grid evidence
A2 = np.column_stack((np.ones(N), x, x**2))   # build design matrix [1, x, x²]
(c2, v2) = ols(A2, y)                         # OLS on quadratic model → coeffs c2=[a2,b2,g2], var v2
a2, b2, g2 = c2                              # unpack α, β, γ

# α,β,γ grid: reuse σ from linear for α,β and fixed range for γ
n2a, n2b, n2g = 50, 50, 50                    # grid resolution
g2a = np.linspace(a2 - 3*e1, a2 + 3*e1, n2a)   # grid in α
g2b = np.linspace(b2 - 3*e2, b2 + 3*e2, n2b)   # grid in β
g2c = np.linspace(-0.5, 1.5, n2g)              # grid in γ
A_q, B_q, C_q = np.meshgrid(g2a, g2b, g2c, indexing='ij')  # 3D grid arrays

# χ²(α,β,γ) calculation
chi2 = (
    s3                                       
    - 2*A_q*s0 - 2*B_q*s1 - 2*C_q*s2           # cross terms
    + A_q**2 * m0                              # α²·N
    + B_q**2 * m2                              # β²·∑x²
    + C_q**2 * m4                              # γ²·∑x⁴
    + 2*A_q*B_q * m1                           # 2αβ·∑x
    + 2*A_q*C_q * m2                           # 2αγ·∑x²
    + 2*B_q*C_q * m3                           # 2βγ·∑x³
)
L2 = -0.5 * chi2 / v2                        # log‑likelihood for quadratic
dg2a, dg2b, dg2c = g2a[1]-g2a[0], g2b[1]-g2b[0], g2c[1]-g2c[0]  # grid spacings
Z2 = (logsumexp(L2) + np.log(dg2a) + 
      np.log(dg2b) + np.log(dg2c))           # log‑evidence for quadratic model

# Bayes factor
logK = Z2 - Z1                               # difference in log‑evidence
K = np.exp(logK)                             # Bayes factor

# print results
print("Log-evidence (linear)   ln Z_L =", Z1)
print("Log-evidence (quadratic) ln Z_Q =", Z2)
print("log Bayes factor ln K        =", logK)
print("Bayes factor K               =", K)
