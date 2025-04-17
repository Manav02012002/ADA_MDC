#!/usr/bin/env python3
# ------------------------------------------------------------
# Bayesian model comparison for MDC‑2
#   linear  : y = a + b x
#   quadratic: y = a + b x + c x^2
# The evidences are computed by direct grid integration.
# ------------------------------------------------------------
import numpy as np
from scipy.special import logsumexp

# ------------------------------------------------------------------
# 0.  Load MDC‑2 data  (adjust the path if needed)
# ------------------------------------------------------------------
data = np.loadtxt("3060309_MDC2.txt")
x, y = data[:, 0], data[:, 1]
N = len(y)

# ------------------------------------------------------------------
# 1.  Pre‑compute sums of powers of x and products with y
#     This lets us write χ² analytically without huge arrays.
# ------------------------------------------------------------------
S0   = N
Sx   = x.sum()
Sx2  = (x**2).sum()
Sx3  = (x**3).sum()
Sx4  = (x**4).sum()
Sy   = y.sum()
Syx  = (y * x).sum()
Syx2 = (y * x**2).sum()
Sy2  = (y**2).sum()       # constant offset; same for every parameter sample

# ------------------------------------------------------------------
# 2.  Helper: ordinary‑least‑squares coefficients and σ² estimate
# ------------------------------------------------------------------
def ols_coeff(design, y):
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    sigma2 = (resid**2).sum() / (len(y) - design.shape[1])
    return beta, sigma2


# ===============  Linear model  (a, b)  ===============
X_lin                = np.vstack([np.ones_like(x), x]).T
(beta_lin, sigma2_l) = ols_coeff(X_lin, y)
a0_l,  b0_l          = beta_lin

# width of the posterior in a, b (from Fisher matrix)
cov_ab               = np.linalg.inv(X_lin.T @ X_lin) * sigma2_l
sigma_a              = np.sqrt(cov_ab[0, 0])
sigma_b              = np.sqrt(cov_ab[1, 1])

# uniform prior box: ±3σ around OLS values
n_a, n_b             = 70, 70        # grid resolution
a_vals               = np.linspace(a0_l - 3*sigma_a, a0_l + 3*sigma_a, n_a)
b_vals               = np.linspace(b0_l - 3*sigma_b, b0_l + 3*sigma_b, n_b)
A, B                 = np.meshgrid(a_vals, b_vals, indexing='ij')

# χ²(a,b) via data moments
chi2_lin = (Sy2
            - 2*A*Sy          - 2*B*Syx
            + A**2 * S0
            + B**2 * Sx2
            + 2*A*B * Sx)

logL_lin = -0.5 * chi2_lin / sigma2_l
dA, dB   = a_vals[1] - a_vals[0], b_vals[1] - b_vals[0]
logZ_lin = logsumexp(logL_lin) + np.log(dA) + np.log(dB)


# ===============  Quadratic model  (a, b, c)  ===============
X_quad                 = np.vstack([np.ones_like(x), x, x**2]).T
(beta_quad, sigma2_q)  = ols_coeff(X_quad, y)
a0_q, b0_q, c0_q       = beta_quad

# same ±3σ in a, b ;  c has a broad prior [-0.5, 1.5]
n_aq, n_bq, n_c        = 50, 50, 50
a_vals_q               = np.linspace(a0_q - 3*sigma_a, a0_q + 3*sigma_a, n_aq)
b_vals_q               = np.linspace(b0_q - 3*sigma_b, b0_q + 3*sigma_b, n_bq)
c_vals                 = np.linspace(-0.5, 1.5, n_c)

Aq, Bq, Cq             = np.meshgrid(a_vals_q, b_vals_q, c_vals, indexing='ij')

chi2_q = (Sy2
          - 2*Aq*Sy          - 2*Bq*Syx          - 2*Cq*Syx2
          + Aq**2 * S0
          + Bq**2 * Sx2
          + Cq**2 * Sx4
          + 2*Aq*Bq * Sx
          + 2*Aq*Cq * Sx2
          + 2*Bq*Cq * Sx3)

logL_q  = -0.5 * chi2_q / sigma2_q
dAq, dBq, dC = (a_vals_q[1] - a_vals_q[0],
                b_vals_q[1] - b_vals_q[0],
                c_vals[1]   - c_vals[0])
logZ_q  = logsumexp(logL_q) + np.log(dAq) + np.log(dBq) + np.log(dC)


# ==================  Bayes factor  ==================
logK = logZ_q - logZ_lin
K    = np.exp(logK)

print("Log‑evidence (linear)    ln Z_L =", logZ_lin)
print("Log‑evidence (quadratic) ln Z_Q =", logZ_q)
print("log Bayes factor ln K    =", logK)
print("Bayes factor      K      =", K)
