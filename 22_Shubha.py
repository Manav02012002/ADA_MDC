"""
Bayesian comparison of two nested models for the MDC‑2 data set

Model 1 (linear)    :  y = α + β x
Model 2 (quadratic) :  y = α + β x + γ x²

log‑evidences are obtained by brute‑force grid integration over
uniform top‑hat priors centred on the OLS solutions.
"""

from __future__ import annotations

import numpy as np
from  scipy.special import logsumexp


# ────────────────────────────────────────────────────────────────
# 1.  Load data + power sums
# ────────────────────────────────────────────────────────────────
PATH = "3060309_MDC2.txt"
x, y = np.loadtxt(PATH, unpack=True)
n = y.size

S0   = n
Sx   = x.sum()
Sx2  = (x**2).sum()
Sx3  = (x**3).sum()
Sx4  = (x**4).sum()

Sy   = y.sum()
Syx  = (y * x).sum()
Syx2 = (y * x**2).sum()
Sy2  = (y**2).sum()          # common offset in χ²


# ────────────────────────────────────────────────────────────────
# 2.  OLS helper
# ────────────────────────────────────────────────────────────────
def ols(design: np.ndarray, target: np.ndarray):
    """Return β̂ and σ̂² from the normal equations."""
    xt = design.T
    beta = np.linalg.solve(xt @ design, xt @ target)
    resid = target - design @ beta
    sigma2 = resid.dot(resid) / (target.size - design.shape[1])
    return beta, sigma2


# ────────────────────────────────────────────────────────────────
# 3.  Linear model  (α, β)
# ────────────────────────────────────────────────────────────────
X_lin                       = np.column_stack((np.ones_like(x), x))
(beta_L, var_L)             = ols(X_lin, y)
alpha_L, beta_L             = beta_L

cov_L                       = np.linalg.inv(X_lin.T @ X_lin) * var_L
sig_alpha, sig_beta         = np.sqrt(np.diag(cov_L))

m_alpha = np.linspace(alpha_L - 3*sig_alpha, alpha_L + 3*sig_alpha, 70)
m_beta  = np.linspace(beta_L  - 3*sig_beta,  beta_L  + 3*sig_beta,  70)
ALPHA, BETA                 = np.meshgrid(m_alpha, m_beta, indexing='ij')

χ2_L = (Sy2
        - 2*ALPHA*Sy      - 2*BETA*Syx
        + ALPHA**2 * S0   + BETA**2 * Sx2
        + 2*ALPHA*BETA*Sx)

lnL_L    = -0.5 * χ2_L / var_L
dα, dβ   = m_alpha[1] - m_alpha[0], m_beta[1] - m_beta[0]
lnZ_L    = logsumexp(lnL_L) + np.log(dα) + np.log(dβ)


# ────────────────────────────────────────────────────────────────
# 4.  Quadratic model  (α, β, γ)
# ────────────────────────────────────────────────────────────────
X_quad                     = np.column_stack((np.ones_like(x), x, x**2))
(beta_Q, var_Q)            = ols(X_quad, y)
alpha_Q, beta_Q, gamma_Q   = beta_Q

m_alpha_q = np.linspace(alpha_Q - 3*sig_alpha, alpha_Q + 3*sig_alpha, 50)
m_beta_q  = np.linspace(beta_Q  - 3*sig_beta,  beta_Q  + 3*sig_beta,  50)
m_gamma   = np.linspace(-0.5, 1.5, 50)

ALPHA_Q, BETA_Q, GAMMA_Q   = np.meshgrid(m_alpha_q, m_beta_q, m_gamma,
                                         indexing='ij')

χ2_Q = (Sy2
        - 2*ALPHA_Q*Sy          - 2*BETA_Q*Syx          - 2*GAMMA_Q*Syx2
        + ALPHA_Q**2 * S0       + BETA_Q**2 * Sx2       + GAMMA_Q**2 * Sx4
        + 2*ALPHA_Q*BETA_Q * Sx + 2*ALPHA_Q*GAMMA_Q*Sx2 + 2*BETA_Q*GAMMA_Q*Sx3)

lnL_Q     = -0.5 * χ2_Q / var_Q
dαq       = m_alpha_q[1] - m_alpha_q[0]
dβq       = m_beta_q[1]  - m_beta_q[0]
dγ        = m_gamma[1]   - m_gamma[0]
lnZ_Q     = logsumexp(lnL_Q) + np.log(dαq) + np.log(dβq) + np.log(dγ)


# ────────────────────────────────────────────────────────────────
# 5.  Bayes factor + report
# ────────────────────────────────────────────────────────────────
lnK = lnZ_Q - lnZ_L
K   = np.exp(lnK)           # may be extremely small!

print("\n--- Bayesian evidence summary ----------------------------------")
print(f"ln Z_linear     = {lnZ_L:.6f}")
print(f"ln Z_quadratic  = {lnZ_Q:.6f}")

print("\nBayes factor (quadratic / linear)")
print(f"ln K            = {lnK:.6f}")
print(f"K               = {K:.6e}")   # << scientific notation

