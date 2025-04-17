import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('3060309_MDC1.txt')
x = data[:, 0]
y = data[:, 1]
n = len(x)

# Compute OLS estimates
x_bar = np.mean(x)
y_bar = np.mean(y)
S_xx = np.sum((x - x_bar) ** 2)
S_xy = np.sum((x - x_bar) * (y - y_bar))
b_hat = S_xy / S_xx
a_hat = y_bar - b_hat * x_bar

# Residuals and variance
residuals = y - (a_hat + b_hat * x)
SSR = np.sum(residuals ** 2)
sigma2_hat = SSR / (n - 2)

# Standard errors and covariance
var_b = sigma2_hat / S_xx
var_a = sigma2_hat * (1 / n + x_bar ** 2 / S_xx)
cov_ab = -x_bar * sigma2_hat / S_xx
se_a = np.sqrt(var_a)
se_b = np.sqrt(var_b)

# Print results
print(f"Number of data points: {n}")
print(f"Estimated slope (b): {b_hat:.5f}")
print(f"Estimated intercept (a): {a_hat:.5f}")
print(f"Estimated noise variance (σ²): {sigma2_hat:.5f}")
print(f"Standard error of a: {se_a:.5f}")
print(f"Standard error of b: {se_b:.5f}")
print(f"Covariance of (a, b): {cov_ab:.5f}")

# Plot with improved aesthetics
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=60, alpha=0.7, label='Data Points')
x_line = np.array([x.min(), x.max()])
y_line = a_hat + b_hat * x_line
plt.plot(x_line, y_line, '--', linewidth=2,
         label=f'Fit: y = {a_hat:.3f} + {b_hat:.3f}x')

plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('Ordinary Least Squares Fit', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.minorticks_on()
plt.tight_layout()
plt.show()
