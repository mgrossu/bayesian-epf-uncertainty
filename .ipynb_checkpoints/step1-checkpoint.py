"""
Thesis Step 1: Bayesian Linear Regression on synthetic data
--------------------------------------------------------------
Goal: verify that Bayesian regression (Normal-Inverse-Gamma conjugate
posterior) recovers known "true" coefficients on fake data, and compare
against plain multi-linear regression (MLR).

Pipeline:
  1. Generate synthetic data from a known linear model y = X @ beta_true + noise
  2. Fit ordinary MLR (statsmodels) -> point estimates
  3. Fit Bayesian linear regression (closed-form NIG posterior) -> full
     posterior distribution over beta and sigma^2
  4. Plot posterior histograms per coefficient, with true value overlaid
  5. Plot residual histogram as a normality diagnostic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import invgamma

rng = np.random.default_rng(42)

# -----------------------------------------------------------------------
# 1. Generate synthetic data
# -----------------------------------------------------------------------
n_samples = 200
n_features = 4  # e.g. stand-ins for: past price, load forecast, wind gen, gas price

# "True" parameters we want our models to recover
beta_true = np.array([2.5, -1.0, 0.8, 3.2])   # coefficients for x1..x4
intercept_true = 5.0
sigma_true = 2.0                               # noise std dev

# Design matrix: random features (you can later replace with real ENTSO-E data)
X = rng.normal(loc=0, scale=1, size=(n_samples, n_features))

# Generate target with known noise
noise = rng.normal(loc=0, scale=sigma_true, size=n_samples)
y = intercept_true + X @ beta_true + noise

# Add intercept column for regression
X_design = sm.add_constant(X)  # shape (n_samples, n_features + 1)
full_beta_true = np.concatenate([[intercept_true], beta_true])

# -----------------------------------------------------------------------
# 2. Fit ordinary MLR (baseline point estimate)
# -----------------------------------------------------------------------
ols_model = sm.OLS(y, X_design).fit()
print("=== OLS (MLR) results ===")
print(ols_model.summary())
print("\nTrue coefficients:      ", full_beta_true)
print("OLS estimated coefficients:", np.asarray(ols_model.params))

# -----------------------------------------------------------------------
# 3. Bayesian linear regression: Normal-Inverse-Gamma conjugate posterior
# -----------------------------------------------------------------------
# Model: y | beta, sigma^2 ~ N(X beta, sigma^2 I)
# Prior: beta | sigma^2 ~ N(mu_0, sigma^2 Lambda_0^-1)
#        sigma^2 ~ InvGamma(a_0, b_0)
#
# This is conjugate, so the posterior is also NIG with closed-form updates.

p = X_design.shape[1]  # number of coefficients (incl. intercept)

# --- Weak/uninformative-ish prior ---
mu_0 = np.zeros(p)            # prior mean for beta
Lambda_0 = np.eye(p) * 0.01   # prior precision (small => weak/uninformative)
a_0 = 1.0                     # prior shape for sigma^2
b_0 = 1.0                     # prior scale for sigma^2

# --- Closed-form posterior updates ---
XtX = X_design.T @ X_design
Xty = X_design.T @ y

Lambda_n = Lambda_0 + XtX
Lambda_n_inv = np.linalg.inv(Lambda_n)
mu_n = Lambda_n_inv @ (Lambda_0 @ mu_0 + Xty)

a_n = a_0 + n_samples / 2
b_n = b_0 + 0.5 * (
    y @ y
    + mu_0 @ Lambda_0 @ mu_0
    - mu_n @ Lambda_n @ mu_n
)

print("\n=== Bayesian (NIG) posterior ===")
print("Posterior mean of beta (mu_n):", mu_n)
print("Posterior shape/scale for sigma^2:", a_n, b_n)
print("Posterior mean of sigma^2:", b_n / (a_n - 1))

# -----------------------------------------------------------------------
# 4. Sample from the posterior for visualization
# -----------------------------------------------------------------------
n_draws = 5000

# sigma^2 ~ InvGamma(a_n, scale=b_n)
sigma2_samples = invgamma.rvs(a_n, scale=b_n, size=n_draws, random_state=rng)

# beta | sigma^2 ~ N(mu_n, sigma^2 * Lambda_n_inv)
beta_samples = np.array([
    rng.multivariate_normal(mu_n, s2 * Lambda_n_inv) for s2 in sigma2_samples
])

# -----------------------------------------------------------------------
# 5. Plots: posterior histograms with true value overlaid
# -----------------------------------------------------------------------
labels = ["intercept"] + [f"x{i+1}" for i in range(n_features)]

fig, axes = plt.subplots(1, p, figsize=(4 * p, 4))
for j in range(p):
    ax = axes[j]
    ax.hist(beta_samples[:, j], bins=40, alpha=0.7, color="steelblue")
    ax.axvline(full_beta_true[j], color="red", linestyle="--", label="true value")
    ax.axvline(mu_n[j], color="black", linestyle=":", label="posterior mean")
    ax.set_title(f"Posterior of {labels[j]}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("posterior_histograms.png", dpi=150)
plt.show()

# Residuals diagnostic (using posterior mean beta as point estimate)
y_pred = X_design @ mu_n
residuals = y - y_pred

plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, color="darkorange", alpha=0.8)
plt.title("Residuals distribution (normality check)")
plt.xlabel("residual")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("residuals_histogram.png", dpi=150)
plt.show()

print("\nDone. Saved posterior_histograms.png and residuals_histogram.png")