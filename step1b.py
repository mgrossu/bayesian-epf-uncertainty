"""
Thesis Step 1b: Bayesian Linear Regression on synthetic data
  -- with scipy.stats distribution OBJECTS + model evidence
--------------------------------------------------------------
Same pipeline as thesis_step1_bayesian_mlr.py, with two changes per
Daniel's feedback:

  (A) The posterior over sigma^2 (and beta | sigma^2) is represented as a
      frozen scipy.stats distribution object, not just a one-off .rvs()
      call. This gives access to .pdf(), .cdf(), .ppf(), and .rvs() on
      the same object.

  (B) The Bayesian model evidence (marginal likelihood) is computed
      analytically via a log_model_evidence() function, tested here on
      the single synthetic model (this will later feed BMA weights once
      there are multiple candidate models to compare).

Pipeline:
  1. Generate synthetic data from a known linear model y = X @ beta_true + noise
  2. Fit ordinary MLR (statsmodels) -> point estimates
  3. Fit Bayesian linear regression (closed-form NIG posterior)
  4. Wrap posterior as scipy.stats distribution objects; demo .pdf/.cdf/.ppf/.rvs
  5. Compute log model evidence and sanity-check the number
  6. Plot posterior histograms per coefficient, with true value overlaid
  7. Plot residual histogram as a normality diagnostic
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import invgamma, multivariate_normal
from scipy.special import gammaln

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

# Design matrix: random features (replace with real ENTSO-E / epftoolbox data later)
X = rng.normal(loc=0, scale=1, size=(n_samples, n_features))

# Generate target with known noise
noise = rng.normal(loc=0, scale=sigma_true, size=n_samples)
y = intercept_true + X @ beta_true + noise

# Add intercept column for regression
X_design = sm.add_constant(X)  # shape (n_samples, n_features + 1)
full_beta_true = np.concatenate([[intercept_true], beta_true])
n, p = X_design.shape

# -----------------------------------------------------------------------
# 2. Fit ordinary MLR (baseline point estimate)
# -----------------------------------------------------------------------
ols_model = sm.OLS(y, X_design).fit()
print("=== OLS (MLR) results ===")
print("True coefficients:         ", full_beta_true)
print("OLS estimated coefficients:", np.asarray(ols_model.params))

# -----------------------------------------------------------------------
# 3. Bayesian linear regression: Normal-Inverse-Gamma conjugate posterior
# -----------------------------------------------------------------------
# Model: y | beta, sigma^2 ~ N(X beta, sigma^2 I)
# Prior: beta | sigma^2 ~ N(mu_0, sigma^2 Lambda_0^-1)
#        sigma^2 ~ InvGamma(a_0, b_0)
# Conjugate, so the posterior is also NIG with closed-form updates.

mu_0 = np.zeros(p)            # prior mean for beta
Lambda_0 = np.eye(p) * 0.01   # prior precision (small => weak/uninformative)
a_0 = 1.0                     # prior shape for sigma^2
b_0 = 1.0                     # prior scale for sigma^2

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
print("Posterior mean of sigma^2:", b_n / (a_n - 1))

# -----------------------------------------------------------------------
# 4. Posterior as scipy.stats distribution OBJECTS (not just .rvs())
# -----------------------------------------------------------------------
# Freeze sigma^2's posterior as a distribution object with parameters baked
# in. This single object then supports .pdf(), .cdf(), .ppf(), and .rvs().
sigma2_posterior = invgamma(a_n, scale=b_n)

# beta | sigma^2 is CONDITIONAL on sigma^2, so we build a fresh frozen
# multivariate_normal object for any given sigma^2 value on demand:
def beta_posterior_given_sigma2(sigma2_value):
    """Frozen multivariate_normal object for beta | sigma^2 = sigma2_value."""
    return multivariate_normal(mean=mu_n, cov=sigma2_value * Lambda_n_inv)

print("\n=== scipy.stats distribution object demo (sigma^2 posterior) ===")
print("pdf at sigma^2=4.0:    ", sigma2_posterior.pdf(4.0))
print("cdf at sigma^2=4.0:    ", sigma2_posterior.cdf(4.0))
print("95% credible interval: ", (sigma2_posterior.ppf(0.025), sigma2_posterior.ppf(0.975)))

# Sampling still uses .rvs(), but now called on the frozen object itself
n_draws = 5000
sigma2_samples = sigma2_posterior.rvs(size=n_draws, random_state=rng)
beta_samples = np.array([
    beta_posterior_given_sigma2(s2).rvs(random_state=rng) for s2 in sigma2_samples
])

# Per-coefficient 95% credible intervals from the samples
beta_ci_lower = np.percentile(beta_samples, 2.5, axis=0)
beta_ci_upper = np.percentile(beta_samples, 97.5, axis=0)
labels = ["intercept"] + [f"x{i+1}" for i in range(n_features)]
print("\nPer-coefficient 95% credible intervals:")
for j in range(p):
    print(f"  {labels[j]}: [{beta_ci_lower[j]:.3f}, {beta_ci_upper[j]:.3f}]  "
          f"(posterior mean {mu_n[j]:.3f}, true {full_beta_true[j]:.3f})")

# -----------------------------------------------------------------------
# 5. Bayesian model evidence (marginal likelihood) -- analytical formula
# -----------------------------------------------------------------------
# p(y | X) for the NIG-conjugate linear model, closed form:
#
#   p(y|X) = (2*pi)^(-n/2) * sqrt(|Lambda_0| / |Lambda_n|)
#            * (b_0^a_0 / b_n^a_n) * Gamma(a_n) / Gamma(a_0)
#
# Computed in log-space via gammaln (raw Gamma(a_n) overflows for even
# moderate n, so always work in log-space here).

def log_model_evidence(Lambda_0, Lambda_n, a_0, a_n, b_0, b_n, n):
    _, logdet_0 = np.linalg.slogdet(Lambda_0)
    _, logdet_n = np.linalg.slogdet(Lambda_n)
    return (
        -0.5 * n * np.log(2 * np.pi)
        + 0.5 * (logdet_0 - logdet_n)
        + a_0 * np.log(b_0) - a_n * np.log(b_n)
        + gammaln(a_n) - gammaln(a_0)
    )

log_ev = log_model_evidence(Lambda_0, Lambda_n, a_0, a_n, b_0, b_n, n_samples)
print("\n=== Bayesian model evidence (sanity check) ===")
print(f"log p(y|X) for this model: {log_ev:.4f}")
print("Sanity check: this is a single finite real number (not NaN/inf) -> formula runs correctly.")
print("Not yet meaningful in isolation -- becomes a BMA weight once compared")
print("against other candidate models' log-evidences via a softmax:")
print("  w_k = exp(log_ev_k - max_log_ev) / sum_j exp(log_ev_j - max_log_ev)")

# -----------------------------------------------------------------------
# 6. Plots: posterior histograms with true value overlaid
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, p, figsize=(4 * p, 4))
for j in range(p):
    ax = axes[j]
    ax.hist(beta_samples[:, j], bins=40, alpha=0.7, color="steelblue")
    ax.axvline(full_beta_true[j], color="red", linestyle="--", label="true value")
    ax.axvline(mu_n[j], color="black", linestyle=":", label="posterior mean")
    ax.set_title(f"Posterior of {labels[j]}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("posterior_histograms_step1b.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------
# 7. Residuals diagnostic (using posterior mean beta as point estimate)
# -----------------------------------------------------------------------
y_pred = X_design @ mu_n
residuals = y - y_pred

plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30, color="darkorange", alpha=0.8)
plt.title("Residuals distribution (normality check)")
plt.xlabel("residual")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("residuals_histogram_step1b.png", dpi=150)
plt.show()

print("\nDone. Saved posterior_histograms_step1b.png and residuals_histogram_step1b.png")