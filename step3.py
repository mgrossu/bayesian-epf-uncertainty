"""
Thesis Step 3: Posterior predictive distribution + evidence-weighted mixture
  -- rolling-origin forecast study on SYNTHETIC data
--------------------------------------------------------------------------
 
  1. Posterior predictive distribution (PPD) for ONE model -- a Student-t,
     derived exactly as in his PDF Section 5.4.
  2. A rolling-origin forecast study: train on data up to time t, forecast
     t+1, roll forward, repeat. At each step, extract a prediction interval
     from the PPD (e.g. 80%) and record whether the true y fell inside it.
     Over many steps, the EMPIRICAL coverage should be close to the NOMINAL
     level (80%) if the model's uncertainty is well-calibrated.
  3. An evidence-weighted MIXTURE of several candidate models' PPDs:
         p(y*|x*) = sum_m w_m * p(y*|x*, M_m)
     where w_m are softmax-normalized log model evidences.
  4. Since a mixture of Student-t's has no closed-form quantile, the
     interval bounds are found by NUMERICALLY SOLVING the mixture CDF
     (matches "quantiles numerical solve" in your notes).
  5. Same rolling-origin coverage check, now for the mixture.
 
Three candidate models are used, differing by which features they include,
so you can see the evidence-weighting prefer the model that actually
matches the true data-generating process.
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from scipy.special import gammaln
from scipy.optimize import brentq
 
rng = np.random.default_rng(42)
 
# =============================================================================
# Part 0: Generate synthetic TIME-ORDERED data
# =============================================================================
# True model uses only features x1 and x2 (x3 is a decoy, irrelevant feature
# included only so the "wrong" candidate models have something extra to
# overfit on -- this lets us see evidence correctly favor the true model).
 
n_total = 300
beta_true = np.array([2.0, -1.5])          # true coefficients for x1, x2 only
intercept_true = 1.0
sigma_true = 1.5
 
x1 = rng.normal(size=n_total)
x2 = rng.normal(size=n_total)
x3 = rng.normal(size=n_total)              # irrelevant decoy feature
noise = rng.normal(scale=sigma_true, size=n_total)
y = intercept_true + beta_true[0]*x1 + beta_true[1]*x2 + noise
 
# Three candidate models, differing by which features they include
def build_design(model_features):
    """model_features: list of arrays to include as columns (excl. intercept)."""
    cols = [np.ones(n_total)] + model_features
    return np.column_stack(cols)
 
X_M1 = build_design([x1])            # M1: underfit -- missing x2
X_M2 = build_design([x1, x2])        # M2: TRUE model
X_M3 = build_design([x1, x2, x3])    # M3: overfit -- extra irrelevant feature
 
models = {
    "M1 (x1 only)": X_M1,
    "M2 (x1, x2) -- TRUE": X_M2,
    "M3 (x1, x2, x3)": X_M3,
}
 
# =============================================================================
# Part 1: Reusable NIG posterior fit + evidence (from step1b/step2)
# =============================================================================
 
def fit_nig_posterior(X, y, mu_0, Lambda_0, a_0, b_0):
    """Fit NIG conjugate posterior. Returns mu_n, Lambda_n, Lambda_n_inv, a_n, b_n, log_evidence."""
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
 
    Lambda_n = Lambda_0 + XtX
    Lambda_n_inv = np.linalg.inv(Lambda_n)
    mu_n = Lambda_n_inv @ (Lambda_0 @ mu_0 + Xty)
 
    a_n = a_0 + n / 2
    b_n = b_0 + 0.5 * (y @ y + mu_0 @ Lambda_0 @ mu_0 - mu_n @ Lambda_n @ mu_n)
 
    _, logdet_0 = np.linalg.slogdet(Lambda_0)
    _, logdet_n = np.linalg.slogdet(Lambda_n)
    log_ev = (
        -0.5 * n * np.log(2 * np.pi)
        + 0.5 * (logdet_0 - logdet_n)
        + a_0 * np.log(b_0) - a_n * np.log(b_n)
        + gammaln(a_n) - gammaln(a_0)
    )
    return mu_n, Lambda_n, Lambda_n_inv, a_n, b_n, log_ev
 
 
# =============================================================================
# Part 2: Posterior predictive distribution (Student-t) -- Daniel's PDF 5.4
# =============================================================================
# y* | x*, D, M ~ Student-t_{nu_n}(mu_star, s2_star)
#   nu_n    = 2*a_n
#   mu_star = x*^T mu_n
#   s2_star = (b_n/a_n) * (1 + x*^T Lambda_n_inv x*)
#
# scipy.stats.t is the STANDARD t distribution (mean 0, scale 1); passing
# loc and scale gives the location-scale version matching the formula above.
 
def posterior_predictive(x_star, mu_n, Lambda_n_inv, a_n, b_n):
    """Return a frozen scipy.stats.t object for y* | x*."""
    nu_n = 2 * a_n
    mu_star = x_star @ mu_n
    s2_star = (b_n / a_n) * (1 + x_star @ Lambda_n_inv @ x_star)
    return student_t(df=nu_n, loc=mu_star, scale=np.sqrt(s2_star))
 
 
# =============================================================================
# Part 3: Rolling-origin forecast study -- SINGLE model (M2, the true one)
# =============================================================================
# "Training 1..n -> new x -> rolls forward (repeat)" from your notes.
# Expanding window: train on all data up to t, forecast t+1, slide forward.
 
def rolling_origin_single_model(X, y, start_window, interval_level=0.80,
                                 mu_0=None, Lambda_0=None, a_0=1.0, b_0=1.0):
    n, p = X.shape
    if mu_0 is None:
        mu_0 = np.zeros(p)
    if Lambda_0 is None:
        Lambda_0 = np.eye(p) * 0.01
 
    alpha = 1 - interval_level
    results = []  # list of dicts: t, y_true, mu_star, lower, upper, inside
 
    for t in range(start_window, n - 1):
        X_train, y_train = X[:t+1], y[:t+1]
        x_star, y_true = X[t+1], y[t+1]
 
        mu_n, Lambda_n, Lambda_n_inv, a_n, b_n, log_ev = fit_nig_posterior(
            X_train, y_train, mu_0, Lambda_0, a_0, b_0
        )
        ppd = posterior_predictive(x_star, mu_n, Lambda_n_inv, a_n, b_n)
 
        lower = ppd.ppf(alpha / 2)
        upper = ppd.ppf(1 - alpha / 2)
        inside = lower <= y_true <= upper
 
        results.append(dict(t=t+1, y_true=y_true, mu_star=ppd.mean(),
                             lower=lower, upper=upper, inside=inside))
    return results
 
 
print("=== Rolling-origin forecast: single model (M2, true model) ===")
single_results = rolling_origin_single_model(X_M2, y, start_window=50, interval_level=0.80)
empirical_coverage = np.mean([r["inside"] for r in single_results])
print(f"Nominal interval level: 80%")
print(f"Empirical coverage over {len(single_results)} forecasts: {empirical_coverage:.1%}")
print("(Should be close to 80% if the model's uncertainty is well-calibrated)")
 
 
# =============================================================================
# Part 4: Rolling-origin forecast study -- EVIDENCE-WEIGHTED MIXTURE
# =============================================================================
 
def mixture_cdf(y_val, components, weights):
    """CDF of the mixture at y_val: sum_m w_m * F_m(y_val)."""
    return sum(w * comp.cdf(y_val) for w, comp in zip(weights, components))
 
def mixture_quantile(q, components, weights, lo=-100, hi=100):
    """Numerically solve mixture_cdf(y) = q for y, via bisection (brentq)."""
    f = lambda y_val: mixture_cdf(y_val, components, weights) - q
    return brentq(f, lo, hi)
 
def softmax_weights(log_evidences):
    log_evidences = np.array(log_evidences)
    shifted = log_evidences - np.max(log_evidences)   # numerically stable
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()
 
 
def rolling_origin_mixture(models_dict, y, start_window, interval_level=0.80,
                            a_0=1.0, b_0=1.0):
    n = len(y)
    alpha = 1 - interval_level
    results = []
    weight_history = []  # track evidence weights over time for plotting
 
    for t in range(start_window, n - 1):
        components = []
        log_evs = []
        mu_stars = []
 
        for name, X in models_dict.items():
            p = X.shape[1]
            mu_0 = np.zeros(p)
            Lambda_0 = np.eye(p) * 0.01
            X_train, y_train = X[:t+1], y[:t+1]
            x_star = X[t+1]
 
            mu_n, Lambda_n, Lambda_n_inv, a_n, b_n, log_ev = fit_nig_posterior(
                X_train, y_train, mu_0, Lambda_0, a_0, b_0
            )
            ppd = posterior_predictive(x_star, mu_n, Lambda_n_inv, a_n, b_n)
            components.append(ppd)
            log_evs.append(log_ev)
            mu_stars.append(ppd.mean())
 
        weights = softmax_weights(log_evs)
        weight_history.append(weights)
 
        y_true = y[t+1]
        lower = mixture_quantile(alpha/2, components, weights)
        upper = mixture_quantile(1 - alpha/2, components, weights)
        inside = lower <= y_true <= upper
        mix_mean = sum(w*m for w, m in zip(weights, mu_stars))
 
        results.append(dict(t=t+1, y_true=y_true, mu_star=mix_mean,
                             lower=lower, upper=upper, inside=inside,
                             weights=weights))
    return results, np.array(weight_history)
 
 
print("\n=== Rolling-origin forecast: evidence-weighted MIXTURE (M1, M2, M3) ===")
mixture_results, weight_history = rolling_origin_mixture(models, y, start_window=50, interval_level=0.80)
mix_coverage = np.mean([r["inside"] for r in mixture_results])
print(f"Nominal interval level: 80%")
print(f"Empirical coverage over {len(mixture_results)} forecasts: {mix_coverage:.1%}")
 
mean_weights = weight_history.mean(axis=0)
print("\nAverage evidence weight per model (should favor M2, the true model):")
for name, w in zip(models.keys(), mean_weights):
    print(f"  {name}: {w:.3f}")
 
 
# =============================================================================
# Part 5: Plots
# =============================================================================
 
# Plot 1: forecasts with prediction intervals for the single model (matches
# your hand-drawn sketch: y_true line, PPD shaded, coverage check)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
 
ts = [r["t"] for r in single_results]
y_trues = [r["y_true"] for r in single_results]
lowers = [r["lower"] for r in single_results]
uppers = [r["upper"] for r in single_results]
mus = [r["mu_star"] for r in single_results]
 
axes[0].fill_between(ts, lowers, uppers, alpha=0.3, color="steelblue", label="80% prediction interval")
axes[0].plot(ts, mus, color="steelblue", label="predictive mean")
axes[0].plot(ts, y_trues, "o", color="black", markersize=2, label="y_true")
axes[0].set_title(f"Single model (M2) rolling-origin forecasts — empirical coverage {empirical_coverage:.1%}")
axes[0].legend(fontsize=8)
 
ts_m = [r["t"] for r in mixture_results]
y_trues_m = [r["y_true"] for r in mixture_results]
lowers_m = [r["lower"] for r in mixture_results]
uppers_m = [r["upper"] for r in mixture_results]
mus_m = [r["mu_star"] for r in mixture_results]
 
axes[1].fill_between(ts_m, lowers_m, uppers_m, alpha=0.3, color="darkorange", label="80% mixture prediction interval")
axes[1].plot(ts_m, mus_m, color="darkorange", label="mixture predictive mean")
axes[1].plot(ts_m, y_trues_m, "o", color="black", markersize=2, label="y_true")
axes[1].set_title(f"Evidence-weighted mixture (M1+M2+M3) — empirical coverage {mix_coverage:.1%}")
axes[1].set_xlabel("time step")
axes[1].legend(fontsize=8)
 
plt.tight_layout()
plt.savefig("rolling_forecasts_coverage.png", dpi=150)
plt.show()
 
# Plot 2: evidence weights over time
plt.figure(figsize=(10, 4))
for i, name in enumerate(models.keys()):
    plt.plot(ts_m, weight_history[:, i], label=name)
plt.title("Evidence-based model weights over time")
plt.xlabel("time step")
plt.ylabel("weight")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("evidence_weights_over_time.png", dpi=150)
plt.show()
 
print("\nDone. Saved rolling_forecasts_coverage.png and evidence_weights_over_time.png")