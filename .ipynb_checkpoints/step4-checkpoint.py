"""
Step 4: non-stationary synthetic data, window comparison, PPD / posterior plots
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, norm, invgamma
from scipy.special import gammaln
from scipy.optimize import brentq

SHOW = False            # set True to open figure windows
os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(7)

# =============================================================================
# 0. Synthetic non-stationary data from a mixture of three components
# =============================================================================
n_total = 700
t_idx = np.arange(n_total)
X_feat = rng.normal(size=(n_total, 3))          # three independent features

# Each component depends on a different feature and has its own noise level.
components = [
    dict(name="M1", feat=0, intercept=1.0,  beta=2.0,  sigma=0.5),   # low noise
    dict(name="M2", feat=1, intercept=-1.0, beta=-2.5, sigma=2.0),   # high noise
    dict(name="M3", feat=2, intercept=0.5,  beta=1.5,  sigma=1.0),   # medium noise
]
K = len(components)
colors = ["tab:blue", "tab:red", "tab:green"]

# Smoothly shifting generating weights: each model peaks in a different period.
centers, width = np.array([120, 350, 580]), 90.0
raw = np.exp(-0.5 * ((t_idx[:, None] - centers[None, :]) / width) ** 2) + 0.03
W_true = raw / raw.sum(axis=1, keepdims=True)

z = np.array([rng.choice(K, p=W_true[t]) for t in t_idx])   # active component
y = np.empty(n_total)
for t in t_idx:
    c = components[z[t]]
    y[t] = c["intercept"] + c["beta"] * X_feat[t, c["feat"]] + c["sigma"] * rng.normal()

# Candidate models to FIT: same three structures (intercept + one feature)
X_M = [np.column_stack([np.ones(n_total), X_feat[:, m]]) for m in range(K)]

# =============================================================================
# 1. Conjugate NIG fit, evidence, PPD   (solve instead of inv)
# =============================================================================
MU0_SCALE, A0, B0 = 0.01, 1.0, 1.0       # weak prior: Lambda_0 = 0.01 I


def fit_nig(X, yv):
    n, p = X.shape
    Lambda_0 = np.eye(p) * MU0_SCALE
    mu_0 = np.zeros(p)
    Lambda_n = Lambda_0 + X.T @ X
    rhs = Lambda_0 @ mu_0 + X.T @ yv
    mu_n = np.linalg.solve(Lambda_n, rhs)
    a_n = A0 + n / 2
    # mu_n^T Lambda_n mu_n = mu_n^T rhs  (no extra matrix product needed)
    b_n = B0 + 0.5 * (yv @ yv + mu_0 @ Lambda_0 @ mu_0 - mu_n @ rhs)
    _, ld0 = np.linalg.slogdet(Lambda_0)
    _, ldn = np.linalg.slogdet(Lambda_n)
    log_ev = (-0.5 * n * np.log(2 * np.pi) + 0.5 * (ld0 - ldn)
              + A0 * np.log(B0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(A0))
    return dict(mu_n=mu_n, Lambda_n=Lambda_n, a_n=a_n, b_n=b_n, log_ev=log_ev)


def ppd(fit, x_star):
    """Frozen Student-t for y* | x*, D, M."""
    v = np.linalg.solve(fit["Lambda_n"], x_star)
    scale2 = fit["b_n"] / fit["a_n"] * (1.0 + x_star @ v)
    return student_t(df=2 * fit["a_n"], loc=x_star @ fit["mu_n"], scale=np.sqrt(scale2))


def softmax(logv):
    e = np.exp(logv - np.max(logv))
    return e / e.sum()


def mixture_cdf(yv, comps, w):
    return sum(wi * c.cdf(yv) for wi, c in zip(w, comps))


def mixture_ppf(q, comps, w):
    """Quantile of the mixture; brentq bracket from the component quantiles."""
    qs = [c.ppf(q) for c in comps]
    lo, hi = min(qs), max(qs)            # F(lo) <= q <= F(hi) by construction
    if hi - lo < 1e-12:
        return lo
    f = lambda v: mixture_cdf(v, comps, w) - q
    f_lo, f_hi = f(lo), f(hi)
    # When one model carries ~all the weight, an endpoint can sit at f ~ 0 and
    # rounding may break the sign change; the endpoint is then the answer.
    if f_lo >= 0:
        return lo
    if f_hi <= 0:
        return hi
    return brentq(f, lo, hi, xtol=1e-9)


def fit_all(s, window):
    """Fit all models using data strictly before target index s."""
    lo = 0 if window is None else max(0, s - window)
    return [fit_nig(X_M[m][lo:s], y[lo:s]) for m in range(K)]


# =============================================================================
# 2. Rolling-origin forecasting study for a given window setting
# =============================================================================
START = 100


def run_study(window, level=0.80):
    alpha = 1 - level
    out = dict(s=[], y=[], lo=[], hi=[], logscore=[], W=[],
               single_inside=[[] for _ in range(K)], single_w=[[] for _ in range(K)],
               single_ls=[[] for _ in range(K)], mean=[])
    for s in range(START, n_total):
        fits = fit_all(s, window)
        comps = [ppd(fits[m], X_M[m][s]) for m in range(K)]
        w = softmax(np.array([f["log_ev"] for f in fits]))
        lo, hi = mixture_ppf(alpha / 2, comps, w), mixture_ppf(1 - alpha / 2, comps, w)
        out["s"].append(s); out["y"].append(y[s]); out["lo"].append(lo); out["hi"].append(hi)
        out["W"].append(w)
        out["mean"].append(sum(wi * c.mean() for wi, c in zip(w, comps)))
        out["logscore"].append(np.log(sum(wi * c.pdf(y[s]) for wi, c in zip(w, comps))))
        for m in range(K):
            l, h = comps[m].ppf(alpha / 2), comps[m].ppf(1 - alpha / 2)
            out["single_inside"][m].append(l <= y[s] <= h)
            out["single_w"][m].append(h - l)
            out["single_ls"][m].append(comps[m].logpdf(y[s]))
    for k in ("s", "y", "lo", "hi", "logscore", "W", "mean"):
        out[k] = np.array(out[k])
    return out


windows = {"expanding": None, "rolling 100": 100, "rolling 40": 40}
results = {}
print(f"{'setting':<14}{'method':<10}{'coverage':>10}{'mean width':>12}{'mean log score':>16}")
for name, win in windows.items():
    r = run_study(win)
    results[name] = r
    inside = (r["y"] >= r["lo"]) & (r["y"] <= r["hi"])
    print(f"{name:<14}{'mixture':<10}{inside.mean():>10.1%}{(r['hi']-r['lo']).mean():>12.2f}"
          f"{r['logscore'].mean():>16.3f}")
    for m in range(K):
        print(f"{'':<14}{'M'+str(m+1):<10}{np.mean(r['single_inside'][m]):>10.1%}"
              f"{np.mean(r['single_w'][m]):>12.2f}{np.mean(r['single_ls'][m]):>16.3f}")
print("(nominal coverage 80%; higher log score is better)")

MAIN = "rolling 100"
R = results[MAIN]

# =============================================================================
# 3. Figures
# =============================================================================
def finish(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150)
    if SHOW:
        plt.show()
    plt.close()


# --- 3a. the generating process ------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
ax[0].stackplot(t_idx, W_true.T, colors=colors, alpha=0.7,
                labels=[f"{c['name']} (sigma={c['sigma']})" for c in components])
ax[0].set_ylabel("generating weight"); ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("Generating mixture weights shift over time")
for m in range(K):
    mask = z == m
    ax[1].plot(t_idx[mask], y[mask], ".", color=colors[m], ms=3)
ax[1].set_ylabel("y (coloured by active component)"); ax[1].set_xlabel("time step")
finish("step4_generating_process")

# --- 3b. forecasts + evidence-based BMA weights ---------------------------------
inside_main = (R["y"] >= R["lo"]) & (R["y"] <= R["hi"])
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].fill_between(R["s"], R["lo"], R["hi"], alpha=0.3, color="darkorange", label="80% mixture interval")
ax[0].plot(R["s"], R["mean"], color="darkorange", lw=1, label="mixture mean")
ax[0].plot(R["s"], R["y"], ".", color="black", ms=3, label="y_true")
ax[0].set_title(f"Mixture-PPD forecasts, window = {MAIN}: coverage {inside_main.mean():.1%} (nominal 80%)")
ax[0].legend(fontsize=8, loc="upper right")
ax[1].stackplot(R["s"], R["W"].T, colors=colors, alpha=0.7, labels=[c["name"] for c in components])
ax[1].set_ylabel("evidence (BMA) weight"); ax[1].legend(fontsize=8, loc="upper right")
ax[1].set_title("Evidence-based BMA weights (model selection criterion, not a weight estimator)")
ax[2].stackplot(t_idx[START:], W_true[START:].T, colors=colors, alpha=0.7)
ax[2].set_ylabel("generating weight"); ax[2].set_xlabel("time step")
finish("step4_forecasts_and_weights")

# --- 3c. window comparison -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
names = list(results)
cov = [np.mean((results[n]["y"] >= results[n]["lo"]) & (results[n]["y"] <= results[n]["hi"])) for n in names]
ls = [results[n]["logscore"].mean() for n in names]
ax[0].bar(names, cov, color="steelblue"); ax[0].axhline(0.8, color="red", ls="--", label="nominal 80%")
ax[0].set_ylim(0, 1); ax[0].set_title("Empirical coverage"); ax[0].legend(fontsize=8)
ax[1].bar(names, ls, color="seagreen"); ax[1].set_title("Mean log predictive score (higher = better)")
finish("step4_window_comparison")

# --- 3d. PPD illustrations -----------------------------------------------------
# Transition case = the forecast time where the evidence weights are most mixed
# (highest entropy), since that is where the mixture PPD can become multimodal.
Wc = np.clip(R["W"], 1e-300, 1.0)
entropy = -(Wc * np.log(Wc)).sum(axis=1)
s_mixed = int(R["s"][np.argmax(entropy)])
print(f"Most mixed weights at t={s_mixed}: {np.round(R['W'][np.argmax(entropy)], 2)}")
cases = [(130, "low-noise period (M1 dominates)"),
         (s_mixed, "most mixed weights (transition)"),
         (350, "high-noise period (M2 dominates)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax_, (s, title) in zip(axes, cases):
    fits = fit_all(s, 100)
    comps = [ppd(fits[m], X_M[m][s]) for m in range(K)]
    w = softmax(np.array([f["log_ev"] for f in fits]))
    lo_ = min(c.ppf(0.001) for c in comps); hi_ = max(c.ppf(0.999) for c in comps)
    grid = np.linspace(lo_, hi_, 800)
    for m in range(K):
        ax_.plot(grid, comps[m].pdf(grid), color=colors[m], lw=1, alpha=0.8,
                 label=f"M{m+1} PPD (w={w[m]:.2f})")
    ax_.plot(grid, sum(wi * c.pdf(grid) for wi, c in zip(w, comps)), "k", lw=2.2, label="mixture PPD")
    # plug-in Gaussian for the highest-weight model (ignores parameter uncertainty)
    mbest = int(np.argmax(w)); f = fits[mbest]
    ax_.plot(grid, norm.pdf(grid, X_M[mbest][s] @ f["mu_n"], np.sqrt(f["b_n"] / (f["a_n"] - 1))),
             "k--", lw=1, label=f"plug-in Gaussian (M{mbest+1})")
    ax_.axvline(y[s], color="purple", ls=":", label="y_true")
    ax_.set_title(f"t={s}: {title}", fontsize=9); ax_.legend(fontsize=7)
finish("step4_ppd_cases")

# --- 3e. posterior vs. PPD, with theoretical densities on histograms ----------
s, m = 350, 1                     # M2 in its high-noise period
fit = fit_all(s, 100)[m]
Xm = X_M[m][s]
p = len(fit["mu_n"])
V = np.linalg.solve(fit["Lambda_n"], np.eye(p))        # posterior scale matrix
a_n, b_n = fit["a_n"], fit["b_n"]
n_draw = 20000
sig2 = invgamma(a_n, scale=b_n).rvs(size=n_draw, random_state=rng)
L = np.linalg.cholesky(V)
beta = fit["mu_n"] + np.sqrt(sig2)[:, None] * (rng.normal(size=(n_draw, p)) @ L.T)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# slope posterior (marginal is Student-t)
grid = np.linspace(beta[:, 1].min(), beta[:, 1].max(), 400)
ax[0].hist(beta[:, 1], bins=60, density=True, alpha=0.5, color="steelblue", label="samples")
ax[0].plot(grid, student_t(df=2 * a_n, loc=fit["mu_n"][1], scale=np.sqrt(b_n / a_n * V[1, 1])).pdf(grid),
           "k", lw=2, label="theory: Student-t")
ax[0].axvline(components[1]["beta"], color="red", ls="--", label="generating slope")
ax[0].set_title("Posterior of slope (M2)"); ax[0].legend(fontsize=8)
# noise variance posterior (inverse gamma)
grid = np.linspace(sig2.min(), np.percentile(sig2, 99.9), 400)
ax[1].hist(sig2, bins=np.linspace(grid[0], grid[-1], 60), density=True, alpha=0.5, color="seagreen", label="samples")
ax[1].plot(grid, invgamma(a_n, scale=b_n).pdf(grid), "k", lw=2, label="theory: Inv-Gamma")
ax[1].axvline(components[1]["sigma"] ** 2, color="red", ls="--", label="generating sigma^2")
ax[1].set_title("Posterior of noise variance (M2)"); ax[1].legend(fontsize=8)
# posterior of the mean function vs. PPD
mean_post = student_t(df=2 * a_n, loc=Xm @ fit["mu_n"], scale=np.sqrt(b_n / a_n * (Xm @ V @ Xm)))
pred = ppd(fit, Xm)
grid = np.linspace(pred.ppf(0.001), pred.ppf(0.999), 600)
ax[2].plot(grid, mean_post.pdf(grid), color="steelblue", lw=2, label="posterior of mean x*^T beta")
ax[2].plot(grid, pred.pdf(grid), "k", lw=2, label="PPD of y* (adds noise)")
ax[2].axvline(y[s], color="purple", ls=":", label="y_true")
ax[2].set_title("Posterior (parameters) vs. PPD (new observation)"); ax[2].legend(fontsize=8)
finish("step4_posterior_vs_ppd")

print("\nSaved figures to ./figures/ (step4_*.png)")