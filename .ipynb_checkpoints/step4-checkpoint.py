"""
Step 4: non-stationary synthetic data, window comparison, PPD / posterior plots
-------------------------------------------------------------------------------

  * Data are generated from a MIXTURE of three component models whose weights
    shift smoothly over time (each component is "the most true" in a different
    period). The components also differ in noise level, so some periods have
    broad predictive distributions and others narrow ones.
  * Linear algebra uses np.linalg.solve instead of explicit inverses.
  * The brentq search interval for mixture quantiles is built automatically
    from the component quantiles (min of lower, max of upper).
  * Expanding window vs. fixed rolling windows (bias-variance trade-off).
  * Plots: PPDs for low/high-noise and transition periods, posterior vs. PPD,
    posterior histograms with their theoretical densities overlaid.

Update (Oct 28):

  * Two nominal interval levels (50% and 80%) instead of one.
  * Tempered BMA weights  w_k  proportional to  exp(eta * log_evidence_k),
    eta = 1 is standard BMA, eta -> 0 gives equal weights. A few values only.
  * Systematic comparison of the four layers (point, plug-in Gaussian,
    single-model PPD, tempered mixture PPD) with the same metrics.
  * Log predictive score computed with scipy.special.logsumexp.

Note on the evidence: the Bayesian model evidence is a
model selection criterion and the standard BMA weight. It is NOT an estimator
of the generating mixture weights, so the plot below compares the two but does
not expect them to agree.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, norm, invgamma
from scipy.special import gammaln, logsumexp
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


def log_softmax(v):
    """Numerically stable log of softmax(v), via logsumexp."""
    v = np.asarray(v, dtype=float)
    return v - logsumexp(v)


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
# 2. Rolling-origin forecasting study, all four layers
# =============================================================================
START = 100
LEVELS = (0.5, 0.8)            # nominal central interval levels
ETAS = (1.0, 0.5, 0.25)        # inverse temperatures (1 = standard BMA)
MAIN_ETA = 0.5
MAIN = "rolling 100"
windows = {"expanding": None, "rolling 100": 100, "rolling 40": 40}


def new_layer():
    return dict(lo={l: [] for l in LEVELS}, hi={l: [] for l in LEVELS}, lps=[], mean=[])


def record(layer, dist, y_obs):
    for l in LEVELS:
        layer["lo"][l].append(dist.ppf((1 - l) / 2))
        layer["hi"][l].append(dist.ppf(1 - (1 - l) / 2))
    layer["lps"].append(dist.logpdf(y_obs))
    layer["mean"].append(dist.mean())


def run_study(window):
    """Layers 2-3 use the model with the highest evidence at each step;
    layer 4 is the tempered mixture (one run per eta)."""
    out = dict(s=[], y=[], L1=[], L2=new_layer(), L3=new_layer(),
               L4={e: new_layer() for e in ETAS}, W={e: [] for e in ETAS})
    for s in range(START, n_total):
        fits = fit_all(s, window)
        comps = [ppd(fits[m], X_M[m][s]) for m in range(K)]
        logev = np.array([f["log_ev"] for f in fits])
        top = int(np.argmax(logev))
        f, mu = fits[top], X_M[top][s] @ fits[top]["mu_n"]
        out["s"].append(s); out["y"].append(y[s])
        out["L1"].append(mu)                                               # layer 1
        record(out["L2"], norm(mu, np.sqrt(f["b_n"] / (f["a_n"] - 1))), y[s])  # layer 2
        record(out["L3"], comps[top], y[s])                                # layer 3
        logpdfs = np.array([c.logpdf(y[s]) for c in comps])
        for e in ETAS:                                                     # layer 4
            logw = log_softmax(e * logev)
            w = np.exp(logw)
            lay = out["L4"][e]
            for l in LEVELS:
                lay["lo"][l].append(mixture_ppf((1 - l) / 2, comps, w))
                lay["hi"][l].append(mixture_ppf(1 - (1 - l) / 2, comps, w))
            lay["lps"].append(logsumexp(logw + logpdfs))                   # stable
            lay["mean"].append(float(w @ np.array([c.mean() for c in comps])))
            out["W"][e].append(w)
    out["s"], out["y"], out["L1"] = map(np.array, (out["s"], out["y"], out["L1"]))
    for lay in [out["L2"], out["L3"]] + list(out["L4"].values()):
        lay["lps"], lay["mean"] = np.array(lay["lps"]), np.array(lay["mean"])
        for l in LEVELS:
            lay["lo"][l], lay["hi"][l] = np.array(lay["lo"][l]), np.array(lay["hi"][l])
    out["W"] = {e: np.array(v) for e, v in out["W"].items()}
    return out


def metrics(lay, yv):
    d = {}
    for l in LEVELS:
        d[f"cov{int(l*100)}"] = np.mean((yv >= lay["lo"][l]) & (yv <= lay["hi"][l]))
        d[f"width{int(l*100)}"] = np.mean(lay["hi"][l] - lay["lo"][l])
    d["lps"] = lay["lps"].mean()
    return d


results = {name: run_study(win) for name, win in windows.items()}

rows = []      # (window, layer, eta, metrics dict)
for name, r in results.items():
    rows.append((name, "1 point", None, dict(rmse=np.sqrt(np.mean((r["y"] - r["L1"]) ** 2)))))
    rows.append((name, "2 plug-in Gaussian", None, metrics(r["L2"], r["y"])))
    rows.append((name, "3 single-model PPD", None, metrics(r["L3"], r["y"])))
    for e in ETAS:
        rows.append((name, "4 mixture PPD", e, metrics(r["L4"][e], r["y"])))

hdr = f"{'window':<13}{'layer':<20}{'eta':>5}{'cov50':>8}{'cov80':>8}{'w50':>7}{'w80':>7}{'LPS':>9}{'RMSE':>7}"
print(hdr)
for name, layer, e, d in rows:
    fmt = lambda k, f: (format(d[k], f) if k in d else "")
    print(f"{name:<13}{layer:<20}{'' if e is None else e:>5}"
          f"{fmt('cov50', '8.1%') :>8}{fmt('cov80', '8.1%'):>8}{fmt('width50', '7.2f'):>7}"
          f"{fmt('width80', '7.2f'):>7}{fmt('lps', '9.3f'):>9}{fmt('rmse', '7.2f'):>7}")
print("(nominal 50% / 80%; LPS = mean log predictive score, higher is better; "
      "layers 2-3 use the highest-evidence model at each step)")

import csv
with open("figures/step4_results.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["window", "layer", "eta", "cov50", "cov80", "width50", "width80", "lps", "rmse"])
    for name, layer, e, d in rows:
        wr.writerow([name, layer, e] + [d.get(k, "") for k in
                     ("cov50", "cov80", "width50", "width80", "lps", "rmse")])

R = results[MAIN]
LM = R["L4"][MAIN_ETA]
WM = R["W"][MAIN_ETA]

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

# --- 3b. forecasts + tempered BMA weights --------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].fill_between(R["s"], LM["lo"][0.8], LM["hi"][0.8], alpha=0.25, color="darkorange", label="80% interval")
ax[0].fill_between(R["s"], LM["lo"][0.5], LM["hi"][0.5], alpha=0.45, color="darkorange", label="50% interval")
ax[0].plot(R["s"], LM["mean"], color="darkorange", lw=1, label="mixture mean")
ax[0].plot(R["s"], R["y"], ".", color="black", ms=3, label="y_true")
cov_txt = ", ".join(f"{int(l*100)}%: {np.mean((R['y']>=LM['lo'][l])&(R['y']<=LM['hi'][l])):.1%}" for l in LEVELS)
ax[0].set_title(f"Mixture-PPD forecasts, {MAIN}, eta={MAIN_ETA}: coverage {cov_txt}")
ax[0].legend(fontsize=8, loc="upper right")
ax[1].stackplot(R["s"], WM.T, colors=colors, alpha=0.7, labels=[c["name"] for c in components])
ax[1].set_ylabel(f"BMA weight (eta={MAIN_ETA})"); ax[1].legend(fontsize=8, loc="upper right")
ax[1].set_title("Tempered evidence weights (model selection criterion, not a weight estimator)")
ax[2].stackplot(t_idx[START:], W_true[START:].T, colors=colors, alpha=0.7)
ax[2].set_ylabel("generating weight"); ax[2].set_xlabel("time step")
finish("step4_forecasts_and_weights")

# --- 3b2. effect of the inverse temperature on the weights ----------------------
fig, ax = plt.subplots(len(ETAS) + 1, 1, figsize=(10, 9), sharex=True)
for i, e in enumerate(ETAS):
    ax[i].stackplot(R["s"], R["W"][e].T, colors=colors, alpha=0.7)
    ax[i].set_ylabel(f"eta = {e}")
ax[-1].stackplot(t_idx[START:], W_true[START:].T, colors=colors, alpha=0.7)
ax[-1].set_ylabel("generating"); ax[-1].set_xlabel("time step")
ax[0].set_title(f"BMA weights for different inverse temperatures ({MAIN})")
finish("step4_eta_weights")

# --- 3c. systematic comparison of the four layers -------------------------------
layers = [("2 plug-in", R["L2"]), ("3 single PPD", R["L3"]), (f"4 mixture\n(eta={MAIN_ETA})", LM)]
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
xs = np.arange(len(layers)); bw = 0.38
for j, l in enumerate(LEVELS):
    vals = [np.mean((R["y"] >= lay["lo"][l]) & (R["y"] <= lay["hi"][l])) for _, lay in layers]
    ax[0].bar(xs + (j - 0.5) * bw, vals, bw, label=f"{int(l*100)}% interval")
    ax[0].hlines(l, xs[0] - 0.5, xs[-1] + 0.5, color="red", ls="--", lw=1)
ax[0].set_xticks(xs); ax[0].set_xticklabels([n for n, _ in layers], fontsize=8)
ax[0].set_ylim(0, 1); ax[0].set_title("Empirical coverage (red = nominal)"); ax[0].legend(fontsize=8)
for j, l in enumerate(LEVELS):
    ax[1].bar(xs + (j - 0.5) * bw, [np.mean(lay["hi"][l] - lay["lo"][l]) for _, lay in layers], bw,
              label=f"{int(l*100)}%")
ax[1].set_xticks(xs); ax[1].set_xticklabels([n for n, _ in layers], fontsize=8)
ax[1].set_title("Mean interval width"); ax[1].legend(fontsize=8, loc="upper center", ncol=2)
ax[1].set_ylim(0, 6.5)
lps_vals = [lay["lps"].mean() for _, lay in layers]
ax[2].plot(xs, lps_vals, "o", color="seagreen", ms=9)
for x_, v_ in zip(xs, lps_vals):
    ax[2].annotate(f"{v_:.3f}", (x_, v_), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
ax[2].set_xticks(xs); ax[2].set_xticklabels([n for n, _ in layers], fontsize=8)
pad = 0.3 * (max(lps_vals) - min(lps_vals) + 0.02)
ax[2].set_ylim(min(lps_vals) - pad, max(lps_vals) + pad)
ax[2].set_title("Mean log predictive score (higher = better)")
fig.suptitle(f"Layer comparison, {MAIN}", y=1.02)
finish("step4_layer_comparison")

# --- 3c2. window x eta comparison ------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
names = list(results); xs = np.arange(len(names)); bw = 0.8 / len(ETAS)
panels = [("cov80", "Coverage of 80% interval", 0.8), ("cov50", "Coverage of 50% interval", 0.5),
          ("lps", "Mean log predictive score", None)]
for a_, (key, title, nominal) in zip(ax, panels):
    for i, e in enumerate(ETAS):
        vals = []
        for n in names:
            lay = results[n]["L4"][e]
            vals.append(metrics(lay, results[n]["y"])[key])
        a_.bar(xs + (i - (len(ETAS) - 1) / 2) * bw, vals, bw, label=f"eta={e}")
    if nominal:
        a_.axhline(nominal, color="red", ls="--", lw=1); a_.set_ylim(0, 1)
    a_.set_xticks(xs); a_.set_xticklabels(names); a_.set_title(title); a_.legend(fontsize=8)
finish("step4_window_eta_comparison")

# --- 3d. PPD illustrations -----------------------------------------------------
# Transition case = the forecast time where the evidence weights are most mixed
# (highest entropy), since that is where the mixture PPD can become multimodal.
Wc = np.clip(WM, 1e-300, 1.0)
entropy = -(Wc * np.log(Wc)).sum(axis=1)
s_mixed = int(R["s"][np.argmax(entropy)])
print(f"Most mixed weights at t={s_mixed}: {np.round(WM[np.argmax(entropy)], 2)}")
cases = [(130, "low-noise period (M1 dominates)"),
         (s_mixed, f"most mixed weights (transition, eta={MAIN_ETA})"),
         (350, "high-noise period (M2 dominates)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax_, (s, title) in zip(axes, cases):
    fits = fit_all(s, 100)
    comps = [ppd(fits[m], X_M[m][s]) for m in range(K)]
    w = np.exp(log_softmax(MAIN_ETA * np.array([f["log_ev"] for f in fits])))
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