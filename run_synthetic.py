"""
run_synthetic.py -- synthetic-data run: the SAME pipeline as run_real.py
-------------------------------------------------------------------------
    python run_synthetic.py [--seed 7] [--n 1200] [--outdir results_synth]

Data come from a mixture of three linear models with periodically shifting weights
(epf_synth.py). The candidate models are the three single-predictor regressions.
Split, tuning, layer comparison and test evaluation are exactly those of
epf_pipeline.run_pipeline. In addition, this script draws the figures that
illustrate the theory (generating process, PPD cases, posterior vs. PPD, effect of eta).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, norm, invgamma

from epf_synth import generate, COMPONENTS
from epf_core import fit_nig, predictive
from epf_forecast import run_origin, mixture_logw
from epf_pipeline import run_pipeline

LEVELS = (0.8, 0.95)
ETAS = (1.0, 0.5, 0.25, 0.1, 0.0)
WINDOWS = {"expanding": None, "rolling 100": 100, "rolling 40": 40}     # in time steps
TRAIN_FRAC = 0.8
BURN_IN = 120

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--outdir", default="results_synth")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
out = lambda f: os.path.join(args.outdir, f)
colors = ["tab:blue", "tab:red", "tab:green"]


def savefig(name):
    plt.tight_layout(); plt.savefig(out(name), dpi=150); plt.close()


D = generate(n=args.n, seed=args.seed)
X, y, z, W, t = D["X"], D["y"], D["z"], D["W"], D["t"]
n = len(y); K = len(COMPONENTS)
n_tv = int(TRAIN_FRAC * n)
names = [c["name"] for c in COMPONENTS]
model_cols = [np.array([c["feat"]]) for c in COMPONENTS]
print(f"synthetic data: n={n}, seed={args.seed}; train+validation 0..{n_tv-1}, "
      f"test {n_tv}..{n-1} (unseen)")

# ---------------------------------------------------- generating process
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
ax[0].stackplot(t, W.T, colors=colors, alpha=0.7,
                labels=[f"{c['name']} (sigma={c['sigma']})" for c in COMPONENTS])
ax[0].axvline(n_tv, color="k", ls="--"); ax[0].text(n_tv + 5, 0.05, "test", fontsize=9)
ax[0].set_ylabel("generating weight"); ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("Generating mixture weights shift over time")
for m in range(K):
    ax[1].plot(t[z == m], y[z == m], ".", color=colors[m], ms=3)
ax[1].axvline(n_tv, color="k", ls="--")
ax[1].set_ylabel("y (coloured by active component)"); ax[1].set_xlabel("time step")
savefig("fig_generating_process.png")

# --------------------------------------------------- shared pipeline
r = run_pipeline(X, y, t, model_cols, names, n_tv, BURN_IN, WINDOWS, ETAS, LEVELS,
                 args.outdir, title="Synthetic,", unit="y", W_true=W)
BEST_W, BEST_ETA, SEL = r["window"], r["eta"], r["sel_idx"]
win = WINDOWS[BEST_W]
stb, test_s = r["studies"][BEST_W], r["test_s"]

# ------------------------------------ effect of eta on the weights (test)
fig, ax = plt.subplots(len(ETAS) + 1, 1, figsize=(10, 2 * (len(ETAS) + 1) + 1), sharex=True)
for i, e in enumerate(ETAS):
    ax[i].stackplot(t[test_s], np.exp(mixture_logw(stb, e)).T, colors=colors, alpha=0.7)
    ax[i].set_ylabel(f"eta = {e}")
ax[-1].stackplot(t[test_s], W[test_s].T, colors=colors, alpha=0.7)
ax[-1].set_ylabel("generating"); ax[-1].set_xlabel("time step")
ax[0].set_title(f"BMA weights for different inverse temperatures ({BEST_W}, test period)")
savefig("fig_eta_weights.png")

# --------------------------------------------------------- PPD cases
Wm = np.exp(mixture_logw(stb, BEST_ETA))
ent = -(np.clip(Wm, 1e-300, 1) * np.log(np.clip(Wm, 1e-300, 1))).sum(axis=1)
i_low = int(np.argmax(W[test_s, 0])); i_high = int(np.argmax(W[test_s, 1])); i_mix = int(np.argmax(ent))
cases = [(i_low, "low-noise period (M1 generates most)"),
         (i_mix, f"most mixed weights (eta={BEST_ETA})"),
         (i_high, "high-noise period (M2 generates most)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax_, (i, ttl) in zip(axes, cases):
    comps, w, s = stb["comps"][i], Wm[i], test_s[i]
    g = np.linspace(min(c.ppf(0.001) for c in comps), max(c.ppf(0.999) for c in comps), 800)
    for m in range(K):
        ax_.plot(g, comps[m].pdf(g), color=colors[m], lw=1, alpha=0.8, label=f"{names[m]} PPD (w={w[m]:.2f})")
    ax_.plot(g, sum(wi * c.pdf(g) for wi, c in zip(w, comps)), "k", lw=2.2, label="mixture PPD")
    ax_.plot(g, stb["plugs"][i][SEL].pdf(g), "k--", lw=1, label=f"plug-in Gaussian ({names[SEL]})")
    ax_.axvline(y[s], color="purple", ls=":", label="observed")
    ax_.set_title(f"t={s}: {ttl}", fontsize=9); ax_.legend(fontsize=7)
savefig("fig_ppd_cases.png")

# ------------------------------------- posterior vs. PPD, theory overlays
s, m, wlen = n_tv, 1, 100            # first test day, model M2, window of 100 steps
lo = s - wlen
Xm = np.column_stack([np.ones(wlen), X[lo:s, 1]]); ym = y[lo:s]
fit = fit_nig(Xm, ym)
xs_ = np.array([1.0, X[s, 1]])
p_ = 2
V = np.linalg.solve(fit["Lam"], np.eye(p_))
a_n, b_n = fit["a"], fit["b"]
shares = np.bincount(z[lo:s], minlength=K) / wlen
true_noise = float(sum(sh * c["sigma"] ** 2 for sh, c in zip(shares, COMPONENTS)))
print(f"window {lo}..{s-1}: share of points generated by M1/M2/M3 = {np.round(shares, 2)}; "
      f"share-weighted noise variance {true_noise:.2f}; posterior mean of sigma^2 (M2) "
      f"{b_n/(a_n-1):.2f}; M2 generating sigma^2 {COMPONENTS[1]['sigma']**2:.2f}")
rng = np.random.default_rng(args.seed + 1)
nd = 20000
sig2 = invgamma(a_n, scale=b_n).rvs(size=nd, random_state=rng)
Lc = np.linalg.cholesky(V)
beta = fit["mu"] + np.sqrt(sig2)[:, None] * (rng.normal(size=(nd, p_)) @ Lc.T)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
g = np.linspace(beta[:, 1].min(), beta[:, 1].max(), 400)
ax[0].hist(beta[:, 1], bins=60, density=True, alpha=0.5, color="steelblue", label="samples")
ax[0].plot(g, student_t(df=2 * a_n, loc=fit["mu"][1], scale=np.sqrt(b_n / a_n * V[1, 1])).pdf(g),
           "k", lw=2, label="theory: Student-t")
ax[0].axvline(COMPONENTS[1]["beta"], color="red", ls="--", label="M2 generating slope")
ax[0].set_title("Posterior of the slope (M2)"); ax[0].legend(fontsize=8)
g = np.linspace(sig2.min(), np.percentile(sig2, 99.9), 400)
ax[1].hist(sig2, bins=np.linspace(g[0], g[-1], 60), density=True, alpha=0.5, color="seagreen", label="samples")
ax[1].plot(g, invgamma(a_n, scale=b_n).pdf(g), "k", lw=2, label="theory: Inv-Gamma")
ax[1].axvline(COMPONENTS[1]["sigma"] ** 2, color="red", ls="--", label="M2 generating sigma^2")
ax[1].set_title("Posterior of the noise variance (M2)\nshares M1/M2/M3 in window: "
                + "/".join(f"{v:.2f}" for v in shares), fontsize=10); ax[1].legend(fontsize=7)
mean_post = student_t(df=2 * a_n, loc=xs_ @ fit["mu"], scale=np.sqrt(b_n / a_n * (xs_ @ V @ xs_)))
ppd, _, _ = predictive(fit, xs_)
g = np.linspace(ppd.ppf(0.001), ppd.ppf(0.999), 600)
ax[2].plot(g, mean_post.pdf(g), color="steelblue", lw=2, label="posterior of the mean x*'beta")
ax[2].plot(g, ppd.pdf(g), "k", lw=2, label="PPD of y* (adds noise)")
ax[2].axvline(y[s], color="purple", ls=":", label="observed")
ax[2].set_title("Posterior of the mean (a point estimate) vs. PPD"); ax[2].legend(fontsize=8)
savefig("fig_posterior_vs_ppd.png")
print(f"\nAll synthetic outputs are in ./{args.outdir}/")
