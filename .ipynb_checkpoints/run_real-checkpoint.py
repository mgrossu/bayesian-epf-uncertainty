"""
run_real.py -- real-data pipeline (Germany, one delivery hour)
----------------------------------------------------------------
    python run_real.py --csv DE.csv [--hour 18] [--selector lasso|omp] [--sizes 2,4,7,11]

Chronological split:  [ train + validation : 80% ] [ test : 20%, the unseen future ]

  1. data        daily table for one hour (epf_data)
  2. selection   feature ranking by Lasso path / OMP, on train+validation only
  3. tuning      on train+validation with a rolling origin:
                   a) model size curve (single-model PPD, log predictive score)
                   b) window x eta grid for the mixture; pick the best by LPS
  4. test        ONE final run on the test part with the frozen settings:
                   all models x layers 1-3, mixture (layer 4), figures, tables
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epf_data import load_hour, split_point, FEATURES
from epf_select import rank_lasso, rank_omp, nested_models
from epf_forecast import (run_origin, summarize, mixture_lps, mixture_logw,
                          mixture_mean, mixture_interval, mixture_pit)

# ----------------------------------------------------------------- config
LEVELS = (0.8, 0.95)                      # nominal interval levels
ETAS = (1.0, 0.5, 0.25, 0.1)              # inverse temperatures tried
WINDOWS = {"expanding": None, "rolling 364": 364, "rolling 91": 91}   # in days
BURN_IN = 364                             # first validation target (days of history)
TRAIN_FRAC = 0.8

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--hour", type=int, default=18)
ap.add_argument("--selector", choices=["lasso", "omp"], default="lasso")
ap.add_argument("--sizes", default="2,4,7,11", help="feature counts of the candidate models")
ap.add_argument("--outdir", default="results_real")
args = ap.parse_args()
SIZES = tuple(int(v) for v in args.sizes.split(","))
os.makedirs(args.outdir, exist_ok=True)
out = lambda f: os.path.join(args.outdir, f)


def savefig(name):
    plt.tight_layout(); plt.savefig(out(name), dpi=150); plt.close()


# ------------------------------------------------------------------ 1. data
df = load_hour(args.csv, args.hour)
X, y = df[FEATURES].values, df["y"].values
n, p = X.shape
n_tv = split_point(n, TRAIN_FRAC)
val_s = list(range(BURN_IN, n_tv))
test_s = list(range(n_tv, n))
print(f"hour {args.hour}: {n} days, {df.index[0].date()} .. {df.index[-1].date()}")
print(f"train+validation: {df.index[0].date()} .. {df.index[n_tv-1].date()} ({n_tv} days; "
      f"validation targets {len(val_s)} from {df.index[BURN_IN].date()})")
print(f"test (unseen):    {df.index[n_tv].date()} .. {df.index[-1].date()} ({len(test_s)} days)")

# --------------------------------------------------------------- 2. selection
orders = {"lasso": rank_lasso(X[:n_tv], y[:n_tv]), "omp": rank_omp(X[:n_tv], y[:n_tv])}
for k, o in orders.items():
    print(f"ranking by {k:<5}: " + " > ".join(FEATURES[j] for j in o))
order = orders[args.selector]

# ------------------------------------------- 3a. model-size curve (validation)
ref_window = 364
curve_models = nested_models(order, range(1, p + 1))
st = run_origin(X, y, curve_models, ref_window, val_s)
curve = pd.DataFrame({"k": range(1, p + 1), "lps": st["logpdf"].mean(axis=0),
                      "feature_added": [FEATURES[j] for j in order]})
curve.to_csv(out("validation_size_curve.csv"), index=False)
print("\nmodel size curve (single-model PPD, validation LPS, window 364):")
print(curve.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
plt.figure(figsize=(7, 4))
plt.plot(curve["k"], curve["lps"], "o-")
for k_ in SIZES:
    plt.axvline(k_, color="gray", ls=":", lw=1)
plt.xlabel("number of features k (ranked by " + args.selector + ")")
plt.ylabel("validation log predictive score")
plt.title("Model size curve (dotted = candidate model sizes)")
savefig("fig_model_size_curve.png")

# ---------------------------------------- 3b. window x eta grid (validation)
model_cols = nested_models(order, SIZES)
names = [f"top{k}" for k in SIZES]
grid = []
for wname, w in WINDOWS.items():
    stv = run_origin(X, y, model_cols, w, val_s)
    for m, nm in enumerate(names):
        grid.append(dict(window=wname, model=nm, eta="", lps=stv["logpdf"][:, m].mean()))
    for eta in ETAS:
        grid.append(dict(window=wname, model="mixture", eta=eta, lps=mixture_lps(stv, eta).mean()))
grid = pd.DataFrame(grid)
grid.to_csv(out("validation_grid.csv"), index=False)
mix = grid[grid.model == "mixture"]
best = mix.loc[mix.lps.idxmax()]
BEST_WINDOW, BEST_ETA = best["window"], float(best["eta"])
print("\nvalidation LPS (mixture) by window and eta:")
print(mix.pivot(index="window", columns="eta", values="lps").round(3).to_string())
print("single models, validation LPS:")
print(grid[grid.model != "mixture"].pivot(index="window", columns="model", values="lps").round(3).to_string())
print(f"\nchosen on validation: window = {BEST_WINDOW}, eta = {BEST_ETA}")
if BEST_ETA in (min(ETAS), max(ETAS)):
    print("  note: the best eta is at the edge of the grid")
json.dump(dict(hour=args.hour, selector=args.selector, order=[FEATURES[j] for j in order],
               sizes=SIZES, window=BEST_WINDOW, eta=BEST_ETA, levels=LEVELS),
          open(out("chosen_settings.json"), "w"), indent=2)

plt.figure(figsize=(7, 4))
for wname in WINDOWS:
    d = mix[mix.window == wname]
    plt.plot(d["eta"].astype(float), d["lps"], "o-", label=wname)
plt.xscale("log"); plt.xlabel("inverse temperature eta"); plt.ylabel("validation log predictive score")
plt.title("Tuning on train+validation (mixture)"); plt.legend()
savefig("fig_tuning_eta_window.png")

# ----------------------------------------------------------------- 4. test
print("\n=== TEST (unseen future), frozen settings ===")
all_rows, studies = [], {}
for wname, w in WINDOWS.items():
    stt = run_origin(X, y, model_cols, w, test_s)
    studies[wname] = stt
    for r in summarize(stt, ETAS, LEVELS, names):
        all_rows.append(dict(window=wname, **r))
res = pd.DataFrame(all_rows)
res.to_csv(out("test_results.csv"), index=False)
cols = ["model", "layer", "eta", "cov80", "cov95", "width80", "width95", "lps", "rmse", "mae"]
show = res[res.window == BEST_WINDOW][cols]
print(f"window = {BEST_WINDOW} (nominal coverage 80% / 95%; LPS higher = better)")
print(show.to_string(index=False, na_rep="", float_format=lambda v: f"{v:.3f}"))
mixrows = res[(res.layer == "L4 mixture PPD")]
print("\nmixture on test, all windows and etas:")
print(mixrows[["window", "eta", "cov80", "cov95", "width80", "width95", "lps"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ------------------------------------------------------------------- figures
stb = studies[BEST_WINDOW]
dates = df.index[test_s]
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
for l, a_ in zip((0.95, 0.8), (0.25, 0.45)):
    lo, hi = mixture_interval(stb, BEST_ETA, l)
    ax[0].fill_between(dates, lo, hi, color="darkorange", alpha=a_, label=f"{round(l*100)}% interval")
ax[0].plot(dates, mixture_mean(stb, BEST_ETA), color="darkorange", lw=1, label="mixture mean")
ax[0].plot(dates, stb["y"], ".", color="black", ms=3, label="observed price")
ax[0].set_ylabel("EUR/MWh"); ax[0].legend(fontsize=8, ncol=4, loc="upper left")
ax[0].set_title(f"Test period, hour {args.hour}: mixture PPD ({BEST_WINDOW}, eta={BEST_ETA})")
W = np.exp(mixture_logw(stb, BEST_ETA))
ax[1].stackplot(dates, W.T, labels=names, alpha=0.8)
ax[1].set_ylabel("BMA weight"); ax[1].legend(fontsize=8, ncol=len(names), loc="upper left")
savefig("fig_test_forecasts_weights.png")

fig, axs = plt.subplots(1, len(LEVELS), figsize=(6.5 * len(LEVELS), 4.8))
for ax_, l in zip(np.atleast_1d(axs), LEVELS):
    c, wd = f"cov{round(l*100)}", f"width{round(l*100)}"
    sub = res[(res.window == BEST_WINDOW) & res.layer.isin(["L2 plug-in", "L3 single PPD"])]
    for layer, mk in (("L2 plug-in", "s"), ("L3 single PPD", "^")):
        d = sub[sub.layer == layer]
        ax_.scatter(d[wd], (d[c] - l).abs(), marker=mk, color="tab:blue", label=layer)
        for _, r in d.iterrows():
            ax_.annotate(r["model"], (r[wd], abs(r[c] - l)), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for wname, mk in zip(WINDOWS, ("o", "D", "P")):
        d = mixrows[mixrows.window == wname]
        ax_.scatter(d[wd], (d[c] - l).abs(), marker=mk, color="darkorange",
                    label=f"L4 mixture, {wname}")
        for _, r in d.iterrows():
            ax_.annotate(f"eta={r['eta']}", (r[wd], abs(r[c] - l)), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax_.set_xlabel("mean interval width (EUR/MWh)"); ax_.set_ylabel("coverage error |empirical - nominal|")
    ax_.set_title(f"{round(l*100)}% intervals (bottom-left is best)"); ax_.legend(fontsize=7)
savefig("fig_coverage_error_vs_width.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
best_single = int(np.argmax(stb["logpdf"].mean(axis=0)))
pit_single = np.array([stb["comps"][i][best_single].cdf(stb["y"][i]) for i in range(len(test_s))])
for a_, pit, ttl in ((ax[0], mixture_pit(stb, BEST_ETA), f"mixture (eta={BEST_ETA})"),
                     (ax[1], pit_single, f"single PPD {names[best_single]}")):
    a_.hist(pit, bins=10, range=(0, 1), density=True, color="steelblue", alpha=0.8)
    a_.axhline(1, color="red", ls="--"); a_.set_title("PIT: " + ttl)
savefig("fig_pit.png")

print(f"\nSaved tables and figures to ./{args.outdir}/")
