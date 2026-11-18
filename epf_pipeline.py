"""
epf_pipeline.py -- the evaluation pipeline shared by the synthetic and real-data runs
--------------------------------------------------------------------------------------
Given a feature matrix, targets and a list of candidate models, it runs exactly the
same procedure on any data set:

  chronological split:  [ train + validation ][ test = unseen future ]

  1. tuning (train+validation, rolling origin): window x eta grid for the mixture,
     chosen by the mean log predictive score (LPS); the single model for the layer
     comparison is chosen on VALIDATION data too (never on test data)
  2. test: ONE run with the frozen settings: every model x layers 1-3, the mixture
     (layer 4) for every eta, tables and figures

The data-specific parts (generating data, loading CSVs, ranking features) live in
run_synthetic.py / run_real.py.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epf_forecast import (run_origin, summarize, mixture_lps, mixture_logw,
                          mixture_mean, mixture_interval, mixture_pit)

COLS = ["model", "layer", "eta", "cov80", "cov95", "width80", "width95", "lps", "rmse", "mae"]
FMT = dict(index=False, na_rep="", float_format=lambda v: f"{v:.3f}")


def run_pipeline(X, y, index, model_cols, names, n_tv, burn_in, windows, etas, levels,
                 outdir, title="", unit="EUR/MWh", W_true=None):
    os.makedirs(outdir, exist_ok=True)
    out = lambda f: os.path.join(outdir, f)

    def savefig(name):
        plt.tight_layout(); plt.savefig(out(name), dpi=150); plt.close()

    n = len(y)
    val_s, test_s = list(range(burn_in, n_tv)), list(range(n_tv, n))

    # ----------------------------------------------- 1. tuning on validation
    grid = []
    for wname, w in windows.items():
        stv = run_origin(X, y, model_cols, w, val_s)
        for m, nm in enumerate(names):
            grid.append(dict(window=wname, model=nm, eta="", lps=stv["logpdf"][:, m].mean()))
        for eta in etas:
            grid.append(dict(window=wname, model="mixture", eta=eta,
                             lps=mixture_lps(stv, eta).mean()))
    grid = pd.DataFrame(grid)
    grid.to_csv(out("validation_grid.csv"), index=False)
    mix = grid[grid.model == "mixture"]
    best = mix.loc[mix.lps.idxmax()]
    BEST_WINDOW, BEST_ETA = best["window"], float(best["eta"])
    print("\nvalidation LPS (mixture) by window and eta:")
    print(mix.pivot(index="window", columns="eta", values="lps").round(3).to_string())
    print("single models, validation LPS:")
    print(grid[grid.model != "mixture"].pivot(index="window", columns="model",
                                              values="lps").round(3).to_string())
    print(f"\nchosen on validation: window = {BEST_WINDOW}, eta = {BEST_ETA}")
    pos = [e for e in etas if e > 0]
    if BEST_ETA == 0.0 or BEST_ETA == min(pos):
        print("  note: best eta is at the lower edge of the grid (eta=0 = equal-weight reference)")
    gs = grid[(grid.window == BEST_WINDOW) & (grid.model != "mixture")]
    SEL_NAME = gs.loc[gs.lps.idxmax(), "model"]
    SEL_IDX = names.index(SEL_NAME)
    print(f"single model chosen on validation (window {BEST_WINDOW}): {SEL_NAME}")
    json.dump(dict(window=BEST_WINDOW, eta=BEST_ETA, single_model=SEL_NAME, levels=levels,
                   models=names), open(out("chosen_settings.json"), "w"), indent=2)

    plt.figure(figsize=(7, 4))
    xs_eta = np.arange(len(etas))
    for wname in windows:
        d = mix[mix.window == wname].set_index("eta").loc[list(etas)]
        plt.plot(xs_eta, d["lps"].values, "o-", label=wname)
    plt.xticks(xs_eta, [str(e) for e in etas])
    plt.xlabel("inverse temperature eta (0 = equal weights)")
    plt.ylabel("validation log predictive score")
    plt.title("Tuning on train+validation (mixture)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    savefig("fig_tuning_eta_window.png")

    # ------------------------------------------------------------ 2. test
    print("\n=== TEST (unseen future), frozen settings ===")
    rows, studies = [], {}
    for wname, w in windows.items():
        stt = run_origin(X, y, model_cols, w, test_s)
        studies[wname] = stt
        for r in summarize(stt, etas, levels, names):
            rows.append(dict(window=wname, **r))
    res = pd.DataFrame(rows)
    res.to_csv(out("test_results.csv"), index=False)
    print(f"window = {BEST_WINDOW} (nominal coverage 80% / 95%; LPS higher = better)")
    print(res[res.window == BEST_WINDOW][COLS].to_string(**FMT))
    mixrows = res[res.layer == "L4 mixture PPD"]
    cmp_rows = res[(res.window == BEST_WINDOW) & (
        ((res.model == SEL_NAME) & res.layer.isin(["L1 point", "L2 plug-in", "L3 single PPD"])) |
        ((res.layer == "L4 mixture PPD") & res.eta.isin([BEST_ETA, 0.0])))][COLS]
    cmp_rows.to_csv(out("test_layer_comparison.csv"), index=False)
    print(f"\nLAYER COMPARISON on test (single model {SEL_NAME} chosen on validation; "
          f"mixture eta={BEST_ETA} tuned on validation; eta=0 = equal-weight reference):")
    print(cmp_rows.to_string(**FMT))
    print("\nmixture on test, all windows and etas:")
    print(mixrows[["window", "eta", "cov80", "cov95", "width80", "width95", "lps"]].to_string(**FMT))

    # -------------------------------------------------------------- figures
    stb = studies[BEST_WINDOW]
    xs = index[test_s]
    nrow = 3 if W_true is not None else 2
    fig, ax = plt.subplots(nrow, 1, figsize=(11, 3.2 * nrow + 1), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2] + [1] * (nrow - 1)))
    for l, a_ in zip((0.95, 0.8), (0.25, 0.45)):
        lo, hi = mixture_interval(stb, BEST_ETA, l)
        ax[0].fill_between(xs, lo, hi, color="darkorange", alpha=a_, label=f"{round(l*100)}% interval")
    ax[0].plot(xs, mixture_mean(stb, BEST_ETA), color="darkorange", lw=1, label="mixture mean")
    ax[0].plot(xs, stb["y"], ".", color="black", ms=3, label="observed")
    ax[0].set_ylabel(unit); ax[0].legend(fontsize=8, ncol=4, loc="upper left")
    ax[0].set_title(f"{title} test period: mixture PPD ({BEST_WINDOW}, eta={BEST_ETA})")
    Wm = np.exp(mixture_logw(stb, BEST_ETA))
    ax[1].stackplot(xs, Wm.T, labels=names, alpha=0.8)
    ax[1].set_ylabel("BMA weight"); ax[1].legend(fontsize=8, ncol=len(names), loc="upper left")
    if W_true is not None:
        ax[2].stackplot(xs, W_true[test_s].T, alpha=0.8)
        ax[2].set_ylabel("generating weight")
    savefig("fig_test_forecasts_weights.png")

    fig, axs = plt.subplots(1, len(levels), figsize=(6.5 * len(levels), 4.8))
    for ax_, l in zip(np.atleast_1d(axs), levels):
        c, wd = f"cov{round(l*100)}", f"width{round(l*100)}"
        sub = res[(res.window == BEST_WINDOW) & res.layer.isin(["L2 plug-in", "L3 single PPD"])]
        for layer, mk in (("L2 plug-in", "s"), ("L3 single PPD", "^")):
            d = sub[sub.layer == layer]
            ax_.scatter(d[wd], (d[c] - l).abs(), marker=mk, color="tab:blue", label=layer)
            for _, r in d.iterrows():
                lab = r["model"] + ("*" if r["model"] == SEL_NAME else "")
                ax_.annotate(lab, (r[wd], abs(r[c] - l)), fontsize=7, xytext=(3, 3),
                             textcoords="offset points")
        for wname, mk in zip(windows, ("o", "D", "P", "X", "v")):
            d = mixrows[mixrows.window == wname]
            ax_.scatter(d[wd], (d[c] - l).abs(), marker=mk, color="darkorange",
                        label=f"L4 mixture, {wname}")
            for _, r in d.iterrows():
                ax_.annotate(f"eta={r['eta']}", (r[wd], abs(r[c] - l)), fontsize=6,
                             xytext=(3, 3), textcoords="offset points")
        ax_.set_xlabel(f"mean interval width ({unit})")
        ax_.set_ylabel("coverage error |empirical - nominal|")
        ax_.set_title(f"{round(l*100)}% intervals (bottom-left is best; * = chosen on validation)")
        ax_.legend(fontsize=7)
    savefig("fig_coverage_error_vs_width.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    pit_single = np.array([stb["comps"][i][SEL_IDX].cdf(stb["y"][i]) for i in range(len(test_s))])
    for a_, pit, ttl in ((ax[0], mixture_pit(stb, BEST_ETA), f"mixture (eta={BEST_ETA})"),
                         (ax[1], pit_single, f"single PPD {SEL_NAME} (chosen on validation)")):
        a_.hist(pit, bins=10, range=(0, 1), density=True, color="steelblue", alpha=0.8)
        a_.axhline(1, color="red", ls="--"); a_.set_title("PIT: " + ttl)
    savefig("fig_pit.png")

    print(f"\nSaved tables and figures to ./{outdir}/")
    return dict(window=BEST_WINDOW, eta=BEST_ETA, sel_name=SEL_NAME, sel_idx=SEL_IDX,
                res=res, grid=grid, studies=studies, test_s=test_s, val_s=val_s)
