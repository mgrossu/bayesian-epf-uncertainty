"""
epf_forecast.py -- rolling-origin forecasting engine and evaluation
-------------------------------------------------------------------
run_origin   for every target day s: fit all candidate models on the window
             [s-window, s) (or all past data if window is None), standardising
             features and target using that window only (no look-ahead), and
             store the layer 1-3 forecasts of every model.
mixture_*    layer 4 for any inverse temperature eta, computed from the store.
summarize    one table: every model x layers 1-3, plus the mixture (layer 4).
"""
import numpy as np
from scipy.special import logsumexp
from epf_core import fit_nig, predictive, tempered_logw, mixture_cdf, mixture_ppf


def run_origin(X, y, model_cols, window, s_values, lam0=0.01):
    """X: (n, p) raw features, y: (n,) prices, model_cols: list of column index arrays."""
    S, K = len(s_values), len(model_cols)
    st = dict(s=np.array(s_values), y=y[s_values], K=K,
              point=np.empty((S, K)), log_ev=np.empty((S, K)),
              logpdf=np.empty((S, K)), plug_logpdf=np.empty((S, K)),
              comps=[], plugs=[])
    for i, s in enumerate(s_values):
        lo = 0 if window is None else max(0, s - window)
        Xw, yw = X[lo:s], y[lo:s]
        mx, sx = Xw.mean(axis=0), Xw.std(axis=0)
        sx = np.where(sx < 1e-8, 1.0, sx)
        my, sy = yw.mean(), max(yw.std(), 1e-8)
        Z, zy, zs = (Xw - mx) / sx, (yw - my) / sy, (X[s] - mx) / sx
        comps, plugs = [], []
        for m, cols in enumerate(model_cols):
            D = np.column_stack([np.ones(len(Z)), Z[:, cols]])
            fit = fit_nig(D, zy, lam0=lam0)
            ppd, plug, pt = predictive(fit, np.concatenate([[1.0], zs[cols]]), my, sy)
            comps.append(ppd); plugs.append(plug)
            st["point"][i, m] = pt
            st["log_ev"][i, m] = fit["log_ev"]
            st["logpdf"][i, m] = ppd.logpdf(y[s])
            st["plug_logpdf"][i, m] = plug.logpdf(y[s])
        st["comps"].append(comps); st["plugs"].append(plugs)
    return st


# ---------------------------------------------------------------- layer 4
def mixture_logw(st, eta):
    return np.array([tempered_logw(le, eta) for le in st["log_ev"]])


def mixture_lps(st, eta):
    """Per-forecast log predictive score of the tempered mixture (logsumexp)."""
    return logsumexp(mixture_logw(st, eta) + st["logpdf"], axis=1)


def mixture_mean(st, eta):
    return (np.exp(mixture_logw(st, eta)) * st["point"]).sum(axis=1)


def mixture_interval(st, eta, level):
    W = np.exp(mixture_logw(st, eta))
    a = (1 - level) / 2
    lo = np.array([mixture_ppf(a, st["comps"][i], W[i]) for i in range(len(W))])
    hi = np.array([mixture_ppf(1 - a, st["comps"][i], W[i]) for i in range(len(W))])
    return lo, hi


def mixture_pit(st, eta):
    W = np.exp(mixture_logw(st, eta))
    return np.array([mixture_cdf(st["y"][i], st["comps"][i], W[i]) for i in range(len(W))])


# ------------------------------------------------------------ evaluation
def _cov_width(lo, hi, y):
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def summarize(st, etas, levels, model_names):
    """Rows: each model x (L1 point, L2 plug-in, L3 single PPD) and mixture L4 per eta."""
    y, K, rows = st["y"], st["K"], []
    for m in range(K):
        e = y - st["point"][:, m]
        rows.append(dict(model=model_names[m], layer="L1 point", eta="",
                         rmse=np.sqrt(np.mean(e ** 2)), mae=np.mean(np.abs(e))))
        for layer, dists, lp in (("L2 plug-in", "plugs", "plug_logpdf"),
                                 ("L3 single PPD", "comps", "logpdf")):
            row = dict(model=model_names[m], layer=layer, eta="", lps=st[lp][:, m].mean())
            for l in levels:
                a = (1 - l) / 2
                lo = np.array([d[m].ppf(a) for d in st[dists]])
                hi = np.array([d[m].ppf(1 - a) for d in st[dists]])
                row[f"cov{round(l*100)}"], row[f"width{round(l*100)}"] = _cov_width(lo, hi, y)
            rows.append(row)
    for eta in etas:
        e = y - mixture_mean(st, eta)
        row = dict(model="mixture", layer="L4 mixture PPD", eta=eta, lps=mixture_lps(st, eta).mean(),
                   rmse=np.sqrt(np.mean(e ** 2)), mae=np.mean(np.abs(e)))
        for l in levels:
            lo, hi = mixture_interval(st, eta, l)
            row[f"cov{round(l*100)}"], row[f"width{round(l*100)}"] = _cov_width(lo, hi, y)
        rows.append(row)
    return rows
