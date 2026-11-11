"""
epf_core.py -- Bayesian machinery shared by the synthetic and real-data pipelines
---------------------------------------------------------------------------------
Conjugate Normal-Inverse-Gamma (NIG) linear regression:

    y | beta, s2 ~ N(X beta, s2 I),  beta | s2 ~ N(0, s2 / lam0 * I),  s2 ~ InvGamma(a0, b0)

  * fit_nig       closed-form posterior + log model evidence
  * predictive    frozen scipy distributions for a new input x*
                  (layer 2 plug-in Gaussian, layer 3 Student-t posterior predictive)
  * tempered_logw tempered BMA weights  w_k ~ exp(eta * log evidence_k)
  * mixture_cdf / mixture_ppf  mixture of frozen distributions

The target is standardised inside each window by the caller; `predictive` maps
the distributions back to the original price scale (so no Jacobian is needed).
"""
import numpy as np
from scipy.stats import t as student_t, norm
from scipy.special import gammaln, logsumexp
from scipy.optimize import brentq


def fit_nig(X, y, lam0=0.01, a0=1.0, b0=1.0):
    """Posterior of the NIG model (prior mean 0) and the log evidence."""
    n, p = X.shape
    Lam_n = lam0 * np.eye(p) + X.T @ X
    rhs = X.T @ y
    mu_n = np.linalg.solve(Lam_n, rhs)                  # solve, not inv
    a_n = a0 + n / 2
    b_n = b0 + 0.5 * (y @ y - mu_n @ rhs)
    logdet_0 = p * np.log(lam0)
    _, logdet_n = np.linalg.slogdet(Lam_n)
    log_ev = (-0.5 * n * np.log(2 * np.pi) + 0.5 * (logdet_0 - logdet_n)
              + a0 * np.log(b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(a0))
    return dict(mu=mu_n, Lam=Lam_n, a=a_n, b=b_n, log_ev=log_ev)


def predictive(fit, x, y_mean=0.0, y_scale=1.0):
    """Distributions for y* given the (standardised) design row x.

    Returns (ppd, plug_in, point):
      ppd     Student-t posterior predictive  (layer 3)
      plug_in Gaussian with plug-in posterior-mean variance  (layer 2)
      point   posterior-mean prediction  (layer 1)
    """
    v = np.linalg.solve(fit["Lam"], x)
    loc = x @ fit["mu"]
    scale = np.sqrt(fit["b"] / fit["a"] * (1.0 + x @ v))
    point = y_mean + y_scale * loc
    ppd = student_t(df=2 * fit["a"], loc=point, scale=y_scale * scale)
    plug = norm(point, y_scale * np.sqrt(fit["b"] / (fit["a"] - 1.0)))
    return ppd, plug, point


def tempered_logw(log_ev, eta):
    """log of softmax(eta * log_ev), stable via logsumexp. eta=1: standard BMA."""
    v = eta * np.asarray(log_ev, dtype=float)
    return v - logsumexp(v)


def mixture_cdf(v, comps, w):
    return sum(wi * c.cdf(v) for wi, c in zip(w, comps))


def mixture_ppf(q, comps, w):
    """Quantile of a mixture of frozen distributions (brentq, automatic bracket)."""
    qs = [c.ppf(q) for c in comps]
    lo, hi = min(qs), max(qs)          # F(lo) <= q <= F(hi) by construction
    if hi - lo < 1e-12:
        return lo
    f = lambda v: mixture_cdf(v, comps, w) - q
    f_lo, f_hi = f(lo), f(hi)
    if f_lo >= 0:                      # one model carries ~all weight
        return lo
    if f_hi <= 0:
        return hi
    return brentq(f, lo, hi, xtol=1e-9)
