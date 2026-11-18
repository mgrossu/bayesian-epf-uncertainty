"""
epf_synth.py -- synthetic data from a mixture of three linear models
----------------------------------------------------------------------
At every time step ONE component generates the observation. Which one is random,
with probabilities (the generating weights) that move smoothly and periodically
over time, so each component is "the most true" in a different period:

    w_k(t)  proportional to  exp( kappa * cos( 2 pi (t - c_k) / period ) )

The components use different predictors and have different noise levels, so some
periods have broad predictive distributions and others narrow ones. The candidate
models fitted later have the same structure as the components (intercept + one
predictor), but no single candidate is the generating process: that is a
mixture. The Bayesian model average is therefore a realistic model to try BMA
on, not the exact generating model.
"""
import numpy as np

COMPONENTS = [
    dict(name="M1", feat=0, intercept=1.0,  beta=2.0,  sigma=0.5),   # low noise
    dict(name="M2", feat=1, intercept=-1.0, beta=-2.5, sigma=2.0),   # high noise
    dict(name="M3", feat=2, intercept=0.5,  beta=1.5,  sigma=1.0),   # medium noise
]


def generate(n=1200, period=400.0, kappa=3.0, seed=7):
    rng = np.random.default_rng(seed)
    K = len(COMPONENTS)
    t = np.arange(n)
    X = rng.normal(size=(n, K))                      # three independent predictors
    centres = period * np.arange(K) / K
    raw = np.exp(kappa * np.cos(2 * np.pi * (t[:, None] - centres[None, :]) / period))
    W = raw / raw.sum(axis=1, keepdims=True)         # generating weights
    z = np.array([rng.choice(K, p=W[i]) for i in t])  # active component per step
    y = np.empty(n)
    for i in t:
        c = COMPONENTS[z[i]]
        y[i] = c["intercept"] + c["beta"] * X[i, c["feat"]] + c["sigma"] * rng.normal()
    return dict(X=X, y=y, z=z, W=W, components=COMPONENTS, t=t)
