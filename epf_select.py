"""
epf_select.py -- feature ranking (Lasso path / OMP) and model sets
-------------------------------------------------------------------
There are many ways to select features and models; this is NOT the focus of the
thesis. The two selectors below only produce a RANKING of the fixed feature
pool. The candidate models are then the nested sets "top-k features".

Call these only with data from the train+validation part (never the test part).
"""
import numpy as np
from sklearn.linear_model import lars_path, orthogonal_mp


def _standardise(X, y):
    sx = X.std(axis=0)
    sx = np.where(sx < 1e-8, 1.0, sx)
    return (X - X.mean(axis=0)) / sx, y - y.mean()


def rank_lasso(X, y):
    """Order in which features enter the Lasso (LARS) path."""
    Z, yc = _standardise(X, y)
    _, _, coefs = lars_path(Z, yc, method="lasso")
    first = [np.argmax(coefs[j] != 0) if np.any(coefs[j] != 0) else np.inf
             for j in range(Z.shape[1])]
    return list(np.argsort(first, kind="stable"))


def rank_omp(X, y):
    """Order in which features are added by Orthogonal Matching Pursuit."""
    Z, yc = _standardise(X, y)
    order = []
    for k in range(1, Z.shape[1] + 1):
        support = np.flatnonzero(orthogonal_mp(Z, yc, n_nonzero_coefs=k))
        order += [int(j) for j in support if j not in order]
    return order


def nested_models(order, sizes):
    """Candidate models: the top-k ranked features for each k in `sizes`."""
    return [np.array(order[:k], dtype=int) for k in sizes]
