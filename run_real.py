"""
run_real.py -- real-data run (Germany, one delivery hour)
----------------------------------------------------------
    python run_real.py --csv DE.csv [--hour 18] [--selector lasso|omp] [--sizes 2,4,7,11]

Data-specific steps (load the CSV, rank features, model-size curve) followed by the
shared pipeline in epf_pipeline.py (tuning on validation, one run on the test part).

Chronological split:  [ train + validation : 80% ] [ test : 20%, the unseen future ]
Feature ranking uses train+validation only (the test part is never touched).
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epf_data import load_hour, split_point, FEATURES
from epf_select import rank_lasso, rank_omp, nested_models
from epf_forecast import run_origin
from epf_pipeline import run_pipeline

# ----------------------------------------------------------------- config
LEVELS = (0.8, 0.95)                      # nominal interval levels
ETAS = (1.0, 0.5, 0.25, 0.1, 0.0)         # inverse temperatures tried (0 = equal weights)
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

# ------------------------------------------------------------------ data
df = load_hour(args.csv, args.hour)
X, y = df[FEATURES].values, df["y"].values
n, p = X.shape
n_tv = split_point(n, TRAIN_FRAC)
print(f"hour {args.hour}: {n} days, {df.index[0].date()} .. {df.index[-1].date()}")
print(f"train+validation: {df.index[0].date()} .. {df.index[n_tv-1].date()} ({n_tv} days; "
      f"validation targets {n_tv - BURN_IN} from {df.index[BURN_IN].date()})")
print(f"test (unseen):    {df.index[n_tv].date()} .. {df.index[-1].date()} ({n - n_tv} days)")

# ------------------------------------------------------------- selection
orders = {"lasso": rank_lasso(X[:n_tv], y[:n_tv]), "omp": rank_omp(X[:n_tv], y[:n_tv])}
for k, o in orders.items():
    print(f"ranking by {k:<5}: " + " > ".join(FEATURES[j] for j in o))
order = orders[args.selector]

# ------------------------------------------- model-size curve (validation)
st = run_origin(X, y, nested_models(order, range(1, p + 1)), 364, list(range(BURN_IN, n_tv)))
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
plt.tight_layout(); plt.savefig(out("fig_model_size_curve.png"), dpi=150); plt.close()

# ----------------------------------------------------- shared pipeline
model_cols = nested_models(order, SIZES)
names = [f"top{k}" for k in SIZES]
r = run_pipeline(X, y, df.index, model_cols, names, n_tv, BURN_IN, WINDOWS, ETAS, LEVELS,
                 args.outdir, title=f"Hour {args.hour},", unit="EUR/MWh")

chosen = json.load(open(out("chosen_settings.json")))
chosen.update(hour=args.hour, selector=args.selector, order=[FEATURES[j] for j in order], sizes=SIZES)
json.dump(chosen, open(out("chosen_settings.json"), "w"), indent=2)
