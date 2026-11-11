"""
epf_data.py -- load the EPF-toolbox CSV and build the daily feature table
-------------------------------------------------------------------------
One row per delivery day d for ONE delivery hour h. Everything used as a
feature is known when the day-ahead auction for day d is run:

  p_lag1/2/3/7   price at hour h on days d-1, d-2, d-3, d-7
  p_min_lag1     minimum price over the 24 hours of day d-1
  p_max_lag1     maximum price over the 24 hours of day d-1
  load           day-ahead load forecast for day d, hour h
  pv_wind        day-ahead PV+wind forecast for day d, hour h
  mon, sat, sun  weekday dummies

The feature pool is deliberately small and FIXED (see the thesis scope).
"""
import numpy as np
import pandas as pd

FEATURES = ["p_lag1", "p_lag2", "p_lag3", "p_lag7", "p_min_lag1", "p_max_lag1",
            "load", "pv_wind", "mon", "sat", "sun"]


def load_hour(csv_path, hour=18):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.columns = ["price", "load", "pv_wind"]
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour
    P = df.pivot(index="date", columns="hour", values="price")
    L = df.pivot(index="date", columns="hour", values="load")
    R = df.pivot(index="date", columns="hour", values="pv_wind")
    if P.isna().any().any():
        raise ValueError("missing hourly values: the CSV is not a complete 24h grid")

    out = pd.DataFrame(index=P.index)
    out["y"] = P[hour]
    for k in (1, 2, 3, 7):
        out[f"p_lag{k}"] = P[hour].shift(k)
    out["p_min_lag1"] = P.min(axis=1).shift(1)
    out["p_max_lag1"] = P.max(axis=1).shift(1)
    out["load"] = L[hour]
    out["pv_wind"] = R[hour]
    dow = out.index.dayofweek
    out["mon"] = (dow == 0).astype(float)
    out["sat"] = (dow == 5).astype(float)
    out["sun"] = (dow == 6).astype(float)
    return out.dropna()[["y"] + FEATURES]


def split_point(n, frac=0.8):
    """Index where the chronological test part (the unseen future) starts."""
    return int(frac * n)
