# src/models/ets.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .common import ensure_series_close, future_index

def forecast(df: pd.DataFrame, horizon: int):
    y = ensure_series_close(df)
    if len(y) < 20:
        raise ValueError("ETS için örnek sayısı yetersiz (<20).")

    cfg_candidates = [
        dict(trend="add", damped_trend=False, seasonal="add", seasonal_periods=7),
        dict(trend="add", damped_trend=False, seasonal=None, seasonal_periods=None),
    ]

    last_err = None
    for cfg in cfg_candidates:
        try:
            model = ExponentialSmoothing(
                y,
                trend=cfg["trend"],
                damped_trend=cfg["damped_trend"],
                seasonal=cfg["seasonal"],
                seasonal_periods=cfg["seasonal_periods"],
                initialization_method="estimated",
            )
            res = model.fit(optimized=True, use_brute=False)
            yhat = res.forecast(horizon)
            if np.isnan(yhat).any():
                raise RuntimeError("ETS forecast produced NaN.")
            idx_future = future_index(y.index[-1], horizon)
            yhat = pd.Series(yhat.values, index=idx_future, name="forecast")
            meta = {"model": "ETS", "cfg": cfg, "train_len": int(len(y)), "horizon": int(horizon)}
            return yhat, meta
        except Exception as e:
            last_err = e
            continue

    # fallback
    model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
    res = model.fit(optimized=True)
    yhat = res.forecast(horizon)
    idx_future = future_index(y.index[-1], horizon)
    yhat = pd.Series(yhat.values, index=idx_future, name="forecast")
    meta = {
        "model": "ETS",
        "cfg": {"trend": "add", "seasonal": None, "damped_trend": False, "seasonal_periods": None},
        "train_len": int(len(y)), "horizon": int(horizon),
        "note": f"fallback used ({type(last_err).__name__ if last_err else None})",
    }
    return yhat, meta