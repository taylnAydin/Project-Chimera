# src/models/arima.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from .common import ensure_series_close, train_val_split, rmse, fit_sarimax, future_index

def _grid_orders(max_p=2, max_d=1, max_q=2) -> List[Tuple[int,int,int]]:
    grid = []
    for p in range(max_p+1):
        for d in range(max_d+1):
            for q in range(max_q+1):
                if p==0 and d==0 and q==0:
                    continue
                grid.append((p,d,q))
    return grid

def select_order(y: pd.Series, orders: Optional[List[Tuple[int,int,int]]] = None, val_ratio: float=0.1) -> Dict[str, Any]:
    if orders is None:
        orders = _grid_orders()
    y_tr, y_val = train_val_split(y, val_ratio)
    best = {"order": None, "rmse": np.inf, "aic": np.inf}
    for od in orders:
        try:
            res = fit_sarimax(y_tr, od, None)
            yhat = res.get_prediction(start=y_val.index[0], end=y_val.index[-1]).predicted_mean.values
            r = rmse(y_val.values, yhat)
            a = float(res.aic) if np.isfinite(res.aic) else np.inf
            if (r < best["rmse"] - 1e-9) or (np.isclose(r, best["rmse"]) and a < best["aic"]):
                best = {"order": od, "rmse": r, "aic": a}
        except Exception:
            continue
    if best["order"] is None:
        best = {"order": (1,1,1), "rmse": np.inf, "aic": np.inf}
    return best

def forecast(df: pd.DataFrame, horizon: int, order: Optional[Tuple[int,int,int]] = None):
    y = ensure_series_close(df)
    if order is None:
        sel = select_order(y)
        order = sel["order"]
        sel_meta = {"select_rmse": sel["rmse"], "select_aic": sel["aic"]}
    else:
        sel_meta = {}
    res = fit_sarimax(y, order, None)
    idx_future = future_index(y.index[-1], horizon)
    fc = res.get_forecast(steps=horizon)
    yhat = pd.Series(fc.predicted_mean, index=idx_future, name="forecast")
    meta = {
        "model": "ARIMA",
        "order": tuple(order),
        "aic": float(res.aic) if np.isfinite(res.aic) else None,
        "bic": float(res.bic) if np.isfinite(res.bic) else None,
        "train_len": int(len(y)),
        "horizon": int(horizon),
        **sel_meta,
    }
    return yhat, meta