# src/models/arimax.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from .common import ensure_series_close, pick_exog, train_val_split, rmse, fit_sarimax, future_index

def select_order(y: pd.Series, exog: pd.DataFrame,
                 orders: Optional[List[Tuple[int,int,int]]]=None, val_ratio: float=0.1) -> Dict[str, Any]:
    if orders is None:
        orders = [(1,1,1), (1,1,2), (2,1,2)]
    y_tr, y_val = train_val_split(y, val_ratio)
    ex_tr, ex_va = exog.loc[y_tr.index], exog.loc[y_val.index]
    best = {"order": None, "rmse": np.inf, "aic": np.inf}
    for od in orders:
        try:
            res = fit_sarimax(y_tr, od, ex_tr)
            yhat = res.get_prediction(start=y_val.index[0], end=y_val.index[-1], exog=ex_va).predicted_mean.values
            r = rmse(y_val.values, yhat)
            a = float(res.aic) if np.isfinite(res.aic) else np.inf
            if (r < best["rmse"] - 1e-9) or (np.isclose(r, best["rmse"]) and a < best["aic"]):
                best = {"order": od, "rmse": r, "aic": a}
        except Exception:
            continue
    return best

def forecast(df: pd.DataFrame, horizon: int, exog_cols: Optional[List[str]]=None, order: Optional[Tuple[int,int,int]]=None):
    y = ensure_series_close(df)
    ex = pick_exog(df, exog_cols)
    if ex is None:
        # exog yoksa ARIMA'ya düş
        from .arima import forecast as arima_forecast
        return arima_forecast(df, horizon, order=None)

    if order is None:
        sel = select_order(y, ex)
        order = sel["order"]
        sel_meta = {"select_rmse": sel["rmse"], "select_aic": sel["aic"]}
    else:
        sel_meta = {}

    res = fit_sarimax(y, order, ex)
    idx_future = future_index(y.index[-1], horizon)
    last_row = ex.iloc[[-1]].to_numpy().repeat(horizon, axis=0)
    ex_future = pd.DataFrame(last_row, index=idx_future, columns=ex.columns)
    fc = res.get_forecast(steps=horizon, exog=ex_future)
    yhat = pd.Series(fc.predicted_mean, index=idx_future, name="forecast")
    meta = {
        "model": "ARIMAX",
        "order": tuple(order),
        "aic": float(res.aic) if np.isfinite(res.aic) else None,
        "bic": float(res.bic) if np.isfinite(res.bic) else None,
        "train_len": int(len(y)),
        "horizon": int(horizon),
        "exog_cols": list(ex.columns),
        **sel_meta,
    }
    return yhat, meta