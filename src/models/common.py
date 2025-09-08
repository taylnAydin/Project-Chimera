# src/models/common.py
from __future__ import annotations
from typing import Tuple, Optional, List
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def ensure_series_close(df: pd.DataFrame) -> pd.Series:
    # close -> Close normalizasyonu
    if "Close" not in df.columns and "close" in df.columns:
        df = df.copy()
        df["Close"] = df["close"]
    if "Close" not in df.columns:
        raise ValueError("'Close' kolonu yok.")

    y = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("index DatetimeIndex olmalı.")
    y.index = pd.to_datetime(y.index)
    if not y.index.is_monotonic_increasing:
        y = y.sort_index()
    if len(y) < 50:
        warnings.warn("Örneklem küçük (<50). Sonuçlar güvenilmez olabilir.", RuntimeWarning)
    return y

def pick_exog(df: pd.DataFrame, exog_cols: Optional[List[str]]) -> Optional[pd.DataFrame]:
    if not exog_cols:
        return None
    cols = [c for c in exog_cols if c in df.columns]
    if not cols:
        return None
    ex = df[cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    ex.index = pd.to_datetime(ex.index)
    if not ex.index.is_monotonic_increasing:
        ex = ex.sort_index()
    return ex

def train_val_split(y: pd.Series, val_ratio: float = 0.1) -> Tuple[pd.Series, pd.Series]:
    n = len(y)
    v = max(1, int(n * val_ratio))
    return y.iloc[:-v], y.iloc[-v:]

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def future_index(last_date: pd.Timestamp, horizon: int, freq: str = "D") -> pd.DatetimeIndex:
    start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq=freq)

def fit_sarimax(y_train: pd.Series, order: Tuple[int,int,int], exog_train: Optional[pd.DataFrame]=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog=y_train,
            exog=exog_train,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend=None,
        )
        res = model.fit(disp=False)
    return res