# -*- coding: utf-8 -*-
"""
Model Selection Agent
=====================

Amaç
-----
- analyzed_data (Close + opsiyonel indikatörler) üzerinden
  ARIMA / ARIMAX / ETS modellerini küçük bir validasyonla karşılaştırmak.
- Skorlamada birincil metrik RMSE (validation), eşitlikte AIC'yi kullanır.
- Seçimi ve tüm rakiplerin skorlarını döndürür.
- LangGraph ile uyumlu `model_selection_node` sağlar.

Girdi
-----
- analyzed_data: pd.DataFrame (DateTimeIndex, 'Close' zorunlu)
- exog_cols: List[str] | None  → ARIMAX için kullanılacak indikatörler
- val_ratio: float (örn. 0.1)
- ets_cfgs: ETS konfig listesi (örn. trend/sezon türleri)
- arima_orders: ARIMA için küçük order ızgarası
- arimax_orders: ARIMAX için küçük order ızgarası

Çıktı
-----
- selection: {
    "chosen": "ARIMA"|"ARIMAX"|"ETS",
    "scores": {
       "ARIMA": {"rmse": .., "aic": .., "order": (p,d,q)},
       "ARIMAX": {"rmse": .., "aic": .., "order": (p,d,q), "exog_cols":[...]},
       "ETS": {"rmse": .., "aic": .., "cfg": {...}}
    },
    "note": Optional[str]
  }
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd

from .forecast import (
    _ensure_series_close,
    _pick_exog,
    _train_val_split,
    _rmse,
    _fit_sarimax,
)
# ETS yalnızca burada (seçim için) gerekirse import edilir
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None  # type: ignore


def _best_arima(y: pd.Series,
                orders: List[Tuple[int,int,int]],
                val_ratio: float = 0.1) -> Dict[str, Any]:
    y_tr, y_val = _train_val_split(y, val_ratio)
    best = {"rmse": np.inf, "aic": np.inf, "order": None}
    for od in orders:
        try:
            res = _fit_sarimax(y_tr, od, exog_train=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = res.get_prediction(start=y_val.index[0], end=y_val.index[-1])
                yhat = pred.predicted_mean.values
            rmse = _rmse(y_val.values, yhat)
            aic = float(res.aic) if np.isfinite(res.aic) else np.inf
            if (rmse < best["rmse"] - 1e-9) or (np.isclose(rmse, best["rmse"]) and aic < best["aic"]):
                best = {"rmse": rmse, "aic": aic, "order": od}
        except Exception:
            continue
    return best


def _best_arimax(y: pd.Series,
                 exog: pd.DataFrame,
                 orders: List[Tuple[int,int,int]],
                 val_ratio: float = 0.1) -> Dict[str, Any]:
    y_tr, y_val = _train_val_split(y, val_ratio)
    ex_tr = exog.loc[y_tr.index]
    ex_va = exog.loc[y_val.index]

    best = {"rmse": np.inf, "aic": np.inf, "order": None}
    for od in orders:
        try:
            res = _fit_sarimax(y_tr, od, ex_tr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = res.get_prediction(start=y_val.index[0], end=y_val.index[-1], exog=ex_va)
                yhat = pred.predicted_mean.values
            rmse = _rmse(y_val.values, yhat)
            aic = float(res.aic) if np.isfinite(res.aic) else np.inf
            if (rmse < best["rmse"] - 1e-9) or (np.isclose(rmse, best["rmse"]) and aic < best["aic"]):
                best = {"rmse": rmse, "aic": aic, "order": od}
        except Exception:
            continue
    return best


def _best_ets(y: pd.Series,
              cfgs: List[Dict[str, Any]],
              val_ratio: float = 0.1) -> Dict[str, Any]:
    if ExponentialSmoothing is None:
        return {"rmse": np.inf, "aic": np.inf, "cfg": None, "note": "statsmodels ETS yok"}

    y_tr, y_val = _train_val_split(y, val_ratio)
    best = {"rmse": np.inf, "aic": np.inf, "cfg": None}

    for cfg in cfgs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                es = ExponentialSmoothing(
                    y_tr,
                    trend=cfg.get("trend"),
                    damped_trend=cfg.get("damped_trend", False),
                    seasonal=cfg.get("seasonal"),
                    seasonal_periods=cfg.get("seasonal_periods"),
                )
                res = es.fit(optimized=True, use_brute=True)
                # validasyon tahmini → in-sample forecast ile val kısmını çıkar
                yhat = res.predict(start=y_val.index[0], end=y_val.index[-1]).values
                rmse = _rmse(y_val.values, yhat)
                # ETS'te AIC doğrudan olmayabilir → yaklaşık BIC/AIC yerine SSE tabanlı
                aic = float(res.sse) if np.isfinite(res.sse) else np.inf

            if (rmse < best["rmse"] - 1e-9) or (np.isclose(rmse, best["rmse"]) and aic < best["aic"]):
                best = {"rmse": rmse, "aic": aic, "cfg": cfg}
        except Exception:
            continue
    return best


def select_model(analyzed_data: pd.DataFrame,
                 exog_cols: Optional[List[str]] = None,
                 *,
                 val_ratio: float = 0.1,
                 arima_orders: Optional[List[Tuple[int,int,int]]] = None,
                 arimax_orders: Optional[List[Tuple[int,int,int]]] = None,
                 ets_cfgs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    ARIMA / ARIMAX / ETS için en iyi konfigürasyonları bul, RMSE+AIC ile en iyiyi seç.

    Returns:
        {
          "chosen": "ARIMA"|"ARIMAX"|"ETS",
          "scores": {"ARIMA": {...}, "ARIMAX": {...}, "ETS": {...}},
          "note": Optional[str]
        }
    """
    y = _ensure_series_close(analyzed_data)

    # default küçük ızgaralar
    if arima_orders is None:
        arima_orders = [(1,0,1), (1,1,1), (2,1,2)]
    if arimax_orders is None:
        arimax_orders = [(1,1,1), (1,1,2)]
    if ets_cfgs is None:
        ets_cfgs = [
            {"trend": "add", "damped_trend": False, "seasonal": None, "seasonal_periods": None},
            {"trend": "add", "damped_trend": False, "seasonal": "add", "seasonal_periods": 7},
        ]

    scores: Dict[str, Dict[str, Any]] = {}

    # 1) ARIMA
    arima_best = _best_arima(y, arima_orders, val_ratio=val_ratio)
    scores["ARIMA"] = arima_best

    # 2) ARIMAX (exog varsa)
    exog = _pick_exog(analyzed_data, exog_cols) if exog_cols else None
    if exog is not None:
        arimax_best = _best_arimax(y, exog, arimax_orders, val_ratio=val_ratio)
        arimax_best["exog_cols"] = list(exog.columns)
    else:
        arimax_best = {"rmse": np.inf, "aic": np.inf, "order": None, "exog_cols": None, "note": "exog yok"}
    scores["ARIMAX"] = arimax_best

    # 3) ETS
    ets_best = _best_ets(y, ets_cfgs, val_ratio=val_ratio)
    scores["ETS"] = ets_best

    # Karar: min RMSE; eşitse min AIC
    def key_fn(item):
        name, sc = item
        return (sc.get("rmse", np.inf), sc.get("aic", np.inf))

    chosen = min(scores.items(), key=key_fn)[0]

    return {"chosen": chosen, "scores": scores, "note": None}


def model_selection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph düğümü gibi çalışır.

    Beklenen state:
        - analyzed_data: pd.DataFrame
        - exog_cols: list[str] | None (örn: ["RSI_14","EMA_12","SMA_14","MACD_line"])
        - val_ratio: float (vars: 0.1)

    Döner:
        {"model_choice": str, "model_scores": dict}
    """
    df = state.get("analyzed_data", None)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("model_selection_node: 'analyzed_data' boş/geçersiz.")
    exog_cols = state.get("exog_cols", None)
    val_ratio = float(state.get("val_ratio", 0.1))

    sel = select_model(df, exog_cols=exog_cols, val_ratio=val_ratio)
    return {"model_choice": sel["chosen"], "model_scores": sel["scores"]}

def _ms_series_len(x: Any) -> int:
    """pd.Series için uzunluk, diğer tipler için güvenli len()."""
    if pd is not None and isinstance(x, pd.Series):
        return int(len(x))
    try:
        return int(len(x))
    except Exception:
        return 0

def _score_from_meta(meta: Dict[str, Any], metric: str) -> Tuple[float, bool]:
    """
    Meta'dan skor çıkarır.
    Returns:
        (score, is_valid)
    """
    m = (metric or "rmse").lower()
    if m == "rmse":
        val = meta.get("select_rmse", meta.get("rmse", None))
    elif m == "aic":
        val = meta.get("aic", None)
    elif m == "bic":
        val = meta.get("bic", None)
    else:
        # bilinmeyen ölçütte rmse'ye düş
        val = meta.get("select_rmse", meta.get("rmse", None))
    if val is None:
        return (float("inf"), False)
    return (float(val), True)

def pick(forecasts: Dict[str, Any], metric: str = "rmse") -> Dict[str, Any]:
    """
    Router'ın beklediği seçim fonksiyonu.

    Args:
        forecasts (dict): forecast.make(...) çıktısı gibi bir sözlük:
            {
              "arima":  pd.Series,
              "arimax": pd.Series,
              "ets":    pd.Series,
              "_meta": {
                  "arima":  {...},  # aic/bic/select_rmse vb.
                  "arimax": {...},
                  "ets":    {...}
              }
            }
        metric (str): "rmse" (vars), "aic" veya "bic".

    Returns:
        dict: {
          "name": "<key>",          # seçilen model adı ("arima"/"arimax"/"ets")
          "metric": "<metric>",
          "score": float,           # seçime esas skor
          "meta": dict,             # modelin meta bilgisi
          "yhat": pd.Series,        # seçilen öngörü serisi
          "horizon": int            # adım sayısı
        }
    """
    if not isinstance(forecasts, dict) or not forecasts:
        return {}

    meta_all = forecasts.get("_meta", {}) or {}
    candidates = [k for k, v in forecasts.items() if not k.startswith("_") and v is not None]
    if not candidates:
        return {}

    best_name: Optional[str] = None
    best_score: float = float("inf")
    best_meta: Dict[str, Any] = {}
    best_series: Any = None

    for name in candidates:
        series = forecasts.get(name)
        meta = meta_all.get(name, {}) or {}
        score, valid = _score_from_meta(meta, metric)

        # meta yoksa/fake ise: deterministik bir fallback uygula
        # (uzun seri daha iyi kabul edilir; skor = 1/len)
        if not valid:
            score = 1.0 / max(1, _ms_series_len(series))

        if score < best_score:
            best_name, best_score, best_meta, best_series = name, score, meta, series

    return {
        "name": best_name,
        "metric": metric,
        "score": float(best_score),
        "meta": best_meta,
        "yhat": best_series,
        "horizon": _ms_series_len(best_series),
    }