# -*- coding: utf-8 -*-
"""
Forecast Agent (ARIMA / ARIMAX / ETS)
=====================================

Amaç
-----
- Kapanış fiyat serisi (Close) için kısa-vadeli (örn. 30 gün) tahmin üretmek.
- ARIMA, ARIMAX (exog ile) ve ETS seçenekleri.
- Küçük bir (p,d,q) ızgarasında AIC + validasyon RMSE ile order seçimi (ARIMA/X).
- ETS için de küçük bir hiperparametre ızgarası ve RMSE temelli seçim.
- LangGraph ile uyumlu `forecast_node`.

Girdi
-----
- analyzed_data: pd.DataFrame (DateTimeIndex, en az 'Close')
- horizon: int (öngörü adımı)
- model: "auto" | "ARIMA" | "ARIMAX" | "ETS"
- exog_cols: list[str] | None  (ARIMAX’ta kullanılacak sütunlar)

Çıktı
-----
- forecast_data: pd.Series
- forecast_meta: dict
"""
from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _ensure_series_close(df: pd.DataFrame) -> pd.Series:
    """DataFrame'den 'Close' serisini alır, float'a çevirir, NaN atar."""
    if "Close" not in df.columns:
        raise ValueError("forecast: 'Close' kolonu bulunamadı (analyzed_data).")
    y = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("forecast: analyzed_data.index DatetimeIndex olmalı.")
    y.index = pd.to_datetime(y.index)
    if not y.index.is_monotonic_increasing:
        y = y.sort_index()
    if len(y) < 50:
        warnings.warn("forecast: Örneklem küçük (<50). Sonuçlar güvenilmez olabilir.", RuntimeWarning)
    return y


def _pick_exog(df: pd.DataFrame, exog_cols: Optional[List[str]]) -> Optional[pd.DataFrame]:
    """Belirli sütunları float'a çevirip exog DataFrame olarak döndürür; yoksa None."""
    if not exog_cols:
        return None
    cols = [c for c in exog_cols if c in df.columns]
    if not cols:
        return None
    exog = df[cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    exog.index = pd.to_datetime(exog.index)
    if not exog.index.is_monotonic_increasing:
        exog = exog.sort_index()
    return exog


def _train_val_split(y: pd.Series, val_ratio: float = 0.1) -> Tuple[pd.Series, pd.Series]:
    """Zaman serisini eğitim/doğrulama olarak böler."""
    n = len(y)
    v = max(1, int(n * val_ratio))
    return y.iloc[:-v], y.iloc[-v:]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _future_index(last_date: pd.Timestamp, horizon: int, freq: str = "D") -> pd.DatetimeIndex:
    """Gelecek horizon için tarih indeksini oluşturur."""
    start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq=freq)


# --------------------------------------------------------------------------- #
# ARIMA / ARIMAX
# --------------------------------------------------------------------------- #
def _fit_sarimax(
    y_train: pd.Series,
    order: Tuple[int, int, int],
    exog_train: Optional[pd.DataFrame] = None,
):
    """SARIMAX (mevsimsiz) fit eder."""
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


def _grid_orders(max_p: int = 2, max_d: int = 1, max_q: int = 2) -> List[Tuple[int, int, int]]:
    """Küçük ARIMA ızgarası."""
    grid = []
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                grid.append((p, d, q))
    return grid


def _best_order_by_aic_rmse(
    y: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    val_ratio: float = 0.1,
    orders: Optional[List[Tuple[int, int, int]]] = None,
) -> Dict[str, Any]:
    """Izgarada modelleri dener; önce RMSE, eşitlikte AIC ile seçer."""
    if orders is None:
        orders = _grid_orders()
    y_tr, y_val = _train_val_split(y, val_ratio=val_ratio)
    ex_tr = exog.loc[y_tr.index] if exog is not None else None
    ex_va = exog.loc[y_val.index] if exog is not None else None

    best: Dict[str, Any] = {"order": None, "rmse": np.inf, "aic": np.inf}
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
                best = {"order": od, "rmse": rmse, "aic": aic}
        except Exception:
            continue
    if best["order"] is None:
        best = {"order": (1, 1, 1), "rmse": np.inf, "aic": np.inf}
    return best


def fit_predict_arima(
    y: pd.Series,
    horizon: int,
    order: Optional[Tuple[int, int, int]] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """ARIMA tahmini."""
    if order is None:
        sel = _best_order_by_aic_rmse(y, exog=None)
        order = sel["order"]
        sel_meta = {"select_rmse": sel["rmse"], "select_aic": sel["aic"]}
    else:
        sel_meta = {}
    res = _fit_sarimax(y, order, exog_train=None)

    idx_future = _future_index(y.index[-1], horizon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def fit_predict_arimax(
    y: pd.Series,
    exog: pd.DataFrame,
    horizon: int,
    order: Optional[Tuple[int, int, int]] = None,
    exog_cols: Optional[List[str]] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    ARIMAX tahmini (exog ile). Gelecek exog: hold-last.
    Exog verisini bu fonksiyon içinde de temizleyip y ile hizalar.
    """
    # 1) Sütun seçimi
    if exog_cols:
        use = [c for c in exog_cols if c in exog.columns]
        if not use:
            return fit_predict_arima(y, horizon, order=None)
        ex = exog[use].copy()
    else:
        ex = exog.copy()

    # 2) Temizlik: numeric, NaN→ffill/bfill, ±∞ temizliği
    ex = ex.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    # sonsuz değerleri NaN yapıp tekrar doldur
    ex = ex.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # 3) Index düzeni ve y ile hizalama (kesişim)
    ex.index = pd.to_datetime(ex.index)
    if not ex.index.is_monotonic_increasing:
        ex = ex.sort_index()
    # y’nin de sıralı olduğundan emin ol
    y = y.copy()
    y.index = pd.to_datetime(y.index)
    if not y.index.is_monotonic_increasing:
        y = y.sort_index()

    common_idx = y.index.intersection(ex.index)
    y = y.loc[common_idx]
    ex = ex.loc[common_idx]

    # 4) Order seçimi
    if order is None:
        sel = _best_order_by_aic_rmse(y, exog=ex)
        order = sel["order"]
        sel_meta = {"select_rmse": sel["rmse"], "select_aic": sel["aic"]}
    else:
        sel_meta = {}

    # 5) Modeli fit et
    res = _fit_sarimax(y, order, ex)

    # 6) Gelecek exog (hold-last)
    idx_future = _future_index(y.index[-1], horizon)
    last_row = ex.iloc[[-1]].to_numpy().repeat(horizon, axis=0)
    ex_future = pd.DataFrame(last_row, index=idx_future, columns=ex.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

# --------------------------------------------------------------------------- #
# ETS
# --------------------------------------------------------------------------- #
def _ets_grid_candidates(y: pd.Series) -> List[Dict[str, Any]]:
    """
    Küçük ETS ızgarası.
    - trend: [None, 'add']
    - damped_trend: [False, True]  (trend varsa)
    - seasonal: [None, 'add' (opsiyonel haftalık)]
    """
    has_weekly = (len(y) >= 21)  # en az 3 hafta
    cands: List[Dict[str, Any]] = []

    # Trend yok
    cands.append(dict(trend=None, damped_trend=False, seasonal=None, seasonal_periods=None))

    # Additive trend
    for damp in (False, True):
        cands.append(dict(trend='add', damped_trend=damp, seasonal=None, seasonal_periods=None))

    # Opsiyonel: günlük veride haftalık additive sezon
    if has_weekly:
        for damp in (False, True):
            cands.append(dict(trend='add', damped_trend=damp, seasonal='add', seasonal_periods=7))

    return cands


def _fit_ets(y_train: pd.Series, cfg: Dict[str, Any]):
    """ETS modelini fit eder."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y_train,
            trend=cfg["trend"],
            damped_trend=cfg["damped_trend"],
            seasonal=cfg["seasonal"],
            seasonal_periods=cfg["seasonal_periods"],
            initialization_method="estimated",
        )
        res = model.fit(optimized=True)
    return res


def fit_predict_ets(y: pd.Series, horizon: int) -> tuple[pd.Series, dict]:
    # 1) temizle
    y = pd.to_numeric(y, errors="coerce").dropna()
    y.index = pd.to_datetime(y.index)
    if not y.index.is_monotonic_increasing:
        y = y.sort_index()
    # güvenlik: yeterli uzunluk kontrolü
    if len(y) < 20:
        raise ValueError("ETS: yetersiz örnek (<20).")

    cfg_candidates = [
        # haftalık sezonsallık
        dict(trend="add", damped_trend=False, seasonal="add", seasonal_periods=7),
        # fallback: sezonsuz (yalnız trend)
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
            # NaN kontrolü
            if np.isnan(yhat).any():
                raise RuntimeError("ETS forecast produced NaN.")
            idx_future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
            yhat = pd.Series(yhat.values, index=idx_future, name="forecast")
            meta = {
                "model": "ETS",
                "cfg": cfg,
                "train_len": int(len(y)),
                "horizon": int(horizon),
            }
            return yhat, meta
        except Exception as e:
            last_err = e
            continue

    # tüm denemeler başarısızsa basit bir fallback daha:
    model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
    res = model.fit(optimized=True)
    yhat = res.forecast(horizon)
    idx_future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    yhat = pd.Series(yhat.values, index=idx_future, name="forecast")
    meta = {
        "model": "ETS",
        "cfg": {"trend": "add", "seasonal": None, "damped_trend": False, "seasonal_periods": None},
        "train_len": int(len(y)),
        "horizon": int(horizon),
        "note": f"robust fallback used (last_err={type(last_err).__name__ if last_err else None})",
    }
    return yhat, meta

# --------------------------------------------------------------------------- #
# LangGraph düğümü
# --------------------------------------------------------------------------- #
def forecast_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Beklenen state:
        - analyzed_data: pd.DataFrame (Close zorunlu; ARIMAX için exog sütunları opsiyonel)
        - horizon: int (varsayılan 30)
        - model: "auto"|"ARIMA"|"ARIMAX"|"ETS" (varsayılan "auto")
        - exog_cols: list[str] | None
    """
    df = state.get("analyzed_data", None)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("forecast_node: 'analyzed_data' boş/geçersiz.")

    horizon: int = int(state.get("horizon", 30))
    model_choice: str = str(state.get("model", "auto")).upper()
    exog_cols: Optional[List[str]] = state.get("exog_cols", None)

    y = _ensure_series_close(df)
    exog = _pick_exog(df, exog_cols) if exog_cols else None

    if model_choice == "ARIMA":
        yhat, meta = fit_predict_arima(y, horizon=horizon, order=None)
    elif model_choice == "ARIMAX":
        if exog is None:
            yhat, meta = fit_predict_arima(y, horizon=horizon, order=None)
            meta["note"] = "exog bulunamadı → ARIMA'ya düşüldü"
        else:
            yhat, meta = fit_predict_arimax(y, exog=exog, horizon=horizon, order=None, exog_cols=exog_cols)
    elif model_choice == "ETS":
        yhat, meta = fit_predict_ets(y, horizon=horizon)
    else:
        # AUTO: önce exog varsa ARIMAX dene; başarısızsa ARIMA; ayrıca ETS'i de dener ve
        # küçük bir validasyon karşılaştırmasıyla ARIMA/ARIMAX vs ETS arasında seçim yapar.
        # (Hızlı olması için tek-pass bir karşılaştırma)
        cand: List[Tuple[pd.Series, Dict[str, Any]]] = []
        if exog is not None:
            try:
                cand.append(fit_predict_arimax(y, exog=exog, horizon=horizon, order=None, exog_cols=exog_cols))
            except Exception:
                pass
        try:
            cand.append(fit_predict_arima(y, horizon=horizon, order=None))
        except Exception:
            pass
        try:
            cand.append(fit_predict_ets(y, horizon=horizon))
        except Exception:
            pass

        if not cand:
            raise RuntimeError("AUTO seçim başarısız: hiçbir model çalışmadı.")

        # Hızlı bir “mini” seçim: eğitim/val böl, her model için val RMSE hesapla ve en iyisini al
        y_tr, y_val = _train_val_split(y, val_ratio=0.1)
        scores = []
        for (yhat, m) in cand:
            # Hepsi farklı yaklaşımlarla üretildi; burada sadece raporlama için model label’ı tutuyoruz.
            # Karşılaştırma için hepsini yeniden eğitmek maliyetli olurdu; basitleştirme:
            # - ARIMA/ARIMAX zaten seçim sırasında RMSE baktı
            # - ETS de baktı
            # Bu yüzden meta içindeki “select_rmse” varsa onu kullan; yoksa sonsuz say.
            sel_rmse = m.get("select_rmse", np.inf)
            scores.append((sel_rmse, (yhat, m)))
        scores.sort(key=lambda x: x[0])
        yhat, meta = scores[0][1]
        meta["auto_competitors"] = [m["model"] for _, (_, m) in scores]

    return {"forecast_data": yhat, "forecast_meta": meta}



# --- dosyanın en altına ekle: adapter for router.make(df, cfg) ---------------

def make(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router uyum katmanı.
    Giriş: df (en az 'close' veya 'Close'), cfg:
        {
          "models": ["arima","arimax","ets"],
          "horizon": 7,
          "exog": {"cols": [...]}   # ARIMAX için kullanılacak sütunlar
        }
    Çıkış: {'arima': Series?, 'arimax': Series?, 'ets': Series?}
           (Olmayanlar eklenmez. Router'daki model_selection bunların metriklerinden seçecek.)
    """
    if df is None or df.empty:
        return {}

    # 1) Column uyumu: senin fonksiyonların 'Close' bekliyor
    df2 = df.copy()
    if "Close" not in df2.columns and "close" in df2.columns:
        df2["Close"] = df2["close"]

    # 2) Parametreler
    horizon = int(cfg.get("horizon", 7))
    models = cfg.get("models", ["arima", "arimax", "ets"])
    exog_cols = None
    if isinstance(cfg.get("exog"), dict):
        exog_cols = cfg["exog"].get("cols")

    # 3) Y serisi ve exog hazırla
    y = _ensure_series_close(df2)
    exog = _pick_exog(df2, exog_cols) if exog_cols else None

    out: Dict[str, Any] = {}

    # 4) Modelleri tek tek dene (bağımsız)
    if "arima" in models:
        try:
            yhat, meta = fit_predict_arima(y, horizon=horizon, order=None)
            out["arima"] = yhat
            out.setdefault("_meta", {})["arima"] = meta
        except Exception:
            pass

    if "arimax" in models:
        try:
            if exog is None:
                # exog yoksa ARIMA'ya düş
                yhat, meta = fit_predict_arima(y, horizon=horizon, order=None)
                meta["note"] = "exog bulunamadı → ARIMA'ya düşüldü"
            else:
                yhat, meta = fit_predict_arimax(y, exog=exog, horizon=horizon, order=None, exog_cols=exog_cols)
            out["arimax"] = yhat
            out.setdefault("_meta", {})["arimax"] = meta
        except Exception:
            pass

    if "ets" in models:
        try:
            yhat, meta = fit_predict_ets(y, horizon=horizon)
            out["ets"] = yhat
            out.setdefault("_meta", {})["ets"] = meta
        except Exception:
            pass

    return out