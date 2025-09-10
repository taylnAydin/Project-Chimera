"""
Forecast Agent (delegate to src.models)
=======================================

- ARIMA / ARIMAX / ETS tahminlerini src/models altındaki modüllere delege eder.
- Tek noktadan:
    - forecast_node(state)  → LangGraph uyumlu
    - make(df, cfg)         → Router adaptörü (çoklu model çıkışı)
    - evaluate_last_k_days  → Son K gün için rolling-origin gerçek vs tahmin tablosu

Beklenen model modülleri:
    src/models/arima.py   → forecast(df, horizon, order=None) -> (yhat, meta)
    src/models/arimax.py  → forecast(df, horizon, exog_cols=None, order=None) -> (yhat, meta)
    src/models/ets.py     → forecast(df, horizon) -> (yhat, meta)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models import arima as M_ARIMA, arimax as M_ARIMAX, ets as M_ETS
from src.models.common import ensure_series_close


# --------------------------------------------------------------------------- #
# İç yardımcılar
# --------------------------------------------------------------------------- #

def _run_single(
    df: pd.DataFrame,
    name: str,
    horizon: int,
    exog_cols: Optional[List[str]]
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Tek bir modeli koştur ve (yhat, meta) döndür.
    - name: "arima" | "arimax" | "ets"
    """
    m = name.lower()
    if m == "arima":
        return M_ARIMA.forecast(df, horizon=horizon, order=None)
    if m == "arimax":
        # exog yoksa modül zaten ARIMA'ya düşer
        return M_ARIMAX.forecast(df, horizon=horizon, exog_cols=exog_cols, order=None)
    if m == "ets":
        return M_ETS.forecast(df, horizon=horizon)
    raise ValueError(f"Bilinmeyen model: {name}")


def _choose_best(cands: List[Tuple[str, pd.Series, Dict[str, Any]]]) -> Tuple[str, pd.Series, Dict[str, Any]]:
    """
    Adaylar arasından 'en düşük RMSE (select_rmse/rmse), eşitlikte en düşük AIC' ile seç.
    """
    if not cands:
        raise RuntimeError("Seçim için aday model yok.")
    def key_fn(item):
        _, _, meta = item
        rmse = meta.get("select_rmse", meta.get("rmse", np.inf))
        aic  = meta.get("aic", np.inf)
        return (rmse, aic)
    return sorted(cands, key=key_fn)[0]


# --------------------------------------------------------------------------- #
# LangGraph düğümü
# --------------------------------------------------------------------------- #

def forecast_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Beklenen state:
        - analyzed_data: pd.DataFrame (Close zorunlu)
        - horizon: int (vars: 30)
        - model: "auto" | "ARIMA" | "ARIMAX" | "ETS" (vars: "auto")
        - exog_cols: list[str] | None (ARIMAX için)
    Döner:
        {"forecast_data": pd.Series, "forecast_meta": dict}
    """
    df = state.get("analyzed_data")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("forecast_node: 'analyzed_data' boş/geçersiz.")

    horizon   = int(state.get("horizon", 30))
    choice    = str(state.get("model", "auto")).lower()
    exog_cols = state.get("exog_cols")

    # Close kontrolü (erken hata)
    _ = ensure_series_close(df)

    if choice in ("arima", "arimax", "ets"):
        yhat, meta = _run_single(df, choice, horizon, exog_cols)
        return {"forecast_data": yhat, "forecast_meta": meta}

    # AUTO: arimax (varsa), arima, ets hepsini dener → en iyisini seç
    candidates: List[Tuple[str, pd.Series, Dict[str, Any]]] = []
    for name in ("arimax", "arima", "ets"):
        try:
            yhat, meta = _run_single(df, name, horizon, exog_cols)
            candidates.append((name.upper(), yhat, meta))
        except Exception:
            continue

    best_name, best_yhat, best_meta = _choose_best(candidates)
    best_meta["auto_competitors"] = [n for n, _, _ in candidates]
    return {"forecast_data": best_yhat, "forecast_meta": best_meta}


# --------------------------------------------------------------------------- #
# Router adaptörü (çoklu model çıktısı)
# --------------------------------------------------------------------------- #

def make(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router uyumluluk katmanı.
    Girdi cfg örneği:
        {
          "models": ["arima","arimax","ets"],
          "horizon": 7,
          "exog": {"cols": ["RSI_14","EMA_21","ATR_14"]}
        }
    Döner:
        {
          "arima":  pd.Series (ops.),
          "arimax": pd.Series (ops.),
          "ets":    pd.Series (ops.),
          "_meta": {
            "arima": {...},
            "arimax": {...},
            "ets": {...}
          }
        }
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    # Close kontrol/normalize
    _ = ensure_series_close(df)

    horizon   = int(cfg.get("horizon", 7))
    models    = [m.lower() for m in cfg.get("models", ["arima", "arimax", "ets"])]
    exog_cols = None
    if isinstance(cfg.get("exog"), dict):
        exog_cols = cfg["exog"].get("cols")

    out: Dict[str, Any] = {}
    for name in models:
        try:
            yhat, meta = _run_single(df, name, horizon, exog_cols)
            out[name] = yhat
            out.setdefault("_meta", {})[name] = meta
        except Exception:
            # bir model fail ederse diğerlerine devam et
            continue

    return out


# --------------------------------------------------------------------------- #
# Son K gün gerçek vs. tahmin (rolling-origin) değerlendirme
# --------------------------------------------------------------------------- #

def evaluate_last_k_days(
    df: pd.DataFrame,
    *,
    model_name: str,
    horizon: int = 7,
    exog_cols: Optional[List[str]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Son k gün için 1-adım ileri OUT-OF-SAMPLE tahmin değerlendirmesi üretir.

    Çalışma şekli:
      - Her gün t için: eğitim verisini [.., t-1] alır, seçili modeli horizon=1 ile fit eder,
        t günü kapanışı için tahmini üretir.
      - Gerçek (actual) ile tahmini (pred) yan yana raporlar, |hata|/% ve MAPE hesaplar.

    Returns
    -------
    {
      "rows": [{"date": "YYYY-MM-DD", "actual": float, "pred": float, "abs_pct_err": float}, ...],
      "mape": float,        # 0-1
      "accuracy": float     # 0-1  (1 - median(|err|/actual))
    }
    """
    # Güvenlik ve normalize
    y = ensure_series_close(df)  # Close doğrulaması
    dff = df.copy()
    dff.index = pd.to_datetime(dff.index)
    if not dff.index.is_monotonic_increasing:
        dff = dff.sort_index()

    # Son k gün listesi (gerçeklerle birlikte)
    idx = dff.index
    if len(idx) < k + 30:  # en azından biraz eğitim geçmişi olsun
        k = max(0, min(k, len(idx) - 1))
    dates_eval = list(idx[-k:]) if k > 0 else []

    rows: List[Dict[str, Any]] = []
    apes: List[float] = []

    # Model anahtarını normalize et
    m = (model_name or "ets").lower()
    m = "arimax" if m == "arima" and exog_cols else m  # exog varsa arimax kullanımı

    # Her gün için 1-adım ileri tahmin
    for dt in dates_eval:
        # Eğitim verisi t-1’e kadar
        train_end_pos = dff.index.get_loc(dt)
        if isinstance(train_end_pos, slice):
            # normalde get_loc int döner; slice dönerse solve et
            train_end_pos = int(np.where(dff.index == dt)[0][0])
        if train_end_pos < 1:
            continue
        train_df = dff.iloc[:train_end_pos]  # dt hariç

        # Modeli horizon=1 ile çalıştır
        try:
            yhat_1, meta_1 = _run_single(train_df, m, horizon=1, exog_cols=exog_cols)
        except Exception:
            continue

        # Tahmin serisinin dt gününü yakala (train_df’in bir gün sonrasını üretmiş olmalı)
        pred_val = None
        try:
            if dt in yhat_1.index:
                pred_val = float(yhat_1.loc[dt])
            else:
                # bazen index ileri tarih olabilir; en yakın ilk değeri al
                pred_val = float(yhat_1.iloc[0])
        except Exception:
            continue

        actual_val = float(dff.loc[dt, "Close"])
        if np.isfinite(actual_val) and np.isfinite(pred_val):
            ape = abs(pred_val - actual_val) / (abs(actual_val) if actual_val != 0 else 1.0)
            apes.append(ape)
            rows.append({
                "date": pd.to_datetime(dt).date().isoformat(),
                "actual": actual_val,
                "pred": pred_val,
                "abs_pct_err": ape
            })

    # Özet metrikler
    if rows:
        mape = float(np.mean(apes))
        accuracy = float(max(0.0, min(1.0, 1.0 - np.median(apes))))  # 1 - median APE, [0,1] kırp
    else:
        mape = float("nan")
        accuracy = float("nan")

    return {"rows": rows, "mape": mape, "accuracy": accuracy}