# agents/anomaly.py
"""
Chimera • Anomaly Agent (Python 3.13 uyumlu)
- Seride (ve tahmin kalıntılarında) olağandışı hareketleri tespit eder.
- Yöntemler:
  1) Fiyat getirisi z-score (rolling) ile spike/crash
  2) Forecast residual z-score (varsa)
  3) Hacim (volume) spike (opsiyonel)

Girdi:
- df: pandas.DataFrame (en az 'close' kolonu; opsiyonel 'volume')
- forecasts: {'y_hat': pandas.Series} (df ile hizalı tahmin serisi) opsiyonel
- cfg: AnomalyConfig (eşikler/ayarlar)

Çıktı:
- List[AnomalyEvent]: {'ts','type','z','value','residual','threshold',...}
"""

from __future__ import annotations
from typing import TypedDict, Literal, List, Dict, Any, Optional
import math

try:
    import pandas as pd  # type: ignore
    import numpy as np   # type: ignore
except Exception:
    pd = None  # type: ignore
    np = None  # type: ignore


# =========================
# Tipler
# =========================

AnomalyType = Literal["price_spike", "price_crash", "residual_spike", "volume_spike"]

class AnomalyEvent(TypedDict, total=False):
    ts: str
    type: AnomalyType
    z: float
    value: float
    residual: float
    threshold: float
    extra: Dict[str, Any]

class AnomalyConfig(TypedDict, total=False):
    # Fiyat getirisi (return) z-score
    window: int                # rolling pencere (default: 30)
    z_threshold: float         # fiyat için z eşiği (default: 3.0)
    use_robust: bool           # robust z-score (MAD) (default: True)
    min_obs: int               # minimum gözlem (default: 20)

    # Residual (forecast - actual)
    residual_check: bool       # residual analizi yap (default: True)
    residual_window: int       # rolling pencere (default: 30)
    residual_z_threshold: float# residual için z eşiği (default: 3.0)
    residual_use_robust: bool  # robust z-score (default: True)

    # Hacim spike
    volume_check: bool         # hacim analizi yap (default: False)
    volume_window: int         # rolling pencere (default: 30)
    volume_z_threshold: float  # hacim için z eşiği (default: 3.5)
    volume_use_robust: bool    # robust z-score (default: True)


# =========================
# Varsayılanlar
# =========================

_DEF: AnomalyConfig = {
    "window": 30,
    "z_threshold": 3.0,
    "use_robust": True,
    "min_obs": 20,
    "residual_check": True,
    "residual_window": 30,
    "residual_z_threshold": 3.0,
    "residual_use_robust": True,
    "volume_check": False,
    "volume_window": 30,
    "volume_z_threshold": 3.5,
    "volume_use_robust": True,
}


# =========================
# Yardımcılar (tip güvenli: Any)
# =========================

def _as_utc_iso(ts: Any) -> str:
    if pd is None:
        return str(ts)
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.isoformat()
    except Exception:
        return str(ts)

def _series_returns(close: Any) -> Any:
    # pd yoksa None döndür
    if pd is None or close is None or not hasattr(close, "pct_change"):
        return None
    try:
        return close.pct_change()
    except Exception:
        return None

def _rolling_z(x: Any, win: int, robust: bool) -> Any:
    if pd is None or np is None or x is None:
        return None
    try:
        minp = max(5, win // 2)
        if robust:
            med = x.rolling(win, min_periods=minp).median()
            mad = (x - med).abs().rolling(win, min_periods=minp).median()
            z = 0.6745 * (x - med) / mad.replace(0, np.nan)
        else:
            mu = x.rolling(win, min_periods=minp).mean()
            sd = x.rolling(win, min_periods=minp).std(ddof=0)
            z = (x - mu) / sd.replace(0, np.nan)
        return z
    except Exception:
        return None

def _add_events_from_z(zs: Any,
                       base_series: Any,
                       pos_threshold: float,
                       neg_threshold: float,
                       pos_type: AnomalyType,
                       neg_type: AnomalyType) -> List[AnomalyEvent]:
    events: List[AnomalyEvent] = []
    if pd is None or zs is None:
        return events
    try:
        nz = zs.dropna()
    except Exception:
        return events
    for ts, z in nz.items():
        try:
            if float(z) >= pos_threshold:
                val = float(base_series.loc[ts]) if ts in getattr(base_series, "index", []) else float("nan")
                events.append(AnomalyEvent(ts=_as_utc_iso(ts), type=pos_type, z=float(z), value=val))
            elif float(z) <= -abs(neg_threshold):
                val = float(base_series.loc[ts]) if ts in getattr(base_series, "index", []) else float("nan")
                events.append(AnomalyEvent(ts=_as_utc_iso(ts), type=neg_type, z=float(z), value=val))
        except Exception:
            continue
    return events


# =========================
# Ana Fonksiyon (Any imzalar)
# =========================

def detect(df: Any,
           forecasts: Dict[str, Any] | None,
           cfg: AnomalyConfig | None = None) -> List[AnomalyEvent]:
    """
    Veride ve (varsa) residual seride anomali tespiti yapar.

    Args:
        df: En azından 'close' kolonu bulunan DataFrame (DatetimeIndex önerilir).
        forecasts: {'y_hat': pandas.Series} gibi tahmin serileri (df ile hizalı).
        cfg: Eşikler ve ayarlar (bkz. AnomalyConfig). Verilmezse makul varsayılanlar.

    Returns:
        List[AnomalyEvent]: Zaman damgası ve detaylarıyla anomali listesi.
    """
    events: List[AnomalyEvent] = []

    # Pandas/Numpy yoksa tespit yapmayalım
    if pd is None or np is None:
        return events

    # df ve kolon kontrolü
    try:
        if df is None or "close" not in df.columns or len(df) < int(_DEF["min_obs"]):  # type: ignore
            return events
    except Exception:
        return events

    cfg = {**_DEF, **(cfg or {})}  # defaults + override

    # close serisi
    try:
        close = df["close"].astype(float)
    except Exception:
        return events

    # --- 1) Fiyat getirisi z-score (price spikes/crashes)
    try:
        ret = _series_returns(close)
        z = _rolling_z(ret, int(cfg["window"]), bool(cfg["use_robust"]))
        events += _add_events_from_z(
            z,
            base_series=close,
            pos_threshold=float(cfg["z_threshold"]),
            neg_threshold=float(cfg["z_threshold"]),
            pos_type="price_spike",
            neg_type="price_crash",
        )
    except Exception:
        pass

    # --- 2) Residual z-score (forecast mevcutsa)
    try:
        if cfg.get("residual_check", True) and forecasts:
            y_hat = forecasts.get("y_hat")
            if y_hat is not None and pd is not None and isinstance(y_hat, pd.Series):
                y_hat = y_hat.astype(float).reindex(close.index)
                residual = close - y_hat
                rz = _rolling_z(residual, int(cfg["residual_window"]), bool(cfg["residual_use_robust"]))
                if rz is not None:
                    nz = rz.dropna()
                    thr = float(cfg["residual_z_threshold"])
                    for ts, zval in nz.items():
                        if abs(float(zval)) >= thr:
                            events.append(AnomalyEvent(
                                ts=_as_utc_iso(ts),
                                type="residual_spike",
                                z=float(zval),
                                residual=float(residual.loc[ts]) if ts in residual.index else float("nan"),
                                threshold=thr
                            ))
    except Exception:
        pass

    # --- 3) Hacim spike (opsiyonel)
    try:
        if cfg.get("volume_check", False) and "volume" in df.columns:
            vol = df["volume"].astype(float)
            vz = _rolling_z(vol.pct_change(), int(cfg["volume_window"]), bool(cfg["volume_use_robust"]))
            if vz is not None:
                nz = vz.dropna()
                thr_v = float(cfg["volume_z_threshold"])
                for ts, zval in nz.items():
                    if float(zval) >= thr_v:
                        events.append(AnomalyEvent(
                            ts=_as_utc_iso(ts),
                            type="volume_spike",
                            z=float(zval),
                            value=float(vol.loc[ts]) if ts in vol.index else float("nan"),
                        ))
    except Exception:
        pass

    # Zaman sırası
    try:
        events.sort(key=lambda e: e.get("ts", ""))
    except Exception:
        pass

    return events