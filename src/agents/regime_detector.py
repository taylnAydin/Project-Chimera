# -*- coding: utf-8 -*-
"""
Regime Detection Agent
======================

Amaç
-----
- analyzed_data üstünden **trend** ve **volatilite** ölçüp son satır için bir
  rejim etiketi üretir:  {bull|sideways|bear} x {low_vol|high_vol}
  Örn: "bull_low_vol", "bear_high_vol" vb.
- İstersek her tarih için satır bazlı rejim kolonunu da ekler.

Girdi
-----
- analyzed_data: pd.DataFrame (DateTimeIndex)
  Zorunlu kolonlar: Close
  Opsiyonel: SMA_50, SMA_200 (yoksa burada hesaplanır)

Çıktı
-----
- regime (str)                 → son satır rejim etiketi
- regime_meta (dict)           → eşikler, ölçümler, skorlar
- (opsiyonel) regime_series    → tarih bazlı etiketler (include_series=True)

Not
---
- Volatilite: 20 günlük günlük-getiri std’si, yıllıklandırılmış (√365 ile).
  “high/low” kararı **dinamik**: son 60 günde medyan vol’a göre
  last_vol > 1.2 * median_vol_60  ⇒ high_vol, aksi halde low_vol.
- Trend: SMA50 vs SMA200 ve Close’un SMA50’den sapması.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


# Default parametreler (gerekirse state içinden override edilebilir)
SMA_FAST = 50
SMA_SLOW = 200
RET_WIN = 20        # volatilite penceresi (gün)
VOL_MED_WIN = 60    # volatilite medyan karşılaştırma penceresi
VOL_MULT = 1.2      # “high vol” eşiği çarpanı (son vol > medyan*VOL_MULT)
SIDE_BAND = 0.01    # Sideways bandı: |Close - SMA50| / Close < %1 => sideways


def _ensure_close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("regime_detection: 'Close' kolonu gerekli (analyzed_data).")
    y = pd.to_numeric(df["Close"], errors="coerce").dropna()
    y.index = pd.to_datetime(y.index)
    if not y.index.is_monotonic_increasing:
        y = y.sort_index()
    return y


def _sma(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win, min_periods=max(5, win // 5)).mean()


def _daily_returns(s: pd.Series) -> pd.Series:
    return s.pct_change()


def _annualized_vol(s: pd.Series, win: int = RET_WIN, scale: float = np.sqrt(365.0)) -> pd.Series:
    rets = _daily_returns(s)
    return rets.rolling(win, min_periods=max(5, win // 5)).std() * scale


def _trend_label(close: pd.Series, sma_fast: pd.Series, sma_slow: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Döner:
        strength: |Close - SMA50| / Close   (0..)
        label: bull/sideways/bear
    """
    strength = (close - sma_fast).abs() / close.clip(lower=1e-9)
    cond_bull = sma_fast > sma_slow
    cond_bear = sma_fast < sma_slow

    label = pd.Series(index=close.index, dtype="object")
    label[cond_bull] = "bull"
    label[cond_bear] = "bear"

    # Sideways bandı (yakınlık), sadece bull/bear etiketini “sideways”e çevirebilir
    near = strength < SIDE_BAND
    label[near] = "sideways"
    return strength, label


def _vol_label(ann_vol: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Dinamik vol eşiği: son 60 gün medyanına göre.
    Döner:
        ratio: ann_vol / median_60
        label: high_vol / low_vol
    """
    med60 = ann_vol.rolling(VOL_MED_WIN, min_periods=max(5, VOL_MED_WIN // 5)).median()
    ratio = ann_vol / med60.replace(0, np.nan)
    label = pd.Series(np.where(ratio > VOL_MULT, "high_vol", "low_vol"), index=ann_vol.index, dtype="object")
    # Erken dönem NaN'leri için varsayılan:
    label[~np.isfinite(ratio)] = "low_vol"
    ratio = ratio.fillna(1.0)
    return ratio, label


def detect_regime(analyzed_data: pd.DataFrame, include_series: bool = False) -> Dict[str, Any]:
    """
    analyzed_data'dan rejim belirler.

    Returns
    -------
    {
      "regime": "bull_low_vol" | ...,
      "regime_meta": {
         "close": float,
         "sma_fast": float,
         "sma_slow": float,
         "trend_label": str,
         "trend_strength": float,
         "ann_vol": float,
         "vol_label": str,
         "vol_ratio": float,
         "params": {...}
      },
      # include_series=True ise:
      "regime_series": pd.Series(dtype='object')
    }
    """
    df = analyzed_data.copy()
    close = _ensure_close(df)

    # SMA'lar yoksa hesapla
    if f"SMA_{SMA_FAST}" in df.columns:
        sma_fast = pd.to_numeric(df[f"SMA_{SMA_FAST}"], errors="coerce")
    else:
        sma_fast = _sma(close, SMA_FAST)

    if f"SMA_{SMA_SLOW}" in df.columns:
        sma_slow = pd.to_numeric(df[f"SMA_{SMA_SLOW}"], errors="coerce")
    else:
        sma_slow = _sma(close, SMA_SLOW)

    # Trend etiketi + güç
    trend_strength, trend_tag = _trend_label(close, sma_fast, sma_slow)

    # Volatilite etiketi
    ann_vol = _annualized_vol(close, win=RET_WIN)
    vol_ratio, vol_tag = _vol_label(ann_vol)

    # Final rejim (satır bazlı)
    regime_series = (trend_tag.fillna("sideways") + "_" + vol_tag.fillna("low_vol"))

    # Son satır
    last_idx = close.index[-1]
    regime = str(regime_series.loc[last_idx])

    meta = {
        "close": float(close.iloc[-1]),
        "sma_fast": float(sma_fast.iloc[-1]),
        "sma_slow": float(sma_slow.iloc[-1]),
        "trend_label": str(trend_tag.iloc[-1]),
        "trend_strength": float(trend_strength.iloc[-1]),
        "ann_vol": float(ann_vol.iloc[-1]) if np.isfinite(ann_vol.iloc[-1]) else None,
        "vol_label": str(vol_tag.iloc[-1]),
        "vol_ratio": float(vol_ratio.iloc[-1]) if np.isfinite(vol_ratio.iloc[-1]) else None,
        "params": {
            "sma_fast": SMA_FAST,
            "sma_slow": SMA_SLOW,
            "ret_win": RET_WIN,
            "vol_med_win": VOL_MED_WIN,
            "vol_mult": VOL_MULT,
            "side_band": SIDE_BAND,
        },
    }

    out: Dict[str, Any] = {"regime": regime, "regime_meta": meta}
    if include_series:
        out["regime_series"] = regime_series
    return out


def regime_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph düğümü.
    Girdi:
      - analyzed_data (pd.DataFrame)
      - include_series (bool, opsiyonel)
    Çıktı:
      - regime, regime_meta (+opsiyonel regime_series)
    """
    df = state.get("analyzed_data", None)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("regime_node: 'analyzed_data' boş/geçersiz.")
    include_series = bool(state.get("include_series", False))
    return detect_regime(df, include_series=include_series)