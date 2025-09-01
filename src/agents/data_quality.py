# -*- coding: utf-8 -*-
"""
Data Quality Agent
==================

OHLCV zaman serileri (kripto/hisse) için **kalite kontrolü** yapar.
Temizlik/onarım yapmaz; **rapor** ve **bayraklar** üretir. LangGraph benzeri
akışlarda düğüm (node) olarak da kullanılabilir.

Girdi
-----
- raw_data: `pd.DataFrame`
    - Index: `pd.DatetimeIndex` (beklenen)
    - Kolonlar: en az ["Open", "High", "Low", "Close"] (Volume opsiyonel)

Çıktı
-----
- check_data_quality(df)  -> Dict[str, Any]
    {
      "shape": (rows, cols),
      "missing_columns": [...],
      "has_datetime_index": bool,
      "is_monotonic_increasing": bool,
      "has_duplicates": bool,
      "na_ratio_per_col": {col: float},
      "zero_volume_ratio": float | None,
      "return_outliers": {"count": int, "threshold": 5.0},
      "candle_flags": {...},          # açık/kapalı/tepe/dip tutarlılıkları
      "gap": {"inferred_freq": str|None, "missing_ratio": float|None, "expected": "1D"},
      "warnings": [str, ...],
      "is_ok": bool,
      "quality_flags": { ... }        # kısa özet bayraklar
    }

- data_quality_node(state) -> {"quality_report": dict, "quality_flags": dict}
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


# Zorunlu kabul ettiğimiz temel fiyat kolonları (Volume opsiyonel)
REQUIRED_COLS = ["Open", "High", "Low", "Close"]


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _zscore(a: np.ndarray) -> np.ndarray:
    """Basit z-score."""
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(a, dtype=float)
    return (a - mu) / sd


def _candle_consistency_flags(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Mum içi tutarlılık bayrakları:
      - Negatif fiyat/volume
      - High < Low
      - Open / Close değerinin [Low, High] dışında kalması
    """
    flags = {
        "price_negative": False,
        "volume_negative": False,
        "high_lt_low": False,
        "open_outside_hilo": False,
        "close_outside_hilo": False,
    }

    # Negatif fiyat/volume kontrolü
    for c in ["Open", "High", "Low", "Close"]:
        if c in df.columns and df[c].dtype.kind in "fi":
            if (df[c] < 0).any():
                flags["price_negative"] = True
    if "Volume" in df.columns and df["Volume"].dtype.kind in "fi":
        if (df["Volume"] < 0).any():
            flags["volume_negative"] = True

    # High >= Low zorunluluğu
    if all(k in df.columns for k in ["High", "Low"]) and \
       df["High"].dtype.kind in "fi" and df["Low"].dtype.kind in "fi":
        if (df["High"] < df["Low"]).any():
            flags["high_lt_low"] = True

    # Open ∈ [Low, High]
    if all(k in df.columns for k in ["Open", "High", "Low"]) and \
       df["Open"].dtype.kind in "fi" and df["High"].dtype.kind in "fi" and df["Low"].dtype.kind in "fi":
        bad_open = (df["Open"] > df["High"]) | (df["Open"] < df["Low"])
        if bad_open.any():
            flags["open_outside_hilo"] = True

    # Close ∈ [Low, High]
    if all(k in df.columns for k in ["Close", "High", "Low"]) and \
       df["Close"].dtype.kind in "fi" and df["High"].dtype.kind in "fi" and df["Low"].dtype.kind in "fi":
        bad_close = (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
        if bad_close.any():
            flags["close_outside_hilo"] = True

    return flags


def _gap_report(df: pd.DataFrame, expect_daily: bool = True) -> Dict[str, Any]:
    """
    Basit gap/sıklık raporu.
    - inferred_freq: pandas'ın tahmin ettiği frekans.
    - missing_ratio: beklenen günlük eksen üzerinde eksik gün oranı.
      (Kriptoda 7/24 → 1D seride beklenen gap ≈ 0; küçük API boşlukları oluşabilir.)
    """
    out = {"inferred_freq": None, "missing_ratio": None, "expected": "1D" if expect_daily else None}

    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return out

    # Sıklık kestirimi
    inf = pd.infer_freq(df.index)
    out["inferred_freq"] = inf

    # Günlük eksende gap oranı
    if expect_daily:
        dt_start, dt_end = df.index[0].normalize(), df.index[-1].normalize()
        full = pd.date_range(dt_start, dt_end, freq="D")
        missing = len(full.difference(df.index.normalize()))
        total = max(1, len(full))
        out["missing_ratio"] = float(missing / total)

    return out


# --------------------------------------------------------------------------- #
# Ana kontrol
# --------------------------------------------------------------------------- #
def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Temel kalite kontrollerini çalıştırır ve rapor döndürür.

    Not: Bu fonksiyon **temizlik yapmaz** (drop/forward-fill vb. yok).
         Ama rapor çıktıları, ileride otomatik düzeltmeler için temel oluşturur.
    """
    report: Dict[str, Any] = {}
    warnings: List[str] = []

    # Şekil
    report["shape"] = tuple(df.shape)

    # 1) Zorunlu kolonlar
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    report["missing_columns"] = missing
    if missing:
        warnings.append(f"Eksik zorunlu kolon(lar): {missing}")

    # 2) Index türü ve sıralama
    has_dt = isinstance(df.index, pd.DatetimeIndex)
    report["has_datetime_index"] = has_dt
    if not has_dt:
        warnings.append("Index DatetimeIndex değil.")

    is_mono = bool(df.index.is_monotonic_increasing) if has_dt else False
    report["is_monotonic_increasing"] = is_mono
    if has_dt and not is_mono:
        warnings.append("Tarih sırası artan değil.")

    # 3) Yinelenen zaman damgası
    has_dupe = bool(df.index.duplicated().any()) if has_dt else False
    report["has_duplicates"] = has_dupe
    if has_dupe:
        warnings.append("Yinelenen zaman damgaları var.")

    # 4) NaN oranları
    na_ratio = {c: float(df[c].isna().mean()) for c in df.columns if df[c].dtype.kind in "fi"}
    report["na_ratio_per_col"] = na_ratio
    if any(r > 0 for r in na_ratio.values()):
        warnings.append("Bazı sayısal kolonlarda NaN mevcut.")

    # 5) Sıfır hacim oranı
    zero_vol_ratio: Optional[float] = None
    if "Volume" in df.columns and df["Volume"].dtype.kind in "fi":
        zero_vol_ratio = float((df["Volume"] == 0).mean())
        report["zero_volume_ratio"] = zero_vol_ratio
        if zero_vol_ratio > 0.05:
            warnings.append(f"Sıfır hacim oranı yüksek: {zero_vol_ratio:.1%}")
    else:
        report["zero_volume_ratio"] = None

    # 6) Getiri outlier sayısı (z-score, 5σ)
    ret_out = {"count": 0, "threshold": 5.0}
    if "Close" in df.columns and df["Close"].dtype.kind in "fi" and len(df) > 5:
        closes = df["Close"].astype(float).to_numpy()
        rets = np.diff(closes) / closes[:-1]
        z = _zscore(rets)
        thr = 5.0
        cnt = int(np.sum(np.abs(z) > thr))
        ret_out = {"count": cnt, "threshold": thr}
        if cnt > 0:
            warnings.append(f"Getiri serisinde {cnt} aykırı (>|{thr}σ|) gözlem var.")
    report["return_outliers"] = ret_out

    # 7) Mum tutarlılığı
    candle_flags = _candle_consistency_flags(df)
    report["candle_flags"] = candle_flags
    if any(candle_flags.values()):
        bads = [k for k, v in candle_flags.items() if v]
        warnings.append(f"Mum içi tutarsızlık(lar): {bads}")

    # 8) Gap/Sıklık raporu (kripto için günlük beklenir)
    gap = _gap_report(df, expect_daily=True)
    report["gap"] = gap
    if gap["missing_ratio"] is not None and gap["missing_ratio"] > 0.01:
        warnings.append(f"Eksik gün oranı yüksek: {gap['missing_ratio']:.1%}")
    if gap["inferred_freq"] and gap["inferred_freq"] not in ("D", "B", "C"):
        # Kripto 7/24 → genelde "D" bekleriz. (B/C iş günleri/özel takvim)
        warnings.append(f"Beklenmeyen frekans: {gap['inferred_freq']}")

    # Genel uygunluk (kritik hatalar yoksa True)
    report["warnings"] = warnings
    report["is_ok"] = not (
        missing
        or (not has_dt)
        or (has_dt and (not is_mono or has_dupe))
        or any(candle_flags.values())
    )

    # Özet bayraklar
    report["quality_flags"] = {
        "cols_ok": len(missing) == 0,
        "index_ok": has_dt and is_mono and not has_dupe,
        "na_ok": max(na_ratio.values() or [0.0]) < 0.01,
        "vol_ok": True if zero_vol_ratio is None else zero_vol_ratio < 0.05,
        "outlier_ok": report["return_outliers"]["count"] == 0,
        "candle_ok": not any(candle_flags.values()),
        "gap_ok": (gap["missing_ratio"] is None) or (gap["missing_ratio"] <= 0.01),
    }

    return report


# --------------------------------------------------------------------------- #
# Node arayüzü
# --------------------------------------------------------------------------- #
def data_quality_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph düğümü gibi çalışır.

    Beklenen state:
        - "raw_data": pd.DataFrame

    Dönen:
        {"quality_report": dict, "quality_flags": dict}
    """
    raw = state.get("raw_data", None)
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("data_quality_node: 'raw_data' boş veya geçersiz.")

    rep = check_data_quality(raw)
    return {"quality_report": rep, "quality_flags": rep["quality_flags"]}