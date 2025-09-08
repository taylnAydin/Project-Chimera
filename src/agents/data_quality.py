# -*- coding: utf-8 -*-
"""
Data Quality Agent
==================

Amaç
----
OHLCV veri çerçevesinde (DateTimeIndex) temel kalite kontrollerini yapmak,
temizleme yapmadan durum/uyarı raporu üretmek ve akışta kullanılacak
bayraklar vermek. (Router, bu raporu kullanarak yönlendirme yapabilir.)

Öne Çıkanlar
------------
- Eşikler parametre: z_thr, zero_vol_thr, na_thr
- Severity: "info" | "warn" | "error"
- Mum içi tutarlılıkta eşitlik durumları kabul (High >= Low, Low ≤ Open/Close ≤ High)
- Meta passthrough: data_retrieval meta'sı (confidence/start/end/symbol) opsiyonel
- LangGraph uyumlu node: `data_quality_node(state)`

Girdi
-----
- DataFrame (DateTimeIndex; kolonlar: Open, High, Low, Close, [Volume, Adj Close])

Çıktı
-----
- report (dict): detaylı kalite raporu
- quality_flags (dict): kısa bayraklar
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


REQUIRED_COLS = ["Open", "High", "Low", "Close"]  # Volume opsiyonel


def _zscore(a: np.ndarray) -> np.ndarray:
    """NaN dayanıklı z-skoru."""
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(a, dtype=float)
    return (a - mu) / sd


def _candle_row_ok(row: pd.Series) -> bool:
    """
    Mum içi kurallar (eşitliklere izin verir):
      - High >= Low
      - Low <= Open <= High (Open varsa)
      - Low <= Close <= High (Close varsa)
      - Volume (varsa) negatif değil
    """
    h = row.get("High")
    l = row.get("Low")
    if pd.isna(h) or pd.isna(l):
        return False
    if h < l:  # eşitlik OK
        return False

    for k in ("Open", "Close"):
        if k in row:
            v = row[k]
            if pd.notna(v) and not (l <= v <= h):
                return False

    if "Volume" in row:
        v = row["Volume"]
        if pd.notna(v) and v < 0:
            return False

    return True


def check_data_quality(
    df: pd.DataFrame,
    *,
    z_thr: float = 5.0,
    zero_vol_thr: float = 0.05,
    na_thr: float = 0.01,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Temel kalite kontrollerini çalıştırır ve rapor üretir.

    Parametreler
    -----------
    df : pd.DataFrame
        OHLCV veri çerçevesi. DateTimeIndex beklenir.
    z_thr : float, default 5.0
        Günlük log/getiri serisinde aykırı sayımı için z-skor eşiği.
    zero_vol_thr : float, default 0.05
        Sıfır hacim oranı bu eşikten büyükse uyarı.
    na_thr : float, default 0.01
        Sayısal kolonlarda NaN oranı eşik üstündeyse uyarı.
    meta : dict | None
        (Opsiyonel) data_retrieval meta bilgisi (confidence/start/end/symbol).

    Dönen
    -----
    report : dict
        {
          "shape": (rows, cols),
          "missing_columns": [...],
          "has_datetime_index": bool,
          "is_monotonic_increasing": bool,
          "has_duplicates": bool,
          "na_ratio_per_col": {col: float},
          "zero_volume_ratio": float | None,
          "return_outliers": {"count": int, "threshold": z_thr},
          "candle_ok": bool,
          "warnings": [str, ...],
          "quality_flags": {
              "cols_ok": bool,
              "index_ok": bool,
              "na_ok": bool,
              "vol_ok": bool,
              "outlier_ok": bool,
              "candle_ok": bool,
              "gap_ok": bool
          },
          "is_ok": bool,          # error seviyesinde sorun yoksa True
          "severity": "info" | "warn" | "error",
          "meta": {...}           # varsa passthrough
        }
    """
    report: Dict[str, Any] = {}
    warnings: List[str] = []

    report["shape"] = tuple(df.shape)

    # --- Kolon kontrolü
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    report["missing_columns"] = missing
    if missing:
        warnings.append(f"Eksik zorunlu kolon(lar): {missing}")

    # --- İndeks tipi ve sıralama
    has_dt = isinstance(df.index, pd.DatetimeIndex)
    report["has_datetime_index"] = has_dt
    if not has_dt:
        warnings.append("Index DatetimeIndex değil.")

    is_mono = bool(df.index.is_monotonic_increasing) if has_dt else False
    report["is_monotonic_increasing"] = is_mono
    if has_dt and not is_mono:
        warnings.append("Tarih sırası artan değil.")

    # --- Yinelenen zaman damgası
    has_dupe = bool(df.index.duplicated().any()) if has_dt else False
    report["has_duplicates"] = has_dupe
    if has_dupe:
        warnings.append("Yinelenen zaman damgaları var.")

    # --- NaN oranları (yalnızca sayısal kolonlar)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    na_ratio = {c: float(df[c].isna().mean()) for c in num_cols}
    report["na_ratio_per_col"] = na_ratio
    na_max = max(na_ratio.values() or [0.0])
    if na_max > 0:
        warnings.append("Bazı sayısal kolonlarda NaN mevcut.")

    # --- Sıfır hacim oranı
    zero_vol_ratio: Optional[float] = None
    if "Volume" in df.columns and pd.api.types.is_numeric_dtype(df["Volume"]):
        zero_vol_ratio = float((df["Volume"] == 0).mean())
        if zero_vol_ratio > zero_vol_thr:
            warnings.append(f"Sıfır hacim oranı yüksek: {zero_vol_ratio:.1%}")
    report["zero_volume_ratio"] = zero_vol_ratio

    # --- Mum içi tutarlılık
    if len(df) > 0:
        candle_ok = bool(df.apply(_candle_row_ok, axis=1).all())
    else:
        candle_ok = False
    if not candle_ok:
        warnings.append("Mum içi tutarsızlık(lar) var.")
    report["candle_ok"] = candle_ok

    # --- Aykırı gün sayısı (getiri tabanlı)
    ret_out = {"count": 0, "threshold": z_thr}
    if "Close" in df.columns and pd.api.types.is_numeric_dtype(df["Close"]) and len(df) > 5:
        closes = df["Close"].astype(float).to_numpy()
        rets = np.diff(closes) / closes[:-1]
        z = _zscore(rets)
        cnt = int(np.sum(np.abs(z) > float(z_thr)))
        ret_out["count"] = cnt
        if cnt > 0:
            warnings.append(f"Getiri serisinde {cnt} aykırı (>|{z_thr}σ|) gözlem var.")
    report["return_outliers"] = ret_out

    # --- Bayraklar
    flags = {
        "cols_ok": len(missing) == 0,
        "index_ok": has_dt and is_mono and not has_dupe,
        "na_ok": na_max < float(na_thr),
        "vol_ok": True if zero_vol_ratio is None else (zero_vol_ratio < float(zero_vol_thr)),
        "outlier_ok": ret_out["count"] == 0,
        "candle_ok": candle_ok,
        "gap_ok": True,  # yer tutucu: istersen takvim boşluk analizi ekleyebiliriz
    }
    report["quality_flags"] = flags

    # --- Severity & is_ok
    if not flags["cols_ok"] or not flags["index_ok"]:
        severity = "error"
    elif not flags["na_ok"] or not flags["vol_ok"] or not flags["candle_ok"]:
        severity = "warn"
    else:
        severity = "info"

    report["severity"] = severity
    report["is_ok"] = severity != "error"

    # --- Uyarılar & meta passthrough
    report["warnings"] = warnings
    if meta:
        report["meta"] = dict(meta)

    return report


def data_quality_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph düğümü gibi çalışır.

    Beklenen state:
        - raw_data: pd.DataFrame (zorunlu)
        - meta: dict | None (opsiyonel; data_retrieval'dan)
        - z_thr, zero_vol_thr, na_thr: opsiyonel eşikler

    Dönen:
        {"quality_report": dict, "quality_flags": dict}
    """
    raw = state.get("raw_data")
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("data_quality_node: 'raw_data' boş veya geçersiz.")

    rep = check_data_quality(
        raw,
        z_thr=float(state.get("z_thr", 5.0)),
        zero_vol_thr=float(state.get("zero_vol_thr", 0.05)),
        na_thr=float(state.get("na_thr", 0.01)),
        meta=state.get("meta"),
    )
    return {"quality_report": rep, "quality_flags": rep["quality_flags"]}

def _years_span(df: pd.DataFrame) -> float:
    """
    DataFrame'in tarih aralığını yıl cinsinden hesaplar.

    Args:
        df (pd.DataFrame): DateTimeIndex'e sahip OHLCV verisi.

    Returns:
        float: Veri kapsamının yıllık uzunluğu. Index uygun değilse 0.0.
    """
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        days = (df.index.max() - df.index.min()).days
        return max(0.0, days / 365.25)
    return 0.0


def _coverage_class(years: float) -> str:
    """
    Yıllık kapsama uzunluğunu kalite sınıfına dönüştürür.

    Args:
        years (float): Veri kapsama uzunluğu (yıl).

    Returns:
        str: "low" (<1y), "medium" (1–3y), "high" (≥3y).
    """
    if years >= 3.0:
        return "high"
    if years >= 1.0:
        return "medium"
    return "low"


def _duplicate_ratio(df: pd.DataFrame) -> float:
    """
    Index'te tekrarlanan zaman damgalarının oranını hesaplar.

    Args:
        df (pd.DataFrame): DateTimeIndex'e sahip OHLCV verisi.

    Returns:
        float: Duplicate oranı [0.0–1.0].
    """
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        dups = int(df.index.duplicated().sum())
        return dups / float(len(df.index))
    return 0.0


def _gap_ratio_daily(df: pd.DataFrame) -> Optional[float]:
    """
    Günlük veride eksik takvim günlerinin oranını hesaplar.

    Args:
        df (pd.DataFrame): DateTimeIndex'e sahip OHLCV verisi.

    Returns:
        float | None: Eksik gün oranı; günlük değilse None.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 2:
        return None
    full = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    missing = len(set(full) - set(df.index.normalize()))
    return missing / float(len(full)) if len(full) > 0 else None


def evaluate(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Router'ın çağırdığı kalite değerlendirme fonksiyonu.

    İçeride `check_data_quality` çalıştırır ve guardrails için özet
    metrikleri döner.

    Args:
        df (pd.DataFrame): OHLCV DataFrame (DateTimeIndex zorunlu).
        cfg (dict | None): Opsiyonel eşikler. Örn:
            {
              "z_thr": 5.0,
              "zero_vol_thr": 0.05,
              "na_thr": 0.01,
              "meta": {...}
            }

    Returns:
        dict: {
          "severity": str,
          "is_ok": bool,
          "flags": dict,
          "nan_ratio_max": float,
          "duplicate_ratio": float,
          "coverage_years": float,
          "coverage_class": str,
          "gap_ratio": float|None,
          "outlier_count": int,
          "zero_volume_ratio": float|None,
          "report": dict   # check_data_quality çıktısı
        }
    """
    cfg = cfg or {}
    rep = check_data_quality(
        df,
        z_thr=float(cfg.get("z_thr", 5.0)),
        zero_vol_thr=float(cfg.get("zero_vol_thr", 0.05)),
        na_thr=float(cfg.get("na_thr", 0.01)),
        meta=cfg.get("meta"),
    )

    nan_ratio_max = max((rep.get("na_ratio_per_col") or {}).values() or [0.0])
    years = _years_span(df)
    qual = {
        "severity": rep.get("severity"),
        "is_ok": bool(rep.get("is_ok", False)),
        "flags": rep.get("quality_flags", {}),
        "nan_ratio_max": float(nan_ratio_max),
        "duplicate_ratio": float(_duplicate_ratio(df)),
        "coverage_years": float(years),
        "coverage_class": _coverage_class(years),
        "gap_ratio": _gap_ratio_daily(df),
        "outlier_count": int((rep.get("return_outliers") or {}).get("count", 0)),
        "zero_volume_ratio": rep.get("zero_volume_ratio"),
        "report": rep,
    }
    return qual