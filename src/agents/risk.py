# -*- coding: utf-8 -*-
"""
Risk Agent
==========
Girdi:
- analyzed_data (pd.DataFrame): OHLC içermeli (Open/High/Low/Close). Son satır "mevcut fiyat".
- regime (str): regime_detector çıktısı, ör. "bull_high_vol", "bear_low_vol", "sideway_high_vol"...
- account_equity (float): toplam sermaye (varsayılan 10_000).
- account_risk_pct (float): işlem başına risk yüzdesi (varsayılan 0.01 = %1).
- direction_override (str|None): "long" / "short" zorla (opsiyonel).
- atr_window (int): ATR periyodu (varsayılan 14).

Çıktı:
- risk_plan (dict): boyut, stop/tp, risk yüzdesi, notlar...
- risk_meta (dict): kullanılan parametreler/ara değerler.
- LangGraph için risk_node(state) -> {"risk_plan": dict, "risk_meta": dict}

Not:
- Rejime göre temel risk katsayısı ve yön önerisi belirleriz.
- ATR tabanlı stop mesafesi kullanır, pozisyon boyutunu risk bütçesine göre hesaplarız.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd


# ----------------------------- yardımcılar ---------------------------------- #
def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(df.columns):
        raise ValueError(f"risk: analyzed_data OHLC içermeli. Eksik: {list(need - set(df.columns))}")
    dff = df.copy()
    dff.index = pd.to_datetime(dff.index)
    if not dff.index.is_monotonic_increasing:
        dff = dff.sort_index()
    return dff


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=max(2, window // 2)).mean()


def _regime_policy(regime: str) -> Dict[str, Any]:
    """Rejime göre temel risk katsayısı, yön ve stop/tp çarpanlarını öner."""
    # varsayılanlar
    pol = dict(
        base_risk_mult=0.5,       # account_risk_pct ile çarpılır
        direction="long",         # veya "short"
        stop_atr_mult=2.5,
        tp_atr_mult=2.0,
        note="default"
    )
    r = regime.lower() if isinstance(regime, str) else "sideway_low_vol"

    if "bull" in r:
        pol["direction"] = "long"
        if "high_vol" in r:
            pol.update(base_risk_mult=0.6, stop_atr_mult=3.0, tp_atr_mult=2.0, note="bull_high_vol")
        else:
            pol.update(base_risk_mult=1.0, stop_atr_mult=2.0, tp_atr_mult=3.0, note="bull_low_vol")
    elif "bear" in r:
        pol["direction"] = "short"
        if "high_vol" in r:
            pol.update(base_risk_mult=0.4, stop_atr_mult=3.5, tp_atr_mult=2.0, note="bear_high_vol")
        else:
            pol.update(base_risk_mult=0.7, stop_atr_mult=2.5, tp_atr_mult=2.5, note="bear_low_vol")
    else:  # sideway
        pol["direction"] = "long"  # nötr; kullanıcı override edebilir
        if "high_vol" in r:
            pol.update(base_risk_mult=0.3, stop_atr_mult=3.0, tp_atr_mult=1.5, note="sideway_high_vol")
        else:
            pol.update(base_risk_mult=0.5, stop_atr_mult=2.0, tp_atr_mult=1.5, note="sideway_low_vol")

    return pol


def _confidence_mult(confidence: Optional[str]) -> float:
    """data_retrieval meta confidence -> risk azaltıcı katsayı."""
    if not confidence:
        return 0.8
    c = str(confidence).lower()
    if c == "high":
        return 1.0
    if c == "medium":
        return 0.7
    if c == "low":
        return 0.4
    return 0.8


# ----------------------------- ana API -------------------------------------- #
def build_risk_plan(
    analyzed_data: pd.DataFrame,
    *,
    regime: str,
    account_equity: float = 10_000.0,
    account_risk_pct: float = 0.01,
    direction_override: Optional[str] = None,
    atr_window: int = 14,
    confidence: Optional[str] = None,  # "high" | "medium" | "low"
    min_pos_notional: float = 10.0,     # çok küçük boyutları engelle
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    ATR tabanlı stop ve risk bütçesi ile pozisyon boyutunu hesaplar.

    Dönüş:
        (risk_plan, risk_meta)
    """
    df = _ensure_ohlc(analyzed_data)
    last = df.iloc[-1]
    price = float(last["Close"])

    atr = _atr(df, window=atr_window).iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        # fallback: fiyatın %2'sini ATR yerine kullan
        atr = 0.02 * price

    pol = _regime_policy(regime)
    if direction_override in ("long", "short"):
        pol["direction"] = direction_override

    # risk bütçesi
    conf_mult = _confidence_mult(confidence)
    risk_pct_effective = account_risk_pct * pol["base_risk_mult"] * conf_mult
    risk_dollars = account_equity * risk_pct_effective

    # stop ve TP mesafesi
    stop_dist = pol["stop_atr_mult"] * atr
    tp_dist = pol["tp_atr_mult"] * atr

    # birim sayısı (notional / stop_dist)
    # spot için units = risk_dollars / stop_dist
    units = risk_dollars / max(1e-9, stop_dist)
    notional = units * price

    # çok küçükse 0'a çek
    if notional < min_pos_notional:
        units = 0.0
        notional = 0.0

    if pol["direction"] == "long":
        stop_price = price - stop_dist
        tp_price = price + tp_dist
    else:
        stop_price = price + stop_dist
        tp_price = price - tp_dist

    plan = {
        "direction": pol["direction"],
        "price": price,
        "units": float(units),
        "notional": float(notional),
        "risk_pct_effective": float(risk_pct_effective),
        "stop_price": float(stop_price),
        "take_profit_price": float(tp_price),
        "atr": float(atr),
        "atr_window": int(atr_window),
        "note": pol["note"],
    }

    meta = {
        "account_equity": float(account_equity),
        "account_risk_pct": float(account_risk_pct),
        "base_risk_mult": float(pol["base_risk_mult"]),
        "confidence": confidence,
        "confidence_mult": float(conf_mult),
        "regime": str(regime),
        "stop_atr_mult": float(pol["stop_atr_mult"]),
        "tp_atr_mult": float(pol["tp_atr_mult"]),
    }

    return plan, meta


# ----------------------------- LangGraph node -------------------------------- #
def risk_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph düğümü gibi çalışır.

    Beklenen state:
        - analyzed_data (pd.DataFrame)  → zorunlu
        - regime (str)                  → zorunlu
        - account_equity (float)        → opsiyonel (10_000)
        - account_risk_pct (float)      → opsiyonel (0.01)
        - direction_override (str)      → opsiyonel
        - atr_window (int)              → opsiyonel
        - confidence (str)              → opsiyonel: high/medium/low

    Döndürülen:
        {"risk_plan": dict, "risk_meta": dict}
    """
    df = state.get("analyzed_data", None)
    regime = state.get("regime", None)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("risk_node: 'analyzed_data' boş/geçersiz.")
    if not regime:
        raise ValueError("risk_node: 'regime' zorunlu.")

    plan, meta = build_risk_plan(
        df,
        regime=str(regime),
        account_equity=float(state.get("account_equity", 10_000.0)),
        account_risk_pct=float(state.get("account_risk_pct", 0.01)),
        direction_override=state.get("direction_override"),
        atr_window=int(state.get("atr_window", 14)),
        confidence=state.get("confidence"),
    )
    return {"risk_plan": plan, "risk_meta": meta}