# -*- coding: utf-8 -*-
"""
Smoke test: Regime Detection Agent
- BTC 5y veriyi çek → indikatörleri hesapla → rejimi bul
- Node arayüzünü de dener
"""
import pandas as pd

from src.agents.data_retrieval import get_stock_data
from src.agents.indicator import calculate_indicators, indicator_node
from src.agents.regime_detector import detect_regime, regime_node


def _prep_analyzed(ticker: str = "BTC"):
    df, _ = get_stock_data(ticker, lookback_years=5, interval="1d")
    analyzed, _ = calculate_indicators(df)
    return analyzed


if __name__ == "__main__":
    analyzed = _prep_analyzed("BTC")
    print(f"[PREP] analyzed shape={analyzed.shape}")

    # 1) Fonksiyon
    out = detect_regime(analyzed, include_series=True)
    print("\n[detect_regime]")
    print("regime:", out["regime"])
    print("meta:", {k: out["regime_meta"][k] for k in ["trend_label", "vol_label", "trend_strength", "vol_ratio"]})
    print(out["regime_series"].tail(3).to_string())

    # 2) Node
    nout = regime_node({"analyzed_data": analyzed, "include_series": False})
    print("\n[regime_node]")
    print("regime:", nout["regime"])
    print("meta.close:", nout["regime_meta"]["close"])
    print("\n✅ Regime detection smoke test is done.")