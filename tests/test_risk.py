# -*- coding: utf-8 -*-
"""
Smoke test: Risk Agent
- BTC 5y verisini çek → indikatörleri hesapla → rejimi bul → risk planı öner.
"""
import pandas as pd
from src.agents.data_retrieval import get_stock_data
from src.agents.indicator import calculate_indicators
from src.agents.regime_detector import detect_regime
from src.agents.risk import build_risk_plan, risk_node

if __name__ == "__main__":
    df, meta = get_stock_data("BTC", lookback_years=5)
    analyzed, _ = calculate_indicators(df)
    regime, rmeta = detect_regime(analyzed)

    print("[INFO] regime:", regime, rmeta)
    plan, pmeta = build_risk_plan(
        analyzed,
        regime=regime,
        account_equity=25_000,
        account_risk_pct=0.01,
        confidence=meta.get("confidence", "high"),
    )
    print("\n[PLAN]", plan)
    print("[META]", pmeta)

    # Node entegrasyonu
    out = risk_node({
        "analyzed_data": analyzed,
        "regime": regime,
        "account_equity": 25_000,
        "account_risk_pct": 0.01,
        "confidence": meta.get("confidence"),
    })
    print("\n[NODE]", out["risk_plan"])