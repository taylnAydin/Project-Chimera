# -*- coding: utf-8 -*-
"""
Smoke test: Model Selection Agent
- BTC verisini al → indikatörleri üret → model_selection çalıştır
- ARIMA/ARIMAX/ETS skorlarını ve seçimi yazdır
"""

import pandas as pd

from src.agents.data_retrieval import get_stock_data
from src.agents.indicator import calculate_indicators
from src.agents.model_selection import select_model, model_selection_node


def _pretty_scores(scores: dict) -> str:
    lines = []
    for name in ("ARIMAX", "ARIMA", "ETS"):
        sc = scores.get(name, {})
        if name in ("ARIMA", "ARIMAX"):
            lines.append(
                f"{name}: rmse={sc.get('rmse'):.4f}  aic={sc.get('aic')}  order={sc.get('order')}  exog={sc.get('exog_cols')}"
            )
        else:
            lines.append(
                f"{name}: rmse={sc.get('rmse'):.4f}  aic≈{sc.get('aic')}  cfg={sc.get('cfg')}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print("\n[PREP] BTC → get_stock_data(5y) + calculate_indicators …")
    df, meta = get_stock_data("BTC", lookback_years=5, interval="1d")
    analyzed, _ = calculate_indicators(df)

    excols = ["RSI_14", "EMA_12", "SMA_14", "MACD_line"]

    print("\n[SELECT] select_model")
    sel = select_model(analyzed, exog_cols=excols, val_ratio=0.1)
    print("chosen:", sel["chosen"])
    print(_pretty_scores(sel["scores"]))

    print("\n[NODE] model_selection_node")
    out = model_selection_node({"analyzed_data": analyzed, "exog_cols": excols, "val_ratio": 0.1})
    print("model_choice:", out["model_choice"])
    print(_pretty_scores(out["model_scores"]))

    print("\n✅ Model selection smoke test tamam.")