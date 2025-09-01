# -*- coding: utf-8 -*-
"""
Smoke test: Forecast Agent (ARIMA / ARIMAX / ETS)
- BTC için 5y analiz + üç modelin çağrısı
- AUTO modda “en iyi” seçimi raporla
"""
import pandas as pd
from src.agents.data_retrieval import get_stock_data
from src.agents.indicator import calculate_indicators
from src.agents.forecast import (
    fit_predict_arima, fit_predict_arimax, fit_predict_ets, forecast_node
)


def _prep_analyzed(ticker: str = "BTC") -> pd.DataFrame:
    df, _meta = get_stock_data(ticker, lookback_years=5, interval="1d")
    analyzed, _ = calculate_indicators(df)  # Close + indikatörler
    return analyzed


def _pp(s: pd.Series, n: int = 5) -> str:
    return s.head(n).to_string()


if __name__ == "__main__":
    print("\n[PREP] BTC analyzed_data oluşturuluyor (5y, daily)…")
    analyzed = _prep_analyzed("BTC")
    print("analyzed shape:", analyzed.shape)

    y = analyzed["Close"]

    # ARIMA
    print("\n[ARIMA] forecast")
    fc_a, meta_a = fit_predict_arima(y, horizon=30)
    print("model:", meta_a["model"], "order:", meta_a["order"])
    print("aic:", meta_a["aic"], "bic:", meta_a["bic"])
    print("train_len:", meta_a["train_len"], "horizon:", meta_a["horizon"])
    print("preview:\n", _pp(fc_a))

    # ARIMAX (exog olarak birkaç indikatör)
    print("\n[ARIMAX] forecast")
    excols = ["RSI_14", "EMA_12", "SMA_14", "MACD_line"]
    fc_ax, meta_ax = fit_predict_arimax(y, exog=analyzed[excols], horizon=30, exog_cols=excols)
    print("model:", meta_ax["model"], "order:", meta_ax["order"])
    print("aic:", meta_ax["aic"], "bic:", meta_ax["bic"])
    print("train_len:", meta_ax["train_len"], "horizon:", meta_ax["horizon"])
    print("exog_cols:", meta_ax["exog_cols"])
    print("preview:\n", _pp(fc_ax))

    # ETS
    print("\n[ETS] forecast")
    fc_e, meta_e = fit_predict_ets(y, horizon=30)
    print("model:", meta_e["model"], "cfg:", meta_e["cfg"])
    print("train_len:", meta_e["train_len"], "horizon:", meta_e["horizon"])
    print("preview:\n", _pp(fc_e))

    # AUTO (ARIMAX vs ARIMA vs ETS küçük karşılaştırma)
    print("\n[AUTO] forecast_node")
    node_out = forecast_node({
        "analyzed_data": analyzed,
        "horizon": 30,
        "model": "auto",
        "exog_cols": excols,
    })
    fc_auto, meta_auto = node_out["forecast_data"], node_out["forecast_meta"]
    print("chosen:", meta_auto["model"], "| competitors:", meta_auto.get("auto_competitors"))
    print("preview:\n", _pp(fc_auto))

    print("\n✅ Forecast (ARIMA/ARIMAX/ETS) smoke test is done.")