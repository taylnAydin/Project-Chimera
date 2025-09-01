# -*- coding: utf-8 -*-
"""
Smoke test: Indicator Agent (aynı coin seti: BTC, ARB, ZRO)
- Hepsinde lookback_years=5 istenir; gerçek kapsama göre HIGH/MEDIUM/LOW doğal çıkar.
- calculate_indicators ve indicator_node çıktılarının şekil/kolon kontrolü yapılır.
"""
import pandas as pd

from src.agents.data_retrieval import get_stock_data, data_retrieval_node
from src.agents.indicator import calculate_indicators, indicator_node


def _pretty(df: pd.DataFrame, n: int = 3) -> str:
    return df.head(n).to_string()


def run_calc(ticker: str):
    print(f"\n[CALC] {ticker} → get_stock_data(5y) + calculate_indicators")
    df, meta = get_stock_data(ticker, lookback_years=5.0, interval="1d")
    print(f"raw shape={df.shape} rows={meta['rows']} conf={meta['confidence']} "
          f"range={meta['start']}→{meta['end']}")

    analyzed, imeta = calculate_indicators(df)  # pandas_ta varsa otomatik kullanır
    must_have = {"SMA_14", "EMA_12", "RSI_14", "MACD_line", "BB_upper"}
    missing = must_have - set(analyzed.columns)
    assert not missing, f"Eksik kolon(lar): {missing}"

    print(f"analyzed shape={analyzed.shape} engine={imeta['engine']}")
    print("added:", imeta["added_columns"][:6], "…")
    # NaN başlangıç pencerelerini atıp birkaç satır göster
    print(_pretty(analyzed.filter(list(must_have)).dropna().head(3)))


def run_node(ticker: str):
    print(f"\n[NODE] {ticker} → data_retrieval_node(5y) + indicator_node")
    out = data_retrieval_node({"ticker": ticker, "lookback_years": 5.0, "interval": "1d"})
    raw = out["raw_data"]; meta = out["meta"]
    print(f"raw shape={raw.shape} rows={meta['rows']} conf={meta['confidence']}")

    nout = indicator_node({"raw_data": raw})
    analyzed = nout["analyzed_data"]; imeta = nout["indicator_meta"]

    for c in ["SMA_14", "EMA_12", "RSI_14", "MACD_line", "BB_upper"]:
        assert c in analyzed.columns, f"{c} bulunamadı"
    print(f"analyzed shape={analyzed.shape} engine={imeta['engine']}")
    print(_pretty(analyzed[["SMA_14", "EMA_12", "RSI_14"]].dropna().head(3)))


if __name__ == "__main__":
    # Aynı set: HIGH (BTC), MEDIUM (ARB), LOW (ZRO gibi yeni coin)
    for tk in ("BTC", "ARB", "ZRO"):
        run_calc(tk)
        run_node(tk)
    print("\n✅ Indicator smoke test is done.")