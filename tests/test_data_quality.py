# -*- coding: utf-8 -*-
"""
Smoke test: Data Quality Agent
- Binance'ten veri çek (BTC 5y)
- check_data_quality & data_quality_node çıktıları
- Negatif/sorunlu örnek için yapay bozuk DataFrame testi
"""

import pandas as pd

from src.agents.data_retrieval import get_stock_data, data_retrieval_node
from src.agents.data_quality import check_data_quality, data_quality_node


def _print_report(title: str, rep: dict, head_df: pd.DataFrame | None = None):
    print(f"\n[{title}]")
    print("is_ok:", rep["is_ok"])
    print("shape:", rep["shape"])
    print("flags:", rep["quality_flags"])
    warns = rep.get("warnings", [])
    if warns:
        print("warnings:")
        for w in warns[:5]:
            print("  -", w)
        if len(warns) > 5:
            print(f"  (+{len(warns)-5} more)")
    if head_df is not None:
        print("\nhead:")
        print(head_df.head(3).to_string())


def test_happy_path_symbol(ticker: str = "BTC"):
    print(f"\n[HAPPY] get_stock_data('{ticker}', 5y) + check_data_quality")
    df, meta = get_stock_data(ticker, lookback_years=5, interval="1d")
    rep = check_data_quality(df)
    _print_report(f"check_data_quality({ticker})", rep, df)


def test_node_symbol(ticker: str = "ETH"):
    print(f"\n[NODE] data_retrieval_node('{ticker}', 5y) + data_quality_node")
    out = data_retrieval_node({"ticker": ticker, "lookback_years": 5})
    raw = out["raw_data"]
    qout = data_quality_node({"raw_data": raw})
    rep = qout["quality_report"]
    _print_report(f"data_quality_node({ticker})", rep, raw)


def test_bad_dataframe():
    print("\n[BAD] Yapay bozuk DataFrame (sıralı olmayan index, dupe, NaN, tutarsız High/Low)")
    # Küçük bir örnek DataFrame üretelim:
    idx = pd.to_datetime(
        ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"]  # sırasız + duplicate
    )
    df = pd.DataFrame(
        {
            "Open":  [100, 101, 102, 103],
            "High":  [105, 104,  99, 108],  # bir satırda High < Low yapalım
            "Low":   [ 98,  99, 103, 100],
            "Close": [102, 100, 101, 101],
            "Volume":[1000, 0,   500,  -5],  # sıfır ve negatif volume
        },
        index=idx,
    )
    # Birkaç NaN da serpiştirelim
    df.loc[idx[0], "Close"] = None

    rep = check_data_quality(df)
    _print_report("check_data_quality(BOZUK)", rep, df)


if __name__ == "__main__":
    test_happy_path_symbol("BTC")
    test_node_symbol("ETH")
    test_bad_dataframe()
    print("\n✅ Data Quality smoke test tamam.")