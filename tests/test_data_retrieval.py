# -*- coding: utf-8 -*-
"""
Smoke test: Binance veri toplama ajanı
- HIGH örnek: 5 yıl (BTC)
- MEDIUM örnek: 2 yıl (ARB)  → 1–3 yıl arası
- LOW örnek: 0.5 yıl (BTC)   → < 1 yıl
- Node testi: ETH (5 yıl)
"""

import pandas as pd
from src.agents.data_retrieval import get_stock_data, data_retrieval_node


def _pretty(df: pd.DataFrame, n: int = 3) -> str:
    return df.head(n).to_string()


def run_case(label: str, ticker: str, years: float, interval: str = "1d"):
    print(f"\n[{label}] get_stock_data('{ticker}', {years}y, {interval})")
    try:
        df, meta = get_stock_data(ticker, lookback_years=years, interval=interval)
        assert isinstance(df, pd.DataFrame), "df DataFrame değil"
        assert len(df) > 0, "Boş DataFrame geldi"
        print(f"shape={df.shape}  rows={meta['rows']}  conf={meta['confidence']}")
        print(f"range: {meta['start']} → {meta['end']}  symbol={meta['symbol']}")
        print(_pretty(df))
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")


def run_node(label: str, ticker: str, years: float = 5.0, interval: str = "1d"):
    print(f"\n[{label}] data_retrieval_node(state={{'ticker':'{ticker}','lookback_years':{years}}})")
    try:
        out = data_retrieval_node({"ticker": ticker, "lookback_years": years, "interval": interval})
        df = out["raw_data"]; meta = out["meta"]
        print(f"shape={df.shape}  rows={meta['rows']}  conf={meta['confidence']}")
        print(_pretty(df))
    except Exception as e:
        print(f"[ERROR][node] {ticker}: {e}")


if __name__ == "__main__":
    # HIGH: ≥3 yıl
    run_case("HIGH", "BTC", 5.0)

    # MEDIUM: 1–3 yıl (ARB ~ 2023’ten beri; ayrıca 2 yıl kısıtlıyoruz)
    run_case("MEDIUM", "ARB", 2.0)

    # LOW: <1 yıl (aynı sembol olsa da lookback kısıt <1 yıl)
    run_case("LOW", "BTC", 0.5)

    # Node örneği
    run_node("NODE", "ETH", 5.0)