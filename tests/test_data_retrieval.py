# -*- coding: utf-8 -*-
"""
Smoke test: Binance veri toplama ajanı
- HIGH örnek: 5 yıl (BTC)
- MEDIUM örnek: 5 yıl istenir ama ARB'nin geçmişi 1–3 yıl → MEDIUM
- LOW örnek: 5 yıl istenir ama yeni bir coin <1 yıl → LOW
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
    # Hepsinde lookback_years = 5.0 → gerçek kapsama göre HIGH / MEDIUM / LOW çıkacak
    run_case("HIGH", "BTC", 5.0)    # ≥3 yıl veri → HIGH
    run_case("MEDIUM", "ARB", 5.0)  # 1–3 yıl veri → MEDIUM
    run_case("LOW", "ZRO", 5.0)     # <1 yıl veri → LOW (örnek yeni coin)