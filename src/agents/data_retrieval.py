# -*- coding: utf-8 -*-
"""
Binance Veri Toplama Ajanı (spot, orijinal OHLCV)
=================================================

- Sembol: "BTC", "ETH" gibi sade ad ver; otomatik olarak "USDT" karşılığını kullanır (BTC -> BTCUSDT).
- Kline endpoint: /api/v3/klines (API key gerektirmez).
- Günlük/ saatlik vb. orijinal veriyi çeker (resample YOK).
- Uzun aralıkları sayfa sayfa çeker, aralarda ufak uyku ile rate limit dostudur.
- Çekilen tarih aralığına göre bir güven puanı (low / medium / high) üretir.

Güven mantığı
-------------
- < 1 yıl veri  → "low" (örneklem az)
- 1 ≤ yıl < 3   → "medium"
- ≥ 3 yıl       → "high"

Dönüş
-----
get_stock_data(...)  -> (df, meta)
  df   : pandas.DataFrame (index: DatetimeIndex[UTC-naive], cols: Open,High,Low,Close,Volume)
  meta : {"symbol","rows","start","end","confidence","interval","raw_symbol"}

LangGraph düğümü:
  data_retrieval_node(state) -> {"raw_data": df, "meta": meta}

Not: Bu modül herhangi bir API anahtarı gerektirmez.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import time
import math
import datetime as dt

import pandas as pd
import requests


# ----------------------------- Yardımcılar ---------------------------------- #

_BINANCE_BASE = "https://api.binance.com"
# Daha sakin istekler için ufak bir bekleme (saniye):
_RATE_SLEEP = 0.25
# Tek sayfada maksimum 1000 kline alınıyor:
_MAX_LIMIT = 1000

_INTERVAL_MAP = {
    # Binance interval string → saniye
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "6h": 6 * 60 * 60,
    "8h": 8 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
    "3d": 3 * 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
    "1M": 30 * 24 * 60 * 60,  # yaklaşık
}


def _to_symbol(usymbol: str) -> str:
    """
    Kullanıcı sembolünü Binance spot sembole çevirir.
    'BTC' -> 'BTCUSDT', 'ETH' -> 'ETHUSDT'
    Kullanıcı zaten 'BTCUSDT' gibi verirse aynen döner.
    """
    usymbol = usymbol.strip().upper()
    if usymbol.endswith("USDT") or usymbol.endswith("USD"):
        return usymbol
    return f"{usymbol}USDT"


def _utc_now_ms() -> int:
    return int(time.time() * 1000)


def _date_to_ms(d: dt.date) -> int:
    """Naive date → UTC midnight (ms)."""
    return int(dt.datetime(d.year, d.month, d.day).timestamp() * 1000)


def _ts_to_iso(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000.0).date().isoformat()


def _confidence_from_span(first_ms: int, last_ms: int) -> str:
    years = (last_ms - first_ms) / (365.0 * 24 * 3600 * 1000)
    if years < 1.0:
        print("⚠️ Veri kapsami < 1 yıl → Güvenilirlik: LOW")
        return "low"
    elif years < 3.0:
        print("⚠️ Veri kapsami 1–3 yıl → Güvenilirlik: MEDIUM")
        return "medium"
    else:
        print("✅ Veri kapsami ≥ 3 yıl → Güvenilirlik: HIGH")
        return "high"


def _fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    max_pages: int = 1000,
) -> List[List]:
    """
    Binance /api/v3/klines sayfa sayfa çeker.
    Dönüş: ham kline listesi (her öğe Binance'in verdiği liste)

    Raises:
        ValueError: sembol/interval hatalı veya veri yoksa.
    """
    all_rows: List[List] = []
    curr = start_ms
    step_sec = _INTERVAL_MAP.get(interval)
    if step_sec is None:
        raise ValueError(f"Geçersiz interval: {interval}")

    page = 0
    while curr < end_ms and page < max_pages:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": curr,
            "endTime": end_ms,
            "limit": _MAX_LIMIT,
        }
        r = requests.get(f"{_BINANCE_BASE}/api/v3/klines", params=params, timeout=15)
        if r.status_code != 200:
            raise ValueError(f"Binance HTTP {r.status_code}: {r.text[:200]}")

        rows = r.json()
        if not rows:
            break

        all_rows.extend(rows)

        # Son barın kapanış zamanı + 1 interval ileri
        last_close_ms = rows[-1][6]  # close time
        # Binance closeTime zaten bir sonraki açık zaman - 1ms şeklinde; +1ms ile ilerletelim
        curr = last_close_ms + 1
        page += 1
        time.sleep(_RATE_SLEEP)

        # Koruma: çok küçük artışla kilitlenmeyi önle
        # (özellikle düşük interval ve kısa aralıklarda)
        if page > 5 and len(rows) < _MAX_LIMIT // 4:
            # Az veri geldiğine göre bitmiş olabilir
            if curr >= end_ms:
                break

    return all_rows


def _rows_to_df(rows: List[List]) -> pd.DataFrame:
    """
    Binance ham kline satırlarını DataFrame’e çevirir (Open, High, Low, Close, Volume).
    """
    if not rows:
        raise ValueError("Boş kline verisi.")

    data: List[Tuple] = []
    for r in rows:
        # Binance /api/v3/klines şeması:
        # [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume,
        #   numberOfTrades, takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, ignore ]
        open_ms = int(r[0])
        o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4]); v = float(r[5])
        data.append((open_ms, o, h, l, c, v))

    df = pd.DataFrame(data, columns=["open_ms", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["open_ms"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    if df.empty or len(df) < 1:
        raise ValueError("DataFrame boş.")
    return df


# ----------------------------- Genel API ------------------------------------ #

def get_stock_data(
    ticker: str,
    *,
    lookback_years: float = 5.0,
    interval: str = "1d",
    end_date: Optional[dt.date] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Binance spot’tan orijinal OHLCV verisi indirir.

    Args:
        ticker (str): "BTC", "ETH" gibi sade ad veya "BTCUSDT" gibi tam sembol.
        lookback_years (float): Geriye dönük kaç yıl (örn. 5.0).
        interval (str): Binance interval ("1d", "1h", "1w", "1M", ...).
        end_date (date|None): Bitiş günü (UTC). None ise bugünün UTC tarihi.

    Returns:
        (pd.DataFrame, dict):
            - df: OHLCV DataFrame (index=DatetimeIndex, cols=Open..Volume)
            - meta: {"symbol","rows","start","end","confidence","interval","raw_symbol"}

    Raises:
        ValueError: Hatalı sembol/interval/veri yoksa.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("ticker zorunludur (örn. 'BTC').")

    raw_symbol = ticker.strip().upper()
    symbol = _to_symbol(raw_symbol)

    if interval not in _INTERVAL_MAP:
        raise ValueError(f"Geçersiz interval: {interval}")

    if end_date is None:
        end_date = dt.datetime.utcnow().date()

    # Başlangıç ve bitiş epoch ms
    end_ms = _date_to_ms(end_date) + (24 * 3600 * 1000) - 1  # gün sonu
    start_days = int(math.ceil(lookback_years * 365))
    start_date = end_date - dt.timedelta(days=start_days)
    start_ms = _date_to_ms(start_date)

    # Kline çek
    rows = _fetch_klines(symbol, interval, start_ms, end_ms)
    if not rows:
        raise ValueError(f"Veri alınamadı: {symbol}")

    # DF'e çevir
    df = _rows_to_df(rows)

    # Meta ve güven
    first_ms = rows[0][0]
    last_ms = rows[-1][6]
    confidence = _confidence_from_span(first_ms, last_ms)

    meta = {
        "raw_symbol": raw_symbol,
        "symbol": symbol,
        "interval": interval,
        "rows": len(df),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "confidence": confidence,
    }

    return df, meta


def data_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph düğümü gibi davranır: state içinden parametreleri alır ve
    {'raw_data': df, 'meta': meta} döndürür.

    Beklenen state alanları:
        - ticker (str)           → zorunlu
        - lookback_years (float) → ops., varsayılan 5.0
        - interval (str)         → ops., varsayılan "1d"

    Örnek:
        out = data_retrieval_node({"ticker": "BTC", "lookback_years": 5})
        df, meta = out["raw_data"], out["meta"]
    """
    ticker = str(state.get("ticker", "")).strip()
    if not ticker:
        raise ValueError("state['ticker'] zorunlu.")

    lookback_years = float(state.get("lookback_years", 5.0))
    interval = str(state.get("interval", "1d"))

    df, meta = get_stock_data(
        ticker=ticker,
        lookback_years=lookback_years,
        interval=interval,
    )
    return {"raw_data": df, "meta": meta}

def _to_date(x: Any) -> dt.date:
    if isinstance(x, dt.date):
        return x
    return pd.to_datetime(x).date()

def _years_between(start_date: dt.date, end_date: dt.date) -> float:
    days = (end_date - start_date).days
    # 0 gelirse Binance en az 1 gün verisine denk getirmek için küçük bir tampon verelim
    return max(0.01, days / 365.25)

def fetch(symbol: str, start: str, end: str, source_cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Router uyumlu giriş noktası.
    Args:
      symbol: "BTC" veya "BTCUSDT"
      start, end: "YYYY-MM-DD"
      source_cfg: {"interval": "1d", ...} (şimdilik interval'ı alıyoruz)

    Returns:
      pd.DataFrame (index=DatetimeIndex, cols: hem Büyük harf hem küçük harf OHLCV)
    """
    cfg = source_cfg or {}
    interval = str(cfg.get("interval", "1d"))

    s_date = _to_date(start)
    e_date = _to_date(end)
    years = _years_between(s_date, e_date)

    # Mevcut fonksiyonunu kullan
    df, _meta = get_stock_data(
        ticker=symbol,
        lookback_years=years,
        interval=interval,
        end_date=e_date,
    )

    # İstenilen aralığa kırp (güvenli)
    try:
        df = df.loc[(df.index.date >= s_date) & (df.index.date <= e_date)]
    except Exception:
        pass

    # Pipeline’daki diğer ajanlar için küçük harf kolonları da ekle
    for C in ["Open", "High", "Low", "Close", "Volume"]:
        c = C.lower()
        if C in df.columns and c not in df.columns:
            df[c] = df[C]

    return df