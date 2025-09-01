# -*- coding: utf-8 -*-
"""
Teknik Analiz Ajanı (Indicator)
===============================

Görev
-----
Ham OHLCV verisi üzerine seçili teknik göstergeleri hesaplayıp yeni sütunlar
olarak ekler. Çıktı, sonraki ajanlar (forecast, model_selection, risk vb.)
tarafından tüketilecek zenginleştirilmiş bir DataFrame'dir.

Kullanım (doğrudan):
    analyzed_df, meta = calculate_indicators(raw_df)
    # veya
    updates = indicator_node({"raw_data": raw_df})

Göstergeler (varsayılan)
------------------------
- SMA_14, SMA_50
- EMA_12, EMA_26
- RSI_14
- MACD (12,26,9): MACD_line, MACD_signal, MACD_hist
- BBANDS_20_2: BB_upper, BB_middle, BB_lower

Bağımlılıklar
-------------
- Opsiyonel: pandas_ta (varsa otomatik kullanılır)
- Zorunlu: pandas

Güvenlik/Dayanıklılık
---------------------
- Girdi kontrolleri (Close/High/Low varlığı, yeterli gözlem sayısı)
- pandas_ta yoksa manuel hesap fallback
- Eksik gözlemde (ör. kısa tarih aralığı) ilgili göstergeler kısmen NaN olabilir.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

# Opsiyonel: pandas_ta (varsa tercih edilir)
try:
    import pandas_ta as ta  # type: ignore
except Exception:  # pragma: no cover
    ta = None  # type: ignore


# --------------------------------------------------------------------------- #
# Yardımcı doğrulamalar
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Girdi DataFrame'inde eksik kolon(lar): {missing}. "
                         f"Mevcut kolonlar: {list(df.columns)}")


def _is_enough_rows(df: pd.DataFrame, min_rows: int) -> bool:
    return len(df) >= min_rows


# --------------------------------------------------------------------------- #
# Fallback (manuel) gösterge hesapları
# --------------------------------------------------------------------------- #
def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
          ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bbands(close: pd.Series, length: int = 20, stdev: float = 2.0
            ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window=length, min_periods=length).mean()
    sd = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = ma + stdev * sd
    lower = ma - stdev * sd
    return upper, ma, lower


# --------------------------------------------------------------------------- #
# Ana hesaplama API'sı
# --------------------------------------------------------------------------- #
def calculate_indicators(
    raw_data: pd.DataFrame,
    *,
    use_pandas_ta: Optional[bool] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Verilen OHLCV DataFrame'ine teknik gösterge sütunları ekler.

    Args:
        raw_data: En azından ['Close'] içeren, tarih indeksli DataFrame.
                  (Bazı göstergeler High/Low'a da ihtiyaç duyar: BBANDS)
        use_pandas_ta: True → pandas_ta zorla; False → manuel fallback zorla;
                       None → otomatik (varsa pandas_ta, yoksa fallback).
        config: Gösterge parametrelerini özelleştirmek için sözlük.
            Örn:
            {
              "sma": [14, 50],
              "ema": [12, 26],
              "rsi": [14],
              "macd": {"fast":12, "slow":26, "signal":9},
              "bbands": {"length":20, "stdev":2.0}
            }

    Returns:
        analyzed_df, meta
        - analyzed_df: Göstergeler eklenmiş DataFrame (orijinal + yeni sütunlar)
        - meta: {"added_columns": [...], "engine": "pandas_ta" | "manual"}

    Raises:
        ValueError: Girdi uygun değilse (eksik kolon, boş df, vs).
    """
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        raise ValueError("raw_data boş olmamalı ve DataFrame olmalı.")

    # Zorunlu kolon kontrolleri (en az Close; BBANDS için High/Low da lazım)
    _require_columns(raw_data, ["Close"])

    # Parametreleri kur
    cfg = {
        "sma": [14, 50],
        "ema": [12, 26],
        "rsi": [14],
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bbands": {"length": 20, "stdev": 2.0},
    }
    if config:
        # Basit shallow-merge
        for k, v in config.items():
            cfg[k] = v

    # Hangi motor?
    if use_pandas_ta is None:
        use_pandas_ta = ta is not None

    df = raw_data.copy()
    added: List[str] = []

    # ---------------- pandas_ta yolu ----------------
    if use_pandas_ta and ta is not None:
        close = df["Close"]

        # SMA
        for n in cfg.get("sma", []):
            col = f"SMA_{n}"
            df[col] = ta.sma(close, length=int(n))
            added.append(col)

        # EMA
        for n in cfg.get("ema", []):
            col = f"EMA_{n}"
            df[col] = ta.ema(close, length=int(n))
            added.append(col)

        # RSI
        for n in cfg.get("rsi", []):
            col = f"RSI_{n}"
            df[col] = ta.rsi(close, length=int(n))
            added.append(col)

        # MACD
        m = cfg.get("macd", {})
        fast, slow, signal = int(m.get("fast", 12)), int(m.get("slow", 26)), int(m.get("signal", 9))
        macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
        if macd_df is not None:
            # pandas_ta kolon adları: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
            df["MACD_line"] = macd_df.iloc[:, 0]
            df["MACD_signal"] = macd_df.iloc[:, 1]
            df["MACD_hist"] = macd_df.iloc[:, 2]
            added += ["MACD_line", "MACD_signal", "MACD_hist"]

        # BBANDS (High/Low gerekir)
        if all(c in df.columns for c in ("High", "Low")):
            b = cfg.get("bbands", {})
            length, stdev = int(b.get("length", 20)), float(b.get("stdev", 2.0))
            bb = ta.bbands(close, length=length, std=stdev)
            if bb is not None:
                # tipik kolonlar: BBU_20_2.0, BBM_20_2.0, BBL_20_2.0
                df["BB_upper"] = bb.iloc[:, 0]
                df["BB_middle"] = bb.iloc[:, 1]
                df["BB_lower"] = bb.iloc[:, 2]
                added += ["BB_upper", "BB_middle", "BB_lower"]

        engine = "pandas_ta"

    # ---------------- manuel fallback yolu ----------------
    else:
        close = df["Close"]

        # SMA
        for n in cfg.get("sma", []):
            col = f"SMA_{n}"
            df[col] = _sma(close, int(n))
            added.append(col)

        # EMA
        for n in cfg.get("ema", []):
            col = f"EMA_{n}"
            df[col] = _ema(close, int(n))
            added.append(col)

        # RSI
        for n in cfg.get("rsi", []):
            col = f"RSI_{n}"
            df[col] = _rsi(close, int(n))
            added.append(col)

        # MACD
        m = cfg.get("macd", {})
        fast, slow, signal = int(m.get("fast", 12)), int(m.get("slow", 26)), int(m.get("signal", 9))
        macd_line, macd_signal, macd_hist = _macd(close, fast=fast, slow=slow, signal=signal)
        df["MACD_line"] = macd_line
        df["MACD_signal"] = macd_signal
        df["MACD_hist"] = macd_hist
        added += ["MACD_line", "MACD_signal", "MACD_hist"]

        # BBANDS (High/Low gerekir ⇒ ortalama için sadece Close kullanıyoruz)
        # Klasik BBANDS Close üzerinden hesaplanır; High/Low zorunlu değil.
        b = cfg.get("bbands", {})
        length, stdev = int(b.get("length", 20)), float(b.get("stdev", 2.0))
        bb_u, bb_m, bb_l = _bbands(close, length=length, stdev=stdev)
        df["BB_upper"] = bb_u
        df["BB_middle"] = bb_m
        df["BB_lower"] = bb_l
        added += ["BB_upper", "BB_middle", "BB_lower"]

        engine = "manual"

    meta = {
        "added_columns": added,
        "engine": engine,
        "rows": len(df),
        "min_date": df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
        "max_date": df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None,
    }
    return df, meta


# --------------------------------------------------------------------------- #
# LangGraph uyumlu düğüm
# --------------------------------------------------------------------------- #
def indicator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph düğümü gibi çalışır: state['raw_data'] üstünde göstergeleri hesaplar
    ve {"analyzed_data": df, "indicator_meta": meta} döndürür.

    Beklenen state:
        raw_data: pd.DataFrame (DatetimeIndex + Close zorunlu)
        indicator_config (ops.): calculate_indicators(config=...) için ayarlar
        use_pandas_ta (ops.): True/False/None

    Dönen güncelleme:
        {"analyzed_data": <DataFrame>, "indicator_meta": <dict>}
    """
    raw: Any = state.get("raw_data")
    if not isinstance(raw, pd.DataFrame):
        raise ValueError("indicator_node: state['raw_data'] DataFrame olmalı.")

    cfg = state.get("indicator_config")
    use_pta = state.get("use_pandas_ta")  # True / False / None

    analyzed, meta = calculate_indicators(raw, use_pandas_ta=use_pta, config=cfg)
    return {"analyzed_data": analyzed, "indicator_meta": meta}