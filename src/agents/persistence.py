# -*- coding: utf-8 -*-
"""
Persistence Agent (SQLite)
==========================

- DB varsayılanı: <project_root>/data/chimera.db  (tek dosya, her run aynı dosyada)
- ENV override:   CHIMERA_DB=/abs/path/to/chimera.db
- Kolay API: open_db, init_db_if_needed, save_run, save_* yardımcıları
- Router köprüsü: run_start/artifacts/backtest/run_end event logları
- LangGraph uyumu: persistence_node(state) (opsiyonel)

Not: Tablo şemasını dışarıdan yönetiyorsan, init_db_if_needed() şu an no-op.
Sadece events tabloyu otomatik oluşturuyoruz (log amaçlı).
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Iterable, Tuple

import os
import json
import sqlite3 as sq
from datetime import datetime

import pandas as pd


# -----------------------------------------------------------------------------
# Konum & Bağlantı
# -----------------------------------------------------------------------------

def _project_root() -> str:
    # .../src/agents/persistence.py -> proje kökü
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))

def _default_db_path() -> str:
    """
    Varsayılan DB yolu:
      - ENV CHIMERA_DB varsa onu kullan (klasörünü de oluştur)
      - Yoksa <project_root>/data/chimera.db (data klasörünü oluştur)
    """
    env = os.getenv("CHIMERA_DB")
    if env:
        abs_env = os.path.abspath(env)
        os.makedirs(os.path.dirname(abs_env), exist_ok=True)
        return abs_env

    data_dir = os.path.join(_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "chimera.db")

DB_PATH = _default_db_path()

def open_db(db_path: Optional[str] = None) -> sq.Connection:
    """
    SQLite bağlantısını açar ve foreign_keys'i aktif eder.
    Dosya/klasör garantisi vardır.
    """
    path = os.path.abspath(db_path or DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sq.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db_if_needed(conn: sq.Connection) -> None:
    """
    Şemayı dışarıdan yönetiyorsan burada sadece foreign_keys'i garanti ediyoruz.
    Dilersen DDL oluşturmayı buraya ekleyebilirsin.
    """
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.commit()


# -----------------------------------------------------------------------------
# Yardımcı dönüşümler
# -----------------------------------------------------------------------------

def _bool(x: Any) -> int:
    return int(bool(x))

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


# -----------------------------------------------------------------------------
# Kayıtlayıcılar (ana tablolar)
# -----------------------------------------------------------------------------

def save_run(
    conn: sq.Connection,
    *,
    ticker: str,
    lookback_years: Optional[float] = None,
    confidence: Optional[str] = None,   # "high"|"medium"|"low"
    regime: Optional[str] = None,       # ör. "bull_high_vol"
    risk_note: Optional[str] = None,
    timestamp: Optional[str] = None,    # ISO; None -> CURRENT_TIMESTAMP
) -> int:
    """
    runs tablosuna bir satır ekler ve run_id döndürür.
    """
    cur = conn.cursor()
    if timestamp is None:
        cur.execute(
            """
            INSERT INTO runs (ticker, lookback_years, confidence, regime, risk_note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticker, lookback_years, confidence, regime, risk_note),
        )
    else:
        cur.execute(
            """
            INSERT INTO runs (timestamp, ticker, lookback_years, confidence, regime, risk_note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, ticker, lookback_years, confidence, regime, risk_note),
        )
    run_id = int(cur.lastrowid)
    conn.commit()
    return run_id


def save_snapshots(
    conn: sq.Connection,
    run_id: int,
    analyzed_df: pd.DataFrame,
) -> int:
    """
    data_snapshots tablosuna (OHLCV + temel indikatörler) toplu insert.
    DataFrame index'i Datetime olmalı.
    """
    if not isinstance(analyzed_df.index, pd.DatetimeIndex):
        raise ValueError("save_snapshots: analyzed_df.index DatetimeIndex olmalı.")

    cols = analyzed_df.columns

    def pick(dt, c: str) -> Optional[float]:
        return float(analyzed_df.at[dt, c]) if c in cols and pd.notna(analyzed_df.at[dt, c]) else None

    rows: list[Tuple] = []
    for dt in analyzed_df.index:
        date_str = pd.to_datetime(dt).date().isoformat()
        rows.append((
            run_id,
            date_str,
            pick(dt, "Open"), pick(dt, "High"), pick(dt, "Low"), pick(dt, "Close"), pick(dt, "Volume"),
            pick(dt, "SMA_14"), pick(dt, "SMA_50"), pick(dt, "EMA_12"), pick(dt, "EMA_26"),
            pick(dt, "RSI_14"), pick(dt, "MACD_line"), pick(dt, "BB_upper"), pick(dt, "BB_middle"), pick(dt, "BB_lower"),
        ))

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO data_snapshots
          (run_id,date,open,high,low,close,volume,
           sma14,sma50,ema12,ema26,rsi14,macd_line,bb_upper,bb_middle,bb_lower)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def save_forecasts(
    conn: sq.Connection,
    run_id: int,
    *,
    model: str,                     # "ARIMA"|"ARIMAX"|"ETS"
    order_text: Optional[str],      # "(1,1,1)" ya da ETS cfg json stringi
    horizon: int,
    forecast_series: pd.Series,     # index: future dates
    aic: Optional[float] = None,
    rmse: Optional[float] = None,
) -> int:
    """
    forecasts tablosuna bir modelin tüm öngörü çizelgesini yazar.
    """
    rows: list[Tuple] = []
    for dt, val in forecast_series.items():
        date_str = pd.to_datetime(dt).date().isoformat()
        rows.append((run_id, model, order_text, horizon, date_str, float(val), aic, rmse))

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO forecasts
          (run_id, model, order_text, horizon, date, forecast_price, aic, rmse)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def save_risk_plan(
    conn: sq.Connection,
    run_id: int,
    plan: Dict[str, Any],
) -> int:
    """
    risk_plans tablosuna tek satır yazar.
    Beklenen plan alanları: direction, price, units, notional, risk_pct_effective,
                            stop_price, take_profit_price, atr, atr_window, note
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO risk_plans
          (run_id, direction, price, units, notional, risk_pct,
           stop_price, take_profit, atr, atr_window, note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            plan.get("direction"),
            plan.get("price"),
            plan.get("units"),
            plan.get("notional"),
            plan.get("risk_pct_effective"),
            plan.get("stop_price"),
            plan.get("take_profit_price"),
            plan.get("atr"),
            plan.get("atr_window"),
            plan.get("note"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_quality_report(
    conn: sq.Connection,
    run_id: int,
    report: Dict[str, Any],
) -> int:
    """
    quality_reports tablosuna tek satır yazar.
    """
    flags = report.get("quality_flags", {})
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO quality_reports
          (run_id, is_ok, severity,
           cols_ok, index_ok, na_ok, vol_ok, outlier_ok, candle_ok, gap_ok,
           warnings, meta_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            _bool(report.get("is_ok")),
            report.get("severity"),
            _bool(flags.get("cols_ok")),
            _bool(flags.get("index_ok")),
            _bool(flags.get("na_ok")),
            _bool(flags.get("vol_ok")),
            _bool(flags.get("outlier_ok")),
            _bool(flags.get("candle_ok")),
            _bool(flags.get("gap_ok")),
            _safe_json(report.get("warnings", [])),
            _safe_json(report.get("meta", {})),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


# -----------------------------------------------------------------------------
# LangGraph düğümü (opsiyonel)
# -----------------------------------------------------------------------------

def persistence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline state'inden kalıcılaşacak parçaları alır ve DB'ye yazar.

    Beklenen (hepsi zorunlu değil; olanı kaydeder):
      - ticker (str), lookback_years (float), meta (data_retrieval)
      - analyzed_data (pd.DataFrame)
      - forecast_data / forecast_meta
      - risk_plan
      - quality_report
      - regime / regime_meta

    Döner: {"run_id": int}
    """
    conn = open_db()
    init_db_if_needed(conn)

    ticker = state.get("ticker") or "UNKNOWN"
    lookback_years = state.get("lookback_years")
    confidence = (state.get("meta") or {}).get("confidence")
    regime = state.get("regime")
    risk_note = (state.get("risk_plan") or {}).get("note")

    run_id = save_run(
        conn,
        ticker=ticker,
        lookback_years=lookback_years,
        confidence=confidence,
        regime=regime,
        risk_note=risk_note,
    )

    # snapshots
    analyzed: Optional[pd.DataFrame] = state.get("analyzed_data")
    if isinstance(analyzed, pd.DataFrame) and not analyzed.empty:
        save_snapshots(conn, run_id, analyzed)

    # forecasts (tek model)
    fcs = state.get("forecast_data")
    fmeta = state.get("forecast_meta")
    if isinstance(fcs, pd.Series) and fmeta:
        order_text = _safe_json(fmeta.get("order")) if fmeta.get("model") != "ETS" else _safe_json(fmeta.get("cfg"))
        save_forecasts(
            conn,
            run_id,
            model=str(fmeta.get("model")),
            order_text=order_text,
            horizon=int(fmeta.get("horizon", len(fcs))),
            forecast_series=fcs,
            aic=fmeta.get("aic"),
            rmse=fmeta.get("rmse"),
        )

    # risk
    plan = state.get("risk_plan")
    if isinstance(plan, dict) and plan:
        save_risk_plan(conn, run_id, plan)

    # quality
    qrep = state.get("quality_report")
    if isinstance(qrep, dict) and qrep:
        save_quality_report(conn, run_id, qrep)

    conn.close()
    return {"run_id": run_id}


# -----------------------------------------------------------------------------
# Router uyumluluk köprüsü (events tablosu)
# -----------------------------------------------------------------------------

def _ensure_events_table(conn: sq.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT     NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            type     TEXT     NOT NULL,
            run_id   TEXT,
            symbol   TEXT,
            payload  TEXT
        )
    """)
    conn.commit()

def _append_event(event_type: str, run_id: str | None, symbol: str | None, payload: Dict[str, Any]) -> None:
    conn = open_db()
    try:
        _ensure_events_table(conn)
        conn.execute(
            "INSERT INTO events(type, run_id, symbol, payload) VALUES (?,?,?,?)",
            (event_type, run_id, symbol, _safe_json(payload)),
        )
        conn.commit()
    finally:
        conn.close()

def save_run_start(run_id: str, symbol: str, cfg: Dict[str, Any]) -> None:
    """
    Router 'start' node'unda çağrılır.
    Basitçe events tablosuna log atar. İstersek burada save_run(...) da çağırabiliriz.
    """
    _append_event("run_start", run_id, symbol, {"cfg": cfg})

def save_run_artifacts(
    run_id: str,
    symbol: str,
    df: Any,
    quality: Dict[str, Any],
    forecasts: Dict[str, Any],
    best: Dict[str, Any],
    regime: Dict[str, Any],
    risk_plan: Dict[str, Any],
    anomalies: Any,
    mode: str,
    halt_reason: Optional[str] = None,
) -> None:
    """Ajan çıktılarının özetini events tablosuna yazar."""
    try:
        df_len = int(len(df)) if df is not None else 0
    except Exception:
        df_len = 0

    try:
        forecast_keys = list((forecasts or {}).keys())
        if "_meta" in forecast_keys:
            forecast_keys.remove("_meta")
    except Exception:
        forecast_keys = []

    try:
        anomalies_count = len(anomalies or [])
    except Exception:
        anomalies_count = 0

    payload = {
        "mode": mode,
        "halt_reason": halt_reason,
        "df_len": df_len,
        "forecast_keys": forecast_keys,
        "quality": quality or {},
        "best": best or {},
        "regime": regime or {},
        "risk_plan": risk_plan or {},
        "anomalies_count": anomalies_count,
    }
    _append_event("artifacts", run_id, symbol, payload)

def save_backtest(run_id: str, symbol: str, backtest: Dict[str, Any]) -> None:
    """Backtest özetini events tablosuna yazar."""
    payload = {
        "metrics": (backtest or {}).get("metrics"),
        "trades_n": len((backtest or {}).get("trades") or []),
        "equity_n": len((backtest or {}).get("equity") or []),
    }
    _append_event("backtest", run_id, symbol, payload)

def save_run_end(run_id: str, status: str, error: Optional[str] = None) -> None:
    """Run bitiş kaydı (success/failed)."""
    _append_event("run_end", run_id, None, {"status": status, "error": error})