# -*- coding: utf-8 -*-
"""
Persistence Agent (SQLite)
- Hibrit kalıcılık: pipeline çıktılarının SQLite'e yazılması.
- Basit API: open_db, init_db_if_needed, save_run, save_* yardımcıları.
- LangGraph uyumlu persistence_node (opsiyonel).

Tablolar, proje kökünde oluşturduğun "chimera.db" içindeki DDL ile uyumludur.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Iterable, Tuple
import os
import json
import sqlite3 as sq
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Konum & Bağlantı
# ---------------------------------------------------------------------------

def _project_root() -> str:
    # .../src/agents/persistence.py -> proje kökü
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))

def _default_db_path() -> str:
    env = os.getenv("CHIMERA_DB")
    if env:
        return env
    return os.path.join(_project_root(), "chimera.db")

DB_PATH = _default_db_path()

def open_db(db_path: Optional[str] = None) -> sq.Connection:
    """
    SQLite bağlantısını açar ve foreign_keys'i aktif eder.
    """
    path = db_path or DB_PATH
    conn = sq.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db_if_needed(conn: sq.Connection) -> None:
    """
    Şema dışarıdan yüklendi varsayımıyla sadece foreign_keys'i garanti eder.
    İstersen buraya DDL'yi de gömebilirsin; şimdilik no-op.
    """
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.commit()

# ---------------------------------------------------------------------------
# Yardımcı dönüşümler
# ---------------------------------------------------------------------------

def _bool(x: Any) -> int:
    return int(bool(x))

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

# ---------------------------------------------------------------------------
# Kayıtlayıcılar
# ---------------------------------------------------------------------------

def save_run(
    conn: sq.Connection,
    *,
    ticker: str,
    lookback_years: Optional[float] = None,
    confidence: Optional[str] = None,   # "high"|"medium"|"low"
    regime: Optional[str] = None,       # "bull_high_vol" vs.
    risk_note: Optional[str] = None,
    timestamp: Optional[str] = None,    # ISO; None -> CURRENT_TIMESTAMP
) -> int:
    """
    runs tablosuna bir satır ekler ve run_id döndürür.
    """
    if timestamp is None:
        # SQLite tarafına bırakmak yerine burada ISO üretmek istersen:
        # timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        # Ama DEFAULT CURRENT_TIMESTAMP da yeterli.
        pass

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
    def pick(c: str) -> Optional[float]:
        return float(analyzed_df.at[dt, c]) if c in cols and pd.notna(analyzed_df.at[dt, c]) else None

    rows: list[Tuple] = []
    for dt in analyzed_df.index:
        date_str = pd.to_datetime(dt).date().isoformat()
        rows.append((
            run_id,
            date_str,
            pick("Open"), pick("High"), pick("Low"), pick("Close"), pick("Volume"),
            pick("SMA_14"), pick("SMA_50"), pick("EMA_12"), pick("EMA_26"),
            pick("RSI_14"), pick("MACD_line"), pick("BB_upper"), pick("BB_middle"), pick("BB_lower"),
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

# ---------------------------------------------------------------------------
# LangGraph düğümü (opsiyonel)
# ---------------------------------------------------------------------------

def persistence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pipeline state'inden kalıcılaşacak parçaları alır ve DB'ye yazar.
    Beklenenler (hepsi zorunlu değil; olanı kaydeder):
      - ticker (str), lookback_years (float), meta (data_retrieval)
      - analyzed_data (pd.DataFrame)
      - forecast_data / forecast_meta  (ya da candidates)
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