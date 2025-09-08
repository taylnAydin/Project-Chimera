# agents/backtester.py
"""
Chimera • Backtester
- Stratejiyi geçmiş veride test eder.
- Sinyal sağlanmazsa forecasts['y_hat'] ile basit yön stratejisi türetir.
- SL/TP uygular, ücret/slippage dikkate alır, metrikleri hesaplar.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from math import sqrt

try:
    import pandas as pd  # type: ignore
    import numpy as np   # type: ignore
except Exception:
    pd = None  # type: ignore
    np = None  # type: ignore


class BacktestResult(TypedDict, total=False):
    metrics: Dict[str, float]
    equity: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]


# ----------------------
# Yardımcı hesaplar
# ----------------------

def _to_iso(ts: Any) -> str:
    if pd is None:
        return str(ts)
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.isoformat()

def _max_drawdown(equity: Any) -> float:
    if pd is None or equity is None or getattr(equity, "empty", True):
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def _sharpe(returns: Any, periods_per_year: int = 252, rf: float = 0.0) -> float:
    if pd is None or returns is None:
        return 0.0
    try:
        nz = returns.dropna()
        if nz.empty:
            return 0.0
        ex = nz - rf / periods_per_year
        mu = ex.mean()
        sd = ex.std(ddof=0)
        if sd == 0 or (np is not None and np.isnan(sd)):
            return 0.0
        return float((mu / sd) * sqrt(periods_per_year))
    except Exception:
        return 0.0

def _cagr(equity: Any, periods_per_year: int = 252) -> float:
    if pd is None or equity is None or getattr(equity, "empty", True):
        return 0.0
    try:
        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        years = max(1e-9, len(equity) / periods_per_year)
        return float((1.0 + total_ret) ** (1.0 / years) - 1.0)
    except Exception:
        return 0.0

def _apply_fees_slippage(px: float, fee_bps: float, slip_bps: float, side: int) -> float:
    """İşlem fiyatını ücret/slippage ile ayarla. side: +1 alım, -1 satım."""
    adj = (fee_bps + slip_bps) / 10000.0
    return px * (1.0 + adj) if side > 0 else px * (1.0 - adj)

def _derive_signals_from_forecast(df: Any, forecasts: Dict[str, Any] | None, hold_threshold: float) -> Any:
    """y_hat'in günlük değişimine göre yön: >th → long, <−th → short, aksi 0."""
    if pd is None or forecasts is None:
        return None
    y_hat = forecasts.get("y_hat")
    if y_hat is None or not isinstance(y_hat, pd.Series):
        return None
    y_hat = y_hat.reindex(df.index).astype(float)
    yret = y_hat.pct_change()
    sig = pd.Series(0, index=df.index)
    sig[yret > hold_threshold] = 1
    sig[yret < -hold_threshold] = -1
    return sig

def _backtest_core(df: Any,
                   signals: Any,
                   start_capital: float,
                   fee_bps: float,
                   slippage_bps: float,
                   position_mode: str,
                   position_size: float,
                   risk_cfg: Optional[Dict[str, float]] = None) -> BacktestResult:
    """
    Basit trade simülasyonu: sinyale göre pozisyon al/sat, SL/TP uygula.
    - position_mode="cash": her işlemde sermayenin 'position_size' oranı kullanılır.
    - position_mode="fixed": her işlemde sabit adet (position_size) alınır/satılır.
    """
    if pd is None or df is None or "close" not in getattr(df, "columns", []):
        return BacktestResult(metrics={}, equity=[], trades=[])

    close = df["close"].astype(float)
    if signals is None or not isinstance(signals, pd.Series):
        return BacktestResult(metrics={}, equity=[], trades=[])

    signals = signals.reindex(close.index).fillna(0).astype(int)

    # Risk parametreleri
    sl = (risk_cfg or {}).get("stop_loss")
    tp = (risk_cfg or {}).get("take_profit")

    # Simülasyon durumları
    equity: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    cash = float(start_capital)
    pos = 0.0           # adet (unit)
    side = 0            # 1: long, -1: short, 0: flat
    entry_px: Optional[float] = None
    entry_idx: Any = None

    for i, (ts, px) in enumerate(close.items()):
        sig = int(signals.iloc[i])

        def _should_exit(cur_px: float) -> bool:
            if side == 0 or entry_px is None:
                return False
            # SL/TP kontrolü
            if sl is not None and tp is not None:
                rr = (cur_px - entry_px) / entry_px * side
                if rr <= -abs(sl) or rr >= abs(tp):
                    return True
            return sig != side  # sinyal yön değiştiyse çık

        # pozisyonu kapat
        if _should_exit(float(px)):
            exit_px = _apply_fees_slippage(float(px), fee_bps, slippage_bps, -side)
            cash += pos * exit_px * side  # short için side=-1 olduğundan doğru işaret
            pnl = pos * (exit_px - entry_px) * side if entry_px is not None else 0.0
            trades.append({
                "entry_ts": _to_iso(entry_idx),
                "entry_px": float(entry_px),
                "side": side,
                "exit_ts": _to_iso(ts),
                "exit_px": float(exit_px),
                "pnl": float(pnl),
            })
            pos, side, entry_px, entry_idx = 0.0, 0, None, None

        # pozisyon yoksa ve sinyal varsa aç
        if side == 0 and sig != 0:
            # alım/satım büyüklüğü
            if position_mode == "cash":
                notional = cash * float(position_size)
                qty = (notional / float(px))
            else:  # "fixed"
                qty = float(position_size)

            trade_px = _apply_fees_slippage(float(px), fee_bps, slippage_bps, sig)

            if sig > 0:  # long aç
                cost = qty * trade_px
                if cost > cash:
                    qty = cash / trade_px  # yettiği kadar al
                    cost = qty * trade_px
                cash -= cost
                pos = qty
                side = 1
            else:        # short aç
                pos = qty
                side = -1
                cash += qty * trade_px  # kısa satış geliri kasaya eklenir

            entry_px = trade_px
            entry_idx = ts

        # equity hesapla
        mtm = cash
        if side != 0 and entry_px is not None:
            mtm += side * pos * float(px)
        equity.append({"ts": _to_iso(ts), "equity": float(mtm)})

    # açık pozisyonu son bar'da kapat
    if side != 0 and entry_px is not None:
        last_ts = close.index[-1]
        exit_px = _apply_fees_slippage(float(close.iloc[-1]), fee_bps, slippage_bps, -side)
        cash += pos * exit_px * side
        pnl = pos * (exit_px - entry_px) * side
        trades.append({
            "entry_ts": _to_iso(entry_idx),
            "entry_px": float(entry_px),
            "side": side,
            "exit_ts": _to_iso(last_ts),
            "exit_px": float(exit_px),
            "pnl": float(pnl),
        })
        pos, side, entry_px, entry_idx = 0.0, 0, None, None
        equity[-1]["equity"] = float(cash)

    # metrikler
    if pd is not None:
        equity_ser = pd.Series([e["equity"] for e in equity],
                               index=[pd.Timestamp(e["ts"]) for e in equity])
        ret_ser = equity_ser.pct_change()
        total_return = float(equity_ser.iloc[-1] / equity_ser.iloc[0] - 1.0)
        cagr = _cagr(equity_ser)
        sharpe = _sharpe(ret_ser)
        max_dd = _max_drawdown(equity_ser)
    else:
        equity_ser = None
        total_return = cagr = sharpe = max_dd = 0.0

    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = float(wins / max(1, len(trades)))

    return BacktestResult(
        metrics={
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "trades": float(len(trades)),
        },
        equity=equity,
        trades=trades
    )


# ----------------------
# Dış API
# ----------------------

def run(df: Any,
        forecasts: Dict[str, Any] | None = None,
        signals: Any = None,
        cfg: Optional[Dict[str, Any]] = None) -> BacktestResult:
    """
    Ana giriş noktası.
    - `signals` verilmişse onu kullanır.
    - `signals` None ise `forecasts['y_hat']` ile yön sinyali türetir.
    """
    cfg = cfg or {}
    start_capital = float(cfg.get("start_capital", 10000.0))
    fee_bps = float(cfg.get("fee_bps", 5.0))
    slippage_bps = float(cfg.get("slippage_bps", 5.0))
    position_mode = str(cfg.get("position_mode", "cash"))
    position_size = float(cfg.get("position_size", 1.0))
    hold_threshold = float(cfg.get("hold_threshold", 0.0005))
    risk_cfg = cfg.get("risk")

    if signals is None:
        signals = _derive_signals_from_forecast(df, forecasts, hold_threshold)

    return _backtest_core(
        df=df,
        signals=signals,
        start_capital=start_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        position_mode=position_mode,
        position_size=position_size,
        risk_cfg=risk_cfg
    )