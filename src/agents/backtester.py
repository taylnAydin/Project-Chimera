"""
Backtester Agent
================

Amaç
-----
Günlük OHLC veri üzerinde basit ve sağlam bir backtest motoru sağlamak:
- Sinyal verilmezse SMA(10/20) crossover sinyali üretir.
- Router'dan gelen 'risk_plan' bilgilerini (risk yüzdesi, ATR çarpanları) tercihli kullanır.
- SL/TP (stop-loss/take-profit) seviyelerini ATR tabanında intraday kontrol eder.
- Maliyetler için slippage (bps) ve işlem ücreti (bps) uygular.
- Çıktı olarak temel metrikler, trade listesi ve equity eğrisi döndürür.

Kullanım
--------
>>> from src.agents import backtester
>>> out = backtester.run(df, forecasts=None, signals=None, cfg={"risk": risk_plan})
>>> out["metrics"], out["trades"], out["equity"]

Beklenen Girdi
--------------
df : pd.DataFrame
    - DatetimeIndex (artan sırada)
    - Zorunlu kolon: 'Close'
    - Tercihen: 'Open', 'High', 'Low' (intraday SL/TP kontrolü için)
forecasts : dict | None
    - Şimdilik kullanılmıyor (ileride sinyal üretimiyle bağlanabilir)
signals : pd.Series | None
    - Değerleri -1/0/+1 olan sinyal serisi (index=df.index). None ise SMA(10/20) ile üretilir.
cfg : dict
    - Aşağıdaki anahtarları içerebilir:
      {
        "risk": {                      # (opsiyonel) router risk_plan'dan taşınır
          "risk_pct_effective": 0.004, # %0.4
          "stop_atr_mult": 2.0,
          "tp_atr_mult": 3.0
        },
        "initial_equity": 10_000.0,
        "atr_window": 14,
        "slippage_bps": 5,             # 0.05%
        "fee_bps": 10,                 # 0.10% (giriş/çıkış başına)
        "long_only": True              # False ise short'a da izin verir
      }

Çıktı
-----
dict:
{
  "metrics": {
     "trades": int,            # işlem adedi
     "win_rate": float,        # kazanma oranı (0-1)
     "total_return": float,    # toplam getiri (0-1)
     "cagr": float,            # yıllıklandırılmış getiri (0-1)
     "max_dd": float,          # maksimum düşüş (0-1)
     "sharpe": float           # naive yıllıklandırılmış Sharpe (rf=0 varsayımı)
  },
  "trades": [
     {
       "entry_date": Timestamp,
       "exit_date":  Timestamp,
       "side": "long"|"short",
       "entry": float,
       "exit": float,
       "qty": float,
       "pnl": float,
       "r_mult": float|nan,    # risk başına R
       "reason": "sl"|"tp"|"signal"|"eod"
     }, ...
  ],
  "equity": pd.Series          # zaman içinde özsermaye
}

Notlar
------
- Sinyal değişimi "crossover anında" (+1/-1) tetiklenir, diğer günler 0'dır.
- ATR, OHLC varsa klasik True Range ile; yoksa Close değişimlerinden türetilmiş
  bir proxy ile hesaplanır.
- Short P&L: qty * (entry - px) (qty > 0 kabulüyle, işaret side ile kontrol edilir).
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import math
import numpy as np
import pandas as pd


# =============================================================================
# Yardımcı Fonksiyonlar
# =============================================================================
def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Girdi DataFrame'ini doğrular ve indeks ile sıralamayı garanti eder.

    Parameters
    ----------
    df : pd.DataFrame
        En az 'Close' kolonu bulunan, DatetimeIndex'li DataFrame.

    Returns
    -------
    pd.DataFrame
        Tarih indeksine göre artan sıralı kopya.

    Raises
    ------
    ValueError
        'Close' kolonu yoksa.
    """
    if "Close" not in df.columns:
        raise ValueError("backtester: 'Close' kolonu zorunludur.")
    dff = df.copy()
    dff.index = pd.to_datetime(dff.index)
    if not dff.index.is_monotonic_increasing:
        dff = dff.sort_index()
    return dff


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Ortalama Gerçek Aralık (ATR) hesabı.

    Eğer 'High'/'Low' mevcut değilse, ATR yerine Close bazlı bir volatilite
    proxy'si kullanılır (ortalama mutlak getiri * fiyat).

    Parameters
    ----------
    df : pd.DataFrame
        OHLC içeren DataFrame (en az Close).
    window : int, default 14
        ATR periyodu.

    Returns
    -------
    pd.Series
        ATR serisi.
    """
    if {"High", "Low", "Close"}.issubset(df.columns):
        h = df["High"].astype(float)
        l = df["Low"].astype(float)
        c = df["Close"].astype(float)
        prev_c = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=max(2, window // 2)).mean()
        return atr
    # Fallback proxy
    c = df["Close"].astype(float)
    proxy = (c.pct_change().abs().rolling(window, min_periods=window // 2).mean() * c)
    return proxy.fillna(method="bfill")


def _sma(sig: pd.Series, n: int) -> pd.Series:
    """
    Basit Hareketli Ortalama (SMA).

    Parameters
    ----------
    sig : pd.Series
        Girdi seri.
    n : int
        Pencere.

    Returns
    -------
    pd.Series
        SMA serisi.
    """
    return sig.rolling(n, min_periods=n).mean()


def _gen_signals(df: pd.DataFrame) -> pd.Series:
    """
    SMA(10/20) crossover tabanlı -1/0/+1 sinyal üretir.

    Kurallar
    --------
    - f = SMA(10), s = SMA(20)
    - İşaret = sign(f - s)
    - İşaret değişiminde sinyal üretilir: +1 (upcross), -1 (downcross)
    - Diğer günler 0 (flat)

    Parameters
    ----------
    df : pd.DataFrame
        OHLC(+) verisi.

    Returns
    -------
    pd.Series
        -1/0/+1 sinyal serisi.
    """
    c = df["Close"].astype(float)
    f = _sma(c, 10)
    s = _sma(c, 20)
    raw = np.sign((f - s).fillna(0.0))
    sig = raw.diff().fillna(0.0)
    out = sig.copy()
    out[(sig > 0)] = 1
    out[(sig < 0)] = -1
    out[(sig == 0)] = 0
    return out.astype(int)


def _annualized_return(equity: pd.Series) -> float:
    """
    Yıllıklandırılmış getiri (CAGR).

    Parameters
    ----------
    equity : pd.Series
        Özsermaye zaman serisi.

    Returns
    -------
    float
        CAGR (0-1).
    """
    if len(equity) < 2:
        return 0.0
    ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    days = (equity.index[-1] - equity.index[0]).days or 1
    years = days / 365.25
    return (1.0 + ret) ** (1.0 / years) - 1.0 if years > 0 else ret


def _max_drawdown(equity: pd.Series) -> float:
    """
    Maksimum gerileme (max drawdown).

    Parameters
    ----------
    equity : pd.Series

    Returns
    -------
    float
        Max DD (0-1, negatif değerli bir oran olarak yorumlanır).
    """
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _sharpe(equity: pd.Series) -> float:
    """
    Naive yıllıklandırılmış Sharpe oranı (rf=0 varsayılarak).

    Parameters
    ----------
    equity : pd.Series

    Returns
    -------
    float
        Sharpe oranı.
    """
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    mu = rets.mean() * 252
    sd = rets.std(ddof=0) * (252 ** 0.5)
    return float(mu / sd) if sd > 0 else 0.0


# =============================================================================
# Ana Backtest Çalıştırıcısı
# =============================================================================
def run(
    df: pd.DataFrame,
    forecasts: Optional[Dict[str, Any]] = None,
    signals: Optional[pd.Series] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Basit kurallı backtest çalıştırır.

    İş Akışı
    --------
    1) Girdi verisi doğrulanır ve sıralanır.
    2) Sinyaller yoksa SMA(10/20) crossover sinyalleri üretilir.
    3) ATR hesaplanır; risk planı varsa risk yüzdesi ve ATR çarpanları okunur.
    4) Sinyal gününde pozisyon açılır (long_only=True ise sadece long).
    5) Pozisyon açıkken her gün SL/TP intraday kontrol edilir; tetiklenirse çıkar.
    6) Karşıt sinyal geldiğinde gün kapanışında pozisyon kapatılır.
    7) Slippage ve işlem ücreti bps cinsinden uygulanır.
    8) Equity, trades ve metrikler döndürülür.

    Parameters
    ----------
    df : pd.DataFrame
        DatetimeIndex'li OHLC(+) DataFrame. 'Close' zorunlu; 'High/Low' varsa SL/TP intraday çalışır.
    forecasts : dict | None
        Şimdilik kullanılmıyor. Gelecekte sinyal türetmek için entegre edilebilir.
    signals : pd.Series | None
        Dışarıdan sağlanan -1/0/+1 sinyalleri. None ise otomatik üretilir.
    cfg : dict | None
        Backtest ayarları ve opsiyonel risk planı. Detaylar dosya üstü docstring'de.

    Returns
    -------
    dict
        metrics, trades, equity anahtarlarını içeren sözlük.

    Raises
    ------
    ValueError
        Girdi DataFrame geçersizse.
    """
    cfg = cfg or {}
    dff = _ensure_ohlc(df)

    # --- Parametreler ---
    init_eq = float(cfg.get("initial_equity", 10_000.0))
    atr_win = int(cfg.get("atr_window", 14))
    slip_bps = float(cfg.get("slippage_bps", 5.0)) / 10_000.0
    fee_bps = float(cfg.get("fee_bps", 10.0)) / 10_000.0
    long_only = bool(cfg.get("long_only", True))

    # Risk planı (router'dan gelebilir)
    risk = cfg.get("risk", {}) or {}
    risk_pct = float(risk.get("risk_pct_effective", 0.004))  # %0.4
    stop_mult = float(risk.get("stop_atr_mult", 2.0))
    tp_mult = float(risk.get("tp_atr_mult", 3.0))

    # --- Sinyaller ---
    if signals is None:
        signals = _gen_signals(dff)
    else:
        signals = signals.reindex(dff.index).fillna(0).astype(int)

    # --- ATR ---
    atr = _atr(dff, window=atr_win).reindex(dff.index).bfill()

    # --- OHLC sütunları ---
    close = dff["Close"].astype(float)
    high = dff["High"].astype(float) if "High" in dff.columns else close
    low = dff["Low"].astype(float) if "Low" in dff.columns else close

    # --- Portföy durumu ---
    equity = pd.Series(index=dff.index, dtype=float)
    equity.iloc[0] = init_eq
    cash = init_eq
    position = 0           # +1 long, -1 short, 0 flat
    qty = 0.0
    entry = 0.0
    stop = math.nan
    tp = math.nan
    trades: List[Dict[str, Any]] = []
    prev_eq = init_eq
    entry_date = None

    # --- Ana döngü ---
    for i, dt in enumerate(dff.index):
        px = float(close.iloc[i])
        atr_i = float(atr.iloc[i]) if (isinstance(atr.iloc[i], (int, float, np.floating)) and atr.iloc[i] > 0) else 0.02 * px

        # (1) Açık pozisyonu SL/TP ve sinyal ile kapatma
        if position != 0:
            hit_sl = (low.iloc[i] <= stop <= high.iloc[i]) if not math.isnan(stop) else False
            hit_tp = (low.iloc[i] <= tp <= high.iloc[i]) if not math.isnan(tp) else False

            exit_price = None
            exit_reason = None

            # Öncelik: SL -> TP
            if hit_sl:
                exit_price = stop
                exit_reason = "sl"
            elif hit_tp:
                exit_price = tp
                exit_reason = "tp"

            # Karşıt sinyalde gün kapanışında çık
            if exit_price is None:
                sig_now = int(signals.iloc[i])
                if (position == 1 and sig_now < 0) or (position == -1 and sig_now > 0):
                    exit_price = px * (1 - slip_bps) if position == 1 else px * (1 + slip_bps)
                    exit_reason = "signal"

            if exit_price is not None:
                fee = abs(qty) * exit_price * fee_bps
                pnl = (exit_price - entry) * qty if position == 1 else (entry - exit_price) * qty
                pnl -= fee
                cash += pnl
                trades.append(dict(
                    entry_date=entry_date,
                    exit_date=dt,
                    side="long" if position == 1 else "short",
                    entry=float(entry),
                    exit=float(exit_price),
                    qty=float(qty),
                    pnl=float(pnl),
                    r_mult=(pnl / (risk_pct * prev_eq)) if prev_eq > 0 else float("nan"),
                    reason=exit_reason
                ))
                position, qty, entry, stop, tp = 0, 0.0, 0.0, math.nan, math.nan
                entry_date = None

        # (2) Giriş
        if position == 0:
            sig_now = int(signals.iloc[i])
            if sig_now == 1 or (sig_now == -1 and not long_only):
                eq_now = cash
                risk_dollars = eq_now * risk_pct
                stop_dist = stop_mult * atr_i if stop_mult * atr_i > 0 else 0.02 * px
                qty = risk_dollars / max(stop_dist, 1e-9)

                # Çok küçük işlemleri atla
                if qty * px >= 10:
                    side = 1 if sig_now == 1 else -1
                    entry_px = px * (1 + slip_bps) if side == 1 else px * (1 - slip_bps)
                    fee = qty * entry_px * fee_bps
                    cash -= fee

                    position = side
                    entry = entry_px
                    entry_date = dt
                    if side == 1:
                        stop = entry - stop_mult * atr_i
                        tp = entry + tp_mult * atr_i
                    else:
                        stop = entry + stop_mult * atr_i
                        tp = entry - tp_mult * atr_i

        # (3) Gün sonu MTM
        mtm = cash
        if position != 0:
            if position == 1:
                mtm += qty * px
            else:  # short
                mtm += qty * (entry - px)
        equity.iloc[i] = mtm
        prev_eq = mtm

    # (4) Son gün pozisyonu kapat (varsa)
    if position != 0:
        px = float(close.iloc[-1])
        exit_price = px * (1 - slip_bps) if position == 1 else px * (1 + slip_bps)
        fee = abs(qty) * exit_price * fee_bps
        pnl = (exit_price - entry) * qty if position == 1 else (entry - exit_price) * qty
        pnl -= fee
        cash += pnl
        trades.append(dict(
            entry_date=entry_date,
            exit_date=dff.index[-1],
            side="long" if position == 1 else "short",
            entry=float(entry),
            exit=float(exit_price),
            qty=float(qty),
            pnl=float(pnl),
            r_mult=(pnl / (risk_pct * equity.iloc[-2])) if len(equity) > 1 and equity.iloc[-2] > 0 else float("nan"),
            reason="eod"
        ))
        equity.iloc[-1] = cash

    # --- Metrikler ---
    pnl_list = [t["pnl"] for t in trades]
    wins = sum(1 for x in pnl_list if x > 0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0

    metrics = dict(
        trades=len(trades),
        win_rate=(wins / len(trades) if trades else 0.0),
        total_return=total_return,
        cagr=_annualized_return(equity),
        max_dd=_max_drawdown(equity),
        sharpe=_sharpe(equity),
    )

    return {
        "metrics": metrics,
        "trades": trades,
        "equity": equity,
    }