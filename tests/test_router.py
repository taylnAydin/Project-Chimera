# tests/test_router.py
import os
from src.agents.router import run_router
from datetime import date


# İlk denemede LLM kapalı tutalım; rapor yine üretilecek (fallback).
# .env içinde REPORTER_USE_LLM=false olmalı.

cfg = {
    "source": {
        "provider": "binance",   # senin data_retrieval.fetch ne bekliyorsa onu kullan
        "interval": "1d",
        # "api_key": "...",
    },
    "indicators": {
        # indicator.enrich ile üretilecek kolonlar (ARIMAX exog için de kullanacağız)
        "sma": [20, 50],
        "ema": [21],
        "rsi": {"period": 14, "colname": "rsi_14"},
        "atr": {"period": 14, "colname": "atr_14"},
        "macd": {"fast": 12, "slow": 26, "signal": 9}
    },
    "guardrails": {
        "min_years_full": 3.0,
        "min_years_analyze": 1.0,
        "max_nan_full": 0.02,
        "max_nan_analyze": 0.05,
        "max_duplicate_full": 0.01,
        "max_duplicate_analyze": 0.03,
        "max_gap_full": 0.02,
        "max_gap_analyze": 0.05,
        "max_outlier_full": 0.05,
        "max_outlier_analyze": 0.10,
        "allow_block_on_no_data": True,
    },
    "forecast": {
        "models": ["arima", "arimax", "ets"],
        "horizon": 7,
        "exog": {
            # ARIMAX için indicator.enrich sonrası oluşacak kolonları kullanıyoruz
            "cols": ["rsi_14", "ema_21", "atr_14"]
        }
    },
    "model_select": {"metric": "rmse"},
    "regime": {
        "halt_on_extreme_vol": True,
        "window": 30
    },
    "risk": {
        "atr_period": 14,
        "risk_per_trade": 0.01,
        "vol_scalar": 1.0,
        "stop_loss": 0.02,
        "take_profit": 0.04,
    },
    "anomaly": {
        "window": 30,
        "z_threshold": 3.0,
        "residual_check": True,
        "residual_z_threshold": 3.0,
        "volume_check": False
    },
    "backtest": {
        "start_capital": 10000,
        "fee_bps": 5,
        "slippage_bps": 5,
        "position_mode": "cash",
        "position_size": 1.0,
        "hold_threshold": 0.0005
    },
    "report": {
        # LLM kapalıysa fallback; sonra true yapıp tekrar deneriz
    },
    "notify": {
        # şimdilik boş; notifier konsola ve data/reports/ klasörüne kaydedecek
    }
}

if __name__ == "__main__":
    symbol = "BTCUSDT"
    start = "2021-01-01"
    end = str(date.today())  

    final = run_router(symbol, start, end, cfg)

    print("\n==== FINAL STATE SUMMARY ====")
    print("run_id:", final.get("run_id"))
    print("mode:", final.get("mode"), "| halt_reason:", final.get("halt_reason"))
    print("errors:", final.get("errors"))
    print("quality.keys:", list((final.get("quality") or {}).keys()))
    print("best:", final.get("best"))
    print("regime:", final.get("regime"))
    print("risk_plan:", final.get("risk_plan"))
    print("backtest.metrics:", (final.get("backtest") or {}).get("metrics"))
    print("report.title:", (final.get("report") or {}).get("title"))