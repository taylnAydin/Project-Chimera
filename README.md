# 🐉 Chimera: Multi-Agent Crypto Forecast & Backtest Pipeline

Chimera is a modular, **multi-agent trading and forecasting pipeline** for crypto assets (currently Binance OHLCV data).  
It combines **data retrieval, forecasting, risk management, anomaly detection, and backtesting** into a single orchestrated system.

---

## ✨ Features

- 📥 **Data Retrieval**: Binance OHLCV (5y+ when available)  
- 📊 **Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands  
- ✅ **Data Quality Checks**: NaN ratio, duplicates, gaps, outliers, zero-volume  
- 🤖 **Forecast Models**: ARIMA, ARIMAX (with exog), ETS → best selected by RMSE/AIC  
- 📈 **Regime Detection**: Trend slope & volatility classification  
- ⚖️ **Risk Management**: ATR-based stop-loss / take-profit, dynamic position sizing  
- 🧪 **Backtester**: SMA crossover strategy with ATR-based SL/TP, slippage & fee simulation  
- 🚨 **Anomaly Agent**: Highlight unusual price/forecast deviations  
- 🧾 **Reporter**: Generates Markdown reports (optionally polished with Gemini API)  
- 🖼️ **Evaluation Add-on**: Last 10-day *real vs. predicted* table with MAPE & accuracy  

---

## 📂 Project Structure
```text

Chimera/
├── src/
│   ├── agents/
│   │   ├── backtester.py       # Rule-based backtest engine
│   │   ├── data_quality.py     # NaN/duplicate/gap/outlier checks
│   │   ├── data_retrieval.py   # Binance OHLCV fetcher
│   │   ├── forecast.py         # Forecast agent (delegates to models)
│   │   ├── guardrails_agent.py # Guardrail logic for routing/blocking
│   │   ├── indicator.py        # Technical indicators (SMA, EMA, RSI, etc.)
│   │   ├── model_selection.py  # Select best forecast by RMSE/AIC
│   │   ├── notifier.py         # Console/file notifier
│   │   ├── persistence.py      # SQLite persistence layer
│   │   ├── regime_detector.py  # Trend & volatility regime classification
│   │   ├── reporter.py         # Markdown report generator (+ Gemini polish)
│   │   └── anomaly.py          # Detect unusual deviations
│   │
│   ├── models/
│   │   ├── arima.py            # ARIMA forecasting
│   │   ├── arimax.py           # ARIMAX forecasting (with exog)
│   │   ├── ets.py              # ETS forecasting
│   │   └── common.py           # Shared helpers (RMSE, SARIMAX fit, etc.)
│   │
│   └── router.py               # LangGraph orchestrator (multi-agent flow)
│
├── data/
│   ├── reports/                # Generated Markdown reports
│   └── chimera.db              # SQLite database (runs, snapshots, metrics)
│
├── tests/
│   ├── test_router.py          # End-to-end pipeline test
│   └── ...                     # Other unit tests
│
├── .env                        # API keys & config (Gemini, etc.)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Quickstart
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/chimera.git
   cd chimera
   ```
   
2.	**Create and activate a virtual environment**
     ```bash
     python3 -m venv venv
     source venv/bin/activate   # On Linux / macOS
     venv\Scripts\activate      # On Windows
     ```

   
3.	**Install dependencies**
     ```bash
     pip install -r requirements.txt
     ```

4. **Set up environment variables**  
   Create a `.env` file in the project root:

   ```bash
   REPORTER_USE_LLM=true
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-1.5-flash
   GEMINI_TIMEOUT=15
   ```

5.	**Run the pipeline**
     ```bash
     python3 -m tests.test_router
     ```

## 📊 Sample Output / Example Report
Chimera generates human-readable **Markdown reports** under `data/reports/`.  
Each report includes:

- Data quality summary  
- Selected forecast model & configuration  
- Last 10 days real vs. predicted prices (with accuracy)  
- Market regime classification (trend & volatility)  
- Risk management plan (entry, stop-loss, take-profit)  
- Backtest summary (metrics & trades)  
- Final trading decision  

🔍 Example report file: [`data/reports/BTCUSDT_run_xxxxx.md`](data/reports/)  

Below is a sample snippet:

```markdown
**Modeling**
- Model: ETS
- Forecast Horizon: 7 days
- Performance: RMSE = 0.14

**Last 10 Days: Real vs. Predicted**
| Date       | Real Close | Predicted | Abs % Error |
|------------|------------|-----------|-------------|
| 2025-09-01 | 109,237.42 | 108,180.57| 0.97%       |
| 2025-09-02 | 111,240.01 | 109,307.39| 1.74%       |
| ...        | ...        | ...       | ...         |

**MAPE:** 0.63% • **Accuracy:** 99.44%
```

## 🤝 Contributing
Contributions are welcome! 🎉 If you’d like to help improve **Chimera**, follow these steps:

1. **Fork the repository**
   ```bash
   git fork https://github.com/your-username/chimera.git
   ```

2. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
   
3. **Make your changes**
   ```bash
   •Follow existing code style and structure.
   •Add or update tests when necessary.
   •Ensure that pytest (or the provided test suite) passes.
   ```
4. **Commit and pushs**
    ```bash
    git push origin feature/your-feature-name
   ```

### 📌 Guidelines
- Follow existing code style and structure.  
- Add or update tests when necessary.  
- Ensure that `pytest` (or the provided test suite) passes.  
- Describe what you changed and why.  
- Reference related issues if applicable.  
- Use **English** in code and documentation (README, comments).  
- Keep commits small and focused.  
- Add docstrings for new functions or modules.  
- Run tests before submitting PRs.

## ⚠️ Disclaimer
This project is developed **for educational and research purposes only**.  
It is **not** intended for live trading or financial advice.  
Use at your own risk.







      
