# SWING_BOT Command Reference & Documentation

## Overview

SWING_BOT is an autonomous momentum-focused swing trading system for NIFTY 500 stocks with comprehensive EOD pipeline, plan auditing, and Teams notifications. The system integrates AI-driven strategy selection, risk management, and automated order placement.

**Current Date:** January 7, 2026
**System Status:** Production Ready
**Market Focus:** NSE (National Stock Exchange of India)

---

## 📋 System Architecture

SWING_BOT operates through a modular pipeline:

1. **Data Fetching** → 2. **Signal Generation** → 3. **Strategy Selection** → 4. **Plan Generation** → 5. **Risk Auditing** → 6. **Order Placement** → 7. **Monitoring**

### Key Components:
- **AI-Driven Strategy Selection**: Multi-strategy ensemble with ML optimization
- **Risk Management**: Position sizing, stop-loss, diversification
- **Market Regime Detection**: SMA200 + ADX/RSI based filtering
- **Multi-Broker Support**: Upstox, ICICI, Indmoney integration
- **Real-time Monitoring**: Telegram alerts, Teams notifications, metrics export

---

## 🚀 Command Reference

### 1. Data Management Commands

#### `fetch_data` - Historical Data Fetching
**Purpose:** Download historical OHLCV data for NIFTY50 stocks and ETFs
**Automation:** Manual (run daily)
**Use Case:** Initial setup, data refresh, backtesting preparation

```bash
python -m src.cli fetch_data --days 365 --out data/nifty50_data.csv --include-etfs
```

**Parameters:**
- `--days`: Number of trading days to fetch (default: 365)
- `--out`: Output CSV file path
- `--include-etfs`: Include NSE ETFs in data fetch
- `--max-workers`: Parallel fetch workers (default: 8)

#### `fetch-and-validate` - Data Fetch with Validation
**Purpose:** Fetch data and immediately validate integrity
**Automation:** Manual (run before trading)
**Use Case:** Ensure data quality before pipeline execution

```bash
python -m src.cli fetch-and-validate --out data/nifty50_data.csv --days 500 --max-age-days 1
```

**Parameters:**
- `--max-age-days`: Maximum age of existing data in days
- `--required-days`: Minimum required trading days
- `--required-symbols`: Minimum required symbols

#### `fetch-all` - Multi-Timeframe Data Fetch
**Purpose:** Fetch data across multiple timeframes for analysis
**Automation:** Manual (run for multi-TF analysis)
**Use Case:** Technical analysis, multi-timeframe strategies

```bash
python -m src.cli fetch-all --symbols RELIANCE.NS,TCS.NS --timeframes 1d,1w,1mo --start-date 2023-01-01 --end-date 2024-12-31
```

### 2. Strategy Analysis & Backtesting

#### `wfo` - Walk-Forward Optimization
**Purpose:** Optimize strategy parameters using walk-forward analysis
**Automation:** Manual (run weekly/monthly)
**Use Case:** Strategy parameter tuning, performance optimization

```bash
python -m src.cli wfo --path data/nifty50_data.csv --strategy Donchian --config config.yaml
```

**Parameters:**
- `--strategy`: Strategy to optimize
- `--config`: Configuration file path
- `--confirm-rsi/macd/hist`: Require additional confirmations

#### `backtest` - Strategy Backtesting
**Purpose:** Test trading strategies on historical data
**Automation:** Manual (run daily for strategy evaluation)
**Use Case:** Strategy performance validation, risk assessment

```bash
python -m src.cli backtest --path data/nifty50_data.csv --out outputs/backtest_results.csv --confirm-rsi
```

**Parameters:**
- `--confirm-rsi/macd/hist`: Require technical confirmations
- `--skip-validation`: Skip data validation

#### `select` - Strategy Selection
**Purpose:** Choose best performing strategy based on backtests
**Automation:** Manual (run after backtesting)
**Use Case:** AI-driven strategy selection for live trading

```bash
python -m src.cli select --path data/nifty50_data.csv --out outputs/selected_strategy.json
```

### 3. Screening & Signal Generation

#### `screener` - Stock Screening
**Purpose:** Generate trading signals using technical analysis
**Automation:** Manual (run intraday)
**Use Case:** Identify potential trading opportunities

```bash
python -m src.cli screener --path data/nifty50_data.csv --out outputs/screener_results.csv --live
```

**Parameters:**
- `--live`: Fetch live quotes and update data
- `--skip-validation`: Skip data validation checks

#### `live-screener` - Real-time Screening
**Purpose:** Screen stocks using live market data
**Automation:** Manual (run intraday)
**Use Case:** Real-time opportunity identification

```bash
python -m src.cli live-screener --include-etfs
```

**Parameters:**
- `--include-etfs`: Include NSE ETFs in screening

#### `multi-tf-excel` - Multi-Timeframe Analysis
**Purpose:** Generate Excel reports with multiple timeframes
**Automation:** Manual (run for detailed analysis)
**Use Case:** Comprehensive technical analysis across timeframes

```bash
python -m src.cli multi-tf-excel --path data/nifty50_data.csv --tfs 1d,1w,1mo --out outputs/multi_tf_analysis.xlsx
```

### 4. Trading Plan Generation

#### `gtt-plan` - GTT Plan Generation
**Purpose:** Create Good Till Triggered order plans
**Automation:** Manual (run EOD)
**Use Case:** Generate executable trading plans with entry/exit levels

```bash
python -m src.cli gtt-plan --path data/nifty50_data.csv --strategy Donchian --top 25 --out outputs/gtt_plan.csv
```

**Parameters:**
- `--strategy`: Primary strategy to use
- `--top`: Number of stocks to select
- `--min-score`: Minimum composite score threshold
- `--fallback-strategies`: Backup strategies if primary fails

#### `plan-audit` - Plan Risk Auditing
**Purpose:** Audit trading plans for risk compliance and validity
**Automation:** Manual (run after plan generation)
**Use Case:** Ensure plans meet risk management criteria

```bash
python -m src.cli plan-audit --plan outputs/gtt_plan.csv --indicators data/nifty50_data.csv --latest data/latest_quotes.csv --out outputs/audited_plan.csv --strict
```

**Parameters:**
- `--strict`: Fail fast on audit failures
- `--config`: Configuration file path

#### `reconcile-plan` - LTP Reconciliation
**Purpose:** Update plan prices with current market prices
**Automation:** Manual (run before order placement)
**Use Case:** Ensure orders reflect current market conditions

```bash
python -m src.cli reconcile-plan --plan outputs/gtt_plan.csv --out outputs/reconciled_plan.csv --adjust-mode soft
```

**Parameters:**
- `--adjust-mode`: soft (adjust within limits) or strict (exact match)
- `--max-entry-ltppct`: Max LTP percentage deviation

### 5. Order Management

#### `gtt-place` - GTT Order Placement
**Purpose:** Place GTT orders on selected broker
**Automation:** Manual (run EOD after reconciliation)
**Use Case:** Execute trading plans in live market

```bash
python -m src.cli gtt-place --plan outputs/audited_plan.csv --out outputs/gtt_orders.json --dry-run false
```

**Parameters:**
- `--dry-run`: Test mode without actual order placement
- `--access-token`: Broker API access token
- `--retries`: Number of retry attempts
- `--rate-limit`: API call rate limiting

#### `gtt-get` - GTT Order Status
**Purpose:** Check status of specific GTT orders
**Automation:** Manual (run for monitoring)
**Use Case:** Monitor order execution and status

```bash
python -m src.cli gtt-get --id ORDER_ID --access-token YOUR_TOKEN
```

#### `gtt-reconcile` - Order Reconciliation
**Purpose:** Sync existing orders with new trading plans
**Automation:** Manual (run before new plan placement)
**Use Case:** Manage position changes and order updates

```bash
python -m src.cli gtt-reconcile --plan outputs/new_plan.csv --dry-run
```

**Parameters:**
- `--dry-run`: Simulate reconciliation without changes
- `--log-path`: Reconciliation log file path

### 6. Reporting & Notifications

#### `final-excel` - Excel Report Generation
**Purpose:** Create comprehensive Excel trading reports
**Automation:** Manual (run EOD)
**Use Case:** Generate detailed trading analysis and documentation

```bash
python -m src.cli final-excel --plan outputs/audited_plan.csv --out outputs/final_report.xlsx --backtest-dir outputs/backtests
```

#### `teams-dashboard` - HTML Dashboard
**Purpose:** Generate interactive HTML dashboard
**Automation:** Manual (run for visualization)
**Use Case:** Web-based trading dashboard and reporting

```bash
python -m src.cli teams-dashboard --plan outputs/gtt_plan.csv --audit outputs/audited_plan.csv --screener outputs/screener_results.csv --out-html outputs/dashboard.html
```

#### `teams-notify` - Teams Notifications
**Purpose:** Send trading summaries to Microsoft Teams
**Automation:** Manual (run after pipeline completion)
**Use Case:** Team communication and alerts

```bash
python -m src.cli teams-notify --plan outputs/audited_plan.csv --date 2024-01-07 --webhook-url YOUR_WEBHOOK
```

### 7. Complete Pipeline Orchestration

#### `orchestrate-eod` - Full EOD Pipeline
**Purpose:** Run complete end-of-day trading pipeline
**Automation:** Can be automated (daily scheduler)
**Use Case:** Complete autonomous daily trading cycle

```bash
python -m src.cli orchestrate-eod --data-out data/nifty50_indicators_full.csv --max-age-days 1 --required-days 500 --top 25 --strict --post-teams --multi-tf --dashboard
```

**Parameters:**
- `--post-teams`: Send Teams notifications
- `--multi-tf`: Generate multi-timeframe analysis
- `--dashboard`: Generate HTML dashboard
- `--metrics`: Enable metrics collection

#### `orchestrate-live` - Live Trading Pipeline
**Purpose:** Run live trading pipeline with order placement
**Automation:** Can be automated (EOD scheduler)
**Use Case:** Complete live trading execution

```bash
python -m src.cli orchestrate-live --data-out data/nifty50_indicators_full.csv --top 25 --strict --post-teams --live --place-gtt --reconcile-gtt --confidence-threshold 0.20
```

**Parameters:**
- `--place-gtt`: Place actual GTT orders
- `--reconcile-gtt`: Reconcile existing orders
- `--confidence-threshold`: Minimum confidence for order placement
- `--enable-ml-filter`: Use ML-based signal filtering
- `--enable-risk-management`: Enable enhanced risk management

#### `hourly-update` - Hourly Updates
**Purpose:** Run hourly screening and notification updates
**Automation:** Can be automated (hourly scheduler)
**Use Case:** Intraday monitoring and alerts

```bash
python -m src.cli hourly-update --data-path data/nifty50_data_today.csv --output-dir outputs/hourly --top 25 --notify-email --notify-telegram
```

**Parameters:**
- `--notify-email`: Send email notifications
- `--notify-telegram`: Send Telegram notifications
- `--force-refresh`: Force data refresh

### 8. Testing & Diagnostics

#### `run-e2e-tests` - End-to-End Testing
**Purpose:** Comprehensive system testing and validation
**Automation:** Manual (run for quality assurance)
**Use Case:** Ensure system reliability and catch regressions

```bash
python -m src.cli run-e2e-tests --output-dir outputs/e2e_tests --verbose --components data signals optimization --regime all --mock-apis
```

**Parameters:**
- `--components`: Specific components to test
- `--regime`: Market regime to test (bullish/bearish/sideways)
- `--performance-benchmark`: Run performance benchmarks

#### `run-full-test` - Full System Test
**Purpose:** Complete system integration testing
**Automation:** Manual (run before deployment)
**Use Case:** Validate entire trading pipeline

```bash
python -m src.cli run-full-test --output-dir outputs/full_test --verbose --mock-apis --generate-report
```

#### `diagnose-universe` - Universe Diagnostics
**Purpose:** Test screening pipeline across all symbols
**Automation:** Manual (run for debugging)
**Use Case:** Identify screening pipeline issues

```bash
python -m src.cli diagnose-universe --max-symbols 50 --verbose --output outputs/universe_diagnostic.csv
```

#### `auto-test` - Automated Performance Testing
**Purpose:** Daily automated performance validation
**Automation:** Can be automated (daily)
**Use Case:** Continuous performance monitoring

```bash
python -m src.cli auto-test --symbol RELIANCE.NS --config config.yaml
```

#### `self-optimize` - Parameter Self-Optimization
**Purpose:** AI-driven parameter optimization
**Automation:** Can be automated (weekly)
**Use Case:** Continuous strategy improvement

```bash
python -m src.cli self-optimize --config config.yaml
```

#### `validate-latest` - Data Validation
**Purpose:** Validate recency and consistency of data files
**Automation:** Manual (run before trading)
**Use Case:** Ensure data integrity

```bash
python -m src.cli validate-latest --data data/nifty50_data.csv --screener outputs/screener_results.csv --plan outputs/gtt_plan.csv
```

### 9. Monitoring & Metrics

#### `metrics-exporter` - Metrics Server
**Purpose:** Export system metrics for monitoring
**Automation:** Can be automated (continuous service)
**Use Case:** System monitoring and alerting

```bash
python -m src.cli metrics-exporter --port 9108 --mode prometheus
```

**Parameters:**
- `--mode`: prometheus or otlp
- `--port`: Server port

---

## 🖥️ Standalone Scripts

### Dashboard & Visualization

#### `run_dashboard.py` - Web Dashboard
**Purpose:** Launch interactive Streamlit web dashboard
**Automation:** Manual (run for visualization)
**Use Case:** Real-time trading monitoring and analysis

```bash
python run_dashboard.py
```
**Access:** http://localhost:8501

#### `diagnose_dashboard.py` - Dashboard Diagnostics
**Purpose:** Troubleshoot dashboard data loading issues
**Automation:** Manual (run when dashboard has issues)
**Use Case:** Debug dashboard data sources and freshness

```bash
python diagnose_dashboard.py
```

### Market Monitoring

#### `monitor_market.py` - Market Monitor
**Purpose:** Monitor market conditions and generate reports
**Automation:** Can be automated (continuous/daily)
**Use Case:** Market surveillance and reporting

```bash
# One-time market check
python monitor_market.py --mode once

# Continuous monitoring
python monitor_market.py --mode continuous --interval 3600

# Daily market report
python monitor_market.py --mode daily --daily-report
```

### Live Trading

#### `live_trader.py` - Live Trading System
**Purpose:** Real-time trading execution system
**Automation:** Manual (requires active supervision)
**Use Case:** Live market trading with risk management

```bash
# Live trading with risk parameters
python live_trader.py --capital 500000 --max-positions 3 --risk-per-trade 0.01

# Trading status check
python live_trader.py --mode status

# Sector analysis
python live_trader.py --mode sector-analysis --symbols RELIANCE.NS,TCS.NS
```

#### `paper_trade.py` - Paper Trading
**Purpose:** Simulated trading for testing strategies
**Automation:** Manual
**Use Case:** Strategy testing without real money

```bash
python paper_trade.py --scan-only
```

### Analysis Tools

#### `walk_forward_test.py` - Walk-Forward Testing
**Purpose:** Advanced strategy validation using walk-forward analysis
**Automation:** Manual
**Use Case:** Robust strategy performance testing

```bash
python walk_forward_test.py --symbol RELIANCE.NS --start-date 2023-01-01 --end-date 2024-12-01
```

#### `backtest_swing_bot.py` - Custom Backtesting
**Purpose:** Specialized backtesting for SWING_BOT strategies
**Automation:** Manual
**Use Case:** Detailed strategy performance analysis

```bash
python backtest_swing_bot.py
```

#### `risk_reward_analysis.py` - Risk Analysis
**Purpose:** Analyze risk-reward profiles of trading plans
**Automation:** Manual
**Use Case:** Risk assessment and optimization

```bash
python risk_reward_analysis.py
```

#### `analyze_stocks.py` - Stock Analysis
**Purpose:** Deep analysis of individual stocks
**Automation:** Manual
**Use Case:** Fundamental and technical stock analysis

```bash
python analyze_stocks.py
```

---

## 🤖 Automation Status

### Automated Components:
- **Daily EOD Pipeline**: `orchestrate-eod` (can be scheduled)
- **Live Trading Pipeline**: `orchestrate-live` (can be scheduled)
- **Hourly Updates**: `hourly-update` (can be scheduled)
- **Metrics Export**: `metrics-exporter` (continuous service)
- **Auto Testing**: `auto-test` (daily schedule)
- **Self-Optimization**: `self-optimize` (weekly schedule)
- **Market Monitoring**: `monitor_market.py` (continuous/daily)

### Manual Components:
- **Data Fetching**: `fetch_data`, `fetch-and-validate`
- **Strategy Analysis**: `wfo`, `backtest`, `select`
- **Signal Generation**: `screener`, `live-screener`
- **Plan Generation**: `gtt-plan`, `plan-audit`, `reconcile-plan`
- **Order Placement**: `gtt-place`, `gtt-get`, `gtt-reconcile`
- **Reporting**: `final-excel`, `teams-dashboard`, `teams-notify`
- **Testing**: `run-e2e-tests`, `run-full-test`, `diagnose-universe`
- **Validation**: `validate-latest`

---

## 📊 System Status Indicators

### Market Regime Detection:
- **ON Conditions**: Above 200-day SMA AND (ADX(14) > 20 OR RSI(14) > 50)
- **OFF Conditions**: Below SMA200 or weak trend momentum
- **Current Status**: Automatically detected and applied

### Risk Management:
- **Position Sizing**: Volatility-adjusted sizing
- **Stop Loss**: Dynamic stop-loss levels
- **Diversification**: Maximum correlation limits
- **Circuit Breakers**: Automatic position reduction on adverse conditions

### Broker Integration:
- **Upstox**: Primary broker (GTT orders, live data)
- **ICICI**: Alternative broker support
- **Indmoney**: Additional broker option

---

## 🚨 Common Workflows

### Daily EOD Trading:
1. `fetch-and-validate` → 2. `screener` → 3. `backtest` → 4. `select` → 5. `gtt-plan` → 6. `plan-audit` → 7. `reconcile-plan` → 8. `gtt-place` → 9. `final-excel` → 10. `teams-notify`

### Live Trading Setup:
1. `orchestrate-live --place-gtt --reconcile-gtt --post-teams`

### System Maintenance:
1. `run-e2e-tests` (weekly) → 2. `self-optimize` (monthly) → 3. `validate-latest` (daily)

### Intraday Monitoring:
1. `hourly-update --notify-telegram` (every hour during market hours)

---

## ⚙️ Configuration

### Key Configuration Files:
- `config.yaml`: Main system configuration
- `.env`: Environment variables (API keys, tokens)
- `requirements.txt`: Python dependencies

### Important Environment Variables:
- `UPSTOX_ACCESS_TOKEN`: Broker API access
- `TEAMS_WEBHOOK_URL`: Teams notifications
- `TELEGRAM_BOT_TOKEN`: Telegram alerts
- `OPENAI_API_KEY`: LLM features (optional)

---

## 🔧 Troubleshooting

### Common Issues:
- **Data Fetch Failures**: Check network connectivity, API limits
- **Order Placement Errors**: Verify broker tokens, account permissions
- **Dashboard Issues**: Run `diagnose_dashboard.py`
- **Pipeline Failures**: Check `validate-latest` for data issues

### Logs Location:
- `outputs/logs/`: All system logs
- `outputs/backtests/`: Backtesting results
- `outputs/gtt/`: Order placement logs

---

## 📞 Support & Documentation

- **README.md**: System overview and setup
- **docs/**: Detailed documentation
- **tests/**: Test suites and examples
- **scripts/**: Automation and utility scripts

For issues or questions, check the logs in `outputs/logs/` and run diagnostic commands like `run-e2e-tests` or `diagnose-universe`.