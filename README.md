# SWING_BOT

Automated NIFTY 50 swing trading system with comprehensive EOD pipeline, plan auditing, and Teams notifications.

## 🚀 Quick Start - One-Line EOD Runbook

```bash
# Complete EOD pipeline in one command (run daily at 15:30 IST)
python -m src.cli orchestrate-eod --data-out data/nifty50_indicators_full.csv --max-age-days 1 --required-days 500 --top 25 --strict --post-teams --multi-tf
```

## 📋 Daily EOD Schedule

| Time (IST) | Activity | Command | Status |
|------------|----------|---------|--------|
| 15:30 | **Complete EOD Pipeline** | `orchestrate-eod --strict --post-teams --multi-tf` | ✅ Automated |
| 15:45 | **Review Teams Notification** | Check Adaptive Card in Teams | Manual |
| 16:00 | **Validate Final Excel** | Check `outputs/gtt/GTT_Delivery_Final.xlsx` | Manual |
| 16:15 | **Plan Audit Review** | Review `outputs/gtt/gtt_plan_audited.csv` | Manual |

## 🏗️ Architecture

### Core Pipeline
1. **Data Fetch** → 2. **Validation** → 3. **Screener** → 4. **Backtest** → 5. **Strategy Selection** → 6. **GTT Plan Generation** → 7. **Plan Audit** → 8. **Final Excel** → 9. **Teams Notification**

### Key Components
- **Data Fetch**: Upstox API integration with AllFetch capability
- **Validation**: Freshness (≤1 day), coverage (≥500 days), symbol count (50)
- **Plan Audit**: Strict validation with canonical pricing and fail-fast mode
- **Teams Integration**: Adaptive Card notifications with audit summaries
- **Multi-TF**: Support for 1m, 15m, 1h, 4h, 1d, 1w, 1mo timeframes

## 📊 Data Requirements

### Validation Checks
- **Freshness**: Data must be ≤1 day old
- **Coverage**: Minimum 500 trading days per symbol
- **Symbols**: Exactly 50 NIFTY 50 stocks
- **Completeness**: No missing OHLCV data

### AllFetch Timeframes
- `1m`: 1-minute candles
- `15m`: 15-minute candles
- `1h`: 1-hour candles
- `4h`: 4-hour candles (resampled)
- `1d`: Daily candles
- `1w`: Weekly candles (resampled)
- `1mo`: Monthly candles (resampled)

## 🔧 Setup & Configuration

### Environment Variables
```bash
# Required
UPSTOX_API_KEY=your_api_key
UPSTOX_API_SECRET=your_api_secret
UPSTOX_ACCESS_TOKEN=your_access_token

# Optional
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...  # For Teams notifications
```

### Dependencies
```bash
pip install -r requirements.txt
```

### Configuration File (config.yaml)
```yaml
audit:
  tick: 0.05
  max_age_days: 1
  strict_mode: true
  max_entry_pct_diff: 0.02
  risk_multiplier: 1.5
  reward_multiplier: 2.0

teams:
  webhook_url: "https://outlook.office.com/webhook/..."

data:
  max_age_days: 1
  required_days: 500
  required_symbols: 50
```

## 📈 Commands

### orchestrate-eod
Run the complete EOD pipeline with all validations and notifications.

```bash
python -m src.cli orchestrate-eod \
  --data-out data/nifty50_indicators_full.csv \
  --max-age-days 1 \
  --required-days 500 \
  --top 25 \
  --strict \
  --post-teams \
  --multi-tf \
  --config config.yaml
```

**Parameters:**
- `--data-out`: Output path for indicators data
- `--max-age-days`: Maximum age of data in days (default: 1)
- `--required-days`: Minimum trading days required (default: 500)
- `--top`: Number of top candidates to select (default: 25)
- `--strict`: Enable strict plan audit mode
- `--post-teams`: Send results to Microsoft Teams
- `--multi-tf`: Generate multi-timeframe workbook
- `--config`: Path to configuration file

### fetch-all (AllFetch)
Fetch data for multiple timeframes simultaneously.

```bash
python -m src.cli fetch-all \
  --symbols "RELIANCE.NS,TCS.NS,HDFCBANK.NS" \
  --timeframes "1m,15m,1h,4h,1d,1w,1mo" \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --out-dir data/multi_tf
```

### teams-notify
Post GTT plan summary to Microsoft Teams.

```bash
python -m src.cli teams-notify \
  --plan outputs/gtt/gtt_plan_audited.csv \
  --date 2024-01-15 \
  --webhook-url "https://outlook.office.com/webhook/..."
```

### plan-audit
Run standalone plan audit with strict validation.

```bash
python -m src.cli plan-audit \
  --plan outputs/gtt/gtt_plan_latest.csv \
  --indicators data/nifty50_indicators_full.csv \
  --latest outputs/screener/screener_latest.csv \
  --out outputs/gtt/gtt_plan_audited.csv \
  --strict \
  --config config.yaml
```

## 🔍 Plan Audit System

### Audit Parameters
- **Tick Size**: 0.05 (₹0.05 for price validation)
- **Max Age**: 1 day (data freshness requirement)
- **Strict Mode**: Fail-fast on any audit failure
- **Entry Tolerance**: ±2% from canonical entry price
- **Risk Multiplier**: 1.5x for stop loss validation
- **Reward Multiplier**: 2.0x for target validation

### Audit Checks
1. **Price Freshness**: Latest data ≤ max_age_days old
2. **Entry Logic**: Entry price within tolerance of pivot-based calculation
3. **Stop Logic**: Stop loss provides adequate risk management
4. **Target Logic**: Target provides sufficient reward potential
5. **Canonical Pricing**: All prices derived from consistent pivot sources

### Audit Output Columns
- `Audit_Flag`: PASS/FAIL status
- `Issues`: Detailed problem description
- `Fix_Suggestion`: Recommended corrective action
- `Pivot_Source`: Source of pivot calculation
- `Entry_Logic`: Entry price derivation logic
- `Stop_Logic`: Stop loss calculation logic
- `Target_Logic`: Target price derivation logic
- `Latest_Close`: Most recent closing price
- `Latest_LTP`: Last traded price
- `Canonical_Entry`: Calculated canonical entry price
- `Canonical_Stop`: Calculated canonical stop price
- `Canonical_Target`: Calculated canonical target price

## 📊 Final Excel Format

### GTT-Delivery-Plan Sheet Columns (Exact Order)
1. Symbol
2. Qty
3. ENTRY_trigger_price
4. TARGET_trigger_price
5. STOPLOSS_trigger_price
6. DecisionConfidence
7. Confidence_Level
8. R (Risk-Reward Ratio)
9. Explanation
10. GTT_Explanation
11. Audit_Flag
12. Issues
13. Fix_Suggestion
14. Pivot_Source
15. Entry_Logic
16. Stop_Logic
17. Target_Logic
18. Latest_Close
19. Latest_LTP
20. Canonical_Entry
21. Canonical_Stop
22. Canonical_Target

### Conditional Formatting
- **DecisionConfidence**: Green (≥4), Yellow (3-4), Red (<3)
- **Audit_Flag**: Green (PASS), Red (FAIL)

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Key Test Categories
- **test_orchestrate_eod.py**: End-to-end pipeline testing
- **test_plan_audit.py**: Audit validation testing
- **test_teams_notifier.py**: Teams integration testing
- **test_fetch_all.py**: Multi-timeframe fetch testing
- **test_data_validation.py**: Data quality validation

### Test Coverage
- Pipeline orchestration
- Plan audit strict mode
- Teams notification formatting
- Multi-TF data fetching
- Data validation guards
- Excel generation with audit columns

## 📁 Output Structure

```
outputs/
├── screener/
│   └── screener_latest.csv
├── backtests/
│   ├── selected_strategy.json
│   ├── AVWAP/
│   ├── Donchian/
│   ├── MR/
│   ├── SEPA/
│   ├── Squeeze/
│   └── VCP/
├── gtt/
│   ├── gtt_plan_latest.csv
│   ├── gtt_plan_audited.csv
│   └── GTT_Delivery_Final.xlsx
├── multi_tf/
│   ├── nifty50_1m.csv
│   ├── nifty50_15m.csv
│   └── ...
└── logs/
    └── validation_20240115_1530.log
```

## 🚨 Error Handling

### Common Issues & Solutions

**Data Fetch Failures:**
- Check Upstox API credentials
- Verify internet connectivity
- Review API rate limits

**Plan Audit Failures (Strict Mode):**
- Review pivot calculations
- Check data freshness
- Validate price tolerances

**Teams Notification Failures:**
- Verify webhook URL
- Check Teams channel permissions
- Review Adaptive Card payload

**Validation Errors:**
- Ensure data is ≤1 day old
- Confirm 500+ trading days coverage
- Verify all 50 NIFTY symbols present

## 🔄 CI/CD Integration

### Automated EOD Pipeline
```yaml
# .github/workflows/eod-pipeline.yml
name: EOD Pipeline
on:
  schedule:
    - cron: '30 10 * * 1-5'  # 15:30 IST Mon-Fri
  workflow_dispatch:

jobs:
  eod:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run EOD Pipeline
        run: |
          python -m src.cli orchestrate-eod \
            --data-out data/nifty50_indicators_full.csv \
            --strict --post-teams --multi-tf
```
## 📄 Documentation

### Operations Documentation
- **[RUNBOOK.md](docs/RUNBOOK.md)**: Complete EOD runbook with exact commands, troubleshooting, and expected outputs
- **[OPERATIONS.md](docs/OPERATIONS.md)**: Daily/weekly/monthly procedures, recovery steps, and housekeeping tasks
- **[ALERTS.md](docs/ALERTS.md)**: Teams webhook setup, notification cards, and escalation procedures

### Scheduling
- **[TaskScheduler_SWING_BOT_EOD.xml](schedule/TaskScheduler_SWING_BOT_EOD.xml)**: Windows Task Scheduler configuration for automated EOD execution
- **[cron_examples.md](schedule/cron_examples.md)**: Linux/Mac cron job examples with timezone handling

## 🛠️ Scripts

### Notification Scripts
- **[post_teams_success.py](scripts/post_teams_success.py)**: Posts success notifications with EOD results summary
- **[post_teams_failure.py](scripts/post_teams_failure.py)**: Posts failure notifications with error details and retry logic

### Maintenance Scripts
- **[calibration_snapshot.py](scripts/calibration_snapshot.py)**: Creates weekly confidence calibration visualizations

### Usage Examples
```bash
# Post success notification after EOD completion
python scripts/post_teams_success.py

# Post failure notification with error details
python scripts/post_teams_failure.py --error-category SYSTEM --error-message "Data fetch timeout"

# Generate weekly calibration snapshot
python scripts/calibration_snapshot.py
```
## 📈 Monitoring & Alerts

### Teams Adaptive Card Features
- **Audit Summary**: PASS/FAIL counts with color coding
- **Top 5 Positions**: Symbol, prices, confidence, audit status
- **Excel Download**: Direct link to final report
- **Error Notifications**: Immediate alerts on pipeline failures

### Log Files
- **Standardized Logging**: All modules use `src/logging_setup.py` with rotating file handlers
- **Module-specific Logs**: `outputs/logs/{module_name}_YYYYMMDD.log` (10MB max, 5 backups)
- **Pipeline Logs**: `outputs/logs/orchestrate_eod_YYYYMMDD_HHMMSS.log` for EOD runs
- **Script Logs**: Individual logs for notification and maintenance scripts
- **Error Logs**: Detailed stack traces with function names and line numbers

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd swing_bot
pip install -r requirements.txt
cp config.yaml.example config.yaml
# Configure environment variables
pytest tests/  # Run tests
```

### Code Standards
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for new features
- Pre-commit hooks for linting

## 📄 License

Proprietary - Internal Use Only

---

**Daily Operations**: Run `orchestrate-eod --strict --post-teams --multi-tf` at 15:30 IST
**Emergency Contacts**: Check Teams notifications and audit logs for issues