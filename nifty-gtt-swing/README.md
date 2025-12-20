# nifty-gtt-swing

Complete end-to-end system for NIFTY50 swing trading: data fetching, signal computation (VCP, SEPA, Donchian, BBKC Squeeze, TS Momentum), backtesting with WFO, portfolio construction (risk parity + BL), GTT order planning, and Excel generation in Delivery mode.

## Quickstart

- Create a Python 3.10+ venv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Set up .env with UPSTOX_ACCESS_TOKEN, UPSTOX_API_KEY, UPSTOX_API_SECRET, and DEFAULT_PRODUCT=D

- Example full pipeline:

```bash
# 1. Fetch fresh NIFTY50 data and compute indicators
python -m src.cli fetch-data --days 500 --out data/nifty50_indicators_renamed.csv

# 2. Screener (Top-25 + CompositeScore)
python -m src.cli screener --path data/nifty50_indicators_renamed.csv --out outputs/screener

# 3. Walk-Forward Optimization for SEPA
python -m src.cli wfo --path data/nifty50_indicators_renamed.csv --strategy SEPA --config config.yaml

# 4. Backtest ALL strategies and select best
python -m src.cli backtest --path data/nifty50_indicators_renamed.csv --out outputs/backtests

# 5. Build GTT plan with fallback
python -m src.cli gtt-plan --path data/nifty50_indicators_renamed.csv --strategy SEPA --fallback-strategies Donchian_Breakout VCP --out outputs/gtt

# 6. Generate Final Excel
python -m src.cli final-excel --plan outputs/gtt/gtt_plan_latest.csv --out outputs/gtt/GTT_Delivery_Final.xlsx

# 7. Place GTT orders LIVE (set --dry-run false for production)
python -m src.cli gtt-place --plan outputs/gtt/gtt_plan_latest.csv --dry-run true
```

## Strategies

- **VCP (Volume Contraction Pattern)**: Progressive BB bandwidth contraction, higher lows, volume dry-up, pivot breakout. Gates on Minervini trend template.
- **SEPA (Stage Emission Price Action)**: 8-point Minervini compliance + tight base + breakout.
- **Donchian Breakout**: Parameterized window (20/55/100), optional volume, middle-line pullback entry.
- **BBKC Squeeze**: Bollinger Bands inside Keltner Channels → breakout with RVOL confirmation.
- **TS Momentum**: 12-month lookback, positive momentum flag.

References: VCP/SEPA from Minervini; Donchian from indicator docs; BB/KC Squeeze from platform examples; TS Momentum from Moskowitz-Ooi-Pedersen (JFE, 2012).

## RSI & Golden Crossover

Informational metrics for decision support (do not affect GTT rules).

- **RSI Status**: Buckets RSI14 into "Oversold (≤30)", "Neutral (30–70)", "Overbought (≥70)".
- **Golden Crossover**: EMA50 crossing EMA200.
  - GoldenBull_Flag: 1 if EMA50 ↑ EMA200 (bullish signal).
  - GoldenBear_Flag: 1 if EMA50 ↓ EMA200 (bearish signal).
  - Dates: Most recent crossover date per symbol (YYYY-MM-DD or empty).

Exposed in screener_latest.csv, plan.csv, final Excel, and dedicated CLI.

## WFO & Portfolio

- **Walk-Forward Optimization**: Rolling/anchored cycles for parameter stability. Reports OOS Sharpe, efficiency.
- **Portfolio Construction**: Risk parity for equal risk contribution; optional Black-Litterman tilt using CompositeScore/RS as views.

## Compliance

- GTT legs: ENTRY ABOVE/BELOW, TARGET/STOPLOSS IMMEDIATE, optional TSL with trailing_gap.
- EDIS required for Delivery SELL legs; authorize in Upstox app.
## Final Excel (Delivery mode)

The `final-excel` command generates a single-sheet Excel summary of the GTT plan for quick review and execution.

### What it contains
- **Columns**: Symbol | GTT_Buy_Price | Stoploss | Sell_Rate | Strategy | Notes | RSI14 | RSI14_Status | GoldenBull_Flag | GoldenBear_Flag | GoldenBull_Date | GoldenBear_Date | Generated_At_IST
- **Sheet**: "GTT-Delivery-Plan" with tabular layout, sorted by Strategy then Symbol.
- **Footer**: Reminders for Delivery product, entry types, and EDIS authorization.

### Command
```bash
python -m src.cli final-excel --plan outputs/gtt/gtt_plan_latest.csv --out outputs/gtt/GTT_Delivery_Final.xlsx
```

### Sample (2-3 rows)
| Symbol | GTT_Buy_Price | Stoploss | Sell_Rate | Strategy | Notes | Generated_At_IST |
|--------|---------------|----------|-----------|----------|-------|------------------|
| RELIANCE | 1522.00 | 1499.00 | 1558.00 | MR | MR dip near EMA20 | 2025-12-16 16:12:00 |
| TCS | 4200.50 | 4150.00 | 4280.00 | Breakout | RVOL=1.8; Donchian breakout | 2025-12-16 16:12:00 |

This Excel is the **single source of truth** for the day's actionable GTT plan.

## RSI-Golden Summary CLI

Export per-symbol latest RSI and Golden crossover info:

```bash
python -m src.cli rsi-golden --path data/nifty50_indicators_renamed.csv --out outputs/rsi_golden_summary.csv
```

Output CSV: Symbol, Date, RSI14, RSI14_Status, EMA50, EMA200, GoldenBull_Flag, GoldenBear_Flag, GoldenBull_Date, GoldenBear_Date (sorted by Symbol).

## Recommended Scheduling (IST)

- **Time of day**: Run pipeline **after market close**, 16:10–16:20 IST: screener → backtest → select → gtt-plan → final-excel → (optionally) gtt-place.
- **Day of week**: Daily on trading days (Mon–Fri). Weekly refresh every Monday EOD to reconfirm best strategy.
- **Day of month**: Monthly robustness sweep on first trading day EOD (re-backtest + select).

## Daily Recency Validation

The system includes robust validations to ensure decisions are based on the **latest NIFTY50 indicators** covering the **last 500 trading days**. If data is stale or incomplete, the pipeline fails fast with clear diagnostics.

### What it guarantees

* **Latest**: `latest_date` in indicators file is within **1 day** of **today (IST)**.
* **Complete**: at least **500 trading days** in the dataset, **50 symbols** present.
* **Consistent**: derived outputs (screener/plan/excel) are produced from the **same latest date**—no stale mismatch.
* **Auditable**: logs capture summaries and any failing checks with actionable fixes.

### Commands

#### 1. Fetch + Validate (Recommended)

Fetch fresh data and validate immediately:

```bash
python -m src.cli fetch-and-validate \
  --out data/indicators_500d.parquet \
  --max-age-days 1 \
  --required-days 500
```

**Success output:**
```
FETCH + VALIDATION OK: latest=2025-12-20, rows=25000, symbols=50, window>=500 days
```

**Failure output:**
```
FETCH + VALIDATION FAILED: Stale data: latest_date=2025-12-18 is 2 days old (max_age_days=1, today_ist=2025-12-20)
Re-run fetch-data; ensure API token valid; verify job schedule at EOD.
```

#### 2. Validate Latest (Post-Pipeline Check)

Validate recency and consistency of all decision files:

```bash
python -m src.cli validate-latest \
  --data data/indicators_500d.parquet \
  --screener outputs/screener/screener_latest.csv \
  --plan outputs/gtt/gtt_plan_latest.csv \
  --excel outputs/gtt/GTT_Delivery_Final.xlsx
```

**Success output:**
```
VALIDATION PASSED: latest=2025-12-20, rows=25000, symbols=50, window>=500 days
```

**Failure example (stale data):**
```
VALIDATION FAILED: Stale data: latest_date=2025-12-18 is 2 days old (max_age_days=1, today_ist=2025-12-20)
```

**Failure example (inconsistent outputs):**
```
VALIDATION FAILED: Inconsistent outputs: screener_latest.csv latest=2025-12-19 vs indicators latest=2025-12-20
```

### Pipeline Guards

All decision commands (`screener`, `backtest`, `select`, `gtt-plan`, `final-excel`) include lightweight validation guards that check recency and coverage. If validation fails, the command exits with code 2.

**Override for testing:** Use `--skip-validation` flag to bypass checks (not recommended for production).

### Suggested Daily Schedule

Run this once per trading day **after market close**:

```bash
# 1) Fetch + validate
python -m src.cli fetch-and-validate --out data/indicators_500d.parquet --max-age-days 1 --required-days 500

# 2) Screener → Backtest/Select → Plan → Excel
python -m src.cli screener   --path data/indicators_500d.parquet --out outputs/screener
python -m src.cli backtest   --path data/indicators_500d.parquet --strategy ALL --out outputs/backtests
python -m src.cli select     --path data/indicators_500d.parquet --out outputs/backtests
python -m src.cli gtt-plan   --path data/indicators_500d.parquet --top 25 --out outputs/gtt
python -m src.cli final-excel --plan outputs/gtt/gtt_plan_latest.csv --out outputs/gtt/GTT_Delivery_Final.xlsx

# 3) Final recency check on decision files (optional)
python -m src.cli validate-latest \
  --data data/indicators_500d.parquet \
  --screener outputs/screener/screener_latest.csv \
  --plan outputs/gtt/gtt_plan_latest.csv \
  --excel outputs/gtt/GTT_Delivery_Final.xlsx
```

### Logs

Validation operations are logged to `outputs/logs/validation_YYYYMMDD_HHMM.log` with detailed summaries and error diagnostics.

## Notes & References

- GTT rules: ENTRY ABOVE/BELOW; TARGET/STOPLOSS IMMEDIATE in multi-leg.
- TSL optional via "trailing_gap" in STOPLOSS (beta).
- SELL legs require one-time EDIS authorization.

References:
- [GTT Orders](https://upstox.com/developer/api-documentation/gtt-orders/)
- [Place GTT Order](https://upstox.com/developer/api-documentation/place-gtt-order/)
- [Modify GTT](https://upstox.com/developer/api-documentation/modify-gtt-order/)
- [TSL Announcement](https://upstox.com/developer/api-documentation/announcements/tsl-gtt-order/)```

Note: For SELL legs, complete EDIS authorization once in Upstox app.
