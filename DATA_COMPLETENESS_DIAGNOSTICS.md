# SWING_BOT Data Completeness Diagnostics

This document describes the data completeness diagnostic tools for SWING_BOT, ensuring reliable historical data for backtesting and signal generation.

## Overview

SWING_BOT relies on complete historical data from Upstox API. The diagnostic tools verify data completeness across multiple timeframes and validate options chain data for F&O stocks.

## Diagnostic Commands

### 1. Single Timeframe Diagnostic (Legacy)

```bash
python -m src.cli diagnose-upstox-data-completeness --days 1000 --sample-size 50 --output outputs/upstox_completeness_report.csv
```

**Parameters:**
- `--days`: Days of historical data to request (default: 1000)
- `--sample-size`: Symbols to sample (default: 50, use 0 for all)
- `--output`: CSV report path
- `--rate-limit`: API call delay in seconds (default: 0.5)
- `--mock`: Use mock data for testing

### 2. Multi-Timeframe Candle Diagnostic (Recommended)

```bash
python -m src.cli diagnose-candle-completeness --timeframes 1d,1w,1mo,5m --days 1000 --sample-size 50 --include-options --output outputs/candle_completeness_report.csv
```

**Parameters:**
- `--timeframes`: Comma-separated timeframes (1d,1w,1mo,5m)
- `--days`: Days of historical data to request (default: 1000)
- `--sample-size`: Symbols to sample (default: 50, use 0 for all)
- `--include-options`: Include F&O options chain verification
- `--output`: CSV report path
- `--rate-limit`: API call delay in seconds (default: 0.5)
- `--mock`: Use mock data for testing

## Supported Timeframes

- `1d`: Daily candles
- `1w`: Weekly candles
- `1mo`: Monthly candles
- `5m`: 5-minute intraday candles

## Options Data Verification

For F&O eligible stocks (top 20 NIFTY 200 by liquidity), the diagnostic verifies:
- Options chain completeness for current expiry
- Strike price coverage (calls + puts)
- Data field completeness (OI, volume, LTP)

## Report Format

### Candle Data Report
```csv
symbol,timeframe,requested_days,returned_bars,start_date,end_date,status,completeness_pct,data_type
RELIANCE.NS,1d,1000,681,2023-04-17,2026-01-08,FULL,99.8,candle
TCS.NS,5m,1000,27581,2023-04-17,2026-01-08,LIMITED,62.5,candle
```

### Options Data Report
```csv
symbol,timeframe,requested_days,returned_bars,start_date,end_date,status,completeness_pct,data_type
RELIANCE.NS,options,1000,45,2026-02-27,,FULL,95.2,options
```

## Status Definitions

- **FULL**: ≥90% completeness (reliable for backtesting)
- **LIMITED**: 50-89% completeness (usable but may affect signals)
- **MISSING**: <50% completeness (unreliable)

## Integration

### Automatic Validation in Live Pipeline

The `orchestrate-live` command automatically validates data completeness:

```bash
python -m src.cli orchestrate-live --data-out data/nifty50_data_today.csv --required-days 500
```

If completeness <90%, warnings are displayed and critical alerts trigger for <80%.

### Manual Validation

```python
from src.data_fetch import validate_candle_completeness

stats = validate_candle_completeness(
    symbols=['RELIANCE.NS', 'TCS.NS'],
    timeframes=['1d', '1w', '5m'],
    days_requested=1000,
    include_options=True
)
```

## Expected Completeness Levels

Based on testing with 50-symbol samples:

| Timeframe | Expected Completeness | Notes |
|-----------|----------------------|-------|
| 1d | 95-100% | ~680-700 bars max from API |
| 1w | 95-100% | ~140-150 weeks coverage |
| 1mo | 90-100% | ~30-36 months coverage |
| 5m | 60-80% | Limited by API bar limits |

## Troubleshooting

### Common Issues

1. **API_ERROR_400**: Invalid instrument keys
   - Check `artifacts/universe/instrument_keys.json`
   - Some symbols may not be available in Upstox

2. **Low Completeness**: API data limits
   - Upstox returns max ~700 bars per request
   - For longer periods, use multiple requests or accept limitations

3. **Rate Limiting**: Too many API calls
   - Increase `--rate-limit` parameter
   - Use smaller `--sample-size`

### Best Practices

1. **Weekly Monitoring**: Run diagnostics weekly to detect API changes
2. **Pre-Backtest Validation**: Always validate before major strategy testing
3. **Sample Testing**: Use `--sample-size 20-50` for regular monitoring
4. **Full Universe**: Use `--sample-size 0` for comprehensive quarterly audits

## API Limitations Discovered

- **Maximum Bars**: ~680-700 bars per request regardless of date range
- **Timeframe Coverage**: 5-minute data most affected by bar limits
- **Options Data**: Current expiry only, rate-limited
- **Instrument Coverage**: ~50% of NIFTY 200 have valid Upstox instrument keys

## Production Readiness

✅ **Data completeness diagnostics integrated into live pipeline**
✅ **Multi-timeframe support including 5-minute candles**
✅ **Options chain verification for F&O stocks**
✅ **Automatic warnings for low completeness**
✅ **Comprehensive CSV reporting**

The diagnostics ensure SWING_BOT operates with reliable data, preventing silent failures in backtesting and signal generation.