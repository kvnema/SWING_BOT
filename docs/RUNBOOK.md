# SWING_BOT EOD Runbook

## Overview
SWING_BOT is an automated trading system that generates GTT (Good Till Triggered) orders for NIFTY 50 stocks based on technical analysis and backtested strategies. This runbook covers the daily End-of-Day (EOD) execution process.

## Prerequisites
- Python 3.8+ with virtual environment
- Upstox API access token (`UPSTOX_ACCESS_TOKEN` in `.env`)
- Teams webhook URL (`TEAMS_WEBHOOK_URL` in `.env`) for notifications
- Minimum 500 trading days of historical data
- Windows/Linux/Mac environment with internet access

## EOD Execution Time
**16:10 - 16:20 IST (Monday - Friday)**

Execute during market close window when data is most current but before after-hours volatility.

## One-Line EOD Command

```bash
python -m src.cli orchestrate-eod \
  --data-out data/indicators_500d.parquet \
  --max-age-days 1 --required-days 500 \
  --top 25 --strict true --post-teams true --multi-tf true
```

## Expected Outputs

After successful execution, verify these artifacts are created:

### Core Outputs
- **Final Excel**: `outputs/gtt/GTT_Delivery_Final.xlsx`
  - GTT-Delivery-Plan sheet with 22 columns including audit flags
  - Portfolio summary with total positions, value, and risk
  - Conditional formatting for confidence levels

- **GTT Plan**: `outputs/gtt/gtt_plan_latest.csv`
  - Raw plan with entry/stop/target prices
  - Confidence scores and explanations

- **Audited Plan**: `outputs/gtt/gtt_plan_audited.csv`
  - Plan with audit flags (PASS/FAIL)
  - Issue tracking and fix suggestions

### Optional Outputs (when enabled)
- **Multi-TF Workbook**: `outputs/multi_tf_nifty50.xlsx`
  - Individual sheets for 1m, 15m, 1h, 4h, 1d, 1w, 1mo timeframes

- **AllFetch Workbook**: `outputs/allfetch_nifty50.xlsx`
  - Single-sheet multi-timeframe data

### Logs
- **Pipeline Logs**: `outputs/logs/orchestrate_eod_<timestamp>.log`
- **Component Logs**: Individual logs for data_fetch, plan_audit, teams_notifier

## Success Indicators

### Console Output
```
🚀 Starting SWING_BOT EOD Orchestration...
✅ Data validated: earliest=2024-01-01, latest=2025-12-20, days=500, symbols=50, rows=25000
✅ Screener completed: 50 symbols → outputs/screener/screener_latest.csv
✅ Backtests completed: selected SEPA
✅ GTT plan built: 25 positions → outputs/gtt/gtt_plan_latest.csv
✅ Final Excel generated → outputs/gtt/GTT_Delivery_Final.xlsx
✅ Plan audit completed → outputs/gtt/gtt_plan_audited.csv
✅ Multi-TF workbook generated → outputs/multi_tf_nifty50.xlsx
✅ Posted to Teams

🎉 SWING_BOT EOD Orchestration Complete!
📅 Latest Date: 2025-12-20
📊 Symbols: 50
📈 Trading Days: 500
✅ Audit Results: 23 PASS, 2 FAIL
📁 Final Excel: outputs/gtt/GTT_Delivery_Final.xlsx
```

### Teams Notification
- Success card with pass/fail counts and top positions
- Links to Excel and CSV files
- Audit summary with confidence levels

## Troubleshooting

### Common Failures

| Error Message | Cause | Fix Steps |
|---------------|-------|-----------|
| `ValidationError: Stale data: latest_date=2025-12-19 is 2 days old` | Data not refreshed today | 1. Check Upstox API access<br>2. Run `python -m src.cli fetch-nifty50 --days 500`<br>3. Verify `.env` has valid `UPSTOX_ACCESS_TOKEN` |
| `ValidationError: Insufficient data: trading_days_count=450 < 500` | Not enough historical data | 1. Increase `--required-days` parameter<br>2. Run fetch with more days: `--days 600`<br>3. Check data source completeness |
| `ValidationError: Missing symbols: expected=50, found=48` | Some stocks failed to fetch | 1. Check API rate limits<br>2. Retry fetch command<br>3. Manual verification of failed symbols |
| `AuditError: Plan audit failed: Strict mode enabled: 5 audit failures found` | Price validation failures | 1. Review `outputs/gtt/gtt_plan_audited.csv`<br>2. Check pivot data freshness<br>3. Adjust risk parameters if needed<br>4. Consider manual override for critical positions |
| `TeamsError: Webhook posting failed` | Teams webhook misconfigured | 1. Verify `TEAMS_WEBHOOK_URL` in `.env`<br>2. Test webhook with `python -m src.cli teams-notify`<br>3. Check network connectivity |
| `ValueError: UPSTOX_ACCESS_TOKEN not set` | Missing API token | 1. Add `UPSTOX_ACCESS_TOKEN=your_token` to `.env`<br>2. Verify token validity<br>3. Check `.env` file permissions |

### Recovery Steps

1. **Check Logs**: Review `outputs/logs/orchestrate_eod_<timestamp>.log` for detailed error information
2. **Verify Environment**: Ensure `.env` file exists with required variables
3. **Test Components**: Run individual commands to isolate failures:
   ```bash
   # Test data fetch
   python -m src.cli fetch-nifty50 --days 500 --out data/test.parquet

   # Test validation
   python -m src.cli validate-data --data data/test.parquet

   # Test plan audit
   python -m src.cli plan-audit --plan outputs/gtt/gtt_plan_latest.csv --strict
   ```
4. **Manual Override**: For urgent situations, use `--strict false` to bypass audit failures
5. **Contact Support**: Escalate to development team with full logs and error messages

## Monitoring

### Daily Checks
- Verify all output files are created with recent timestamps
- Check Teams notification was received
- Review audit pass/fail ratio (target: >90% pass rate)
- Monitor execution time (<10 minutes)

### Weekly Reviews
- Analyze strategy performance metrics
- Review confidence calibration
- Check for recurring failures

## Emergency Contacts

- **Development Team**: For technical issues and code fixes
- **Trading Desk**: For trading-related decisions and overrides
- **IT Support**: For infrastructure and access issues

---

*Last Updated: December 21, 2025*