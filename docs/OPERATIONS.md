# SWING_BOT Operations Manual

## Overview
This manual covers operational procedures for maintaining and monitoring the SWING_BOT trading system. Includes daily, weekly, and monthly maintenance tasks, recovery procedures, and data management guidelines.

## Daily Operations

### EOD Execution (16:10-16:20 IST)
1. **Execute orchestrate-eod command** (see RUNBOOK.md)
2. **Verify outputs**:
   - Check `outputs/gtt/GTT_Delivery_Final.xlsx` exists and has data
   - Verify audit results in console output
   - Confirm Teams notification received
3. **Review audit results**:
   - Open `outputs/gtt/gtt_plan_audited.csv`
   - Check PASS/FAIL ratio (target: >90% pass)
   - Review any FAIL reasons and fix suggestions
4. **Backup critical outputs** to secure storage

### Optional Pre-Market Tasks
```bash
# Pre-market plan audit (if needed)
python -m src.cli plan-audit \
  --plan outputs/gtt/gtt_plan_latest.csv \
  --data data/indicators_500d.parquet \
  --screener outputs/screener/screener_latest.csv \
  --out outputs/gtt/gtt_plan_premarket_audit.csv \
  --strict true
```

### Monitoring Checks
- **Execution Time**: Should complete in <10 minutes
- **Data Freshness**: Latest date should be current trading day
- **Symbol Coverage**: All 50 NIFTY 50 stocks present
- **Audit Quality**: >90% positions should pass audit

## Weekly Operations

### Monday: Strategy Refresh
```bash
# Run full backtest cycle
python -m src.cli backtest-all --strategies SEPA,VCP,Donchian,MR,Squeeze,AVWAP

# Select best strategy
python -m src.cli select-strategy --backtest-dir outputs/backtests

# Generate confidence calibration snapshot
python scripts/calibration_snapshot.py
```

### Friday: End-of-Week Review
1. **Review weekly performance**:
   - Check strategy selection results
   - Analyze confidence calibration trends
   - Review any recurring audit failures

2. **Data maintenance**:
   - Verify data freshness (should be updated daily)
   - Check for data quality issues
   - Archive weekly logs

## Monthly Operations

### First Monday: Comprehensive Review
```bash
# Run walk-forward optimization
python -m src.cli wfo-sweep --months 12 --step-size 1

# Review risk parameters
python -m src.cli risk-review --portfolio outputs/gtt/gtt_plan_latest.csv

# Generate monthly reports
python -m src.cli monthly-report --output outputs/reports/monthly_$(date +%Y%m).xlsx
```

### Risk and Compliance Review
1. **Sector concentration**: Ensure no sector exceeds 25% of portfolio
2. **Risk limits**: Verify total portfolio risk within acceptable bounds
3. **Strategy performance**: Review 3-month rolling returns
4. **Audit compliance**: Check 95%+ audit pass rate over month

### System Health Check
- Review all log files for errors
- Check disk space usage
- Verify backup integrity
- Update dependencies if needed

## Data Management

### Data Retention Policy
- **Raw data**: Keep 2 years of daily indicators
- **Backtest results**: Keep 1 year of detailed results
- **GTT plans**: Keep 6 months of executed plans
- **Logs**: Keep 3 months of detailed logs
- **Reports**: Keep 2 years of monthly reports

### Housekeeping Tasks
```bash
# Monthly cleanup (run on last day of month)
find outputs/logs -name "*.log" -mtime +90 -delete
find outputs/backtests -name "*.csv" -mtime +365 -exec gzip {} \;
find outputs/gtt -name "gtt_plan_*.csv" -mtime +180 -exec gzip {} \;

# Archive monthly outputs
tar -czf archives/$(date +%Y%m)_outputs.tar.gz outputs/
```

### Backup Strategy
- **Daily**: Critical outputs (Excel, audited plans) to secure storage
- **Weekly**: Full outputs directory to cloud backup
- **Monthly**: Complete system backup including code and configuration

## Recovery Procedures

### Data Corruption Recovery
1. **Identify corruption**: Check validation errors in logs
2. **Fresh data fetch**:
   ```bash
   python -m src.cli fetch-nifty50 --days 600 --out data/recovery.parquet
   ```
3. **Validate and replace**:
   ```bash
   python -m src.cli validate-data --data data/recovery.parquet
   mv data/recovery.parquet data/indicators_500d.parquet
   ```

### Strategy Failure Recovery
1. **Fallback strategy selection**:
   ```bash
   python -m src.cli select-strategy --backtest-dir outputs/backtests --fallback Donchian
   ```
2. **Manual parameter adjustment** if needed
3. **Re-run orchestration** with adjusted parameters

### System Outage Recovery
1. **Check system status**: Network, API access, disk space
2. **Delayed execution**: Run EOD process after market close if possible
3. **Partial recovery**: Execute individual components if full pipeline fails
4. **Communication**: Notify stakeholders of delays and expected completion

## Monitoring and Alerts

### Automated Monitoring
- **Teams notifications**: Success/failure alerts for EOD execution
- **Log monitoring**: Automated scanning for critical errors
- **Performance tracking**: Execution time and success rate metrics

### Manual Monitoring
- **Daily**: Check EOD execution completion
- **Weekly**: Review strategy performance
- **Monthly**: Comprehensive system health check

### Alert Escalation
1. **Warning**: Teams notification with details
2. **Critical**: Immediate notification to on-call personnel
3. **Emergency**: Trading desk notification for immediate action

## Configuration Management

### Environment Variables
- `UPSTOX_ACCESS_TOKEN`: API access (rotate quarterly)
- `TEAMS_WEBHOOK_URL`: Notification endpoint (test monthly)
- `LOG_LEVEL`: Set to INFO for production

### Configuration Files
- `config.yaml`: Strategy parameters (version controlled)
- `requirements.txt`: Python dependencies (update monthly)
- `.env`: Environment variables (not version controlled)

### Version Control
- **Code**: Maintained in Git with tagged releases
- **Configuration**: Version controlled with change tracking
- **Documentation**: Updated with system changes

## Performance Benchmarks

### Execution Times
- **Full EOD pipeline**: <10 minutes
- **Data fetch**: <5 minutes
- **Backtest cycle**: <15 minutes
- **Plan audit**: <2 minutes

### Quality Metrics
- **Data completeness**: 100% symbol coverage
- **Audit pass rate**: >90%
- **Strategy stability**: <10% weekly strategy changes
- **System uptime**: 99.5%+

## Emergency Contacts

### Primary Contacts
- **System Administrator**: For infrastructure issues
- **Development Lead**: For code and logic issues
- **Trading Operations**: For trading-related decisions

### Escalation Path
1. **Level 1**: System administrator (infrastructure)
2. **Level 2**: Development team (code/logic)
3. **Level 3**: Trading desk (business decisions)

---

*Last Updated: December 21, 2025*