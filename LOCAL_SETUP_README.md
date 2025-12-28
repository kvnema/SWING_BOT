# SWING_BOT Local GTT Monitor Setup

This guide helps you set up automated GTT order monitoring on your local Windows machine.

## Prerequisites

1. **Python Environment**: Virtual environment set up with dependencies installed
2. **Environment Variables**: `.env` file configured with broker credentials (ICICI Direct or Upstox)
3. **Windows Task Scheduler**: For automated scheduling (or run continuously)

## Supported Brokers

### ICICI Direct (Recommended for New Setups)
- ✅ GTT Orders support
- ⚠️ Session tokens require periodic manual refresh
- 📖 See [ICICI_SETUP_README.md](ICICI_SETUP_README.md) for detailed setup

### Upstox (Legacy)
- ✅ JWT tokens with automatic refresh
- ✅ Good Till Triggered orders
- ⚠️ Requires manual authorization code for token refresh

## Quick Setup

### 1. Choose Your Broker

#### For ICICI Direct:
```powershell
# Follow the ICICI setup guide
.\setup_icici.ps1 -Setup
```

#### For Upstox:
```bash
# Copy and edit environment file
copy .env.example .env
# Edit .env with your Upstox credentials
```

### 2. Test Setup
```powershell
# Test ICICI setup
.\setup_icici.ps1 -Test

# Or test Upstox setup
.\setup_gtt_scheduler.ps1 -Test
```

### 3. Choose Running Mode

#### Option A: Windows Task Scheduler (Recommended)
```powershell
# Install scheduled task
.\setup_gtt_scheduler.ps1 -Install
```

This creates a Windows Task Scheduler task that runs at:
- 8:15 AM, 9:15 AM - 3:15 PM (hourly), 4:30 PM IST
- Only on weekdays (Monday-Friday)
- With automatic retry and error handling

#### Option B: Continuous Mode
```powershell
# Run continuously (checks schedule every minute)
.\setup_gtt_scheduler.ps1 -Continuous
```

## Schedule Details

| Time (IST) | Description |
|------------|-------------|
| 8:15 AM   | Pre-market analysis |
| 9:15 AM   | Market open analysis |
| 10:15 AM  | Mid-morning check |
| 11:15 AM  | Late morning check |
| 12:15 PM  | Noon analysis |
| 1:15 PM   | Early afternoon check |
| 2:15 PM   | Mid-afternoon check |
| 3:15 PM   | Pre-close analysis |
| 4:30 PM   | End-of-day analysis |

## Monitoring & Logs

### View Logs
```bash
# Main scheduler logs
type logs\gtt_scheduler.log

# Individual run logs
type outputs\logs\*.log
```

### Check Task Status
```cmd
# View scheduled task
schtasks /query /tn "SWING_BOT_GTT_Monitor"

# Run task manually
schtasks /run /tn "SWING_BOT_GTT_Monitor"
```

### Stop Continuous Mode
- Press `Ctrl+C` in the PowerShell window

## Troubleshooting

### Task Scheduler Issues
```powershell
# Check task history
.\Get-ScheduledTaskInfo.ps1 -TaskName "SWING_BOT_GTT_Monitor"

# Delete and reinstall task
.\setup_gtt_scheduler.ps1 -Uninstall
.\setup_gtt_scheduler.ps1 -Install
```

### Environment Issues
```powershell
# Test environment setup
.\setup_gtt_scheduler.ps1 -Test

# Check environment variables
Get-ChildItem env: | Where-Object {$_.Name -like "*UPSTOX*"}
```

### Permission Issues
- Make sure you're running PowerShell as Administrator
- Check that the virtual environment is accessible
- Verify file permissions on logs and outputs directories

## Notifications

Configure notifications in your `.env` file:

```env
# Teams webhook
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/your-webhook-url

# Telegram bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email (fallback)
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

## Manual Testing

### Single Run
```bash
python scheduled_gtt_monitor.py
```

### Full Pipeline Test
```bash
python -m src.cli orchestrate-eod
```

### Check GTT Orders
```bash
python -c "from src.upstox_gtt import get_all_gtt_orders; import os; print(get_all_gtt_orders(os.getenv('UPSTOX_ACCESS_TOKEN')))"
```

## File Structure

```
swing-bot/
├── local_gtt_scheduler.py      # Continuous scheduler
├── scheduled_gtt_monitor.py    # Main monitoring logic
├── run_gtt_monitor.bat         # Batch script for Task Scheduler
├── setup_gtt_scheduler.ps1     # Setup/uninstall script
├── SWING_BOT_GTT_Task.xml      # Task Scheduler configuration
├── logs/
│   └── gtt_scheduler.log       # Scheduler logs
└── outputs/
    └── gtt/
        └── gtt_monitor_state.json  # State persistence
```

## Stopping the Service

### Task Scheduler Mode
```powershell
.\setup_gtt_scheduler.ps1 -Uninstall
```

### Continuous Mode
- Press `Ctrl+C` in the running PowerShell window

## Support

Check the logs for errors:
- `logs/gtt_scheduler.log` - Scheduler activity
- `outputs/logs/` - Individual run logs

Common issues:
1. **Environment variables not set** - Check `.env` file
2. **Virtual environment not activated** - Run from project root
3. **Network connectivity** - Check internet connection for Upstox API
4. **Token expired** - Refresh Upstox access token

For more help, check the main `AUTOMATED_DEPLOYMENT_README.md`.

---

# 📊 Current Market Analysis & Portfolio Recommendations

*Last Updated: December 28, 2025*

## Current Market Regime: OFF (ADX: 18.6, RSI: 45.0)
- **No signals generated** - Risk management working as designed
- **Framework ready** - All trade plans prepared for regime flip

## Sector-Based Correlation Insights

### 🔴 High Correlation Pairs (Avoid Overlap):
- **ASIANPAINT + TITAN** (0.7-0.8): Both consumer defensives, similar momentum
- **LT + TATAMOTORS** (0.4-0.6): Both cyclicals, similar economic sensitivity

### 🟢 Low Correlation Pairs (Best Diversification):
- **TITAN + UPL** (-0.1-0.2): Luxury consumer vs commodity chemicals
- **ASIANPAINT + LT** (0.2-0.4): Defensive consumer vs cyclical infrastructure
- **TATAMOTORS + BAJAJFINSV** (0.4-0.6): Auto cyclical vs financial services

## Recommended Portfolio Construction

### For ₹5L Portfolio (3-4 positions):
```
ASIANPAINT: 30-35% (₹1.5L-₹1.75L) - High-conviction primary position
BAJAJFINSV or TITAN: 25-30% (₹1.25L-₹1.5L) - Secondary, correlation control
LT or UPL: 20-25% (₹1L-₹1.25L) - Cyclical/thematic exposure
Cash: Remainder until full regime ON
```

### For ₹10L Portfolio (4-5 positions):
```
ASIANPAINT: 30-35% (₹3L-₹3.5L) - High-conviction primary
BAJAJFINSV or TITAN: 25-30% (₹2.5L-₹3L) - Secondary diversification
LT or UPL: 20-25% (₹2L-₹2.5L) - Cyclical exposure
TATAMOTORS: 10-15% (₹1L-₹1.5L) - Contrarian diversification
Cash: Remainder until regime confirmation
```

*Sector Caps: Consumer 40%, Cyclicals 30%, Financials 20%, Chemicals 10%*

## Risk Management Framework
- **Max 40% in any sector** (Consumer: 40%, Cyclical: 30%, Financial: 20%, Chemicals: 10%)
- **Trailing stops active** on all positions
- **Regime monitoring** via Telegram alerts
- **Weekly rebalancing** based on relative strength changes

## Next Steps
1. **Monitor regime** - Wait for ADX >20 or RSI >50 (Telegram alerts active)
2. **Execute entry checklists** when regime flips ON
3. **Scale positions** based on conviction and correlation
4. **Track correlations** quarterly for rebalancing
5. **Watch catalysts**: Budget (likely Feb 1, 2026), RBI MPC (Feb 4-6, 2026)

## Next Steps
1. **Monitor regime** - Wait for ADX >20 or RSI >50 (Telegram alerts active)
2. **Execute entry checklists** when regime flips ON
3. **Scale positions** based on conviction and correlation
4. **Track correlations** quarterly for rebalancing
5. **Watch catalysts**: Budget (likely Feb 1, 2026), RBI MPC (Feb 4-6, 2026)

*Post-holiday trading (Dec 30 onward) may bring volume resurgence and regime signals.*

---

# 🔧 Maintenance & Reliability Guide

## Expected Uptime: 95%+ (with proper maintenance)

### Daily Maintenance (5-10 minutes)
```bash
# 1. Refresh Upstox token (required daily)
# Visit Upstox developer portal and update UPSTOX_ACCESS_TOKEN in .env

# 2. Run data pipeline during market hours
python -m src.cli orchestrate-eod

# 3. Check system status
python -c "from src.regime_detector import check_regime; print(check_regime())"

# 4. Monitor logs
type logs/gtt_scheduler.log | tail -20
```

### Weekly Maintenance (15 minutes)
```powershell
# Check Task Scheduler
schtasks /query /tn "SWING_BOT_GTT_Monitor"

# Test Telegram alerts
python -c "
import os
from src.notifications_router import send_telegram_message
send_telegram_message('SWING_BOT: Weekly maintenance check - ' + str(datetime.now()))
"

# Verify data freshness
python -c "
import os
from datetime import datetime
file_time = datetime.fromtimestamp(os.path.getmtime('outputs/screener/screener_latest.csv'))
hours_old = (datetime.now() - file_time).total_seconds() / 3600
print(f'Data age: {hours_old:.1f} hours')
"
```

### Monthly Maintenance (30 minutes)
```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Full system test
python -m src.cli orchestrate-eod
python -c "from src.live_screener import run_screener; run_screener()"

# Backup configuration
copy .env .env.backup
copy config.yaml config.yaml.backup
```

## Critical Dependencies & Failure Points

### 🔴 Critical (System Down)
- **Upstox API Token**: Expires daily - Must refresh
- **Internet Connectivity**: Required for data/API calls
- **Windows Task Scheduler**: Core automation fails if disabled

### 🟡 Important (Degraded Performance)
- **Data Staleness**: >24h old data affects signals
- **Telegram Alerts**: Notifications fail silently
- **Python Dependencies**: Outdated packages may break

### 🟢 Optional (Nice-to-have)
- **Teams Webhook**: Backup notification channel
- **Advanced Logging**: Detailed error tracking

## Troubleshooting Quick Reference

### "No signals generated"
- Check regime status: ADX should be >20, RSI >50
- Verify data freshness: Run orchestrate-eod
- Check logs for errors

### "Telegram alerts not working"
```bash
# Test Telegram connection
python -c "
import os
from src.notifications_router import send_telegram_message
send_telegram_message('Test message')
"
```

### "Task Scheduler not running"
```powershell
# Check and restart
schtasks /query /tn "SWING_BOT_GTT_Monitor"
schtasks /run /tn "SWING_BOT_GTT_Monitor"
```

### "Import errors"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## Reliability Metrics to Monitor

- **Data Freshness**: <24 hours during market days
- **Token Validity**: Refresh daily before market open
- **Alert Delivery**: Test weekly
- **Pipeline Success**: 99%+ success rate (Upstox API dependent)
- **Regime Detection**: Accurate OFF/ON states

## Backup & Recovery

### Configuration Backup
```bash
# Weekly backup
copy .env .env.$(date +%Y%m%d)
copy config.yaml config.yaml.$(date +%Y%m%d)
```

### Data Recovery
- Historical data: Re-run orchestrate-eod for specific dates
- Lost signals: Check outputs/ directory for backups
- Token issues: Re-authorize via Upstox developer portal

## Support & Escalation

1. **Self-service**: Check logs, run diagnostics above
2. **Quick fixes**: Token refresh, dependency updates
3. **System rebuild**: Full reinstall if corrupted
4. **External help**: Upstox API issues, Windows system problems

*With daily token refresh and weekly checks, expect 95%+ uptime and reliable signal generation.*