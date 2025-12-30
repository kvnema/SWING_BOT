# SWING_BOT Token Auto-Refresh Setup Guide
# ======================================

## Overview
SWING_BOT now includes automatic Upstox API token management to prevent authentication failures.

## Features
- ✅ Automatic token expiration detection
- ✅ Interactive token refresh process
- ✅ Integration with main orchestration
- ✅ Scheduled monitoring options
- ✅ Windows Task Scheduler support

## Quick Start

### 1. Check Current Token Status
```bash
python src/token_manager.py --info
```

### 2. Manual Token Refresh (when expired)
```bash
python src/token_manager.py --refresh
```

### 3. Automated Orchestration (includes token check)
```bash
python -m src.cli orchestrate-eod --data-out data/indicators.parquet --required-days 30 --top 5
```

## Setup Options

### Option 1: Windows Task Scheduler (Recommended)
1. Open Task Scheduler
2. Create new task:
   - Name: "SWING_BOT Token Refresh"
   - Trigger: Daily at 6 AM
   - Action: Start a program
   - Program: `powershell.exe`
   - Arguments: `-ExecutionPolicy Bypass -File "C:\path\to\SWING_BOT\token_refresh.ps1" -Once`

### Option 2: Batch File (Simple)
1. Create scheduled task to run `token_refresh.bat` every 6 hours

### Option 3: Continuous Monitoring
```bash
# Run in background (stops with Ctrl+C)
python src/token_scheduler.py
```

## Token Manager Commands

```bash
# Show token information
python src/token_manager.py --info

# Check if token is valid
python src/token_manager.py --check

# Force token refresh
python src/token_manager.py --refresh

# Show scheduler status
python src/token_scheduler.py --status

# Run one-time check
python src/token_scheduler.py --once
```

## Integration

### Automatic Checks
- **Orchestration**: Token checked before every EOD run
- **CLI Integration**: Automatic refresh prompts when needed
- **Error Handling**: Graceful fallback with clear error messages

### Monitoring Files
- `data/token_status.json` - Current token status
- `data/token_scheduler_status.json` - Scheduler status
- `logs/token_manager.log` - Detailed logs
- `logs/token_scheduler.log` - Scheduler logs

## Token Expiration Handling

### Before Expiration (15+ hours left)
- ✅ Automatic monitoring
- ✅ No action required

### Near Expiration (6 hours left)
- ⚠️  Warning logged
- ✅ Automatic refresh attempted

### After Expiration
- ❌ API calls fail with 401 errors
- 🔄 Manual refresh required
- 📧 Clear instructions provided

## Troubleshooting

### Token Refresh Failed
```bash
# Check logs
tail logs/token_manager.log

# Manual refresh
python src/token_manager.py --refresh
```

### Scheduler Not Working
```bash
# Check scheduler status
python src/token_scheduler.py --status

# Run manual check
python src/token_scheduler.py --once
```

### Permission Issues
- Ensure write access to `.env` file
- Check Python execution permissions
- Verify network connectivity for Upstox API

## Security Notes

- ✅ Tokens stored securely in `.env` file
- ✅ No sensitive data in logs
- ✅ Automatic cleanup of expired tokens
- ✅ Secure OAuth2 flow for refresh

## Production Deployment

For production environments:
1. Set up monitoring alerts for token expiration
2. Configure automated email notifications
3. Use service accounts where possible
4. Implement token rotation policies

---
**Status**: ✅ Token management system active
**Next Check**: Automatic monitoring enabled
**Token Valid**: ✅ Until 2025-12-25 03:30:00