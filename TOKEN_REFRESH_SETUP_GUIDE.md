# 🚀 SWING_BOT Automated Token Refresh Setup Guide

## Overview

This guide covers setting up automated Upstox API token refresh to prevent authentication failures and ensure continuous access to real market data.

## ✅ What's Been Implemented

### Core Components
- **`src/token_manager.py`** - Complete token lifecycle management
- **`src/token_scheduler.py`** - Automated monitoring system
- **`token_refresh.bat`** - Windows batch script for Task Scheduler
- **`token_refresh.ps1`** - PowerShell script for Task Scheduler
- **CLI Integration** - Token checking built into `orchestrate-eod` command

### Features
- ✅ JWT token validation and expiration detection
- ✅ OAuth2 refresh flow implementation
- ✅ Automated token refresh scheduling
- ✅ Comprehensive logging and error handling
- ✅ CLI integration with graceful failure handling
- ✅ Multiple deployment options (manual, scheduled, CLI)

## 🔧 Setup Instructions

### 1. Environment Variables

Ensure your `.env` file contains:
```bash
UPSTOX_API_KEY=your_api_key_here
UPSTOX_API_SECRET=your_api_secret_here
UPSTOX_ACCESS_TOKEN=your_current_access_token_here
```

### 2. Test Token Manager

```bash
# Check token status
python src/token_manager.py --info

# Force refresh token
python src/token_manager.py --refresh

# Auto-check and refresh if needed
python src/token_manager.py --auto
```

### 3. Test CLI Integration

```bash
# This will now check token validity before proceeding
python -m src.cli orchestrate-eod --data-out data/test.parquet --required-days 30 --top 3
```

## ⏰ Automated Scheduling Setup

### Option A: Windows Task Scheduler (Recommended)

#### Using PowerShell Script (Preferred)
1. Open Windows Task Scheduler
2. Create new task:
   - **Name**: `SWING_BOT_Token_Refresh`
   - **Trigger**: Daily at 6:00 AM, 12:00 PM, 6:00 PM, 12:00 AM
   - **Action**: Start a program
     - **Program**: `powershell.exe`
     - **Arguments**: `-ExecutionPolicy Bypass -File "C:\path\to\SWING_BOT\token_refresh.ps1"`
     - **Start in**: `C:\path\to\SWING_BOT`

#### Using Batch Script
1. Open Windows Task Scheduler
2. Create new task:
   - **Name**: `SWING_BOT_Token_Refresh_Batch`
   - **Trigger**: Daily every 6 hours
   - **Action**: Start a program
     - **Program**: `C:\path\to\SWING_BOT\token_refresh.bat`
     - **Start in**: `C:\path\to\SWING_BOT`

### Option B: Manual Monitoring

Run token checks manually or integrate into your existing automation:

```bash
# Check daily
python src/token_manager.py --auto

# Or integrate into existing scripts
python -c "from src.token_manager import UpstoxTokenManager; UpstoxTokenManager().check_and_refresh_token()"
```

## 📊 Monitoring & Logs

### Log Files
- `logs/token_manager.log` - Token operations and refresh attempts
- `logs/token_scheduler.log` - Scheduled monitoring activities

### Status Checking
```bash
# Quick status
python src/token_manager.py --info

# Detailed validation
python src/token_manager.py --check
```

### Data Files
- `data/token_status.json` - Current token status and metadata

## 🔍 Troubleshooting

### Common Issues

#### 1. Token Refresh Failures
**Symptoms**: Token refresh fails with OAuth2 errors
**Solution**:
- Check API credentials in `.env`
- Verify Upstox app permissions
- Check internet connectivity
- Review `logs/token_manager.log` for details

#### 2. CLI Integration Issues
**Symptoms**: Orchestration fails with token errors
**Solution**:
- Run `python src/token_manager.py --auto` manually
- Check token validity with `--info` flag
- Verify `.env` file is accessible

#### 3. Scheduling Issues
**Symptoms**: Automated refresh not working
**Solution**:
- Verify Task Scheduler task is enabled
- Check execution permissions
- Review Windows Event Viewer for errors
- Test scripts manually first

#### 4. Unicode/Encoding Errors
**Symptoms**: Emoji characters cause encoding issues
**Solution**:
- Use PowerShell instead of CMD for execution
- Set console encoding: `chcp 65001` in batch files
- Or modify logging to avoid Unicode characters

### Emergency Token Refresh

If automated refresh fails:

```bash
# Force manual refresh
python src/token_manager.py --refresh

# Or get new token via OAuth flow
python -c "from src.token_manager import UpstoxTokenManager; tm = UpstoxTokenManager(); tm.get_new_token()"
```

## 📈 Performance & Reliability

### Refresh Timing
- Checks every 6 hours (configurable in scheduler)
- Refreshes when < 24 hours remaining
- Graceful handling of refresh failures

### Error Handling
- Comprehensive logging for all operations
- Graceful degradation (continues with existing token if refresh fails)
- CLI integration prevents orchestration failures

### Security
- Tokens stored securely in `.env` file
- No sensitive data in logs
- OAuth2 flow handles credential security

## 🎯 Next Steps

1. **Configure Task Scheduler** using the PowerShell script
2. **Test the integration** with your regular SWING_BOT runs
3. **Monitor logs** for any issues
4. **Set up alerts** if needed for critical failures

## 📞 Support

If you encounter issues:
1. Check the logs in `logs/token_manager.log`
2. Run diagnostic commands above
3. Verify your Upstox API credentials
4. Ensure proper file permissions

The system is designed to be robust and self-healing, automatically handling token lifecycle management.