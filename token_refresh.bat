@echo off
REM SWING_BOT Token Auto-Refresh Batch Script
REM Run this script to check and refresh Upstox tokens automatically

echo Starting SWING_BOT Token Refresh Check...
python src/token_scheduler.py --once

if %ERRORLEVEL% EQU 0 (
    echo Token check completed successfully
) else (
    echo Token check failed - manual intervention may be required
)

pause