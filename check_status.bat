@echo off
REM SWING_BOT Status Checker
REM Quick check if SWING_BOT ran today

echo ========================================
echo SWING_BOT Daily Status Check
echo %DATE% %TIME%
echo ========================================

cd /d "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT"

REM Check if dashboard was updated today
if exist "outputs\dashboard_today.html" (
    echo ✅ Dashboard exists
    for %%A in ("outputs\dashboard_today.html") do echo Last modified: %%~tA
) else (
    echo ❌ Dashboard missing - SWING_BOT may not have run today
)

REM Check for recent log files
echo.
echo Recent log files:
dir /b /o-d outputs\logs\*.log 2>nul | findstr /r ".*"

REM Check live positions
echo.
echo Live positions check:
python -c "
try:
    from src.live_trade_tracker import get_live_positions
    positions = get_live_positions()
    print(f'📊 Current live positions: {len(positions)}')
    if positions:
        for pos in positions:
            print(f'  • {pos.get(\"symbol\", \"Unknown\")}: ₹{pos.get(\"current_price\", 0):.2f}')
except Exception as e:
    print(f'❌ Error checking positions: {e}')
"

echo.
echo ========================================
echo 💡 If SWING_BOT hasn't run today:
echo    Run: manual_daily_run.bat
echo ========================================

pause