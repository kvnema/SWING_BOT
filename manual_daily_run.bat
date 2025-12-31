@echo off
REM SWING_BOT Manual Daily Runner
REM Run this script daily at 4:10 PM for autonomous operation

echo ========================================
echo SWING_BOT Manual Daily Execution
echo %DATE% %TIME%
echo ========================================

cd /d "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the complete EOD pipeline
echo Running SWING_BOT EOD Orchestration...
python -m src.cli orchestrate-live --data-out outputs/daily_data.csv --place-gtt --confidence-threshold 0.20 --top 10

REM Check result
if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo ✅ SWING_BOT completed successfully!
    echo ========================================
    echo Next run: Tomorrow at 4:10 PM
    echo Dashboard: outputs/dashboard_today.html
) else (
    echo ========================================
    echo ❌ SWING_BOT failed! Check logs.
    echo ========================================
)

echo Finished at %DATE% %TIME%
pause