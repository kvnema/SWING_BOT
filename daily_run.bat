@echo off
REM SWING_BOT Daily Automation Script
REM Runs the complete EOD pipeline: fetch → test → optimize → execute

echo ========================================
echo SWING_BOT Daily Automation
echo %DATE% %TIME%
echo ========================================

cd /d "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Step 1: Run full EOD orchestration with live order placement
echo Step 1: Running EOD Orchestration with Live Orders...
python -m src.cli orchestrate-live --data-out outputs/daily_data.csv --place-gtt --confidence-threshold 0.42 --top 10

REM Check if orchestration succeeded
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: EOD Orchestration failed!
    goto :error
)

REM Step 2: Run self-optimization for next day
echo Step 2: Running Self-Optimization...
python -m src.cli self-optimize

REM Step 3: Send success notification
echo Step 3: Sending Success Notification...
python scripts/post_teams_success.py

echo ========================================
echo SWING_BOT Daily Run Complete!
echo ========================================

goto :end

:error
echo ========================================
echo SWING_BOT Daily Run FAILED!
echo ========================================
python scripts/post_teams_failure.py

:end
echo Finished at %DATE% %TIME%