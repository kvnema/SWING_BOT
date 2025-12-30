@echo off
REM SWING_BOT GTT Monitor Batch Script
REM This script is called by Windows Task Scheduler

echo [%DATE% %TIME%] Starting SWING_BOT GTT Monitor...

REM Change to the script directory
cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)

REM Load environment variables from .env file if it exists
if exist .env (
    for /f "tokens=*" %%a in (.env) do (
        set %%a
    )
)

REM Run the GTT monitor
python scheduled_gtt_monitor.py
set EXITCODE=%errorlevel%

REM Deactivate virtual environment
call .venv\Scripts\deactivate.bat

echo [%DATE% %TIME%] SWING_BOT GTT Monitor completed with exit code %EXITCODE%

REM Exit with the same code as the Python script
exit /b %EXITCODE%