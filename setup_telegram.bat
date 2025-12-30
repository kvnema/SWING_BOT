@echo off
REM SWING_BOT Telegram Environment Setup

echo 🚀 Setting up SWING_BOT Telegram Environment Variables
echo.

REM Set the bot token and chat ID
set TELEGRAM_BOT_TOKEN=8486307857:AAHt4XXRokWf_Uv49NIVozp3lj1W-seqMg4
set TELEGRAM_CHAT_ID=%1

if "%TELEGRAM_CHAT_ID%"=="" (
    echo ❌ Usage: setup_telegram.bat YOUR_CHAT_ID
    echo.
    echo Example: setup_telegram.bat -123456789
    echo.
    echo To get your chat ID:
    echo 1. Start conversation with @swingkopal_bot
    echo 2. Send any message
    echo 3. Visit: https://api.telegram.org/bot8486307857:AAHt4XXRokpWf_Uv49NIVozp3lj1W-seqMg4/getUpdates
    echo 4. Find "chat":{"id":YOUR_ID,...}
    exit /b 1
)

echo ✅ Environment variables set:
echo TELEGRAM_BOT_TOKEN=%TELEGRAM_BOT_TOKEN%
echo TELEGRAM_CHAT_ID=%TELEGRAM_CHAT_ID%
echo.

echo 🧪 Testing connection...
python scripts\test_telegram.py %TELEGRAM_BOT_TOKEN% %TELEGRAM_CHAT_ID%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo 🎉 Setup complete! Telegram reports are active.
    echo Daily reports will be sent ~16:45 IST on weekdays.
) else (
    echo.
    echo ❌ Setup failed. Please check your chat ID.
)

echo.
pause