param(
    [Parameter(Mandatory=$true)]
    [string]$ChatId
)

Write-Host "🚀 Setting up SWING_BOT Telegram Environment Variables" -ForegroundColor Green
Write-Host ""

# Set environment variables
$env:TELEGRAM_BOT_TOKEN = "8486307857:AAHt4XXRokWf_Uv49NIVozp3lj1W-seqMg4"
$env:TELEGRAM_CHAT_ID = $ChatId

Write-Host "✅ Environment variables set:" -ForegroundColor Green
Write-Host "TELEGRAM_BOT_TOKEN=$env:TELEGRAM_BOT_TOKEN"
Write-Host "TELEGRAM_CHAT_ID=$env:TELEGRAM_CHAT_ID"
Write-Host ""

Write-Host "🧪 Testing connection..." -ForegroundColor Yellow
& python scripts\test_telegram.py $env:TELEGRAM_BOT_TOKEN $env:TELEGRAM_CHAT_ID

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 Setup complete! Telegram reports are active." -ForegroundColor Green
    Write-Host "Daily reports will be sent ~16:45 IST on weekdays."
    Write-Host "Instant alerts will be sent for parameter changes and errors."
} else {
    Write-Host ""
    Write-Host "❌ Setup failed. Please check your chat ID." -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to continue"