# SWING_BOT Token Auto-Refresh PowerShell Script
# Use this with Windows Task Scheduler for automated token management

param(
    [switch]$Once,
    [switch]$Status,
    [int]$IntervalHours = 6
)

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "python"
$TokenScript = Join-Path $ScriptPath "src\token_scheduler.py"

Write-Host "SWING_BOT Token Auto-Refresh" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

if ($Status) {
    & $PythonExe $TokenScript --status
} elseif ($Once) {
    Write-Host "Running one-time token check..."
    & $PythonExe $TokenScript --once
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Token check completed successfully" -ForegroundColor Green
    } else {
        Write-Host "Token check failed - manual intervention may be required" -ForegroundColor Red
    }
} else {
    Write-Host "Starting continuous token monitoring (Ctrl+C to stop)..."
    try {
        & $PythonExe $TokenScript --interval $IntervalHours
    } catch {
        Write-Host "Scheduler stopped." -ForegroundColor Yellow
    }
}