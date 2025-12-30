# SWING_BOT Hourly Update Runner
# Executed by Windows Task Scheduler during market hours

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ScriptDir "logs\hourly_update_$(Get-Date -Format 'yyyyMMdd_HHmm').log"

# Ensure log directory exists
$logDir = Split-Path $LogFile -Parent
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage
}

Write-Log "Starting SWING_BOT Hourly Update" "INFO"

try {
    # Change to script directory
    Push-Location $ScriptDir
    Write-Log "Changed to directory: $ScriptDir"

    # Check if it's a market day (Monday-Friday)
    $dayOfWeek = (Get-Date).DayOfWeek
    if ($dayOfWeek -eq 'Saturday' -or $dayOfWeek -eq 'Sunday') {
        Write-Log "Skipping execution - weekend day: $dayOfWeek" "INFO"
        exit 0
    }

    # Check market hours (9:15 AM to 3:15 PM IST)
    $currentTime = Get-Date
    $marketStart = Get-Date "09:15"
    $marketEnd = Get-Date "15:15"

    if ($currentTime -lt $marketStart -or $currentTime -gt $marketEnd) {
        Write-Log "Skipping execution - outside market hours: $($currentTime.ToString('HH:mm'))" "INFO"
        exit 0
    }

    Write-Log "Market hours check passed - proceeding with update" "INFO"

    # Activate virtual environment
    $venvPath = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
    if (Test-Path $venvPath) {
        Write-Log "Activating virtual environment"
        & $venvPath
    } else {
        Write-Log "Virtual environment not found at $venvPath" "WARNING"
    }

    # Run the hourly update command
    $command = "python -m src.cli hourly-update --data-path data/nifty50_data_today.csv --output-dir outputs/hourly --top 25 --notify-email --notify-telegram"
    Write-Log "Executing command: $command"

    $process = Start-Process -FilePath "python" -ArgumentList "-m src.cli hourly-update --data-path data/nifty50_data_today.csv --output-dir outputs/hourly --top 25 --notify-email --notify-telegram" -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$LogFile.stdout" -RedirectStandardError "$LogFile.stderr"

    # Check exit code
    if ($process.ExitCode -eq 0) {
        Write-Log "SWING_BOT Hourly Update completed successfully" "SUCCESS"
    } else {
        Write-Log "SWING_BOT Hourly Update failed with exit code: $($process.ExitCode)" "ERROR"

        # Include error output in log
        if (Test-Path "$LogFile.stderr") {
            $errorContent = Get-Content "$LogFile.stderr" -Raw
            if ($errorContent) {
                Write-Log "Error output: $errorContent" "ERROR"
            }
        }
    }

} catch {
    Write-Log "Exception during hourly update: $($_.Exception.Message)" "ERROR"
    Write-Log "Stack trace: $($_.Exception.StackTrace)" "ERROR"
} finally {
    # Cleanup temp files
    if (Test-Path "$LogFile.stdout") { Remove-Item "$LogFile.stdout" -ErrorAction SilentlyContinue }
    if (Test-Path "$LogFile.stderr") { Remove-Item "$LogFile.stderr" -ErrorAction SilentlyContinue }

    Pop-Location
}

Write-Log "SWING_BOT Hourly Update script finished" "INFO"