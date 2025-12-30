# SWING_BOT Hourly Update Scheduler
# Runs SWING_BOT hourly updates during market hours (9:15 AM - 3:15 PM IST)
# Sends results via Excel, Email, and Telegram

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Test,
    [switch]$Status
)

$ScriptPath = $PSScriptRoot
$PythonExe = "python"
$CliCommand = "python -m src.cli hourly-update --data-path data/nifty50_data_today.csv --output-dir outputs/hourly --top 25 --notify-email --notify-telegram"

# Market hours: 9:15 AM to 3:15 PM IST (Monday to Friday)
$MarketStart = "09:15"
$MarketEnd = "15:15"
$TaskName = "SWING_BOT_Hourly_Update"

function Test-Command {
    Write-Host "🧪 Testing SWING_BOT Hourly Update..." -ForegroundColor Cyan

    try {
        # Change to script directory
        Push-Location $ScriptPath

        # Activate virtual environment if it exists
        if (Test-Path ".venv\Scripts\Activate.ps1") {
            & ".venv\Scripts\Activate.ps1"
        }

        # Run test command
        Write-Host "Running: $CliCommand" -ForegroundColor Yellow
        Invoke-Expression $CliCommand

        Write-Host "✅ Test completed successfully!" -ForegroundColor Green

    } catch {
        Write-Host "❌ Test failed: $($_.Exception.Message)" -ForegroundColor Red
    } finally {
        Pop-Location
    }
}

function Install-ScheduledTask {
    Write-Host "📅 Installing SWING_BOT Hourly Update scheduled task..." -ForegroundColor Cyan

    # Check if task already exists
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-Host "⚠️  Task '$TaskName' already exists. Removing old task..." -ForegroundColor Yellow
        Uninstall-ScheduledTask
    }

    # Create the scheduled task
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$PSScriptRoot\run_hourly_update.ps1`""

    # Create trigger for market hours (every hour from 9:15 to 15:15, Mon-Fri)
    $triggers = @()
    $hours = 9, 10, 11, 12, 13, 14, 15

    foreach ($hour in $hours) {
        $time = "{0:D2}:15" -f $hour
        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $time
        $triggers += $trigger
    }

    # Special case for 3:15 PM (market close)
    $closeTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $MarketEnd
    $triggers += $closeTrigger

    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

    $task = New-ScheduledTask -Action $action -Trigger $triggers -Settings $settings -Principal $principal -Description "SWING_BOT Hourly Update - Runs during market hours to generate trading signals"

    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force

    Write-Host "✅ Scheduled task '$TaskName' installed successfully!" -ForegroundColor Green
    Write-Host "📅 Schedule: Monday-Friday, $MarketStart to $MarketEnd (hourly)" -ForegroundColor Cyan
}

function Uninstall-ScheduledTask {
    Write-Host "🗑️  Uninstalling SWING_BOT Hourly Update scheduled task..." -ForegroundColor Cyan

    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "✅ Scheduled task '$TaskName' uninstalled successfully!" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Task '$TaskName' not found or already removed." -ForegroundColor Yellow
    }
}

function Get-TaskStatus {
    Write-Host "📊 Checking SWING_BOT Hourly Update task status..." -ForegroundColor Cyan

    try {
        $task = Get-ScheduledTask -TaskName $TaskName
        Write-Host "✅ Task '$TaskName' is installed" -ForegroundColor Green
        Write-Host "📅 State: $($task.State)" -ForegroundColor Cyan
        Write-Host "⏰ Next Run: $($task.NextRunTime)" -ForegroundColor Cyan

        # Show recent task history
        Write-Host "`n📋 Recent Task History:" -ForegroundColor Cyan
        $history = Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-TaskScheduler/Operational'; ProviderName='Microsoft-Windows-TaskScheduler'; ID=201,202} -MaxEvents 5 -ErrorAction SilentlyContinue |
            Where-Object { $_.Message -like "*$TaskName*" } |
            Select-Object TimeCreated, Message

        if ($history) {
            $history | ForEach-Object {
                Write-Host "  $($_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss')) - $($_.Message)" -ForegroundColor Gray
            }
        } else {
            Write-Host "  No recent history found" -ForegroundColor Gray
        }

    } catch {
        Write-Host "❌ Task '$TaskName' is not installed" -ForegroundColor Red
        Write-Host "💡 Run with -Install parameter to create the scheduled task" -ForegroundColor Yellow
    }
}

# Main logic
if ($Test) {
    Test-Command
} elseif ($Install) {
    Install-ScheduledTask
} elseif ($Uninstall) {
    Uninstall-ScheduledTask
} elseif ($Status) {
    Get-TaskStatus
} else {
    Write-Host "SWING_BOT Hourly Update Scheduler" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\schedule_hourly.ps1 -Install     # Install scheduled task"
    Write-Host "  .\schedule_hourly.ps1 -Uninstall   # Remove scheduled task"
    Write-Host "  .\schedule_hourly.ps1 -Test        # Test the update command"
    Write-Host "  .\schedule_hourly.ps1 -Status      # Check task status"
    Write-Host ""
    Write-Host "The task runs hourly during market hours (9:15 AM - 3:15 PM IST, Mon-Fri)" -ForegroundColor Gray
    Write-Host "Results are sent via Excel, Email, and Telegram notifications." -ForegroundColor Gray
}