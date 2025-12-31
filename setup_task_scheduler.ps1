# SWING_BOT Task Scheduler Setup
# Run this script as Administrator to set up daily automation

$taskName = "SWING_BOT_EOD"
$scriptPath = "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT\daily_run.bat"

Write-Host "Setting up SWING_BOT Daily Automation..." -ForegroundColor Green
Write-Host "Task Name: $taskName" -ForegroundColor Yellow
Write-Host "Script Path: $scriptPath" -ForegroundColor Yellow

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Task '$taskName' already exists. Removing old task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Create new scheduled task
Write-Host "Creating new scheduled task..." -ForegroundColor Green

$action = New-ScheduledTaskAction -Execute $scriptPath -WorkingDirectory "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT"

# Schedule: Monday-Friday at 4:10 PM
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "16:10"

# Run with highest privileges
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

# Settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

# Register the task
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "SWING_BOT EOD Orchestration - Automated trading system execution at market close"

# Verify the task was created
$task = Get-ScheduledTask -TaskName $taskName
if ($task) {
    Write-Host "✅ Task '$taskName' created successfully!" -ForegroundColor Green
    Write-Host "Next run: $($task.NextRunTime)" -ForegroundColor Cyan
    Write-Host "Status: $($task.State)" -ForegroundColor Cyan
} else {
    Write-Host "❌ Failed to create task!" -ForegroundColor Red
}

Write-Host "`nTo manually run the task:" -ForegroundColor White
Write-Host "Start-ScheduledTask -TaskName '$taskName'" -ForegroundColor Gray

Write-Host "`nTo view task details:" -ForegroundColor White
Write-Host "Get-ScheduledTask -TaskName '$taskName' | Select-Object *" -ForegroundColor Gray

Write-Host "`n🎯 SWING_BOT is now scheduled for daily autonomous operation!" -ForegroundColor Green