# SWING_BOT GTT Monitor Setup Script
# This script sets up Windows Task Scheduler for automated GTT monitoring

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Test,
    [switch]$Continuous
)

$TaskName = "SWING_BOT_GTT_Monitor"
$TaskXmlPath = "$PSScriptRoot\SWING_BOT_GTT_Task.xml"
$BatchScriptPath = "$PSScriptRoot\run_gtt_monitor.bat"
$PythonScriptPath = "$PSScriptRoot\local_gtt_scheduler.py"

function Test-Environment {
    Write-Host "Testing environment setup..." -ForegroundColor Cyan

    # Check if .env file exists
    if (!(Test-Path ".env")) {
        Write-Warning "Warning: .env file not found. Please copy .env.example to .env and configure your credentials."
        return $false
    }

    # Check virtual environment
    if (!(Test-Path ".venv")) {
        Write-Error "Error: Virtual environment not found. Run: python -m venv .venv"
        return $false
    }

    # Check required environment variables
    $requiredVars = @('UPSTOX_ACCESS_TOKEN', 'UPSTOX_API_KEY', 'UPSTOX_API_SECRET')
    $envContent = Get-Content ".env" -ErrorAction SilentlyContinue
    $missingVars = @()

    foreach ($var in $requiredVars) {
        $found = $false
        foreach ($line in $envContent) {
            if ($line -match "^$var=") {
                $found = $true
                break
            }
        }
        if (!$found) {
            $missingVars += $var
        }
    }

    if ($missingVars.Count -gt 0) {
        Write-Error "Missing required environment variables in .env: $($missingVars -join ', ')"
        return $false
    }

    Write-Host "Environment setup looks good!" -ForegroundColor Green
    return $true
}

function Install-Task {
    Write-Host "Installing Windows Task Scheduler task..." -ForegroundColor Cyan

    if (!(Test-Path $TaskXmlPath)) {
        Write-Error "Task XML file not found: $TaskXmlPath"
        return $false
    }

    if (!(Test-Path $BatchScriptPath)) {
        Write-Error "Batch script not found: $BatchScriptPath"
        return $false
    }

    try {
        # Delete existing task if it exists
        schtasks /delete /tn $TaskName /f 2>$null

        # Import the new task
        schtasks /create /tn $TaskName /xml $TaskXmlPath

        Write-Host "Task Scheduler task installed successfully!" -ForegroundColor Green
        Write-Host "Task Details:" -ForegroundColor Yellow
        Write-Host "   Name: $TaskName"
        Write-Host "   Schedule: 8:15 AM, 9:15 AM - 3:15 PM (hourly), 4:30 PM"
        Write-Host "   Working Directory: $PSScriptRoot"
        return $true
    }
    catch {
        Write-Error "Failed to install task: $($_.Exception.Message)"
        return $false
    }
}

function Uninstall-Task {
    Write-Host "Uninstalling Windows Task Scheduler task..." -ForegroundColor Cyan

    try {
        schtasks /delete /tn $TaskName /f
        Write-Host "Task Scheduler task uninstalled successfully!" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Error "Failed to uninstall task: $($_.Exception.Message)"
        return $false
    }
}

function Test-Run {
    Write-Host "Running test execution..." -ForegroundColor Cyan

    try {
        & $BatchScriptPath
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Host "Test run completed successfully!" -ForegroundColor Green
            return $true
        } else {
            Write-Error "Test run failed with exit code: $exitCode"
            return $false
        }
    }
    catch {
        Write-Error "Test run exception: $($_.Exception.Message)"
        return $false
    }
}

function Start-Continuous {
    Write-Host "Starting continuous GTT scheduler..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow

    try {
        & python $PythonScriptPath
    }
    catch {
        Write-Error "Continuous scheduler failed: $($_.Exception.Message)"
    }
}

# Main execution
if ($Install) {
    if (Test-Environment) {
        Install-Task
    }
}
elseif ($Uninstall) {
    Uninstall-Task
}
elseif ($Test) {
    if (Test-Environment) {
        Test-Run
    }
}
elseif ($Continuous) {
    if (Test-Environment) {
        Start-Continuous
    }
}
else {
    Write-Host "SWING_BOT GTT Monitor Setup Script" -ForegroundColor Cyan
    Write-Host "==================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\setup_gtt_scheduler.ps1 -Install     # Install Task Scheduler task"
    Write-Host "  .\setup_gtt_scheduler.ps1 -Uninstall   # Remove Task Scheduler task"
    Write-Host "  .\setup_gtt_scheduler.ps1 -Test        # Test run the monitor"
    Write-Host "  .\setup_gtt_scheduler.ps1 -Continuous  # Run continuous scheduler"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\setup_gtt_scheduler.ps1 -Test"
    Write-Host "  .\setup_gtt_scheduler.ps1 -Install"
    Write-Host ""
    Write-Host "Make sure to configure your .env file first!" -ForegroundColor Yellow
}