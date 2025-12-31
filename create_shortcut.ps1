# SWING_BOT Daily Reminder Setup (No Admin Required)

# Create a desktop shortcut for easy daily execution
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\SWING_BOT_Daily_Run.lnk")
$Shortcut.TargetPath = "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT\manual_daily_run.bat"
$Shortcut.WorkingDirectory = "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT"
$Shortcut.Description = "SWING_BOT Daily Trading Execution - Run at 4:10 PM"
$Shortcut.Save()

Write-Host "✅ Desktop shortcut created: SWING_BOT_Daily_Run.lnk"
Write-Host "Double-click this shortcut daily at 4:10 PM to run SWING_BOT"