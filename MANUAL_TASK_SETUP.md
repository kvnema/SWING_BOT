# 🎯 MANUAL TASK SCHEDULER SETUP (Administrator Required)

Since automated setup requires admin privileges, here's how to set it up manually:

## Step 1: Open Task Scheduler as Administrator
1. Press `Win + R`, type `taskschd.msc`, press Enter
2. Right-click "Task Scheduler" → "Run as administrator"

## Step 2: Create New Task
1. Click "Action" → "Create Task..."
2. **General Tab:**
   - Name: `SWING_BOT_EOD`
   - Description: `SWING_BOT EOD Orchestration - Automated trading system`
   - ✅ "Run with highest privileges"
   - ✅ "Run only when user is logged on" (or configure service account)

## Step 3: Triggers Tab
1. Click "New..."
2. **Settings:**
   - Begin the task: "On a schedule"
   - Weekly: ✅ Monday, Tuesday, Wednesday, Thursday, Friday
   - Start: 4:10:00 PM
   - ✅ "Enabled"

## Step 4: Actions Tab
1. Click "New..."
2. **Settings:**
   - Action: "Start a program"
   - Program/script: `C:\Users\K01340\SWING_BOT_GIT\SWING_BOT\daily_run.bat`
   - Start in: `C:\Users\K01340\SWING_BOT_GIT\SWING_BOT`

## Step 5: Conditions Tab
1. ✅ "Start the task only if the computer is on AC power"
2. ✅ "Start only if the following network connection is available: Any connection"

## Step 6: Settings Tab
1. ✅ "Allow task to be run on demand"
2. ✅ "Run task as soon as possible after a scheduled start is missed"
3. ✅ "If the task fails, restart every: 5 minutes" (up to 3 times)
4. ⏰ "Stop the task if it runs longer than: 30 minutes"
5. ✅ "If the running task does not end when requested, force it to stop"

## Step 7: Verify
1. Click "OK" to save
2. Find "SWING_BOT_EOD" in the task list
3. Right-click → "Run" to test immediately
4. Check "Next Run Time" shows tomorrow at 4:10 PM

## Alternative: Command Line Setup
If you prefer command line (run PowerShell as Administrator):

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT\daily_run.bat" -WorkingDirectory "C:\Users\K01340\SWING_BOT_GIT\SWING_BOT"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "16:10"
Register-ScheduledTask -TaskName "SWING_BOT_EOD" -Action $action -Trigger $trigger -Description "SWING_BOT EOD Orchestration"
```

## Testing the Setup
After creating the task:
1. Right-click the task → "Run"
2. Check the logs in `outputs/logs/`
3. Verify dashboard updates: `outputs/dashboard_today.html`
4. Confirm Telegram notifications arrive

---
**Once set up, SWING_BOT will run autonomously every trading day at 4:10 PM!** 🚀