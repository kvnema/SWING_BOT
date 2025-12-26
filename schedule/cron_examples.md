# SWING_BOT Cron Examples

## Linux/Mac Cron Setup

### Prerequisites
- Python virtual environment activated
- Proper PATH environment for cron
- Working directory permissions
- Log file permissions

### Basic EOD Cron Job (16:10 IST, Mon-Fri)

```bash
# Add to crontab (crontab -e)
10 16 * * 1-5 cd /path/to/SWING_BOT && /usr/bin/python3 -m src.cli orchestrate-eod --data-out data/indicators_500d.parquet --max-age-days 1 --required-days 500 --top 25 --strict true --post-teams true --multi-tf true >> outputs/logs/eod_cron_$(date +\%Y\%m\%d).log 2>&1
```

### Cron Job with Virtual Environment

```bash
# If using virtual environment
10 16 * * 1-5 cd /path/to/SWING_BOT && source .venv/bin/activate && python -m src.cli orchestrate-eod --data-out data/indicators_500d.parquet --max-age-days 1 --required-days 500 --top 25 --strict true --post-teams true --multi-tf true >> outputs/logs/eod_cron_$(date +\%Y\%m\%d).log 2>&1
```

### Cron Job with Error Handling

```bash
# With error notification
10 16 * * 1-5 cd /path/to/SWING_BOT && /usr/bin/python3 -m src.cli orchestrate-eod --data-out data/indicators_500d.parquet --max-age-days 1 --required-days 500 --top 25 --strict true --post-teams true --multi-tf true 2>&1 | tee outputs/logs/eod_cron_$(date +\%Y\%m\%d).log | grep -q "❌" && echo "EOD Failed - check logs" | mail -s "SWING_BOT EOD Alert" admin@company.com
```

## Cron Schedule Reference

### Time Fields
```
* * * * * command
│ │ │ │ │
│ │ │ │ └─ Day of week (0-7, 0 or 7 = Sunday)
│ │ │ └─── Month (1-12)
│ │ └──── Day of month (1-31)
│ └────── Hour (0-23)
└──────── Minute (0-59)
```

### IST Time Zone Setup

For systems not in IST, adjust the cron time accordingly:

```bash
# If system is in UTC, add 5.5 hours (16:10 IST = 10:40 UTC)
40 10 * * 1-5 cd /path/to/SWING_BOT && /usr/bin/python3 -m src.cli orchestrate-eod --data-out data/indicators_500d.parquet --max-age-days 1 --required-days 500 --top 25 --strict true --post-teams true --multi-tf true >> outputs/logs/eod_cron_$(date +\%Y\%m\%d).log 2>&1

# If system is in EST (UTC-5), add 10.5 hours (16:10 IST = 5:40 EST)
40 5 * * 1-5 cd /path/to/SWING_BOT && /usr/bin/python3 -m src.cli orchestrate-eod --data-out data/indicators_500d.parquet --max-age-days 1 --required-days 500 --top 25 --strict true --post-teams true --multi-tf true >> outputs/logs/eod_cron_$(date +\%Y\%m\%d).log 2>&1
```

## Alternative Scheduling Methods

### Systemd Timer (Linux)

Create `/etc/systemd/system/swingbot-eod.service`:
```ini
[Unit]
Description=SWING_BOT EOD Orchestration
After=network.target

[Service]
Type=oneshot
User=swingbot
WorkingDirectory=/path/to/SWING_BOT
ExecStart=/usr/bin/python3 -m src.cli orchestrate-eod --data-out data/indicators_500d.parquet --max-age-days 1 --required-days 500 --top 25 --strict true --post-teams true --multi-tf true
```

Create `/etc/systemd/system/swingbot-eod.timer`:
```ini
[Unit]
Description=Run SWING_BOT EOD daily
Requires=swingbot-eod.service

[Timer]
OnCalendar=Mon..Fri 16:10:00 Asia/Kolkata
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:
```bash
sudo systemctl enable swingbot-eod.timer
sudo systemctl start swingbot-eod.timer
```

### Launchd (macOS)

Create `~/Library/LaunchAgents/com.swingbot.eod.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.swingbot.eod</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>-m</string>
        <string>src.cli</string>
        <string>orchestrate-eod</string>
        <string>--data-out</string>
        <string>data/indicators_500d.parquet</string>
        <string>--max-age-days</string>
        <string>1</string>
        <string>--required-days</string>
        <string>500</string>
        <string>--top</string>
        <string>25</string>
        <string>--strict</string>
        <string>true</string>
        <string>--post-teams</string>
        <string>true</string>
        <string>--multi-tf</string>
        <string>true</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/SWING_BOT</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>16</integer>
        <key>Minute</key>
        <integer>10</integer>
        <key>Weekday</key>
        <integer>1</integer>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>16</integer>
        <key>Minute</key>
        <integer>10</integer>
        <key>Weekday</key>
        <integer>2</integer>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>16</integer>
        <key>Minute</key>
        <integer>10</integer>
        <key>Weekday</key>
        <integer>3</integer>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>16</integer>
        <key>Minute</key>
        <integer>10</integer>
        <key>Weekday</key>
        <integer>4</integer>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>16</integer>
        <key>Minute</key>
        <integer>10</integer>
        <key>Weekday</key>
        <integer>5</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/path/to/SWING_BOT/outputs/logs/eod_launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/SWING_BOT/outputs/logs/eod_launchd_error.log</string>
</dict>
</plist>
```

Load the job:
```bash
launchctl load ~/Library/LaunchAgents/com.swingbot.eod.plist
```

## Testing Cron Jobs

### Manual Testing
```bash
# Test the command manually first
cd /path/to/SWING_BOT
/usr/bin/python3 -m src.cli orchestrate-eod --data-out data/indicators_500d.parquet --max-age-days 1 --required-days 500 --top 25 --strict true --post-teams true --multi-tf true

# Test with time override for testing
/usr/bin/python3 -m src.cli orchestrate-eod --data-out data/indicators_500d.parquet --max-age-days 7 --required-days 500 --top 25 --strict false --post-teams false --multi-tf false
```

### Cron Debugging
```bash
# Check cron logs
grep CRON /var/log/syslog

# Test cron environment
* * * * * env > /tmp/cron_env.log

# Test with full paths
* * * * * /usr/bin/python3 --version >> /tmp/python_test.log 2>&1
```

## Monitoring Cron Jobs

### Log Rotation
```bash
# Add to logrotate.conf
/path/to/SWING_BOT/outputs/logs/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
}
```

### Health Checks
```bash
# Daily health check script
#!/bin/bash
LOG_DIR="/path/to/SWING_BOT/outputs/logs"
TODAY=$(date +%Y%m%d)

# Check if EOD ran today
if [ -f "$LOG_DIR/eod_cron_$TODAY.log" ]; then
    if grep -q "🎉 SWING_BOT EOD Orchestration Complete" "$LOG_DIR/eod_cron_$TODAY.log"; then
        echo "EOD completed successfully"
    else
        echo "EOD failed - check logs"
        # Send alert
    fi
else
    echo "EOD did not run today"
    # Send alert
fi
```

---

*Last Updated: December 21, 2025*