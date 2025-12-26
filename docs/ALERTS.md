# SWING_BOT Alerts and Notifications

## Overview
SWING_BOT uses Microsoft Teams Adaptive Cards for automated notifications of EOD execution results. This document covers webhook setup, card layouts, and escalation procedures.

## Teams Webhook Setup

### Prerequisites
- Microsoft Teams workspace with posting permissions
- Webhook URL from Teams channel configuration

### Configuration Steps

1. **Create Teams Webhook**:
   - Go to Teams channel → "⋯" menu → "Connectors"
   - Search for "Incoming Webhook" → Add
   - Configure name: "SWING_BOT EOD Alerts"
   - Copy webhook URL

2. **Environment Configuration**:
   ```bash
   # Add to .env file
   TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/your-webhook-url
   ```

3. **Test Webhook**:
   ```bash
   python -m src.cli teams-notify --message "Test notification from SWING_BOT"
   ```

### Security Notes
- Store webhook URL securely (not in version control)
- Rotate webhook URLs quarterly
- Monitor for unauthorized usage

## Success Notification Card

### Trigger
- Sent when `orchestrate-eod --post-teams true` completes successfully
- All pipeline steps completed without critical failures

### Card Layout

```json
{
  "type": "AdaptiveCard",
  "version": "1.4",
  "body": [
    {
      "type": "TextBlock",
      "text": "🎉 SWING_BOT EOD Complete",
      "weight": "Bolder",
      "size": "Large",
      "color": "Good"
    },
    {
      "type": "FactSet",
      "facts": [
        {
          "title": "Latest Date:",
          "value": "2025-12-20"
        },
        {
          "title": "Audit Results:",
          "value": "23 PASS, 2 FAIL"
        },
        {
          "title": "Positions Generated:",
          "value": "25"
        }
      ]
    },
    {
      "type": "TextBlock",
      "text": "Top Positions:",
      "weight": "Bolder"
    },
    {
      "type": "FactSet",
      "facts": [
        {
          "title": "RELIANCE.NS",
          "value": "Entry: ₹2500 | Stop: ₹2400 | Target: ₹2600 | Conf: 4.5 | Audit: PASS"
        },
        {
          "title": "TCS.NS",
          "value": "Entry: ₹3200 | Stop: ₹3100 | Target: ₹3300 | Conf: 4.2 | Audit: PASS"
        }
      ]
    },
    {
      "type": "ActionSet",
      "actions": [
        {
          "type": "Action.OpenUrl",
          "title": "📊 View Excel",
          "url": "file://outputs/gtt/GTT_Delivery_Final.xlsx"
        },
        {
          "type": "Action.OpenUrl",
          "title": "📋 View Plan CSV",
          "url": "file://outputs/gtt/gtt_plan_audited.csv"
        }
      ]
    }
  ]
}
```

### Card Fields
- **Title**: "🎉 SWING_BOT EOD Complete"
- **Latest Date**: Most recent data date (YYYY-MM-DD)
- **Audit Results**: "X PASS, Y FAIL" count
- **Positions Generated**: Total number of positions created
- **Top Positions**: Top 5 positions with Symbol, Entry/Stop/Target prices, Confidence, Audit status
- **Action Buttons**: Links to Excel and CSV files

## Email Fallback Notifications

### Overview
When Teams notifications fail, SWING_BOT automatically falls back to email notifications using SMTP or Microsoft Graph. This ensures critical alerts are never missed.

### Email Provider Setup

#### SMTP Configuration (Recommended)
```bash
# Environment variables
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=alerts@company.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=alerts@company.com
TO_EMAILS=ops@company.com,traders@company.com
```

#### Microsoft Graph Configuration (Alternative)
```bash
# Environment variables
GRAPH_TENANT_ID=your-tenant-id
GRAPH_CLIENT_ID=your-client-id
GRAPH_CLIENT_SECRET=your-client-secret
MAILBOX_UPN=alerts@company.com
TO_EMAILS=ops@company.com,traders@company.com
```

### Fallback Logic

#### Success Notifications
1. **Primary**: Teams Adaptive Card
2. **Fallback**: HTML email with embedded results table
3. **Attachments**: Final Excel, Audited Plan CSV (< 10MB)

#### Failure Notifications
1. **Primary**: Teams failure card
2. **Fallback**: HTML email with error details and troubleshooting
3. **Attachments**: Log files, audit reports

### Email Templates

#### Success Email Layout
```
Subject: 🎉 SWING_BOT EOD Complete - {date}

📊 Results Summary
- Pass Count: X
- Fail Count: Y
- Total Positions: Z

📈 Top Positions
[Embedded HTML table with Symbol, Prices, Confidence, Audit Status]

📎 Attachments
- GTT_Delivery_Final.xlsx
- gtt_plan_audited.csv
- dashboard/index.html (if generated)
```

#### Failure Email Layout
```
Subject: 🚨 SWING_BOT EOD Failed - {stage}

❌ Failure Details
- Stage: {pipeline_stage}
- Error: {error_message}
- Time: {timestamp}

🔧 Troubleshooting
- Hint 1: {troubleshooting_step}
- Hint 2: {troubleshooting_step}

📎 Attachments
- Recent log files
- System diagnostics
```

### Configuration Options

```yaml
# config.yaml
notifications:
  teams_enabled: true
  email_enabled: true
  fallback_enabled: true
  email_provider: smtp  # smtp or graph
```

### Testing Email Fallbacks

```bash
# Test success email
python scripts/post_teams_success.py  # Will fallback to email if Teams fails

# Test failure email
python scripts/post_teams_failure.py --error-category SYSTEM --error-message "Test failure"

# Force email-only mode
TEAMS_WEBHOOK_URL="" python -m src.cli orchestrate-eod --post-teams [other-flags]
```

## Notification Router

### Architecture
The notification router (`src/notifications_router.py`) orchestrates multi-channel delivery:

1. **Attempt Teams**: Send Adaptive Card
2. **On Failure**: Automatically switch to email
3. **Retry Logic**: Exponential backoff for transient failures
4. **Logging**: Track delivery success/failure by channel

### Router Configuration
```python
from notifications_router import notify_eod_success, notify_eod_failure

# Success notification
success = notify_eod_success(
    webhook_url=teams_url,
    latest_date="2025-12-21",
    pass_count=15,
    fail_count=2,
    top_rows=top_positions_df,
    file_links=file_paths
)

# Failure notification
success = notify_eod_failure(
    webhook_url=teams_url,
    stage="plan-audit",
    error_msg="Audit validation failed",
    hints=["Check data freshness", "Verify price tolerances"],
    file_links=log_paths
)
```

## Failure Notification Card

### Trigger
- Sent when any critical pipeline step fails
- Includes error category and recommended fix steps

### Card Layout

```json
{
  "type": "AdaptiveCard",
  "version": "1.4",
  "body": [
    {
      "type": "TextBlock",
      "text": "❌ SWING_BOT EOD Failed",
      "weight": "Bolder",
      "size": "Large",
      "color": "Attention"
    },
    {
      "type": "FactSet",
      "facts": [
        {
          "title": "Failure Stage:",
          "value": "plan-audit"
        },
        {
          "title": "Error Time:",
          "value": "2025-12-20 16:15 IST"
        },
        {
          "title": "Error Message:",
          "value": "Plan audit failed: Strict mode enabled: 5 audit failures found"
        }
      ]
    },
    {
      "type": "TextBlock",
      "text": "🔧 Recommended Fix Steps:",
      "weight": "Bolder"
    },
    {
      "type": "TextBlock",
      "text": "1. Review outputs/gtt/gtt_plan_audited.csv for FAIL reasons\n2. Check data freshness (max_age_days=1)\n3. Verify pivot calculations\n4. Consider adjusting risk parameters\n5. Re-run with --strict false if urgent",
      "wrap": true
    },
    {
      "type": "ActionSet",
      "actions": [
        {
          "type": "Action.OpenUrl",
          "title": "📋 View Logs",
          "url": "file://outputs/logs/"
        },
        {
          "type": "Action.OpenUrl",
          "title": "🔄 Retry EOD",
          "url": "cmd://run-orchestrate-eod"
        }
      ]
    }
  ]
}
```

### Failure Categories

| Stage | Description | Common Causes | Fix Steps |
|-------|-------------|---------------|-----------|
| **data-fetch** | Failed to fetch market data | API token invalid, network issues, rate limits | 1. Check UPSTOX_ACCESS_TOKEN<br>2. Verify internet connectivity<br>3. Wait for rate limit reset<br>4. Manual data fetch |
| **data-validate** | Data validation failed | Stale data, missing symbols, insufficient history | 1. Check data age (< 1 day)<br>2. Verify symbol coverage (50 stocks)<br>3. Ensure 500+ trading days<br>4. Re-fetch data |
| **screener** | Screener execution failed | Data format issues, calculation errors | 1. Check input data format<br>2. Verify column mappings<br>3. Review screener logs |
| **backtest** | Strategy backtesting failed | Missing strategy data, calculation errors | 1. Check backtest directory<br>2. Verify strategy configurations<br>3. Review backtest logs |
| **plan-generate** | GTT plan creation failed | No valid candidates, confidence issues | 1. Check screener results<br>2. Review confidence thresholds<br>3. Adjust strategy selection |
| **excel-generate** | Final Excel creation failed | Template issues, data formatting | 1. Check Excel template<br>2. Verify data types<br>3. Review Excel generation logs |
| **plan-audit** | Plan audit validation failed | Price mismatches, strict mode | 1. Review audit failures<br>2. Check pivot data<br>3. Adjust risk parameters<br>4. Use --strict false |
| **teams-post** | Teams notification failed | Webhook invalid, network issues | 1. Verify TEAMS_WEBHOOK_URL<br>2. Test webhook connectivity<br>3. Check Teams permissions |

## Escalation Procedures

### Alert Levels

#### **Level 1: Warning (Yellow)**
- Non-critical failures (e.g., multi-TF workbook generation fails)
- System continues to function
- **Action**: Log issue, monitor for recurrence

#### **Level 2: Critical (Orange)**
- EOD pipeline fails but data is intact
- Trading decisions can be made manually
- **Action**: Immediate notification to operations team

#### **Level 3: Emergency (Red)**
- Complete system failure or data corruption
- Trading cannot proceed
- **Action**: Immediate escalation to trading desk and development team

### Escalation Timeline
- **Immediate**: Teams notification sent
- **5 minutes**: Operations team alerted
- **15 minutes**: Development team notified for critical issues
- **30 minutes**: Trading desk notified for emergency situations

### Communication Channels
1. **Primary**: Teams channel notifications
2. **Secondary**: Email alerts to operations@company.com
3. **Emergency**: Phone call to on-call personnel

## Monitoring and Testing

### Regular Testing
```bash
# Test webhook connectivity
python scripts/post_teams_success.py

# Test failure notification
python scripts/post_teams_failure.py --stage plan-audit --error "Test failure message"
```

### Monitoring Checks
- **Daily**: Verify notifications received
- **Weekly**: Test webhook endpoints
- **Monthly**: Review alert patterns and effectiveness

### Alert Effectiveness Metrics
- **Delivery Rate**: >99% of notifications delivered
- **Response Time**: <5 minutes average response to critical alerts
- **False Positive Rate**: <1% non-actionable alerts

## Troubleshooting

### Common Issues

#### Webhook Not Working
```
Error: 400 Bad Request
```
**Fix**: Verify webhook URL is correct and active

#### Card Not Displaying
```
Error: Invalid JSON
```
**Fix**: Check Adaptive Card JSON syntax

#### No Notifications Received
**Fix**: Check TEAMS_WEBHOOK_URL environment variable

### Debug Commands
```bash
# Test basic webhook
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Test"}' $TEAMS_WEBHOOK_URL

# View recent logs
tail -f outputs/logs/teams_notifier_*.log
```

---

*Last Updated: December 21, 2025*