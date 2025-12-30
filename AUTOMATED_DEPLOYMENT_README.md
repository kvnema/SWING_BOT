# SWING_BOT Automated GTT Monitor

This system automatically monitors and updates GTT (Good Till Triggered) orders based on SWING_BOT's trading signals, running on a schedule during market hours.

## Features

- 🕐 **Automated Scheduling**: Runs every hour from 9:15 AM to 3:30 PM IST, plus 8:15 AM and 4:30 PM
- 🤖 **Smart Order Management**: Automatically places new orders, modifies existing ones, and detects significant price changes
- 📱 **Multi-Channel Notifications**: Sends updates via Microsoft Teams, Telegram, and email
- ☁️ **Cloud-Ready**: Deploy to AWS Lambda or run in Docker containers
- 🔄 **State Persistence**: Tracks order changes and maintains history

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Scheduler     │───▶│  SWING_BOT      │───▶│  Order Monitor  │
│  (Cron/AWS)     │    │  Pipeline       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Upstox API      │    │ Notifications   │
                       │  (GTT Orders)    │    │ (Teams/Telegram)│
                       └──────────────────┘    └─────────────────┘
```

## Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Clone and setup**:
```bash
git clone <your-repo>
cd swing-bot
```

2. **Create environment file**:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Deploy with Docker Compose**:
```bash
docker-compose up -d
```

4. **Check logs**:
```bash
docker-compose logs -f swing-bot-monitor
```

### Option 2: AWS Lambda Deployment

1. **Install AWS CLI and configure**:
```bash
aws configure
```

2. **Run deployment script**:
```bash
chmod +x deploy_aws_lambda.sh
./deploy_aws_lambda.sh
```

3. **Set environment variables** in AWS Lambda console:
   - `UPSTOX_ACCESS_TOKEN`
   - `UPSTOX_API_KEY`
   - `UPSTOX_API_SECRET`
   - `TEAMS_WEBHOOK_URL` (optional)
   - `TELEGRAM_BOT_TOKEN` (optional)
   - `TELEGRAM_CHAT_ID` (optional)

## Environment Variables

### Required
- `UPSTOX_ACCESS_TOKEN`: Your Upstox API access token
- `UPSTOX_API_KEY`: Your Upstox API key
- `UPSTOX_API_SECRET`: Your Upstox API secret

### Notifications (Optional)
- `TEAMS_WEBHOOK_URL`: Microsoft Teams webhook for notifications
- `TELEGRAM_BOT_TOKEN`: Telegram bot token
- `TELEGRAM_CHAT_ID`: Telegram chat ID for notifications

### Email (Optional)
- `SMTP_SERVER`: SMTP server address
- `SMTP_PORT`: SMTP port (587 for TLS)
- `SMTP_USERNAME`: SMTP username
- `SMTP_PASSWORD`: SMTP password

## Schedule Configuration

The system runs automatically at these times (IST):

- **8:15 AM**: Pre-market analysis
- **9:15 AM to 3:15 PM**: Hourly during market hours
- **4:30 PM**: End-of-day analysis

### AWS EventBridge Schedule
```
Hourly: cron(15 3-9 * * ? *)    # 9:15 AM - 3:15 PM IST
Morning: cron(15 2 * * ? *)     # 8:15 AM IST
Evening: cron(30 10 * * ? *)    # 4:30 PM IST
```

## Order Management Logic

### Change Detection
- **New Orders**: Positions in latest plan but not in active GTT orders
- **Modify Orders**: Existing positions with >0.5% price change or >5 paise difference
- **Cancel Orders**: Positions removed from latest plan (logged but not auto-cancelled)

### Price Thresholds
Orders are modified when entry/stop/target prices change by more than:
- 0.5% of current price, OR
- ₹0.05 (5 paise), whichever is larger

## Notification Examples

### New Orders Placed
```
🤖 SWING_BOT GTT Update (14:15 27/12)

🆕 New Orders: 2
  • RELIANCE
  • INFY

✅ Successfully Placed: 2
```

### Orders Modified
```
🤖 SWING_BOT GTT Update (11:15 27/12)

📝 Modified Orders: 1
  • RELIANCE

✅ Successfully Modified: 1
```

## Monitoring & Troubleshooting

### Check Logs
```bash
# Docker
docker-compose logs swing-bot-monitor

# AWS Lambda
aws logs tail /aws/lambda/swing-bot-gtt-monitor --follow
```

### Manual Testing
```bash
# Test locally
python scheduled_gtt_monitor.py

# Test in Docker
docker-compose exec swing-bot-monitor python scheduled_gtt_monitor.py
```

### Health Checks
```bash
# Docker health
docker-compose ps

# API health (if using nginx)
curl http://localhost:8080/health
```

## File Structure

```
swing-bot/
├── scheduled_gtt_monitor.py    # Main monitoring script
├── deploy_aws_lambda.sh       # AWS deployment script
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── nginx.conf                 # Nginx reverse proxy config
├── src/
│   ├── upstox_gtt.py         # GTT order management
│   ├── notifications_router.py # Notification handling
│   └── ...                   # Other SWING_BOT modules
├── outputs/
│   └── gtt/                  # GTT monitoring state
└── logs/                     # Application logs
```

## Security Considerations

- Store API credentials as environment variables, never in code
- Use IAM roles with minimal permissions for AWS deployments
- Rotate Upstox tokens regularly
- Monitor CloudWatch logs for security events

## Cost Estimation

### AWS Lambda (Free Tier)
- 1M requests/month free
- ~$0.20 per 1M requests beyond free tier
- ~$1-2/month for typical usage

### Docker (Self-hosted)
- VPS costs: $5-20/month depending on provider
- No additional AWS charges

## Support

For issues or questions:
1. Check the logs in `logs/cron.log`
2. Verify environment variables are set correctly
3. Test manually with `python scheduled_gtt_monitor.py`
4. Check Upstox API status and token validity

## Development

To modify the monitoring logic:

1. Edit `scheduled_gtt_monitor.py` for core logic
2. Update `src/notifications_router.py` for notifications
3. Modify `src/upstox_gtt.py` for order management
4. Test locally before deploying

## License

[Your License Here]