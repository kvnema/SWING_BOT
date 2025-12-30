# ICICI Direct Breeze API Setup Guide

This guide will help you set up ICICI Direct Breeze API integration for the SWING_BOT system.

## Prerequisites

1. **ICICI Direct Trading Account**: You must have an active ICICI Direct trading account
2. **Python Environment**: Ensure you have Python 3.6+ installed
3. **SWING_BOT Installation**: The SWING_BOT system should be properly installed

## Step 1: Register for ICICI Direct Breeze API

1. Visit the [ICICI Direct Breeze API Portal](https://api.icicidirect.com/apiuser/home)
2. Log in with your ICICI Direct credentials
3. Navigate to "Register an App"
4. Fill in the required details:
   - **App Name**: SWING_BOT (or any name you prefer)
   - **Redirect URL**: `https://127.0.0.1`
5. Submit the registration form
6. You will receive:
   - **API Key** (AppKey)
   - **API Secret** (Secret Key)

⚠️ **Important**: Keep your API Key and Secret Key secure and never share them publicly.

## Step 2: Configure Environment Variables

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your ICICI Direct credentials:
   ```env
   # ICICI Direct Breeze API Configuration
   ICICI_API_KEY=your_actual_api_key_here
   ICICI_API_SECRET=your_actual_api_secret_here
   ICICI_SESSION_TOKEN=your_session_token_here
   ```

## Step 3: Authenticate and Get Session Token

The ICICI Direct API uses session-based authentication. You'll need to authenticate via browser to get a session token.

### Option 1: Using the Session Manager (Recommended)

1. Activate your virtual environment:
   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

2. Run the ICICI session manager:
   ```bash
   python -m src.icici_token_manager --refresh
   ```

3. Follow the on-screen instructions:
   - Open the provided URL in your browser
   - Log in to your ICICI Direct account
   - Grant permissions to the app
   - Copy the API Session value from the browser URL
   - Paste it when prompted

4. The session token will be automatically saved to your `.env` file.

### Option 2: Manual Authentication

1. Generate the login URL:
   ```python
   import urllib.parse
   api_key = "your_api_key_here"
   encoded_key = urllib.parse.quote_plus(api_key)
   login_url = f"https://api.icicidirect.com/apiuser/login?api_key={encoded_key}"
   print(login_url)
   ```

2. Open the URL in your browser and complete authentication

3. Extract the API Session from the browser URL (it will look like `api_session=ABC123...`)

4. Exchange the API Session for a Session Token:
   ```python
   import requests
   import json

   api_session = "your_api_session_here"
   api_key = "your_api_key_here"

   url = "https://api.icicidirect.com/breezeapi/api/v1/customerdetails"
   headers = {'Content-Type': 'application/json'}
   data = json.dumps({
       'SessionToken': api_session,
       'AppKey': api_key
   })

   response = requests.post(url, headers=headers, data=data)
   result = response.json()

   if result.get('Status') == 200:
       session_token = result['Success']['session_token']
       print(f"Session Token: {session_token}")
   ```

5. Add the session token to your `.env` file.

## Step 4: Test the Integration

1. Test session validity:
   ```bash
   python -m src.icici_token_manager --check
   ```

2. Test API connectivity:
   ```python
   from src.icici_gtt import ICICIAPIClient
   from src.icici_token_manager import ICICISessionManager
   import os

   # Load credentials
   manager = ICICISessionManager()
   client = ICICIAPIClient(manager.api_key, manager.api_secret, manager.session_token)

   # Test customer details
   response = client.get_customer_details()
   print("Customer Details:", response)

   # Test quotes
   response = client.get_quotes("NIFTY", "NFO", product_type="futures")
   print("NIFTY Quote:", response)
   ```

## Step 5: Configure SWING_BOT for ICICI Direct

1. Update `config.yaml` to use ICICI Direct as the default broker:
   ```yaml
   broker:
     default: icici  # Change from 'upstox' to 'icici'
   ```

2. Ensure ICICI Direct is enabled in the configuration:
   ```yaml
   broker:
     icici:
       enabled: true
   ```

## Step 6: Run SWING_BOT with ICICI Direct

You can now run SWING_BOT with ICICI Direct integration:

```bash
# Run pipeline with ICICI Direct
python -m src.pipeline

# Run GTT monitoring
python -m src.scheduled_gtt_monitor
```

## API Features Supported

### ✅ Implemented Features

- **Authentication**: OAuth 2.0 with session tokens
- **Market Data**: Live quotes, historical data
- **Portfolio**: Positions, holdings, funds
- **Orders**: Place, modify, cancel regular orders
- **GTT Orders**: Single leg and three-leg OCO orders
- **Square Off**: Position squaring
- **Margin Calculator**: Margin requirements calculation

### 🔄 Session Management

ICICI Direct sessions may expire. The system includes automatic session validation and re-authentication:

- **Automatic Checks**: Session validity is checked before API calls
- **Re-authentication**: Automatic prompts for re-authentication when sessions expire
- **Status Monitoring**: Session status is logged and monitored

### 📊 Supported Instruments

- **NSE Stocks**: Equity cash, margin, BTST
- **NFO Derivatives**: Futures and Options
- **BSE**: Limited support (BSE equities not available via Breeze API)

## Troubleshooting

### Common Issues

1. **Session Expired**: Run `python -m src.icici_token_manager --refresh`

2. **Invalid API Key/Secret**: Double-check your credentials in `.env`

3. **Network Issues**: Ensure stable internet connection

4. **Rate Limits**: Breeze API allows 100 calls/minute, 5000 calls/day

5. **Instrument Not Found**: Verify stock codes using the security master file

### Error Codes

- **400**: Bad Request - Check parameters
- **401**: Unauthorized - Session expired, re-authenticate
- **403**: Forbidden - Insufficient permissions
- **404**: Not Found - Invalid endpoint or instrument
- **429**: Rate Limited - Too many requests
- **500**: Server Error - Retry later

### Getting Help

- **API Documentation**: https://api.icicidirect.com/breezeapi/documents/index.html
- **Support Email**: breezeapi@icicisecurities.com
- **Community**: Check GitHub issues for common problems

## Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use environment variables** for all sensitive data
3. **Rotate API keys** regularly
4. **Monitor API usage** to detect unauthorized access
5. **Use HTTPS** for all API communications

## Advanced Configuration

### Custom GTT Settings

```yaml
broker:
  icici:
    gtt:
      default_product: options  # or futures
      default_exchange: NFO     # or NSE
```

### Session Monitoring

The system automatically monitors session health. Check logs at:
- `logs/icici_session_manager.log`
- `data/icici_session_status.json`

### Rate Limiting

The API client includes automatic retry logic with exponential backoff for rate-limited requests.

---

🎉 **Congratulations!** Your SWING_BOT is now configured to work with ICICI Direct Breeze API. Happy trading!