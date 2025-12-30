# ICICI Direct Setup Guide for SWING_BOT

## Overview
This guide helps you set up ICICI Direct API integration for automated trading with SWING_BOT.

## Prerequisites
1. **ICICI Direct Account**: Active trading account with ICICI Direct
2. **API Access**: Register for Breeze API access
3. **Python Environment**: Virtual environment with dependencies

## Step 1: Register for ICICI Direct Breeze API

### 1.1 Get API Credentials
1. Visit: https://api.icicidirect.com/
2. Log in with your ICICI Direct credentials
3. Navigate to "API" or "Developer" section
4. Register for Breeze API access
5. Note down your:
   - **API Key** (App Key)
   - **API Secret** (Secret Key)

### 1.2 Update Environment Variables
Update your `.env` file with ICICI credentials:

```bash
ICICI_API_KEY=your_actual_api_key_here
ICICI_API_SECRET=your_actual_api_secret_here
ICICI_SESSION_TOKEN=  # Will be set during authentication
```

## Step 2: Initial Authentication

### 2.1 Run Authentication Script
```powershell
python -c "
from src.icici_token_manager import ICICISessionManager
manager = ICICISessionManager()
token = manager.authenticate_via_browser()
if token:
    print('✅ Authentication successful!')
else:
    print('❌ Authentication failed')
"
```

### 2.2 Manual Authentication Process
1. The script will display a login URL
2. Open the URL in your browser
3. Log in to ICICI Direct
4. Grant API permissions
5. Copy the `api_session` value from the browser URL
6. Paste it when prompted

### 2.3 Verify Setup
```powershell
python -c "
from src.icici_token_manager import ICICISessionManager
manager = ICICISessionManager()
info = manager.get_session_info()
print('Session Info:', info)
"
```

## Step 3: Test API Integration

### 3.1 Test Basic Connectivity
```powershell
python -c "
from src.icici_gtt import ICICIAPIClient
from dotenv import load_dotenv
import os

load_dotenv()
client = ICICIAPIClient(
    os.getenv('ICICI_API_KEY'),
    os.getenv('ICICI_API_SECRET'),
    os.getenv('ICICI_SESSION_TOKEN')
)

# Test customer details
response = client.get_customer_details()
print('API Test Result:', response)
"
```

### 3.2 Test Market Data
```powershell
python -c "
from src.icici_gtt import ICICIAPIClient
from dotenv import load_dotenv
import os

load_dotenv()
client = ICICIAPIClient(
    os.getenv('ICICI_API_KEY'),
    os.getenv('ICICI_API_SECRET'),
    os.getenv('ICICI_SESSION_TOKEN')
)

# Test live quote
quote = client.get_live_quote('NSE', 'RELIANCE')
print('Live Quote:', quote)
"
```

## Step 4: Configure for SWING_BOT

### 4.1 Update Configuration
The system will automatically use ICICI Direct when credentials are available.

### 4.2 Test Full Pipeline
```powershell
# Test the complete pipeline with ICICI
python -m src.cli orchestrate-eod --broker icici
```

## Step 5: Session Management

### 5.1 Automatic Session Monitoring
ICICI Direct sessions may expire. The system includes monitoring:

```powershell
# Check session status
python -c "
from src.icici_token_manager import ICICISessionManager
manager = ICICISessionManager()
status = manager.check_and_refresh_session()
print('Session Status:', status)
"
```

### 5.2 Scheduled Session Refresh
Set up Windows Task Scheduler for periodic session checks:

```powershell
# Create a session monitor script
.\setup_icici_session_monitor.ps1
```

## Important Notes

### Session Expiry
- ICICI Direct session tokens expire periodically
- The system will detect expired sessions and prompt for re-authentication
- **Not fully automatic** - Manual intervention required when sessions expire

### Limitations
- ICICI Direct doesn't support fully automatic token refresh like some other brokers
- Session renewal requires browser login when tokens expire
- Consider Fyers or Dhan for truly automated setups

### Supported Features
- ✅ GTT Orders (Good Till Triggered)
- ✅ Live Market Data
- ✅ Historical Data
- ✅ Portfolio Management
- ✅ Order Management

### Troubleshooting

**Authentication Issues:**
```powershell
# Clear session and re-authenticate
python -c "
from src.icici_token_manager import ICICISessionManager
manager = ICICISessionManager()
manager.update_session_token_in_env('')
print('Session cleared. Run authentication again.')
"
```

**API Errors:**
- Check your API credentials
- Verify account permissions
- Ensure sufficient balance for orders

**Session Expired:**
- Re-run the authentication process
- Update the session token in `.env`

## Alternative: Consider Fyers for Full Automation

If you need **truly automatic token refresh**, consider switching to **Fyers**:

### Why Fyers?
- ✅ OAuth 2.0 with automatic refresh tokens
- ✅ No manual intervention required
- ✅ Tokens refresh automatically in background
- ✅ Similar API capabilities

### Fyers Setup (Alternative)
If you prefer Fyers for automation:

1. Open Fyers account
2. Get API credentials
3. Use the existing Fyers integration in the codebase
4. Enjoy fully automated token management

Let me know if you want to switch to Fyers instead!