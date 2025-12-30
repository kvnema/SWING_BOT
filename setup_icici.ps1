# SWING_BOT ICICI Direct Setup and Test Script

param(
    [switch]$Setup,
    [switch]$Test,
    [switch]$Authenticate,
    [switch]$Status
)

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "python"

Write-Host "SWING_BOT ICICI Direct Setup" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

if ($Setup) {
    Write-Host "Setting up ICICI Direct integration..." -ForegroundColor Yellow

    # Check if .env exists
    if (!(Test-Path ".env")) {
        Write-Host "❌ .env file not found. Please create it first." -ForegroundColor Red
        exit 1
    }

    Write-Host "✅ .env file found" -ForegroundColor Green
    Write-Host "Please update your .env file with ICICI credentials:" -ForegroundColor Yellow
    Write-Host "ICICI_API_KEY=your_api_key_here" -ForegroundColor White
    Write-Host "ICICI_API_SECRET=your_api_secret_here" -ForegroundColor White
    Write-Host "ICICI_SESSION_TOKEN=" -ForegroundColor White

    Read-Host "Press Enter after updating .env file"
}

if ($Authenticate) {
    Write-Host "Starting ICICI Direct authentication..." -ForegroundColor Yellow

    $authScript = @"
from src.icici_token_manager import ICICISessionManager
import os

try:
    manager = ICICISessionManager()
    token = manager.authenticate_via_browser()
    if token:
        print("✅ Authentication successful!")
        print(f"Session Token: {token[:20]}...")
    else:
        print("❌ Authentication failed")
except Exception as e:
    print(f"❌ Error: {e}")
"@

    $authScript | Out-File -FilePath "temp_auth.py" -Encoding UTF8
    & $PythonExe temp_auth.py
    Remove-Item "temp_auth.py" -ErrorAction SilentlyContinue
}

if ($Test) {
    Write-Host "Testing ICICI Direct API integration..." -ForegroundColor Yellow

    $testScript = @"
from src.icici_gtt import ICICIAPIClient
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('ICICI_API_KEY')
api_secret = os.getenv('ICICI_API_SECRET')
session_token = os.getenv('ICICI_SESSION_TOKEN')

if not all([api_key, api_secret, session_token]):
    print("❌ Missing ICICI credentials in .env")
    print("Required: ICICI_API_KEY, ICICI_API_SECRET, ICICI_SESSION_TOKEN")
    exit(1)

try:
    client = ICICIAPIClient(api_key, api_secret, session_token)

    # Test 1: Customer Details
    print("Testing customer details...")
    customer = client.get_customer_details()
    if customer.get('status_code') == 200:
        print("✅ Customer details: OK")
    else:
        print(f"❌ Customer details failed: {customer}")

    # Test 2: Live Quote
    print("Testing live quote...")
    quote = client.get_live_quote('NSE', 'RELIANCE')
    if quote and 'Success' in quote:
        print("✅ Live quote: OK")
    else:
        print(f"❌ Live quote failed: {quote}")

    # Test 3: Historical Data
    print("Testing historical data...")
    hist = client.get_historical_data('NSE', 'RELIANCE', '1day', '2024-01-01', '2024-01-02')
    if hist and 'Status' in hist:
        print("✅ Historical data: OK")
    else:
        print(f"❌ Historical data failed: {hist}")

    print("🎉 All tests passed! ICICI Direct is ready.")

except Exception as e:
    print(f"❌ Test failed with error: {e}")
"@

    $testScript | Out-File -FilePath "temp_test.py" -Encoding UTF8
    & $PythonExe temp_test.py
    Remove-Item "temp_test.py" -ErrorAction SilentlyContinue
}

if ($Status) {
    Write-Host "Checking ICICI Direct session status..." -ForegroundColor Yellow

    $statusScript = @"
from src.icici_token_manager import ICICISessionManager
from dotenv import load_dotenv
import os

load_dotenv()

try:
    manager = ICICISessionManager()
    info = manager.get_session_info()

    print("ICICI Direct Session Status:")
    print(f"  Has Session Token: {info['has_session_token']}")
    print(f"  Token Length: {info['session_token_length']}")
    print(f"  Is Valid: {info['is_valid']}")

    if info['decoded_info']:
        print(f"  Decoded Info: {info['decoded_info']}")

    if info['is_valid']:
        print("✅ Session is active and valid")
    else:
        print("❌ Session is invalid or expired")
        print("Run: .\setup_icici.ps1 -Authenticate")

except Exception as e:
    print(f"❌ Status check failed: {e}")
"@

    $statusScript | Out-File -FilePath "temp_status.py" -Encoding UTF8
    & $PythonExe temp_status.py
    Remove-Item "temp_status.py" -ErrorAction SilentlyContinue
}

if (-not ($Setup -or $Test -or $Authenticate -or $Status)) {
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\setup_icici.ps1 -Setup          # Initial setup instructions" -ForegroundColor White
    Write-Host "  .\setup_icici.ps1 -Authenticate   # Authenticate and get session token" -ForegroundColor White
    Write-Host "  .\setup_icici.ps1 -Test           # Test API integration" -ForegroundColor White
    Write-Host "  .\setup_icici.ps1 -Status         # Check session status" -ForegroundColor White
    Write-Host "" -ForegroundColor White
    Write-Host "Example workflow:" -ForegroundColor Cyan
    Write-Host "  1. .\setup_icici.ps1 -Setup" -ForegroundColor White
    Write-Host "  2. Update .env with API credentials" -ForegroundColor White
    Write-Host "  3. .\setup_icici.ps1 -Authenticate" -ForegroundColor White
    Write-Host "  4. .\setup_icici.ps1 -Test" -ForegroundColor White
}