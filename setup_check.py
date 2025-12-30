#!/usr/bin/env python3
"""
SWING_BOT Setup Helper
Helps configure Telegram alerts and validate system setup
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Please copy .env.example to .env and fill in your credentials")
        return False

    required_vars = [
        'UPSTOX_API_KEY',
        'UPSTOX_API_SECRET',
        'UPSTOX_ACCESS_TOKEN'
    ]

    optional_vars = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]

    missing_required = []
    missing_optional = []

    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)

    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)

    if missing_required:
        print("❌ Missing required environment variables:")
        for var in missing_required:
            print(f"   - {var}")
        return False

    print("✅ Required API credentials configured")

    if missing_optional:
        print("⚠️  Optional features not configured:")
        for var in missing_optional:
            print(f"   - {var} (needed for Telegram alerts)")
    else:
        print("✅ Telegram alerts configured")

    return True

def test_api_connection():
    """Test API connectivity."""
    try:
        from src.data_fetch import calculate_market_regime
        print("🔄 Testing API connection...")
        regime = calculate_market_regime('NSE_INDEX|Nifty 50', days=50)  # Quick test
        print("✅ API connection successful")
        print(f"   Current regime: {regime.get('regime_status', 'UNKNOWN')}")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def test_telegram_alerts():
    """Test Telegram alert functionality."""
    try:
        from src.notifications import TelegramNotifier
        notifier = TelegramNotifier()

        if not notifier.enabled:
            print("⚠️  Telegram alerts not configured (skipping test)")
            return True

        print("🔄 Testing Telegram alerts...")
        # Send a test message
        success = notifier.send_message(
            "🤖 *SWING_BOT Setup Test*\n\nSystem is configured and ready!",
            parse_mode="Markdown"
        )

        if success:
            print("✅ Telegram alerts working")
            return True
        else:
            print("❌ Telegram alerts failed")
            return False

    except Exception as e:
        print(f"❌ Telegram test failed: {e}")
        return False

def show_setup_instructions():
    """Show setup instructions."""
    print("\n" + "="*60)
    print("SWING_BOT SETUP INSTRUCTIONS")
    print("="*60)

    print("\n1. API Configuration:")
    print("   - Sign up for Upstox API access")
    print("   - Get your API key, secret, and access token")
    print("   - Add them to your .env file")

    print("\n2. Telegram Alerts (Optional but Recommended):")
    print("   - Message @BotFather on Telegram: /newbot")
    print("   - Follow instructions to create your bot")
    print("   - Get your bot token from BotFather")
    print("   - Message @userinfobot to get your chat ID")
    print("   - Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")

    print("\n3. Test Your Setup:")
    print("   python setup_check.py")

    print("\n4. Start Monitoring:")
    print("   python monitor_market.py --mode continuous")

    print("\n5. Paper Trading:")
    print("   python paper_trade.py --scan-only")

    print("\n6. Walk-Forward Testing:")
    print("   python walk_forward_test.py --symbol RELIANCE.NS --start-date 2023-01-01 --end-date 2024-12-01")

def main():
    print("SWING_BOT System Setup Check")
    print("="*40)

    # Check environment
    env_ok = check_env_file()
    if not env_ok:
        show_setup_instructions()
        sys.exit(1)

    # Test API
    api_ok = test_api_connection()
    if not api_ok:
        print("\n❌ API test failed - check your credentials")
        sys.exit(1)

    # Test Telegram
    telegram_ok = test_telegram_alerts()

    # Test Live Trading (optional)
    live_ok = test_live_trading_setup()

    print("\n" + "="*40)
    if api_ok and telegram_ok and live_ok:
        print("🎉 SWING_BOT is fully configured!")
        print("\nAvailable modes:")
        print("- Paper Trading: python paper_trade.py")
        print("- Live Trading: python live_trader.py")
        print("- Market Monitor: python monitor_market.py --mode continuous")
        print("- Sector Analysis: python live_trader.py --mode sector-analysis")
    else:
        print("⚠️  Partial setup - check warnings above")
        show_setup_instructions()

if __name__ == "__main__":
    main()