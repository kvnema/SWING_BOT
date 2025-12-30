#!/usr/bin/env python3
"""
Test SWING_BOT Telegram Connection

Usage: python scripts/test_telegram.py BOT_TOKEN CHAT_ID
"""

import requests
import sys
from datetime import datetime

def test_telegram_connection(bot_token, chat_id):
    """Test Telegram connection by sending a test message."""

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": f"🧪 *SWING_BOT Telegram Test*\n\n✅ Connection successful!\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M IST')}\n\n🚀 Your self-improvement reports are now active!",
            "parse_mode": "Markdown"
        }

        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            print("✅ Telegram test message sent successfully!")
            print("🎉 SWING_BOT Telegram reports are now active!")
            return True
        else:
            print(f"❌ Telegram test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error testing Telegram: {e}")
        return False

def main():
    """Main function."""

    if len(sys.argv) != 3:
        print("Usage: python scripts/test_telegram.py BOT_TOKEN CHAT_ID")
        print("\nExample:")
        print("python scripts/test_telegram.py 8486307857:AAHt4XXRokpWf_Uv49NIVozp3lj1W-seqMg4 YOUR_CHAT_ID")
        print("\nTo get your CHAT_ID:")
        print("1. Start a conversation with @swingkopal_bot")
        print("2. Send any message to the bot")
        print("3. Visit: https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates")
        print("4. Find your chat ID in the JSON response")
        sys.exit(1)

    bot_token = sys.argv[1]
    chat_id = sys.argv[2]

    print("🚀 Testing SWING_BOT Telegram Connection")
    print("=" * 45)
    print(f"🤖 Bot Token: {bot_token[:20]}...")
    print(f"📱 Chat ID: {chat_id}")

    success = test_telegram_connection(bot_token, chat_id)

    if success:
        print("\n📋 Set these environment variables for production:")
        print(f"TELEGRAM_BOT_TOKEN={bot_token}")
        print(f"TELEGRAM_CHAT_ID={chat_id}")
        print("\n🔄 Daily reports will be sent ~16:45 IST (weekdays)")
        print("⚡ Instant alerts will be sent for parameter changes and errors")
    else:
        print("\n❌ Please check your bot token and chat ID")

if __name__ == "__main__":
    main()