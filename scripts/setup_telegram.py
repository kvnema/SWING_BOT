#!/usr/bin/env python3
"""
Get Telegram Chat ID for SWING_BOT

This script helps you get your Telegram chat ID for bot notifications.
"""

import requests
import os
import sys

def get_chat_id(bot_token, username=None):
    """Get chat ID from username or provide instructions."""

    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not set")
        return None

    if username:
        print(f"🔍 Looking for chat with username: @{username}")
        print("Note: This requires the bot to have been added to the chat first")
        print("If this doesn't work, please follow the manual steps below.")

        try:
            # Try to get updates to find chat ID
            url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('ok') and data.get('result'):
                    for update in data['result']:
                        if 'message' in update:
                            chat = update['message']['chat']
                            if chat.get('username') == username:
                                chat_id = chat['id']
                                print(f"✅ Found chat ID for @{username}: {chat_id}")
                                return chat_id

                print("⚠️  Could not find chat with that username in recent updates")
                print("Make sure:")
                print("1. You've started a conversation with the bot")
                print("2. You've sent at least one message to the bot")
                print("3. The username is correct")

        except Exception as e:
            print(f"❌ Error getting chat ID: {e}")

    print("\n📋 Manual steps to get Chat ID:")
    print("1. Start a conversation with your bot: @swingkopal_bot")
    print("2. Send any message to the bot")
    print("3. Visit: https://api.telegram.org/bot" + bot_token + "/getUpdates")
    print("4. Look for 'chat': {'id': YOUR_CHAT_ID, ...}")
    print("5. The 'id' field is your chat ID (usually a negative number for private chats)")

    return None

def test_telegram_connection(bot_token, chat_id):
    """Test Telegram connection by sending a test message."""

    if not bot_token or not chat_id:
        print("❌ Bot token or chat ID missing")
        return False

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": "🧪 *SWING_BOT Telegram Test*\n\n✅ Connection successful!\n📅 " + __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M IST"),
            "parse_mode": "Markdown"
        }

        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            print("✅ Telegram test message sent successfully!")
            return True
        else:
            print(f"❌ Telegram test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error testing Telegram: {e}")
        return False

def main():
    """Main function to get chat ID and test connection."""

    print("🚀 SWING_BOT Telegram Setup Helper")
    print("=" * 40)

    # Get bot token
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or "8486307857:AAHt4XXRokWf_Uv49NIVozp3lj1W-seqMg4"
    chat_name = "swingkopal"

    print(f"🤖 Bot Token: {bot_token[:20]}...")
    print(f"👤 Chat Name: @{chat_name}")

    # Try to get chat ID
    chat_id = get_chat_id(bot_token, chat_name)

    if chat_id:
        print(f"\n✅ Chat ID found: {chat_id}")

        # Test connection
        print("\n🧪 Testing Telegram connection...")
        if test_telegram_connection(bot_token, chat_id):
            print("\n🎉 Setup complete! Telegram reports are ready.")
            print("Environment variables to set:")
            print(f"  TELEGRAM_BOT_TOKEN={bot_token}")
            print(f"  TELEGRAM_CHAT_ID={chat_id}")
        else:
            print("\n❌ Connection test failed. Please check your setup.")
    else:
        print("\n📝 Please get your chat ID manually and run:")
        print(f"python scripts/setup_telegram.py {bot_token} YOUR_CHAT_ID")

if __name__ == "__main__":
    main()