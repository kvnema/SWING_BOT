#!/usr/bin/env python3
import requests

token = '8486307857:AAHt4XXRokWf_Uv49NIVozp3lj1W-seqMg4'
url = f'https://api.telegram.org/bot{token}/getUpdates'
response = requests.get(url, timeout=10)

if response.status_code == 200:
    data = response.json()
    if data.get('result'):
        print("📨 Found recent interactions!")
        for update in data['result'][-1:]:  # Get the most recent
            if 'message' in update:
                chat = update['message']['chat']
                chat_id = chat['id']
                username = chat.get('username', 'N/A')
                first_name = chat.get('first_name', 'N/A')
                print(f"✅ YOUR CHAT ID: {chat_id}")
                print(f"   Username: @{username}")
                print(f"   Name: {first_name}")
                print()
                print("🎉 Use this chat ID to complete setup!")
                break
        else:
            print("❌ No messages found in recent updates")
    else:
        print("📭 No recent interactions found")
        print()
        print("📋 To get your chat ID:")
        print("1. Open Telegram")
        print("2. Search for @swingkopal_bot")
        print("3. Send any message to start conversation")
        print("4. Run this script again")
else:
    print(f"❌ API Error: {response.status_code}")