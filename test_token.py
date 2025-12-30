#!/usr/bin/env python3
import requests

token = '8486307857:AAHt4XXRokWf_Uv49NIVozp3lj1W-seqMg4'
url = f'https://api.telegram.org/bot{token}/getMe'
response = requests.get(url, timeout=10)

print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print(f'Bot: @{data["result"]["username"]}')
    print('✅ Bot token is valid!')
else:
    print(f'❌ Error: {response.text}')