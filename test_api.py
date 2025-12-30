import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
if not ACCESS_TOKEN:
    print('No access token')
    exit(1)

headers = {'Authorization': f'Bearer {ACCESS_TOKEN}', 'Accept': 'application/json'}
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

url = f'https://api.upstox.com/v2/historical-candle/NSE_EQ|INE009A01021/day/{end_date.strftime("%Y-%m-%d")}/{start_date.strftime("%Y-%m-%d")}'
print(f'URL: {url}')

response = requests.get(url, headers=headers)
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    candles = data.get('data', {}).get('candles', [])
    print(f'Candles returned: {len(candles)}')
    if candles:
        print('First candle:', candles[0])
        print('Last candle:', candles[-1])
else:
    print('Response:', response.text)