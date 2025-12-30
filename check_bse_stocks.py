import requests
import gzip
import json

# Check BSE instruments
url = 'https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz'
response = requests.get(url)

if response.status_code == 200:
    decompressed = gzip.decompress(response.content)
    instruments = json.loads(decompressed.decode('utf-8'))

    print(f'BSE Total: {len(instruments)} instruments')

    # Sample some popular BSE stocks
    popular_bse = []
    for inst in instruments:
        symbol = inst.get('trading_symbol', '')
        name = inst.get('name', '')
        if any(keyword in name.upper() for keyword in ['TCS', 'INFOSYS', 'RELIANCE', 'HDFC', 'ICICI', 'BAJAJ', 'MARUTI']):
            popular_bse.append({
                'symbol': symbol,
                'name': name,
                'key': inst.get('instrument_key'),
                'type': 'BSE_EQ'
            })

    print(f'Found {len(popular_bse)} popular BSE stocks:')
    for stock in popular_bse[:10]:
        print(f'  {stock["symbol"]}: {stock["name"]} ({stock["type"]})')
else:
    print(f'Failed to fetch BSE data: {response.status_code}')