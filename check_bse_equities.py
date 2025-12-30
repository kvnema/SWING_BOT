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

    # Check instrument types
    types = {}
    for inst in instruments[:1000]:  # Sample first 1000
        itype = inst.get('instrument_type', 'unknown')
        types[itype] = types.get(itype, 0) + 1

    print(f'Instrument types in sample: {types}')

    # Look for actual equity stocks
    equity_stocks = []
    for inst in instruments:
        symbol = inst.get('trading_symbol', '')
        name = inst.get('name', '')
        itype = inst.get('instrument_type', '')

        # Look for common equity patterns
        if itype in ['A', 'B', 'CE'] and len(symbol) <= 10 and not any(char in symbol for char in ['-', '.', '/']):
            if len(equity_stocks) < 20:
                equity_stocks.append({
                    'symbol': symbol,
                    'name': name,
                    'key': inst.get('instrument_key'),
                    'type': itype
                })

    print(f'Found {len(equity_stocks)} potential equity stocks:')
    for stock in equity_stocks:
        print(f'  {stock["symbol"]}: {stock["name"]} ({stock["type"]})')
else:
    print(f'Failed to fetch BSE data: {response.status_code}')