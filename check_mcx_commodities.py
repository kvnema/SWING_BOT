import requests
import gzip
import json

# Check MCX commodities
url = 'https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz'
response = requests.get(url)

if response.status_code == 200:
    decompressed = gzip.decompress(response.content)
    instruments = json.loads(decompressed.decode('utf-8'))

    print(f'MCX Total: {len(instruments)} instruments')

    # Check instrument types
    types = {}
    for inst in instruments[:1000]:  # Sample first 1000
        itype = inst.get('instrument_type', 'unknown')
        types[itype] = types.get(itype, 0) + 1

    print(f'Instrument types in sample: {types}')

    # Look for different commodities
    commodities = []
    seen_symbols = set()
    for inst in instruments:
        symbol = inst.get('trading_symbol', '')
        name = inst.get('name', '')
        itype = inst.get('instrument_type', '')

        # Get unique commodity types
        if symbol not in seen_symbols and len(commodities) < 20:
            commodities.append({
                'symbol': symbol,
                'name': name,
                'key': inst.get('instrument_key'),
                'type': itype
            })
            seen_symbols.add(symbol)

    print(f'Found {len(commodities)} different commodities:')
    for comm in commodities:
        print(f'  {comm["symbol"]}: {comm["name"]} ({comm["type"]})')
else:
    print(f'Failed to fetch MCX data: {response.status_code}')