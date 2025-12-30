import gzip
import json
from urllib.request import urlopen

# Download and check instrument types
url = 'https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz'
print('Downloading...')
with urlopen(url) as response:
    with gzip.open(response, 'rt', encoding='utf-8') as f:
        data = json.load(f)

print(f'Total instruments: {len(data)}')

# Check NSE instruments
nse_instruments = [i for i in data if i.get('exchange') == 'NSE']
print(f'NSE instruments: {len(nse_instruments)}')

# Get unique instrument types for NSE
types = set()
for i in nse_instruments[:1000]:  # Sample first 1000
    types.add(i.get('instrument_type', 'UNKNOWN'))

print(f'Instrument types in NSE: {sorted(types)}')

# Look for ETF-like instruments
etf_like = [i for i in nse_instruments if 'ETF' in i.get('trading_symbol', '').upper()][:5]
print(f'Sample ETF-like symbols: {[i.get("trading_symbol") for i in etf_like]}')