import gzip
import json
import os

# Create synthetic BOD data with 4 popular ETFs for testing
synthetic_instruments = [
    {
        "segment": "NSE_EQ",
        "name": "NIPPON INDIA ETF NIFTY BEES",
        "exchange": "NSE",
        "isin": "INF204KB14I2",
        "instrument_type": "ETF",
        "instrument_key": "NSE_EQ|INF204KB14I2",
        "trading_symbol": "NIFTYBEES",
        "short_name": "NIFTYBEES",
        "lot_size": 1,
        "tick_size": 0.01,
        "exchange_token": "105831",
        "security_type": "NORMAL"
    },
    {
        "segment": "NSE_EQ",
        "name": "SBI NIFTY 50 ETF",
        "exchange": "NSE",
        "isin": "INF200KA1FS1",
        "instrument_type": "ETF",
        "instrument_key": "NSE_EQ|INF200KA1FS1",
        "trading_symbol": "SETFNIF50",
        "short_name": "SETFNIF50",
        "lot_size": 1,
        "tick_size": 0.01,
        "exchange_token": "104063",
        "security_type": "NORMAL"
    },
    {
        "segment": "NSE_EQ",
        "name": "ICICI Prudential Nifty ETF",
        "exchange": "NSE",
        "isin": "INF109KC1NT3",
        "instrument_type": "ETF",
        "instrument_key": "NSE_EQ|INF109KC1NT3",
        "trading_symbol": "ICICINIFTY",
        "short_name": "ICICINIFTY",
        "lot_size": 1,
        "tick_size": 0.01,
        "exchange_token": "105850",
        "security_type": "NORMAL"
    },
    {
        "segment": "NSE_EQ",
        "name": "Kotak Nifty ETF",
        "exchange": "NSE",
        "isin": "INF174K014P6",
        "instrument_type": "ETF",
        "instrument_key": "NSE_EQ|INF174K014P6",
        "trading_symbol": "KOTAKNIFTY",
        "short_name": "KOTAKNIFTY",
        "lot_size": 1,
        "tick_size": 0.01,
        "exchange_token": "105858",
        "security_type": "NORMAL"
    }
]

# Write to gzipped JSON file
output_path = "c:/Users/K01340/SWING_BOT_GIT/SWING_BOT/artifacts/bod/complete.json.gz"
with gzip.open(output_path, 'wt', encoding='utf-8') as f:
    json.dump(synthetic_instruments, f, indent=2)

print(f"Created synthetic BOD file: {output_path}")
print(f"Contains {len(synthetic_instruments)} ETF instruments")