import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Upstox API credentials from .env
API_KEY = os.getenv('UPSTOX_API_KEY')
API_SECRET = os.getenv('UPSTOX_API_SECRET')
ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')

# Instrument keys for Upstox (Nifty 50)
instrument_keys = {
    'ADANIPORTS': 'NSE_EQ|INE742F01042',
    'ADANIENT': 'NSE_EQ|INE423A01024',
    'APOLLOHOSP': 'NSE_EQ|INE437A01024',
    'ASIANPAINT': 'NSE_EQ|INE021A01026',
    'AXISBANK': 'NSE_EQ|INE238A01034',
    'BAJAJ-AUTO': 'NSE_EQ|INE917I01010',
    'BAJFINANCE': 'NSE_EQ|INE296A01024',
    'BAJAJFINSV': 'NSE_EQ|INE918I01026',
    'BPCL': 'NSE_EQ|INE029A01023',
    'BHARTIARTL': 'NSE_EQ|INE397D01024',
    'BRITANNIA': 'NSE_EQ|INE216A01030',
    'CIPLA': 'NSE_EQ|INE059A01026',
    'COALINDIA': 'NSE_EQ|INE522F01014',
    'DIVISLAB': 'NSE_EQ|INE361B01024',
    'DRREDDY': 'NSE_EQ|INE089A01023',
    'EICHERMOT': 'NSE_EQ|INE066A01021',
    'GRASIM': 'NSE_EQ|INE047A01021',
    'HCLTECH': 'NSE_EQ|INE860A01027',
    'HDFCBANK': 'NSE_EQ|INE040A01034',
    'HDFCLIFE': 'NSE_EQ|INE795G01014',
    'HEROMOTOCO': 'NSE_EQ|INE158A01026',
    'HINDALCO': 'NSE_EQ|INE038A01020',
    'HINDUNILVR': 'NSE_EQ|INE030A01027',
    'ICICIBANK': 'NSE_EQ|INE090A01021',
    'INDUSINDBK': 'NSE_EQ|INE095A01012',
    'INFY': 'NSE_EQ|INE009A01021',
    'ITC': 'NSE_EQ|INE154A01025',
    'JSWSTEEL': 'NSE_EQ|INE019A01038',
    'KOTAKBANK': 'NSE_EQ|INE237A01028',
    'LT': 'NSE_EQ|INE018A01030',
    'M&M': 'NSE_EQ|INE101A01026',
    'MARUTI': 'NSE_EQ|INE585B01010',
    'NTPC': 'NSE_EQ|INE733E01010',
    'NESTLEIND': 'NSE_EQ|INE239A01024',
    'ONGC': 'NSE_EQ|INE213A01029',
    'POWERGRID': 'NSE_EQ|INE752E01010',
    'RELIANCE': 'NSE_EQ|INE002A01018',
    'SBILIFE': 'NSE_EQ|INE123W01016',
    'SHREECEM': 'NSE_EQ|INE070A01015',
    'SBIN': 'NSE_EQ|INE062A01020',
    'SUNPHARMA': 'NSE_EQ|INE044A01036',
    'TCS': 'NSE_EQ|INE467B01029',
    'TATACONSUM': 'NSE_EQ|INE192A01025',
    'TATAMOTORS': 'NSE_EQ|INE155A01022',
    'TATASTEEL': 'NSE_EQ|INE081A01020',
    'TECHM': 'NSE_EQ|INE669C01036',
    'TITAN': 'NSE_EQ|INE280A01028',
    'ULTRACEMCO': 'NSE_EQ|INE481G01011',
    'UPL': 'NSE_EQ|INE628A01036',
    'WIPRO': 'NSE_EQ|INE075A01022',
    'NIFTY': 'NSE_INDEX|Nifty 50'
}

nifty_50_stocks = list(instrument_keys.keys())


def fetch_ohlc_v3(symbols: List[str], tf: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch OHLC data for given symbols and timeframe using Upstox OHLC Quotes V3.

    Supported tf: 'I1' (1min), 'I30' (30min), '1d' (daily)
    """
    if not ACCESS_TOKEN:
        raise ValueError("UPSTOX_ACCESS_TOKEN not set in .env")

    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Accept': 'application/json'
    }

    all_data = []

    for symbol in symbols:
        try:
            instrument_key = instrument_keys.get(symbol, symbol)
            # Assuming endpoint for OHLC V3
            url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{tf}/{end.strftime('%Y-%m-%d')}/{start.strftime('%Y-%m-%d')}"

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data_json = response.json()
                candles = data_json.get('data', {}).get('candles', [])

                if candles:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['Date'] = df['timestamp']
                    df['Symbol'] = symbol
                    df = df[['Symbol', 'Date', 'open', 'high', 'low', 'close', 'volume']]
                    df.columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    all_data.append(df)
            else:
                print(f"Failed to fetch {symbol} for {tf}: {response.status_code}")

        except Exception as e:
            print(f"Error fetching {symbol} for {tf}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True).sort_values(['Symbol', 'Date']).reset_index(drop=True)
    else:
        return pd.DataFrame()


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample OHLC data to target timeframe.

    Rules:
    - Open: first
    - High: max
    - Low: min
    - Close: last
    - Volume: sum
    """
    df = df.set_index('Date')
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }

    if tf == '15m':
        resampled = df.resample('15min').agg(ohlc_dict)
    elif tf == '1h':
        resampled = df.resample('1h').agg(ohlc_dict)
    elif tf == '4h':
        resampled = df.resample('4h').agg(ohlc_dict)
    elif tf == '1w':
        resampled = df.resample('W-FRI').agg(ohlc_dict)  # Weekly ending Friday
    elif tf == '1mo':
        resampled = df.resample('ME').agg(ohlc_dict)  # Monthly end
    else:
        raise ValueError(f"Unsupported resample tf: {tf}")

    resampled = resampled.dropna().reset_index()
    resampled['Symbol'] = df['Symbol'].iloc[0]  # Assuming single symbol
    return resampled[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


def compute_indicators_for_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Compute indicators for the given timeframe.
    Assumes df has OHLCV columns.
    """
    d = df.copy()
    d['Date'] = pd.to_datetime(d['Date'])

    # Basic indicators (adjust spans if needed for TF)
    span_20 = 20
    span_50 = 50
    span_200 = 200
    rsi_window = 14
    atr_window = 14
    bb_window = 20
    donchian_window = 20
    rvol_window = 20
    avwap_window = 60

    # EMAs
    d['EMA20'] = d['Close'].ewm(span=span_20, adjust=False).mean()
    d['EMA50'] = d['Close'].ewm(span=span_50, adjust=False).mean()
    d['EMA200'] = d['Close'].ewm(span=span_200, adjust=False).mean()

    # RSI14
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    d['RSI14'] = 100 - (100 / (1 + rs))

    # ATR14
    high_low = d['High'] - d['Low']
    high_close = (d['High'] - d['Close'].shift(1)).abs()
    low_close = (d['Low'] - d['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d['ATR14'] = true_range.rolling(atr_window).mean()

    # Bollinger Bands
    sma20 = d['Close'].rolling(bb_window).mean()
    std = d['Close'].rolling(bb_window).std()
    d['BB_MA20'] = sma20
    d['BB_Upper'] = sma20 + (std * 2)
    d['BB_Lower'] = sma20 - (std * 2)
    d['BB_BandWidth'] = (d['BB_Upper'] - d['BB_Lower']) / sma20

    # Donchian
    d['DonchianH20'] = d['High'].rolling(donchian_window).max()
    d['DonchianL20'] = d['Low'].rolling(donchian_window).min()

    # RVOL20
    avg_volume_20 = d['Volume'].rolling(rvol_window).mean()
    d['RVOL20'] = d['Volume'] / avg_volume_20

    # AVWAP60 (simplified as EMA200 for now)
    d['AVWAP60'] = d['EMA200']

    # Keltner Channels
    d['KC_Upper'] = d['BB_MA20'] + (d['ATR14'] * 1.5)
    d['KC_Lower'] = d['BB_MA20'] - (d['ATR14'] * 1.5)

    # For RS, need index data - placeholder
    d['RS_vs_Index'] = np.nan
    d['RS_ROC20'] = np.nan
    d['IndexUpRegime'] = 0

    return d


def latest_window(df: pd.DataFrame, bars_needed: int) -> pd.DataFrame:
    """
    Slice the latest window of bars_needed.
    """
    if len(df) > bars_needed:
        return df.tail(bars_needed).reset_index(drop=True)
    return df