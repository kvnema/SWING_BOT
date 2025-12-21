import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from .data_io import save_dataset

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

# NIFTY 50 stocks list
nifty_50_stocks = [
    'ADANIPORTS.NS', 'ADANIENT.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
    'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS',
    'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'SBIN.NS',
    'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
    'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS', 'NIFTY.NS'
]


def fetch_nifty50_data(days=500, out_path='data/nifty50_indicators_renamed.csv'):
    """Fetch NIFTY50 data, compute indicators, and save to CSV."""
    print("Starting NIFTY 50 Data Fetch and Processing...")

    if not ACCESS_TOKEN:
        raise ValueError("UPSTOX_ACCESS_TOKEN not set in .env")

    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Accept': 'application/json'
    }

    all_data = []
    failed_stocks = []

    for i, stock in enumerate(nifty_50_stocks):
        try:
            print(f"  [{i+1}/{len(nifty_50_stocks)}] Fetching {stock}...", end=" ", flush=True)

            # Fetch data for specified days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            symbol = stock.replace('.NS', '')
            instrument_key = instrument_keys.get(symbol, symbol)
            url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{end_date.strftime('%Y-%m-%d')}/{start_date.strftime('%Y-%m-%d')}"

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data_json = response.json()
                candles = data_json.get('data', {}).get('candles', [])

                if candles:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['Date'] = df['timestamp'].dt.date
                    df['Symbol'] = symbol
                    df = df[['Symbol', 'Date', 'open', 'high', 'low', 'close', 'volume']]
                    df.columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    all_data.append(df)
                    print("OK")
                else:
                    print("FAILED (No data)")
                    failed_stocks.append(stock)
            else:
                print(f"FAILED (API Error: {response.status_code})")
                failed_stocks.append(stock)

        except Exception as e:
            print(f"FAILED (Error: {str(e)[:30]})")
            failed_stocks.append(stock)

        time.sleep(0.2)  # Rate limiting

    print(f"\nSuccessfully fetched {len(all_data)} stocks out of {len(nifty_50_stocks)}")

    if len(all_data) == 0:
        raise RuntimeError("Failed to fetch data for any stocks.")

    # Combine all data
    df_all = pd.concat(all_data, ignore_index=True)
    df_all['Date'] = pd.to_datetime(df_all['Date']).dt.date
    df_all = df_all.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    # Get NIFTY data for RS and IndexUpRegime
    nifty_data = df_all[df_all['Symbol'] == 'NIFTY'].set_index('Date')['Close'] if 'NIFTY' in df_all['Symbol'].unique() else None

    processed_frames = []
    for symbol in df_all['Symbol'].unique():
        if symbol == 'NIFTY':
            continue

        stock_data = df_all[df_all['Symbol'] == symbol].copy()

        if len(stock_data) >= 200:
            # Exponential Moving Averages
            stock_data['EMA20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
            stock_data['EMA50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
            stock_data['EMA200'] = stock_data['Close'].ewm(span=200, adjust=False).mean()

            # RSI14
            delta = stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            stock_data['RSI14'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = stock_data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = stock_data['Close'].ewm(span=26, adjust=False).mean()
            stock_data['MACD'] = ema12 - ema26
            stock_data['MACDSignal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
            stock_data['MACDHist'] = stock_data['MACD'] - stock_data['MACDSignal']

            # ATR14
            high_low = stock_data['High'] - stock_data['Low']
            high_close = (stock_data['High'] - stock_data['Close'].shift(1)).abs()
            low_close = (stock_data['Low'] - stock_data['Close'].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            stock_data['ATR14'] = true_range.rolling(14).mean()

            # Bollinger Bands
            sma20 = stock_data['Close'].rolling(20).mean()
            std = stock_data['Close'].rolling(20).std()
            stock_data['BB_MA20'] = sma20
            stock_data['BB_Upper'] = sma20 + (std * 2)
            stock_data['BB_Lower'] = sma20 - (std * 2)
            stock_data['BB_BandWidth'] = (stock_data['BB_Upper'] - stock_data['BB_Lower']) / sma20

            # Donchian Channel
            stock_data['DonchianH20'] = stock_data['High'].rolling(20).max()
            stock_data['DonchianL20'] = stock_data['Low'].rolling(20).min()

            # RVOL20
            avg_volume_20 = stock_data['Volume'].rolling(20).mean()
            stock_data['RVOL20'] = stock_data['Volume'] / avg_volume_20

            # RS vs NIFTY
            if nifty_data is not None:
                common_dates = stock_data.set_index('Date').index.intersection(nifty_data.index)
                stock_aligned = stock_data.set_index('Date').loc[common_dates]['Close']
                nifty_aligned = nifty_data.loc[common_dates]
                stock_data = stock_data.set_index('Date')
                stock_data['RS_vs_Index'] = stock_aligned / nifty_aligned
                stock_data['RS_ROC20'] = stock_data['RS_vs_Index'].pct_change(20) * 100
                stock_data = stock_data.reset_index()
            else:
                stock_data['RS_vs_Index'] = np.nan
                stock_data['RS_ROC20'] = np.nan

            # Keltner Channels (simplified)
            stock_data['KC_Upper'] = stock_data['BB_MA20'] + (stock_data['ATR14'] * 1.5)
            stock_data['KC_Lower'] = stock_data['BB_MA20'] - (stock_data['ATR14'] * 1.5)
            stock_data['Squeeze'] = ((stock_data['BB_BandWidth'] < stock_data['BB_BandWidth'].rolling(20).mean()) & 
                                     (stock_data['ATR14'] < stock_data['ATR14'].rolling(20).mean())).astype(int)

            # AVWAP60 (simplified as EMA200 for now, or compute properly)
            stock_data['AVWAP60'] = stock_data['EMA200']  # Placeholder

            # IndexUpRegime
            if nifty_data is not None:
                nifty_ema50 = nifty_data.ewm(span=50, adjust=False).mean()
                stock_data = stock_data.set_index('Date')
                stock_data['IndexUpRegime'] = (nifty_data > nifty_ema50).astype(int)
                stock_data = stock_data.reset_index()
            else:
                stock_data['IndexUpRegime'] = 0

            processed_frames.append(stock_data)

    if processed_frames:
        df_processed = pd.concat(processed_frames, ignore_index=True)
        df_processed = df_processed.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        # Save data in appropriate format
        save_dataset(df_processed, out_path)
        print(f"Processed data saved to {out_path}")
        print(f"Total records: {len(df_processed)}")
        print(f"Stocks processed: {len(df_processed['Symbol'].unique())}")
    else:
        raise RuntimeError("No data processed.")


if __name__ == "__main__":
    fetch_nifty50_data()