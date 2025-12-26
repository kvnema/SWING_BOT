import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

# Import centralized configuration
from .config import (
    API_KEY, API_SECRET, ACCESS_TOKEN, INSTRUMENT_KEYS, NIFTY_50_STOCKS,
    NSE_ETFS, ALL_INSTRUMENTS, API_RATE_LIMIT_DELAY, MAX_RETRIES,
    RETRY_BACKOFF_FACTOR, MIN_DATA_POINTS, DEFAULT_LOOKBACK_DAYS
)
from .data_io import save_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache for API responses to avoid redundant calls
_api_cache: Dict[str, Dict] = {}
_cache_file = Path('data/api_cache.json')

def load_instrument_keys() -> Dict[str, str]:
    """Load instrument keys from artifacts/universe/instrument_keys.json if available, fallback to config."""
    universe_file = Path('artifacts/universe/instrument_keys.json')

    if universe_file.exists():
        try:
            with open(universe_file, 'r') as f:
                universe_data = json.load(f)

            # Convert to symbol -> instrument_key mapping
            instrument_keys = {}
            for symbol, data in universe_data.items():
                instrument_keys[symbol] = data.get('instrument_key', symbol)

            logger.info(f"Loaded {len(instrument_keys)} instrument keys from {universe_file}")
            return instrument_keys

        except Exception as e:
            logger.warning(f"Failed to load instrument keys from {universe_file}: {e}")
            logger.info("Falling back to hardcoded instrument keys")

    return INSTRUMENT_KEYS

def _load_cache() -> Dict[str, Dict]:
    """Load API response cache from disk."""
    global _api_cache
    if _cache_file.exists():
        try:
            with open(_cache_file, 'r') as f:
                _api_cache = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            _api_cache = {}
    return _api_cache

def _save_cache():
    """Save API response cache to disk."""
    try:
        _cache_file.parent.mkdir(exist_ok=True)
        with open(_cache_file, 'w') as f:
            json.dump(_api_cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

def _make_api_request(url: str, headers: Dict[str, str], max_retries: int = MAX_RETRIES) -> Optional[Dict]:
    """Make API request with retry logic and caching."""
    cache_key = url

    # Check cache first
    if cache_key in _api_cache:
        cached_data = _api_cache[cache_key]
        cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
        if (datetime.now() - cache_time).days < 1:  # Cache for 1 day
            logger.debug(f"Using cached data for {url}")
            return cached_data['data']

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                # Cache successful response
                _api_cache[cache_key] = {
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }
                _save_cache()
                return data
            elif response.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) * RETRY_BACKOFF_FACTOR
                logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                break

        except requests.exceptions.RequestException as e:
            wait_time = (2 ** attempt) * RETRY_BACKOFF_FACTOR
            logger.warning(f"Request failed (attempt {attempt + 1}): {e}, waiting {wait_time}s")
            time.sleep(wait_time)

    return None

def fetch_single_instrument(symbol: str, days: int, headers: Dict[str, str]) -> Tuple[str, Optional[pd.DataFrame], str]:
    """Fetch data for a single instrument with error handling."""
    try:
        logger.info(f"Fetching {symbol}...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Remove .NS suffix for instrument key lookup
        clean_symbol = symbol.replace('.NS', '')
        instrument_key = INSTRUMENT_KEYS.get(clean_symbol, clean_symbol)

        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{end_date.strftime('%Y-%m-%d')}/{start_date.strftime('%Y-%m-%d')}"

        data = _make_api_request(url, headers)
        if not data:
            return symbol, None, "API_ERROR"

        candles = data.get('data', {}).get('candles', [])
        if not candles:
            return symbol, None, "NO_DATA"

        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['Date'] = df['timestamp'].dt.date
        df['Symbol'] = clean_symbol
        df = df[['Symbol', 'Date', 'open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        return symbol, df, "SUCCESS"

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
        return symbol, None, f"ERROR: {str(e)}"

def fetch_nifty50_data(days: int = DEFAULT_LOOKBACK_DAYS, out_path: str = 'data/nifty50_indicators_renamed.csv', include_etfs: bool = True, max_workers: int = 8) -> None:
    """
    Optimized data fetching with parallel processing, caching, and error handling.

    Args:
        days: Number of days of historical data to fetch
        out_path: Output path for processed data
        include_etfs: Whether to include ETFs in the fetch
        max_workers: Maximum number of parallel threads
    """
    logger.info("Starting optimized NSE data fetch and processing...")

    if not ACCESS_TOKEN:
        raise ValueError("UPSTOX_ACCESS_TOKEN not set in .env")

    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Accept': 'application/json'
    }

    # Load instrument keys dynamically
    global INSTRUMENT_KEYS
    INSTRUMENT_KEYS = load_instrument_keys()

    # Load cache
    _load_cache()

    # Filter instruments to only those available in our universe
    available_symbols = set(INSTRUMENT_KEYS.keys())
    instruments_to_fetch = [sym for sym in ALL_INSTRUMENTS if sym.replace('.NS', '') in available_symbols]
    
    logger.info(f"Fetching {len(instruments_to_fetch)} instruments from available universe (out of {len(ALL_INSTRUMENTS)} total)")
    if len(instruments_to_fetch) == 0:
        raise RuntimeError("No instruments available in universe file. Run ETF universe update first.")

    all_data = []
    failed_instruments = []

    # Parallel fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(fetch_single_instrument, symbol, days, headers): symbol
            for symbol in instruments_to_fetch
        }

        # Process completed tasks
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol_result, df_result, status = future.result()

                if df_result is not None and status == "SUCCESS":
                    all_data.append(df_result)
                    logger.info(f"✓ {symbol}: SUCCESS")
                else:
                    failed_instruments.append((symbol, status))
                    logger.warning(f"✗ {symbol}: {status}")

            except Exception as e:
                failed_instruments.append((symbol, f"THREAD_ERROR: {str(e)}"))
                logger.error(f"Thread error for {symbol}: {str(e)}")

    # Log summary
    success_count = len(all_data)
    total_count = len(instruments_to_fetch)
    logger.info(f"Fetch complete: {success_count}/{total_count} successful")

    if failed_instruments:
        logger.warning(f"Failed instruments ({len(failed_instruments)}):")
        for symbol, reason in failed_instruments[:10]:  # Show first 10 failures
            logger.warning(f"  {symbol}: {reason}")
        if len(failed_instruments) > 10:
            logger.warning(f"  ... and {len(failed_instruments) - 10} more")

    if success_count == 0:
        raise RuntimeError("Failed to fetch data for any instruments.")

    # Combine all data
    logger.info("Combining and processing data...")
    df_all = pd.concat(all_data, ignore_index=True)
    df_all['Date'] = pd.to_datetime(df_all['Date']).dt.date
    df_all = df_all.sort_values(['Symbol', 'Date']).reset_index(drop=True)

    # Get NIFTY data for RS and IndexUpRegime
    nifty_data = df_all[df_all['Symbol'] == 'NIFTY'].set_index('Date')['Close'] if 'NIFTY' in df_all['Symbol'].unique() else None

    # Process each symbol with optimized indicator calculations
    processed_frames = []
    symbols_processed = 0

    for symbol in df_all['Symbol'].unique():
        if symbol == 'NIFTY':
            continue

        stock_data = df_all[df_all['Symbol'] == symbol].copy()
        logger.info(f"Processing {symbol}: {len(stock_data)} data points")

        if len(stock_data) >= MIN_DATA_POINTS:
            # Vectorized indicator calculations for better performance
            try:
                stock_data = calculate_indicators_optimized(stock_data, nifty_data)
                processed_frames.append(stock_data)
                symbols_processed += 1

                if symbols_processed % 5 == 0:
                    logger.info(f"Processed {symbols_processed} symbols...")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        else:
            logger.warning(f"Skipping {symbol}: insufficient data ({len(stock_data)} < {MIN_DATA_POINTS})")

    if processed_frames:
        df_processed = pd.concat(processed_frames, ignore_index=True)
        df_processed = df_processed.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        # Save data in appropriate format
        save_dataset(df_processed, out_path)
        logger.info(f"✓ Processed data saved to {out_path}")
        logger.info(f"✓ Total records: {len(df_processed):,}")
        logger.info(f"✓ Symbols processed: {symbols_processed}")
    else:
        raise RuntimeError("No data processed successfully.")

def calculate_indicators_optimized(stock_data: pd.DataFrame, nifty_data: Optional[pd.Series] = None) -> pd.DataFrame:
    """Optimized indicator calculations using vectorized operations."""
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

    # AVWAP60 (simplified as EMA200 for now)
    stock_data['AVWAP60'] = stock_data['EMA200']

    # IndexUpRegime
    if nifty_data is not None:
        nifty_ema50 = nifty_data.ewm(span=50, adjust=False).mean()
        stock_data = stock_data.set_index('Date')
        stock_data['IndexUpRegime'] = (nifty_data > nifty_ema50).astype(int)
        stock_data = stock_data.reset_index()
    else:
        stock_data['IndexUpRegime'] = 0

    return stock_data


def fetch_live_quotes() -> pd.DataFrame:
    """
    Fetch live quotes for all NIFTY 50 stocks.
    
    Returns:
        DataFrame with columns: Symbol, Open, High, Low, Close, Volume, InstrumentToken
    """
    from .config import NIFTY_50_STOCKS, INSTRUMENT_KEYS
    from .ltp_reconcile import fetch_live_quotes as _fetch_ltp_quotes
    import os
    
    # Get access token
    access_token = os.environ.get('UPSTOX_ACCESS_TOKEN')
    if not access_token:
        print("Warning: UPSTOX_ACCESS_TOKEN not set, returning empty DataFrame")
        return pd.DataFrame(columns=['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'InstrumentToken'])
    
    # Get instrument tokens for NIFTY 50 stocks
    instrument_tokens = []
    symbol_map = {}  # token -> symbol mapping
    
    for symbol in NIFTY_50_STOCKS:
        # Remove .NS suffix for lookup
        clean_symbol = symbol.replace('.NS', '')
        token = INSTRUMENT_KEYS.get(clean_symbol)
        if token:
            instrument_tokens.append(token)
            symbol_map[token] = clean_symbol
    
    if not instrument_tokens:
        print("Warning: No instrument tokens found, returning empty DataFrame")
        return pd.DataFrame(columns=['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'InstrumentToken'])
    
    # Fetch live quotes
    try:
        ltp_df = _fetch_ltp_quotes(instrument_tokens, access_token)
        
        if ltp_df.empty:
            return pd.DataFrame(columns=['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'InstrumentToken'])
        
        # Transform to expected format
        result_rows = []
        for _, row in ltp_df.iterrows():
            ohlc = row.get('ohlc', {})
            if ohlc:
                result_rows.append({
                    'Symbol': symbol_map.get(row['instrument_token'], row.get('symbol', '')),
                    'Open': ohlc.get('open', 0),
                    'High': ohlc.get('high', 0),
                    'Low': ohlc.get('low', 0),
                    'Close': row.get('last_price', ohlc.get('close', 0)),  # Use last_price as Close
                    'Volume': ohlc.get('volume', 0),
                    'InstrumentToken': row['instrument_token']
                })
        
        return pd.DataFrame(result_rows)
        
    except Exception as e:
        print(f"Error fetching live quotes: {e}")
        return pd.DataFrame(columns=['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'InstrumentToken'])


def fetch_all_timeframes(symbols: List[str], timeframes: List[str], start_date: str, end_date: str, out_dir: str) -> Dict[str, str]:
    """
    Fetch data for multiple symbols across multiple timeframes.
    
    Args:
        symbols: List of symbols (with .NS suffix)
        timeframes: List of timeframes like ['1m', '15m', '1h', '4h', '1d', '1w', '1mo']
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        out_dir: Output directory path
        
    Returns:
        Dict mapping timeframe to output file path
    """
    import os
    from pathlib import Path
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Map timeframe strings to Upstox interval strings
    tf_mapping = {
        '1m': '1minute',
        '15m': '15minute', 
        '1h': '1hour',
        '4h': '4hour',
        '1d': 'day',
        '1w': 'week',
        '1mo': 'month'
    }
    
    # Get API headers
    access_token = os.environ.get('UPSTOX_ACCESS_TOKEN')
    if not access_token:
        raise ValueError("UPSTOX_ACCESS_TOKEN environment variable not set")
        
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }
    
    results = {}
    
    for tf in timeframes:
        if tf not in tf_mapping:
            raise ValueError(f"Unsupported timeframe: {tf}")
            
        interval = tf_mapping[tf]
        all_data = []
        
        print(f"Fetching {tf} ({interval}) data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                # Remove .NS suffix for instrument key lookup
                clean_symbol = symbol.replace('.NS', '')
                instrument_key = INSTRUMENT_KEYS.get(clean_symbol, clean_symbol)
                
                # Build API URL
                url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{end_date}/{start_date}"
                
                data = _make_api_request(url, headers)
                if not data:
                    print(f"  {symbol}: API error")
                    continue
                    
                candles = data.get('data', {}).get('candles', [])
                if not candles:
                    print(f"  {symbol}: No data")
                    continue
                    
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['Date'] = df['timestamp'].dt.date
                df['Symbol'] = clean_symbol
                df = df[['Symbol', 'Date', 'open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                all_data.append(df)
                print(f"  {symbol}: OK ({len(df)} records)")
                
            except Exception as e:
                print(f"  {symbol}: Error - {str(e)}")
                continue
        
        if all_data:
            # Combine all symbols for this timeframe
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['Symbol', 'Date'])
            
            # Save to file
            out_file = out_path / f"nifty50_{tf}.csv"
            combined_df.to_csv(out_file, index=False)
            results[tf] = str(out_file)
            
            print(f"✅ {tf}: Saved {len(combined_df)} records to {out_file}")
        else:
            print(f"❌ {tf}: No data fetched")
    
    return results


if __name__ == "__main__":
    fetch_nifty50_data()