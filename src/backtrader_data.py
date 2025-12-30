"""
SWING_BOT Backtrader Data Feed Handler

Handles loading historical data from SWING_BOT's data pipeline into Backtrader.
Supports both individual stock data and benchmark data for regime filtering.
"""

import backtrader as bt
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SwingBotData(bt.feeds.DataBase):
    """
    Custom Backtrader data feed for SWING_BOT data format

    Expected CSV format:
    Date,Open,High,Low,Close,Volume,Symbol
    """

    params = (
        ('datetime', 0),  # Date is column 0
        ('open', 1),      # Open is column 1
        ('high', 2),      # High is column 2
        ('low', 3),       # Low is column 3
        ('close', 4),     # Close is column 4
        ('volume', 5),    # Volume is column 5
        ('symbol', 6),    # Symbol is column 6
    )

    def __init__(self, dataname=None, **kwargs):
        # Store dataname before calling super().__init__
        self._dataname = dataname
        self._data = []
        self._idx = 0  # Initialize index
        
        super().__init__(**kwargs)
        
        # Load data after super().__init__ - use p.dataname if available
        data_file = self._dataname or self.p.dataname
        if data_file is not None:
            try:
                df = pd.read_csv(data_file)
                logger.info(f"Loading data from {data_file}, shape: {df.shape}")

                # Skip header row and load data
                for i, (_, row) in enumerate(df.iloc[1:].iterrows()):
                    try:
                        dt = pd.to_datetime(row.iloc[self.p.datetime])
                        dt_num = bt.utils.date2num(dt)
                        o = float(row.iloc[self.p.open])
                        h = float(row.iloc[self.p.high])
                        l = float(row.iloc[self.p.low])
                        c = float(row.iloc[self.p.close])
                        v = int(row.iloc[self.p.volume])
                        self._data.append((dt_num, o, h, l, c, v))
                        if i < 2:  # Log first few
                            logger.info(f"Parsed row {i}: {dt} O:{o} H:{h} L:{l} C:{c} V:{v}")
                    except Exception as e:
                        logger.warning(f"Error parsing row {i}: {e}")
                        continue

                logger.info(f"Loaded {len(self._data)} data points from {data_file}")
            except Exception as e:
                logger.error(f"Error loading data from {data_file}: {e}")
        else:
            logger.warning("No dataname provided to SwingBotData")

    def __len__(self):
        return len(self._data)

    def _load(self):
        if self._idx >= len(self._data):
            return False

        dt, o, h, l, c, v = self._data[self._idx]
        self.lines.datetime[0] = dt
        self.lines.open[0] = o
        self.lines.high[0] = h
        self.lines.low[0] = l
        self.lines.close[0] = c
        self.lines.volume[0] = v

        self._idx += 1
        return True


def load_stock_data(symbol, data_dir='data', start_date=None, end_date=None):
    """
    Load historical data for a specific stock

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
        data_dir: Directory containing data files
        start_date: Start date for filtering (optional)
        end_date: End date for filtering (optional)

    Returns:
        pandas DataFrame with OHLCV data
    """

    # Try different file patterns - prioritize individual files for historical data
    possible_files = [
        f"{data_dir}/{symbol}_2020-01-01_2025-12-01.csv",  # Full historical range first
        f"{data_dir}/{symbol}_2023-01-01_2024-12-01.csv",  # Recent data
        f"{data_dir}/{symbol}_2024-01-01_2024-12-01.csv",  # Current year
        f"{data_dir}/{symbol}.csv",  # Individual symbol file
        f"{data_dir}/full_universe_100d.csv",  # Try full universe data last
        f"{data_dir}/nifty50_data.csv",  # Try consolidated NIFTY50 data last
    ]

    df = None
    for file_path in possible_files:
        try:
            if 'full_universe_100d.csv' in file_path or 'nifty50_data.csv' in file_path:
                # Load consolidated data and filter for symbol
                full_df = pd.read_csv(file_path)
                df = full_df[full_df['Symbol'] == symbol].copy()
                if not df.empty:
                    logger.info(f"Loaded data for {symbol} from {file_path} (filtered from consolidated data)")
                    break
                else:
                    continue
            else:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded data for {symbol} from {file_path}")
                break
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    if df is None:
        logger.error(f"No data file found for {symbol}")
        return None

    # Standardize column names
    column_mapping = {
        'date': 'Date',
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'vol': 'Volume',
    }

    df = df.rename(columns=column_mapping)

    # Ensure Date column exists and is datetime
    if 'Date' not in df.columns:
        # Assume first column is date if not named
        date_col = df.columns[0]
        df = df.rename(columns={date_col: 'Date'})

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Filter date range if specified
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    logger.info(f"After date filtering: {len(df)} bars for {symbol} from {df.index.min() if not df.empty else 'NaT'} to {df.index.max() if not df.empty else 'NaT'}")

    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col} for {symbol}")
            return None

    # Clean data - only check required columns for NaN
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.dropna(subset=required_cols)  # Remove rows with NaN values in required columns
    df = df[df['Volume'] > 0]  # Remove zero volume bars

    # Keep only required columns for backtesting
    df = df[required_cols]

    # Ensure data is sorted by date
    df = df.sort_index()

    # Ensure datetime index has no timezone info (Backtrader prefers naive datetime)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep Date as index for Backtrader (don't reset to column)
    # df = df.reset_index()
    # df.rename(columns={'index': 'Date'}, inplace=True)

    # Ensure Date column is datetime (but keep as index)
    # df['Date'] = pd.to_datetime(df['Date'])

    logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
    return df


def load_benchmark_data(benchmark='NIFTY50', data_dir='data'):
    """
    Load benchmark index data for regime filtering

    Args:
        benchmark: Benchmark symbol (default: 'NIFTY50')
        data_dir: Directory containing data files

    Returns:
        pandas DataFrame with OHLCV data
    """
    return load_stock_data(benchmark, data_dir)


def create_data_feeds(symbols, benchmark='NIFTY50', data_dir='data',
                     start_date=None, end_date=None):
    """
    Create Backtrader data feeds for multiple symbols

    Args:
        symbols: List of stock symbols
        benchmark: Benchmark symbol for regime filter
        data_dir: Directory containing data files
        start_date: Start date for data filtering
        end_date: End date for data filtering

    Returns:
        List of Backtrader data feeds
    """

    data_feeds = []

    # Load benchmark data first (with date filtering to match stock data)
    # Temporarily disabled due to data format issues
    # benchmark_files = [
    #     f"{data_dir}/{benchmark}_{start_date}_{end_date}.csv",
    #     f"{data_dir}/nifty50_data_extended.csv",
    #     f"{data_dir}/nifty50_data.csv"
    # ]
    # 
    # benchmark_feed = None
    # for benchmark_file in benchmark_files:
    #     if os.path.exists(benchmark_file):
    #         try:
    #             benchmark_feed = SwingBotData(dataname=benchmark_file, name=benchmark)
    #             logger.info(f"Created benchmark feed for {benchmark} from {benchmark_file}")
    #             data_feeds.append(benchmark_feed)
    #             break
    #         except Exception as e:
    #             logger.warning(f"Could not load benchmark data from {benchmark_file}: {e}")
    #             continue
    # 
    # if not benchmark_feed:
    #     logger.warning(f"Could not load benchmark data for {benchmark} - skipping benchmark")

    # Load stock data
    for symbol in symbols:
        stock_file = f"{data_dir}/{symbol}_{start_date}_{end_date}.csv"
        if stock_file is not None:
            logger.info(f"Creating data feed for {symbol} from {stock_file}")
            stock_feed = SwingBotData(dataname=stock_file, name=symbol)
            data_feeds.append(stock_feed)
            logger.info(f"Created data feed for {symbol}")
        else:
            logger.warning(f"Could not load data for {symbol} - skipping")

    logger.info(f"Created {len(data_feeds)} total data feeds")
    return data_feeds


def get_nifty50_universe():
    """
    Get the NIFTY50 universe symbols

    Returns:
        List of NIFTY50 stock symbols
    """
    nifty50_symbols = [
        'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
        'BAJAJFINSV', 'BHARTIARTL', 'BPCL', 'BRITANNIA', 'CIPLA',
        'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GRASIM',
        'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO',
        'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'INFY', 'ITC',
        'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI',
        'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE',
        'SBILIFE', 'SBIN', 'SHREECEM', 'SUNPHARMA', 'TATACONSUM',
        'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN',
        'ULTRACEMCO', 'UPL', 'WIPRO'
    ]
    return nifty50_symbols


def validate_data_quality(df, symbol):
    """
    Validate data quality for backtesting

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name for logging

    Returns:
        bool: True if data passes quality checks
    """

    issues = []

    # Check for sufficient data
    if len(df) < 252:  # At least 1 year of data
        issues.append(f"Insufficient data: {len(df)} bars (need at least 252)")

    # Check for missing values
    missing_data = df.isnull().sum().sum()
    if missing_data > 0:
        issues.append(f"Missing data points: {missing_data}")

    # Check for zero prices
    zero_prices = (df[['Open', 'High', 'Low', 'Close']] == 0).sum().sum()
    if zero_prices > 0:
        issues.append(f"Zero prices found: {zero_prices}")

    # Check for negative prices
    negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).sum().sum()
    if negative_prices > 0:
        issues.append(f"Negative prices found: {negative_prices}")

    # Check for invalid OHLC relationships
    invalid_ohlc = (
        (df['High'] < df['Low']) |
        (df['High'] < df['Open']) |
        (df['High'] < df['Close']) |
        (df['Low'] > df['Open']) |
        (df['Low'] > df['Close'])
    ).sum()
    if invalid_ohlc > 0:
        issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")

    if issues:
        logger.warning(f"Data quality issues for {symbol}:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    logger.info(f"Data quality check passed for {symbol}")
    return True


# Example usage and testing
if __name__ == "__main__":
    # Test data loading
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
    feeds = create_data_feeds(symbols, start_date='2023-01-01', end_date='2024-12-31')

    print(f"Created {len(feeds)} data feeds")
    for feed in feeds:
        print(f"- {feed._name}: {len(feed)} bars")