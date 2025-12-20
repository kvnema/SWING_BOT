import pandas as pd
from typing import Tuple
import os


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV/Parquet into a DataFrame. Supports .csv, .parquet."""
    if path.endswith('.parquet') or path.endswith('.parq'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError('Unsupported file format: ' + path)
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV or Parquet format based on file extension."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if path.endswith('.parquet') or path.endswith('.parq'):
        df.to_parquet(path, index=False)
    elif path.endswith('.csv'):
        df.to_csv(path, index=False)
    else:
        raise ValueError('Unsupported file format: ' + path)


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, list]:
    required = [
        'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume',
        'EMA20', 'EMA50', 'EMA200', 'RSI14', 'MACD', 'MACDSignal', 'MACDHist',
        'ATR14', 'BB_MA20', 'BB_Upper', 'BB_Lower', 'BB_BandWidth',
        'RVOL20', 'DonchianH20', 'DonchianL20', 'RS_vs_Index', 'RS_ROC20',
        'KC_Upper', 'KC_Lower', 'Squeeze', 'AVWAP60', 'IndexUpRegime'
    ]
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


def split_by_symbol(df: pd.DataFrame) -> dict:
    """Return a dict Symbol -> DataFrame (sorted by Date)."""
    out = {}
    for sym, g in df.groupby('Symbol'):
        dfg = g.copy()
        dfg['Date'] = pd.to_datetime(dfg['Date'])
        dfg = dfg.sort_values('Date').reset_index(drop=True)
        out[sym] = dfg
    return out
