"""Mean Reversion (MR) Strategy with Oversold Bounce Logic."""
import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import compute_all_indicators
from ..patterns import detect_weekly_patterns


def signal(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None,
           rsi_oversold: int = 30, bb_lower_mult: float = 0.5) -> pd.DataFrame:
    """
    Mean reversion strategy focusing on oversold bounces

    Args:
        df: Daily stock data
        benchmark_df: Benchmark index data for RS calculations
        rsi_oversold: RSI oversold threshold
        bb_lower_mult: Distance from BB lower for entry

    Returns:
        DataFrame with MR signals and components
    """
    # Compute all indicators
    d = compute_all_indicators(df, benchmark_df)

    # Oversold conditions
    d['RSI_Oversold'] = d['RSI14'] <= rsi_oversold
    d['BB_Oversold'] = d['Close'] <= (d['BB_Lower'] + (d['BB_Upper'] - d['BB_Lower']) * bb_lower_mult)

    # Mean reversion setup: price rejection at support
    d['Support_Rejection'] = (
        (d['Low'] < d['BB_Lower']) &
        (d['Close'] > d['BB_Lower'])  # Closed above lower BB
    )

    # Trend context (counter-trend entries)
    d['Counter_Trend'] = d['Close'] < d['EMA20']  # Below short-term trend

    # Volume confirmation (not excessive)
    d['Volume_OK'] = (d['RVOL20'] >= 0.8) & (d['RVOL20'] <= 2.0)

    # Volatility filters (prefer moderate volatility for MR)
    d['Volatility_OK'] = (d['ATR_pct'] > 0.8) & (d['ATR_pct'] < 2.5)

    # Mansfield RS filter (can be weaker for MR)
    if 'Mansfield_RS' in d.columns:
        d['RS_OK'] = d['Mansfield_RS'] > 0.8
    else:
        d['RS_OK'] = True

    # Weekly pattern confirmation (reversal patterns)
    weekly_patterns = detect_weekly_patterns(d)
    d['Weekly_Reversal'] = (
        weekly_patterns['hammer'] |
        weekly_patterns['bullish_engulfing'] |
        weekly_patterns['morning_star'] |
        weekly_patterns['piercing']
    )

    # Primary MR signal
    d['MR_Signal'] = (
        d['RSI_Oversold'] &
        d['BB_Oversold'] &
        d['Support_Rejection'] &
        d['Counter_Trend'] &
        d['Volume_OK'] &
        d['Volatility_OK'] &
        d['RS_OK'] &
        d['Weekly_Reversal']
    ).astype(int)

    # High-quality signal: extreme oversold with strong rejection
    d['MR_High_Quality'] = (
        d['MR_Signal'].astype(bool) &
        (d['RSI14'] <= 25) &  # Very oversold
        (d['Close'] > d['BB_Lower'] * 1.02) &  # Strong rejection
        (d['RSI14'].shift(1) < d['RSI14'])  # RSI improving
    ).astype(int)

    return d[[
        'Symbol', 'Date', 'MR_Signal', 'MR_High_Quality',
        'RSI_Oversold', 'BB_Oversold', 'Support_Rejection', 'Counter_Trend',
        'Volume_OK', 'Volatility_OK', 'RS_OK', 'Weekly_Reversal',
        'RSI14', 'BB_Lower', 'BB_Upper', 'EMA20'
    ]]


def get_entry_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get entry setup details for MR signals

    Args:
        df: DataFrame with signals

    Returns:
        DataFrame with entry details
    """
    entries = df[df['MR_Signal'] == 1].copy()

    if entries.empty:
        return pd.DataFrame()

    # Entry on support rejection (next day open or current close)
    entries['entry_price'] = entries['Close']

    # Stop loss below recent low or BB lower
    entries['stop_loss'] = np.minimum(
        entries['BB_Lower'] * 0.98,  # Slightly below BB
        entries['Low'].rolling(20).min()
    )

    # Target at mean reversion levels (mid-BB or EMA)
    entries['target_price'] = entries[['BB_Lower', 'EMA20']].mean(axis=1) * 1.05

    # Alternative target at full BB upper
    entries['target_alt'] = entries['BB_Upper']

    return entries[[
        'entry_price', 'stop_loss', 'target_price', 'target_alt',
        'BB_Lower', 'BB_Upper', 'EMA20', 'RSI14'
    ]]