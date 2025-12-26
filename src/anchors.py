"""
Event-Anchored AVWAP (Anchored Volume Weighted Average Price)
Computes AVWAP from significant events: earnings, gaps, swing points, high volume days
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


def detect_earnings_dates(events_df: pd.DataFrame) -> pd.Series:
    """
    Extract earnings dates from events DataFrame

    Args:
        events_df: DataFrame with earnings dates

    Returns:
        Series of earnings dates indexed by symbol
    """
    if events_df.empty or 'earnings_date' not in events_df.columns:
        return pd.Series(dtype='datetime64[ns]')

    earnings_dates = events_df.set_index('symbol')['earnings_date']
    return pd.to_datetime(earnings_dates)


def detect_major_gaps(daily_df: pd.DataFrame, gap_threshold: float = 0.05) -> pd.DataFrame:
    """
    Detect major price gaps (up or down)

    Args:
        daily_df: Daily OHLCV data
        gap_threshold: Minimum gap size as percentage

    Returns:
        DataFrame with gap events
    """
    df = daily_df.copy()

    # Calculate gap from previous close to current open
    df['prev_close'] = df.groupby('Symbol')['Close'].shift(1)
    df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']

    # Major gaps
    major_gaps = df[abs(df['gap_pct']) >= gap_threshold].copy()
    major_gaps['gap_type'] = np.where(major_gaps['gap_pct'] > 0, 'gap_up', 'gap_down')
    major_gaps['gap_size'] = abs(major_gaps['gap_pct'])

    return major_gaps[['Symbol', 'Date', 'gap_type', 'gap_size', 'gap_pct']]


def detect_swing_points(daily_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Detect 52-week swing highs and lows

    Args:
        daily_df: Daily OHLCV data
        window: Lookback window (default 252 trading days ≈ 52 weeks)

    Returns:
        DataFrame with swing points
    """
    df = daily_df.copy()

    # 52-week high/low (rolling 252 days)
    df['52w_high'] = df.groupby('Symbol')['High'].rolling(window).max()
    df['52w_low'] = df.groupby('Symbol')['Low'].rolling(window).min()

    # Detect new highs/lows
    df['is_52w_high'] = df['High'] == df['52w_high']
    df['is_52w_low'] = df['Low'] == df['52w_low']

    swing_points = df[df['is_52w_high'] | df['is_52w_low']].copy()
    swing_points['swing_type'] = np.where(swing_points['is_52w_high'], '52w_high', '52w_low')

    return swing_points[['Symbol', 'Date', 'swing_type', 'High', 'Low']]


def detect_high_volume_days(daily_df: pd.DataFrame, volume_threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect high volume accumulation/distribution days

    Args:
        daily_df: Daily OHLCV data
        volume_threshold: Volume multiple above average

    Returns:
        DataFrame with high volume days
    """
    df = daily_df.copy()

    # Calculate rolling volume average
    df['vol_avg_20'] = df.groupby('Symbol')['Volume'].rolling(20).mean()
    df['vol_multiple'] = df['Volume'] / df['vol_avg_20']

    # High volume days
    high_vol = df[df['vol_multiple'] >= volume_threshold].copy()
    high_vol['vol_type'] = 'high_volume'

    return high_vol[['Symbol', 'Date', 'vol_type', 'Volume', 'vol_multiple']]


def compute_anchored_vwap(daily_df: pd.DataFrame, anchor_date: pd.Timestamp,
                         window_days: int = 60) -> float:
    """
    Compute VWAP anchored to a specific event date

    Args:
        daily_df: Daily OHLCV data for a single symbol
        anchor_date: Date to anchor VWAP calculation
        window_days: Number of days to include in VWAP (forward from anchor)

    Returns:
        Anchored VWAP price
    """
    # Filter data from anchor date onwards
    mask = (daily_df['Date'] >= anchor_date)
    window_data = daily_df[mask].head(window_days)

    if window_data.empty:
        return np.nan

    # Calculate VWAP: sum(price * volume) / sum(volume)
    price_volume = window_data['Close'] * window_data['Volume']
    total_volume = window_data['Volume'].sum()

    if total_volume == 0:
        return np.nan

    vwap = price_volume.sum() / total_volume
    return vwap


def get_event_anchors(symbol: str, daily_df: pd.DataFrame,
                     events_df: Optional[pd.DataFrame] = None,
                     max_anchors: int = 3) -> List[Dict]:
    """
    Get significant event anchors for a symbol

    Args:
        symbol: Stock symbol
        daily_df: Daily data for the symbol
        events_df: Events data (earnings, etc.)
        max_anchors: Maximum number of anchors to return

    Returns:
        List of anchor dictionaries with date, type, and price
    """
    anchors = []

    # Filter data for this symbol
    symbol_data = daily_df[daily_df['Symbol'] == symbol].copy()
    if symbol_data.empty:
        return anchors

    symbol_data = symbol_data.sort_values('Date')

    # 1. Earnings dates
    if events_df is not None:
        earnings_dates = detect_earnings_dates(events_df)
        if symbol in earnings_dates.index:
            earnings_date = earnings_dates[symbol]
            anchors.append({
                'date': earnings_date,
                'type': 'earnings',
                'price': None  # Will be computed as VWAP
            })

    # 2. Major gaps
    gaps = detect_major_gaps(symbol_data)
    for _, gap in gaps.tail(max_anchors).iterrows():  # Most recent gaps
        anchors.append({
            'date': gap['Date'],
            'type': gap['gap_type'],
            'price': symbol_data.loc[symbol_data['Date'] == gap['Date'], 'Close'].iloc[0]
        })

    # 3. 52-week swing points
    swings = detect_swing_points(symbol_data)
    for _, swing in swings.tail(max_anchors).iterrows():  # Most recent swings
        anchors.append({
            'date': swing['Date'],
            'type': swing['swing_type'],
            'price': swing['High'] if swing['swing_type'] == '52w_high' else swing['Low']
        })

    # 4. High volume days
    high_vol = detect_high_volume_days(symbol_data)
    for _, vol_day in high_vol.tail(max_anchors).iterrows():  # Most recent high vol days
        anchors.append({
            'date': vol_day['Date'],
            'type': 'high_volume',
            'price': symbol_data.loc[symbol_data['Date'] == vol_day['Date'], 'Close'].iloc[0]
        })

    # Sort by date (most recent first) and limit
    anchors.sort(key=lambda x: x['date'], reverse=True)
    return anchors[:max_anchors]


def compute_event_anchored_avwap(symbol: str, daily_df: pd.DataFrame,
                                events_df: Optional[pd.DataFrame] = None,
                                window_days: int = 60) -> Dict:
    """
    Compute event-anchored AVWAP for a symbol

    Args:
        symbol: Stock symbol
        daily_df: Daily OHLCV data
        events_df: Events data
        window_days: VWAP window in days

    Returns:
        Dict with anchor information and AVWAP prices
    """
    anchors = get_event_anchors(symbol, daily_df, events_df)

    if not anchors:
        return {'symbol': symbol, 'anchors': [], 'avwap_prices': []}

    avwap_prices = []
    symbol_data = daily_df[daily_df['Symbol'] == symbol]

    for anchor in anchors:
        if anchor['date'] in symbol_data['Date'].values:
            vwap = compute_anchored_vwap(symbol_data, anchor['date'], window_days)
            if not np.isnan(vwap):
                anchor['avwap'] = vwap
                avwap_prices.append(vwap)

    return {
        'symbol': symbol,
        'anchors': anchors,
        'avwap_prices': avwap_prices,
        'stack_check': len([p for p in avwap_prices if symbol_data['Close'].iloc[-1] > p]) >= 2
    }


def get_avwap_reclaim_signal(daily_df_or_symbol, avwap_column_or_daily_df=None,
                           events_df: Optional[pd.DataFrame] = None,
                           min_anchors: int = 2):
    """
    Check if price has reclaimed AVWAP

    Args:
        daily_df_or_symbol: Either DataFrame (for simple reclaim) or symbol string (for event-based)
        avwap_column_or_daily_df: Either AVWAP column name (for simple) or daily_df (for event-based)
        events_df: Events data for event-based reclaim
        min_anchors: Minimum anchors for event-based reclaim

    Returns:
        Series of booleans (for simple) or Dict (for event-based)
    """
    # Simple reclaim: DataFrame + column name
    if isinstance(daily_df_or_symbol, pd.DataFrame) and isinstance(avwap_column_or_daily_df, str):
        daily_df = daily_df_or_symbol
        avwap_column = avwap_column_or_daily_df
        return daily_df['Close'] > daily_df[avwap_column]

    # Event-based reclaim: symbol + DataFrame
    elif isinstance(daily_df_or_symbol, str) and isinstance(avwap_column_or_daily_df, pd.DataFrame):
        symbol = daily_df_or_symbol
        daily_df = avwap_column_or_daily_df

        anchor_data = compute_event_anchored_avwap(symbol, daily_df, events_df)

        if not anchor_data['avwap_prices']:
            return {'signal': False, 'reason': 'no_anchors', 'anchors': []}

        current_price = daily_df[daily_df['Symbol'] == symbol]['Close'].iloc[-1]
        reclaimed_anchors = [p for p in anchor_data['avwap_prices'] if current_price > p]

        signal = len(reclaimed_anchors) >= min_anchors

        return {
            'signal': signal,
            'current_price': current_price,
            'reclaimed_count': len(reclaimed_anchors),
            'total_anchors': len(anchor_data['avwap_prices']),
            'anchors': anchor_data['anchors'],
            'reason': 'insufficient_reclaim' if not signal else 'valid_reclaim'
        }

    else:
        raise ValueError("Invalid arguments for get_avwap_reclaim_signal")