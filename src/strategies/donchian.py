"""Enhanced Donchian Breakout Strategy with Pattern Confirmation and AVWAP Logic."""
import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import compute_all_indicators, enhanced_donchian, donchian_breakout_signal
from ..patterns import detect_weekly_patterns, get_last_weekly_pattern
from ..anchors import get_avwap_reclaim_signal


def signal(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None,
           short_window: int = 20, long_window: int = 55) -> pd.DataFrame:
    """
    Enhanced Donchian breakout strategy with pattern confirmation and AVWAP logic

    Args:
        df: Daily stock data
        benchmark_df: Benchmark index data for RS calculations
        short_window: Short Donchian period
        long_window: Long Donchian period

    Returns:
        DataFrame with Donchian signals and components
    """
    # Compute all indicators
    d = compute_all_indicators(df, benchmark_df)

    # Enhanced Donchian channels
    dh_short, dl_short, dm_short, dw_short = enhanced_donchian(d, short_window)
    dh_long, dl_long, dm_long, dw_long = enhanced_donchian(d, long_window)

    d['DonchianH_short'] = dh_short
    d['DonchianL_short'] = dl_short
    d['DonchianM_short'] = dm_short
    d['DonchianW_short'] = dw_short

    d['DonchianH_long'] = dh_long
    d['DonchianL_long'] = dl_long
    d['DonchianM_long'] = dm_long
    d['DonchianW_long'] = dw_long

    # Donchian breakout signals
    d['Donchian_Breakout_Short'] = donchian_breakout_signal(d, short_window)
    d['Donchian_Breakout_Long'] = donchian_breakout_signal(d, long_window)

    # Trend filters
    d['Short_Trend_Up'] = d['Close'] > d['EMA20']
    d['Long_Trend_Up'] = d['Close'] > d['EMA50']

    # Volume confirmation
    d['Volume_OK'] = d['RVOL20'] >= 1.2

    # Volatility filters
    d['Volatility_OK'] = ~d['high_vol']  # Prefer low volatility for breakouts

    # Mansfield RS filter (if benchmark available)
    if 'Mansfield_RS' in d.columns:
        d['RS_OK'] = d['Mansfield_RS'] > 1.0
    else:
        d['RS_OK'] = True

    # Weekly pattern confirmation
    weekly_patterns = detect_weekly_patterns(d)
    bullish_pattern_cols = ['hammer', 'bullish_engulfing', 'morning_star', 'three_white_soldiers', 'piercing']
    d['Weekly_Pattern_Bullish'] = weekly_patterns[bullish_pattern_cols].any(axis=1)

    # AVWAP reclaim signal (simple 60-day AVWAP)
    d['AVWAP_Reclaim'] = d['Close'] > d['AVWAP60']

    # Primary breakout signal: short Donchian breakout
    primary_breakout = (
        (d['Donchian_Breakout_Short'] == 1) &
        d['Short_Trend_Up'] &
        d['Volume_OK'] &
        d['Volatility_OK'] &
        d['RS_OK']
    )

    # Enhanced signal with pattern confirmation
    d['Donchian_Breakout'] = (
        primary_breakout &
        (d['Weekly_Pattern_Bullish'] | d['AVWAP_Reclaim'])
    ).astype(int)

    # Additional filters for signal quality
    d['Donchian_High_Quality'] = (
        d['Donchian_Breakout'].astype(bool) &
        (d['DonchianW_short'] > d['DonchianW_short'].rolling(20).mean()) &  # Expanding channel
        (d['Close'] > d['DonchianM_short'])  # Above midline
    ).astype(int)

    return d[[
        'Symbol', 'Date', 'Donchian_Breakout', 'Donchian_High_Quality',
        'DonchianH_short', 'DonchianL_short', 'DonchianM_short', 'DonchianW_short',
        'DonchianH_long', 'DonchianL_long', 'DonchianM_long', 'DonchianW_long',
        'Short_Trend_Up', 'Long_Trend_Up', 'Volume_OK', 'Volatility_OK', 'RS_OK',
        'Weekly_Pattern_Bullish', 'AVWAP_Reclaim'
    ]]


def get_entry_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get entry setup details for Donchian breakouts

    Args:
        df: DataFrame with signals

    Returns:
        DataFrame with entry details
    """
    entries = df[df['Donchian_Breakout'] == 1].copy()

    if entries.empty:
        return pd.DataFrame()

    # Entry on breakout above high
    entries['entry_price'] = entries['DonchianH_short']

    # Stop loss below recent low or channel low
    entries['stop_loss'] = np.minimum(
        entries['DonchianL_short'],
        entries['Low'].rolling(10).min()
    )

    # Target at longer channel high or 2:1 RR
    risk = entries['entry_price'] - entries['stop_loss']
    entries['target_price'] = entries['entry_price'] + (risk * 2)

    # Alternative target at long Donchian high
    entries['target_alt'] = entries['DonchianH_long']

    return entries[[
        'entry_price', 'stop_loss', 'target_price', 'target_alt',
        'DonchianH_short', 'DonchianL_short', 'DonchianH_long'
    ]]
