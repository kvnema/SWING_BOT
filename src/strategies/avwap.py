"""AVWAP (Anchored VWAP) Strategy with Event-Based Anchors."""
import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import compute_all_indicators, avwap
from ..patterns import detect_weekly_patterns
from ..anchors import get_event_anchors, compute_event_anchored_avwap, get_avwap_reclaim_signal


def signal(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    AVWAP strategy based on event-anchored VWAP levels

    Args:
        df: Daily stock data
        benchmark_df: Benchmark index data for RS calculations

    Returns:
        DataFrame with AVWAP signals and components
    """
    # Compute all indicators
    d = compute_all_indicators(df, benchmark_df)

    # Get event anchors
    symbol = d['Symbol'].iloc[0] if not d.empty else 'UNKNOWN'
    event_anchors = get_event_anchors(symbol, d)

    # Compute event-anchored VWAP (simplified for now - use standard AVWAP)
    d['AVWAP_Earnings'] = d['AVWAP60']  # Placeholder
    d['AVWAP_Gaps'] = d['AVWAP60']      # Placeholder
    d['AVWAP_Swings'] = d['AVWAP60']    # Placeholder
    d['AVWAP_HighVol'] = d['AVWAP60']   # Placeholder

    # Composite AVWAP (weighted average of all anchors)
    avwap_cols = ['AVWAP_Earnings', 'AVWAP_Gaps', 'AVWAP_Swings', 'AVWAP_HighVol']
    d['AVWAP_Composite'] = d[avwap_cols].mean(axis=1, skipna=True)

    # AVWAP reclaim signals
    d['AVWAP_Reclaim_Earnings'] = get_avwap_reclaim_signal(d, 'AVWAP_Earnings')
    d['AVWAP_Reclaim_Gaps'] = get_avwap_reclaim_signal(d, 'AVWAP_Gaps')
    d['AVWAP_Reclaim_Swings'] = get_avwap_reclaim_signal(d, 'AVWAP_Swings')
    d['AVWAP_Reclaim_HighVol'] = get_avwap_reclaim_signal(d, 'AVWAP_HighVol')
    d['AVWAP_Reclaim_Composite'] = get_avwap_reclaim_signal(d, 'AVWAP_Composite')

    # Trend filters
    d['Trend_Up'] = d['Close'] > d['EMA20']
    d['Strong_Trend'] = d['Close'] > d['EMA50']

    # Volume confirmation
    d['Volume_OK'] = d['RVOL20'] >= 1.1

    # Volatility filters (prefer moderate volatility)
    d['Volatility_OK'] = (d['ATR_pct'] > 1.0) & (d['ATR_pct'] < 3.0)

    # Mansfield RS filter
    if 'Mansfield_RS' in d.columns:
        d['RS_OK'] = d['Mansfield_RS'] > 1.0
    else:
        d['RS_OK'] = True

    # Weekly pattern confirmation
    weekly_patterns = detect_weekly_patterns(d)
    bullish_pattern_cols = ['hammer', 'bullish_engulfing', 'morning_star', 'three_white_soldiers', 'piercing']
    d['Weekly_Pattern_Bullish'] = weekly_patterns[bullish_pattern_cols].any(axis=1)

    # Primary AVWAP signal: reclaim of composite AVWAP
    d['AVWAP_Signal'] = (
        d['AVWAP_Reclaim_Composite'] &
        d['Trend_Up'] &
        d['Volume_OK'] &
        d['Volatility_OK'] &
        d['RS_OK'] &
        d['Weekly_Pattern_Bullish']
    ).astype(int)

    # High-quality signal: multiple anchor reclaims
    anchor_reclaims = [
        'AVWAP_Reclaim_Earnings', 'AVWAP_Reclaim_Gaps',
        'AVWAP_Reclaim_Swings', 'AVWAP_Reclaim_HighVol'
    ]
    d['Multiple_Reclaims'] = d[anchor_reclaims].sum(axis=1) >= 2

    d['AVWAP_High_Quality'] = (
        d['AVWAP_Signal'].astype(bool) &
        d['Multiple_Reclaims'] &
        d['Strong_Trend']
    ).astype(int)

    return d[[
        'Symbol', 'Date', 'AVWAP_Signal', 'AVWAP_High_Quality',
        'AVWAP_Earnings', 'AVWAP_Gaps', 'AVWAP_Swings', 'AVWAP_HighVol', 'AVWAP_Composite',
        'AVWAP_Reclaim_Earnings', 'AVWAP_Reclaim_Gaps', 'AVWAP_Reclaim_Swings',
        'AVWAP_Reclaim_HighVol', 'AVWAP_Reclaim_Composite', 'Multiple_Reclaims',
        'Trend_Up', 'Strong_Trend', 'Volume_OK', 'Volatility_OK', 'RS_OK',
        'Weekly_Pattern_Bullish'
    ]]


def get_entry_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get entry setup details for AVWAP signals

    Args:
        df: DataFrame with signals

    Returns:
        DataFrame with entry details
    """
    entries = df[df['AVWAP_Signal'] == 1].copy()

    if entries.empty:
        return pd.DataFrame()

    # Entry on AVWAP reclaim
    entries['entry_price'] = entries['Close']  # Entry at current close

    # Stop loss below recent low or AVWAP level
    entries['stop_loss'] = np.minimum(
        entries[['AVWAP_Earnings', 'AVWAP_Gaps', 'AVWAP_Swings', 'AVWAP_Composite']].min(axis=1),
        entries['Low'].rolling(10).min()
    )

    # Target at resistance levels or 2:1 RR
    risk = entries['entry_price'] - entries['stop_loss']
    entries['target_price'] = entries['entry_price'] + (risk * 2)

    # Alternative target at next resistance
    entries['target_alt'] = entries[['DonchianH20', 'DonchianH55']].max(axis=1)

    return entries[[
        'entry_price', 'stop_loss', 'target_price', 'target_alt',
        'AVWAP_Composite', 'DonchianH20', 'DonchianH55'
    ]]