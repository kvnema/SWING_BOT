"""Enhanced SEPA and VCP Strategies with Pattern Recognition."""
import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import compute_all_indicators
from ..patterns import detect_weekly_patterns


def signal(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Enhanced SEPA and VCP strategies with pattern confirmation

    Args:
        df: Daily stock data
        benchmark_df: Benchmark index data for RS calculations

    Returns:
        DataFrame with SEPA and VCP signals
    """
    # Compute all indicators
    d = compute_all_indicators(df, benchmark_df)

    # === SEPA Strategy (Trend Continuation with Tight Base) ===

    # Strong trend setup (Minervini-style)
    d['SEPA_Trend_OK'] = (
        (d['Close'] > d['EMA20']) &
        (d['EMA20'] > d['EMA50']) &
        (d['EMA50'] > d['EMA200']) &
        (d['RSI14'] > 50)  # Momentum confirmation
    )

    # Tight consolidation base
    bw_percentile = d['BB_BandWidth'].rolling(60).rank(pct=True)
    d['SEPA_Tight_Base'] = bw_percentile <= 0.2  # Bottom 20% of bandwidth

    # Volume dry-up in consolidation
    vol_ma = d['Volume'].rolling(20).mean()
    d['SEPA_Volume_Dry'] = d['Volume'] < (vol_ma * 0.8)

    # Breakout above resistance
    d['SEPA_Breakout'] = (
        (d['Close'] > d['DonchianH20']) &
        (d['Close'] > d['DonchianH55']) &
        (d['RVOL20'] >= 1.5)  # Volume confirmation
    )

    # Weekly pattern confirmation
    weekly_patterns = detect_weekly_patterns(d)
    bullish_pattern_cols = ['hammer', 'bullish_engulfing', 'morning_star', 'three_white_soldiers', 'piercing']
    d['SEPA_Weekly_Bullish'] = weekly_patterns[bullish_pattern_cols].any(axis=1)

    # Mansfield RS filter
    if 'Mansfield_RS' in d.columns:
        d['SEPA_RS_OK'] = d['Mansfield_RS'] > 1.1  # Strong RS for trend continuation
    else:
        d['SEPA_RS_OK'] = True

    # Primary SEPA signal
    d['SEPA_Flag'] = (
        d['SEPA_Trend_OK'] &
        d['SEPA_Tight_Base'] &
        d['SEPA_Volume_Dry'] &
        d['SEPA_Breakout'] &
        d['SEPA_Weekly_Bullish'] &
        d['SEPA_RS_OK']
    ).astype(int)

    # === VCP Strategy (Volume Climax Pattern) ===

    # Cup/handle formation (higher lows, contracting volatility)
    d['VCP_Higher_Lows'] = d['Low'] > d['Low'].shift(5).rolling(20).min()
    bb_slope = d['BB_BandWidth'].diff().rolling(10).mean()
    d['VCP_Contracting'] = bb_slope < 0

    # Volume climax followed by dry-up
    vol_climax = d['RVOL20'] > 2.0
    d['VCP_Volume_Dry'] = d['Volume'] < (vol_ma * 0.7)  # More restrictive

    # Handle formation (small pullback)
    d['VCP_Handle'] = (
        (d['Close'] < d['Close'].rolling(20).max() * 0.95) &  # Within 5% of high
        (d['Close'] > d['Close'].rolling(20).max() * 0.85)    # Not too deep
    )

    # Breakout from handle
    d['VCP_Breakout'] = (
        (d['Close'] > d['Close'].rolling(20).max()) &
        (d['RVOL20'] >= 1.8)  # Strong volume on breakout
    )

    # Weekly pattern confirmation
    d['VCP_Weekly_Bullish'] = weekly_patterns[bullish_pattern_cols].any(axis=1)

    # RS filter (can be moderate for VCP)
    if 'Mansfield_RS' in d.columns:
        d['VCP_RS_OK'] = d['Mansfield_RS'] > 0.9
    else:
        d['VCP_RS_OK'] = True

    # Primary VCP signal
    d['VCP_Flag'] = (
        d['VCP_Higher_Lows'] &
        d['VCP_Contracting'] &
        d['VCP_Volume_Dry'] &
        d['VCP_Handle'] &
        d['VCP_Breakout'] &
        d['VCP_Weekly_Bullish'] &
        d['VCP_RS_OK']
    ).astype(int)

    # High-quality signals
    d['SEPA_High_Quality'] = (
        d['SEPA_Flag'].astype(bool) &
        (bw_percentile <= 0.1) &  # Very tight base
        (d['RSI14'] > 60)  # Strong momentum
    ).astype(int)

    d['VCP_High_Quality'] = (
        d['VCP_Flag'].astype(bool) &
        vol_climax.shift(5).rolling(10).max().astype(bool) &  # Volume climax in recent past
        (d['BB_BandWidth'] < d['BB_BandWidth'].rolling(40).quantile(0.1))  # Very tight
    ).astype(int)

    return d[[
        'Symbol', 'Date', 'SEPA_Flag', 'SEPA_High_Quality', 'VCP_Flag', 'VCP_High_Quality',
        'SEPA_Trend_OK', 'SEPA_Tight_Base', 'SEPA_Volume_Dry', 'SEPA_Breakout',
        'SEPA_Weekly_Bullish', 'SEPA_RS_OK',
        'VCP_Higher_Lows', 'VCP_Contracting', 'VCP_Volume_Dry', 'VCP_Handle',
        'VCP_Breakout', 'VCP_Weekly_Bullish', 'VCP_RS_OK'
    ]]


def get_sepa_entry_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get entry setup details for SEPA signals

    Args:
        df: DataFrame with signals

    Returns:
        DataFrame with entry details
    """
    entries = df[df['SEPA_Flag'] == 1].copy()

    if entries.empty:
        return pd.DataFrame()

    # Entry on breakout
    entries['entry_price'] = entries['Close']

    # Stop loss below breakout base
    entries['stop_loss'] = entries['DonchianL20'].rolling(20).min()

    # Target at next resistance or 2.5:1 RR
    risk = entries['entry_price'] - entries['stop_loss']
    entries['target_price'] = entries['entry_price'] + (risk * 2.5)
    entries['target_alt'] = entries['DonchianH55']

    return entries[[
        'entry_price', 'stop_loss', 'target_price', 'target_alt',
        'DonchianH20', 'DonchianL20', 'DonchianH55'
    ]]


def get_vcp_entry_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get entry setup details for VCP signals

    Args:
        df: DataFrame with signals

    Returns:
        DataFrame with entry details
    """
    entries = df[df['VCP_Flag'] == 1].copy()

    if entries.empty:
        return pd.DataFrame()

    # Entry on handle breakout
    entries['entry_price'] = entries['Close']

    # Stop loss below handle low
    entries['stop_loss'] = entries['Low'].rolling(20).min()

    # Target at pattern height projection
    pattern_high = entries['Close'].rolling(40).max()
    pattern_low = entries['Close'].rolling(40).min()
    pattern_height = pattern_high - pattern_low
    entries['target_price'] = entries['entry_price'] + pattern_height
    entries['target_alt'] = entries['DonchianH55']

    return entries[[
        'entry_price', 'stop_loss', 'target_price', 'target_alt',
        'Low', 'DonchianH55'
    ]]
