"""Enhanced BB/KC Squeeze Strategy with Expansion Breakouts."""
import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import compute_all_indicators, bb_inside_kc
from ..patterns import detect_weekly_patterns


def signal(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Enhanced squeeze strategy with expansion breakouts and pattern confirmation

    Args:
        df: Daily stock data
        benchmark_df: Benchmark index data for RS calculations

    Returns:
        DataFrame with squeeze signals and components
    """
    # Compute all indicators
    d = compute_all_indicators(df, benchmark_df)

    # Squeeze detection
    d['Squeeze_Active'] = bb_inside_kc(d)

    # Squeeze duration (consecutive squeeze days)
    d['Squeeze_Duration'] = d['Squeeze_Active'].groupby(
        (d['Squeeze_Active'] != d['Squeeze_Active'].shift()).cumsum()
    ).cumsum()

    # Breakout conditions
    d['Upper_Breakout'] = (d['Close'] > d['KC_Upper']) & (d['Close'].shift(1) <= d['KC_Upper'].shift(1))
    d['Lower_Breakout'] = (d['Close'] < d['KC_Lower']) & (d['Close'].shift(1) >= d['KC_Lower'].shift(1))

    # Volume confirmation on breakout
    d['Breakout_Volume'] = d['RVOL20'] >= 1.5

    # Volatility expansion (squeeze release)
    d['Volatility_Expansion'] = d['BB_BandWidth'] > d['BB_BandWidth'].shift(5)

    # Trend filters
    d['Trend_Up'] = d['Close'] > d['EMA20']
    d['Trend_Down'] = d['Close'] < d['EMA20']

    # Weekly pattern confirmation
    weekly_patterns = detect_weekly_patterns(d)
    bullish_pattern_cols = ['hammer', 'bullish_engulfing', 'morning_star', 'three_white_soldiers', 'piercing']
    d['Weekly_Pattern_Bullish'] = weekly_patterns[bullish_pattern_cols].any(axis=1)
    d['Weekly_Pattern_Bearish'] = False  # Placeholder for future bearish patterns

    # Mansfield RS filter (if benchmark available)
    if 'Mansfield_RS' in d.columns:
        d['RS_OK'] = d['Mansfield_RS'] > 0.9  # Slightly relaxed for squeeze
    else:
        d['RS_OK'] = True

    # Primary squeeze breakout signal (bullish)
    d['Squeeze_Breakout_Bullish'] = (
        d['Squeeze_Active'].shift(1) &  # Was in squeeze previous day
        d['Upper_Breakout'] &
        d['Breakout_Volume'] &
        d['Volatility_Expansion'] &
        d['Trend_Up'] &
        d['RS_OK'] &
        (d['Weekly_Pattern_Bullish'] | (d['Squeeze_Duration'] >= 5))  # Long squeeze or pattern
    ).astype(int)

    # Bearish squeeze breakout
    d['Squeeze_Breakout_Bearish'] = (
        d['Squeeze_Active'].shift(1) &
        d['Lower_Breakout'] &
        d['Breakout_Volume'] &
        d['Volatility_Expansion'] &
        d['Trend_Down'] &
        d['RS_OK'] &
        (d['Weekly_Pattern_Bearish'] | (d['Squeeze_Duration'] >= 5))
    ).astype(int)

    # Combined signal (legacy compatibility)
    d['BBKC_Squeeze_Flag'] = d['Squeeze_Breakout_Bullish']
    d['SqueezeBreakout_Flag'] = d['BBKC_Squeeze_Flag']

    # Signal quality indicators
    d['Squeeze_High_Quality'] = (
        d['Squeeze_Breakout_Bullish'].astype(bool) &
        (d['Squeeze_Duration'] >= 7) &  # Long consolidation
        (d['BB_BandWidth'] > d['BB_BandWidth'].rolling(20).quantile(0.8))  # Strong expansion
    ).astype(int)

    return d[[
        'Symbol', 'Date', 'BBKC_Squeeze_Flag', 'SqueezeBreakout_Flag',
        'Squeeze_Breakout_Bullish', 'Squeeze_Breakout_Bearish', 'Squeeze_High_Quality',
        'Squeeze_Active', 'Squeeze_Duration', 'Upper_Breakout', 'Lower_Breakout',
        'Breakout_Volume', 'Volatility_Expansion', 'Trend_Up', 'Trend_Down',
        'Weekly_Pattern_Bullish', 'Weekly_Pattern_Bearish', 'RS_OK'
    ]]


def get_entry_setup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get entry setup details for squeeze breakouts

    Args:
        df: DataFrame with signals

    Returns:
        DataFrame with entry details
    """
    # Bullish entries
    bullish_entries = df[df['Squeeze_Breakout_Bullish'] == 1].copy()

    if not bullish_entries.empty:
        # Entry on breakout above KC upper
        bullish_entries['entry_price'] = bullish_entries['KC_Upper']

        # Stop loss below squeeze low or KC lower
        bullish_entries['stop_loss'] = np.minimum(
            bullish_entries['KC_Lower'],
            bullish_entries['Low'].rolling(20).min()
        )

        # Target at BB upper or 3:1 RR for high-volatility breakouts
        risk = bullish_entries['entry_price'] - bullish_entries['stop_loss']
        bullish_entries['target_price'] = bullish_entries['entry_price'] + (risk * 3)
        bullish_entries['target_alt'] = bullish_entries['BB_Upper']
        bullish_entries['direction'] = 'long'

    # Bearish entries
    bearish_entries = df[df['Squeeze_Breakout_Bearish'] == 1].copy()

    if not bearish_entries.empty:
        # Entry on breakout below KC lower
        bearish_entries['entry_price'] = bearish_entries['KC_Lower']

        # Stop loss above squeeze high or KC upper
        bearish_entries['stop_loss'] = np.maximum(
            bearish_entries['KC_Upper'],
            bearish_entries['High'].rolling(20).max()
        )

        # Target at BB lower or 3:1 RR
        risk = bearish_entries['stop_loss'] - bearish_entries['entry_price']
        bearish_entries['target_price'] = bearish_entries['entry_price'] - (risk * 3)
        bearish_entries['target_alt'] = bearish_entries['BB_Lower']
        bearish_entries['direction'] = 'short'

    # Combine entries
    all_entries = pd.concat([bullish_entries, bearish_entries], ignore_index=True)

    if all_entries.empty:
        return pd.DataFrame()

    return all_entries[[
        'entry_price', 'stop_loss', 'target_price', 'target_alt', 'direction',
        'KC_Upper', 'KC_Lower', 'BB_Upper', 'BB_Lower', 'Squeeze_Duration'
    ]]
