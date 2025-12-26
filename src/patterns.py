"""
Weekly Candlestick Pattern Recognition
Evaluates patterns on last completed weekly candle (no repainting)
"""

import pandas as pd
import numpy as np


def is_hammer(open_price: float, high: float, low: float, close: float,
              body_ratio_threshold: float = 0.3, shadow_ratio_threshold: float = 2.0) -> bool:
    """
    Hammer pattern: small body, long lower shadow, little/no upper shadow

    Args:
        open_price, high, low, close: OHLC prices
        body_ratio_threshold: max body size as % of total range
        shadow_ratio_threshold: min lower shadow vs body ratio

    Returns:
        True if hammer pattern
    """
    body = abs(close - open_price)
    total_range = high - low
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    if total_range == 0:
        return False

    body_ratio = body / total_range
    shadow_ratio = lower_shadow / body if body > 0 else float('inf')

    return (body_ratio <= body_ratio_threshold and
            shadow_ratio >= shadow_ratio_threshold and
            upper_shadow <= body)  # minimal upper shadow


def is_bullish_engulfing(prev_open: float, prev_close: float,
                        curr_open: float, curr_close: float) -> bool:
    """
    Bullish engulfing: current candle completely engulfs previous bearish candle

    Args:
        prev_open, prev_close: previous candle
        curr_open, curr_close: current candle

    Returns:
        True if bullish engulfing pattern
    """
    # Previous candle must be bearish
    if prev_close >= prev_open:
        return False

    # Current candle must be bullish
    if curr_close <= curr_open:
        return False

    # Current body must engulf previous body
    prev_body_high = max(prev_open, prev_close)
    prev_body_low = min(prev_open, prev_close)

    return curr_close >= prev_body_high and curr_open <= prev_body_low


def is_morning_star(open3: float, close3: float, open2: float, close2: float,
                   open1: float, close1: float, star_body_ratio: float = 0.1) -> bool:
    """
    Morning star: three-candle reversal pattern

    Args:
        open3, close3: first candle (bearish)
        open2, close2: star candle (small body)
        open1, close1: third candle (bullish)
        star_body_ratio: max body size for star as % of total range

    Returns:
        True if morning star pattern
    """
    # First candle bearish
    if close3 >= open3:
        return False

    # Third candle bullish
    if close1 <= open1:
        return False

    # Star candle: small body, gaps down from first, gaps up from third
    star_body = abs(close2 - open2)
    star_range = max(open2, close2, open3, close3, open1, close1) - min(open2, close2, open3, close3, open1, close1)

    if star_range == 0:
        return False

    # Star gaps down from first candle
    first_low = min(open3, close3)
    star_gaps_down = max(open2, close2) < first_low

    # Star gaps up from third candle
    third_high = max(open1, close1)
    star_gaps_up = min(open2, close2) > third_high

    return (star_body / star_range <= star_body_ratio and
            star_gaps_down and star_gaps_up)


def is_three_white_soldiers(open3: float, close3: float, open2: float, close2: float,
                           open1: float, close1: float, overlap_threshold: float = 0.1) -> bool:
    """
    Three white soldiers: three consecutive bullish candles with progressive highs/lows

    Args:
        open3, close3: first soldier
        open2, close2: second soldier
        open1, close1: third soldier
        overlap_threshold: max allowed overlap between consecutive candles

    Returns:
        True if three white soldiers pattern
    """
    # All candles must be bullish
    if not (close3 > open3 and close2 > open2 and close1 > open1):
        return False

    # Progressive highs: each close higher than previous
    progressive_highs = close1 > close2 > close3

    # Progressive lows: each open higher than previous open (with small overlap allowed)
    overlap1 = abs(open2 - close3) / close3
    overlap2 = abs(open1 - close2) / close2

    progressive_lows = (open2 >= open3 * (1 - overlap_threshold) and
                       open1 >= open2 * (1 - overlap_threshold))

    return progressive_highs and progressive_lows


def is_piercing(prev_open: float, prev_close: float,
               curr_open: float, curr_close: float, penetration_threshold: float = 0.5) -> bool:
    """
    Piercing pattern: bullish candle opens below previous low, closes above previous midpoint

    Args:
        prev_open, prev_close: previous candle (bearish)
        curr_open, curr_close: current candle (bullish)
        penetration_threshold: minimum penetration into previous body

    Returns:
        True if piercing pattern
    """
    # Previous candle must be bearish
    if prev_close >= prev_open:
        return False

    # Current candle must be bullish
    if curr_close <= curr_open:
        return False

    # Current open below previous low
    prev_low = min(prev_open, prev_close)
    opens_below_prev_low = curr_open < prev_low

    # Current close penetrates at least halfway into previous body
    prev_body_mid = (prev_open + prev_close) / 2
    penetration = (curr_close - prev_low) / (prev_body_mid - prev_low) if prev_body_mid > prev_low else 0

    return opens_below_prev_low and penetration >= penetration_threshold


def detect_weekly_patterns_simple(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect weekly patterns on the last completed weekly candle

    Args:
        weekly_df: DataFrame with weekly OHLCV data, sorted by date

    Returns:
        DataFrame with pattern detection columns added
    """
    df = weekly_df.copy()

    # Initialize pattern columns
    df['is_hammer'] = False
    df['is_bullish_engulfing'] = False
    df['is_morning_star'] = False
    df['is_three_white_soldiers'] = False
    df['is_piercing'] = False

    # Need at least some candles for multi-candle patterns
    if len(df) < 3:
        return df

    for i in range(len(df)):
        # Single candle patterns
        if i < len(df):
            row = df.iloc[i]
            df.loc[df.index[i], 'is_hammer'] = is_hammer(
                row['Open'], row['High'], row['Low'], row['Close']
            )

        # Two-candle patterns
        if i >= 1:
            prev_row = df.iloc[i-1]
            curr_row = df.iloc[i]

            df.loc[df.index[i], 'is_bullish_engulfing'] = is_bullish_engulfing(
                prev_row['Open'], prev_row['Close'],
                curr_row['Open'], curr_row['Close']
            )

            df.loc[df.index[i], 'is_piercing'] = is_piercing(
                prev_row['Open'], prev_row['Close'],
                curr_row['Open'], curr_row['Close']
            )

        # Three-candle patterns
        if i >= 2:
            row3 = df.iloc[i-2]  # First candle
            row2 = df.iloc[i-1]  # Star/confirmation candle
            row1 = df.iloc[i]    # Last candle

            df.loc[df.index[i], 'is_morning_star'] = is_morning_star(
                row3['Open'], row3['Close'],
                row2['Open'], row2['Close'],
                row1['Open'], row1['Close']
            )

            df.loc[df.index[i], 'is_three_white_soldiers'] = is_three_white_soldiers(
                row3['Open'], row3['Close'],
                row2['Open'], row2['Close'],
                row1['Open'], row1['Close']
            )

    # Classify patterns
    df['pattern_type'] = 'none'
    df['pattern_name'] = ''

    reversal_patterns = ['is_hammer', 'is_bullish_engulfing', 'is_morning_star', 'is_piercing']
    continuation_patterns = ['is_three_white_soldiers']

    for pattern in reversal_patterns:
        df.loc[df[pattern], 'pattern_type'] = 'reversal'
        df.loc[df[pattern], 'pattern_name'] = pattern.replace('is_', '').replace('_', ' ').title()

    for pattern in continuation_patterns:
        df.loc[df[pattern], 'pattern_type'] = 'continuation'
        df.loc[df[pattern], 'pattern_name'] = pattern.replace('is_', '').replace('_', ' ').title()

    return df


def detect_weekly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect weekly candlestick patterns for the given DataFrame

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with pattern detection columns
    """
    result_df = df.copy()

    # Initialize pattern columns
    patterns = ['hammer', 'bullish_engulfing', 'morning_star', 'three_white_soldiers', 'piercing']
    for pattern in patterns:
        result_df[pattern] = False

    # Group by symbol and detect patterns
    if 'Symbol' in result_df.columns:
        # Multi-symbol data
        for symbol in result_df['Symbol'].unique():
            symbol_mask = result_df['Symbol'] == symbol
            symbol_data = result_df[symbol_mask].copy()

            if len(symbol_data) < 3:  # Need at least 3 candles for some patterns
                continue

            # Process patterns for this symbol
            for i in range(len(symbol_data)):
                if i >= 2:  # Need previous candles
                    # Hammer
                    if is_hammer(
                        symbol_data.iloc[i]['Open'],
                        symbol_data.iloc[i]['High'],
                        symbol_data.iloc[i]['Low'],
                        symbol_data.iloc[i]['Close']
                    ):
                        result_df.loc[symbol_data.index[i], 'hammer'] = True

                    # Bullish Engulfing
                    if is_bullish_engulfing(
                        symbol_data.iloc[i-1]['Open'], symbol_data.iloc[i-1]['Close'],
                        symbol_data.iloc[i]['Open'], symbol_data.iloc[i]['Close']
                    ):
                        result_df.loc[symbol_data.index[i], 'bullish_engulfing'] = True

                    # Morning Star
                    if is_morning_star(
                        symbol_data.iloc[i-2]['Open'], symbol_data.iloc[i-2]['Close'],
                        symbol_data.iloc[i-1]['Open'], symbol_data.iloc[i-1]['Close'],
                        symbol_data.iloc[i]['Open'], symbol_data.iloc[i]['Close']
                    ):
                        result_df.loc[symbol_data.index[i], 'morning_star'] = True

                    # Three White Soldiers
                    if is_three_white_soldiers(
                        symbol_data.iloc[i-2]['Open'], symbol_data.iloc[i-2]['Close'],
                        symbol_data.iloc[i-1]['Open'], symbol_data.iloc[i-1]['Close'],
                        symbol_data.iloc[i]['Open'], symbol_data.iloc[i]['Close']
                    ):
                        result_df.loc[symbol_data.index[i], 'three_white_soldiers'] = True

                    # Piercing
                    if is_piercing(
                        symbol_data.iloc[i-1]['Open'], symbol_data.iloc[i-1]['Close'],
                        symbol_data.iloc[i]['Open'], symbol_data.iloc[i]['Close']
                    ):
                        result_df.loc[symbol_data.index[i], 'piercing'] = True
    else:
        # Single symbol data - use the simple detection
        result_df = detect_weekly_patterns_simple(result_df)

    return result_df


def get_last_weekly_pattern(weekly_df: pd.DataFrame) -> dict:
    """
    Get the pattern from the last completed weekly candle

    Args:
        weekly_df: DataFrame with weekly data and pattern columns

    Returns:
        dict with pattern information
    """
    if weekly_df.empty:
        return {'type': 'none', 'name': '', 'high': None, 'low': None}

    last_row = weekly_df.iloc[-1]

    return {
        'type': last_row.get('pattern_type', 'none'),
        'name': last_row.get('pattern_name', ''),
        'high': last_row['High'],
        'low': last_row['Low'],
        'close': last_row['Close'],
        'date': last_row.name if hasattr(last_row, 'name') else None
    }