"""Dynamic Breakout II Strategy
Inspired by QuantConnect's 'Dynamic Breakout II Strategy'
Adapted for NSE stocks with adaptive thresholds and volatility-based breakout detection.
"""

import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import compute_all_indicators, dynamic_breakout_levels
from ..patterns import detect_weekly_patterns


def signal(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Dynamic breakout strategy with adaptive thresholds based on volatility and trend.

    Features:
    - Volatility-adjusted breakout levels
    - Trend confirmation requirements
    - Dynamic lookback periods
    - False breakout filters

    Args:
        df: Daily stock data
        benchmark_df: Benchmark index data for RS calculations

    Returns:
        DataFrame with breakout signals and components
    """
    # Compute all indicators
    d = compute_all_indicators(df, benchmark_df)

    # Dynamic lookback periods based on volatility
    volatility_regime = pd.cut(d['ATR20'] / d['Close'],
                              bins=[0, 0.015, 0.03, 0.06, 1],
                              labels=['Very_Low', 'Low', 'Medium', 'High'])

    lookback_map = {
        'Very_Low': 50,  # Longer lookback for stable markets
        'Low': 40,
        'Medium': 30,
        'High': 20       # Shorter lookback for volatile markets
    }

    d['Lookback_Period'] = volatility_regime.map(lookback_map).fillna(30)

    # Dynamic breakout levels
    for i, row in d.iterrows():
        lookback = int(row['Lookback_Period'])
        if i >= lookback:
            window_data = d.iloc[i-lookback:i+1]

            # Resistance levels (highest highs)
            d.loc[i, 'Resistance_Level'] = window_data['High'].max()
            d.loc[i, 'Resistance_Level_2'] = window_data['High'].nlargest(2).iloc[-1]

            # Support levels (lowest lows)
            d.loc[i, 'Support_Level'] = window_data['Low'].min()
            d.loc[i, 'Support_Level_2'] = window_data['Low'].nsmallest(2).iloc[-1]

            # Mid-range breakout levels
            d.loc[i, 'Mid_High'] = window_data['High'].quantile(0.75)
            d.loc[i, 'Mid_Low'] = window_data['Low'].quantile(0.25)

    # Fill NaN values with current close
    d['Resistance_Level'] = d['Resistance_Level'].fillna(d['Close'])
    d['Resistance_Level_2'] = d['Resistance_Level_2'].fillna(d['Close'])
    d['Support_Level'] = d['Support_Level'].fillna(d['Close'])
    d['Support_Level_2'] = d['Support_Level_2'].fillna(d['Close'])
    d['Mid_High'] = d['Mid_High'].fillna(d['Close'])
    d['Mid_Low'] = d['Mid_Low'].fillna(d['Close'])

    # Breakout signals with trend confirmation
    d['Bullish_Breakout'] = (
        (d['Close'] > d['Resistance_Level']) &
        (d['Close'] > d['Resistance_Level_2']) &
        (d['Close'] > d['Mid_High'])
    )

    d['Bearish_Breakout'] = (
        (d['Close'] < d['Support_Level']) &
        (d['Close'] < d['Support_Level_2']) &
        (d['Close'] < d['Mid_Low'])
    )

    # Trend confirmation (require uptrend for bullish breakouts)
    d['Trend_Up'] = d['Close'] > d['EMA20']
    d['Strong_Trend'] = (d['Close'] > d['EMA20']) & (d['EMA20'] > d['EMA50'])

    # Volume confirmation
    d['Volume_Breakout'] = d['Volume'] > d['Volume_MA20'] * 1.2

    # False breakout filters
    # Check if price returns within breakout level within next few days
    d['False_Breakout_Filter'] = True  # Default to true

    for i in range(len(d)):
        if i < len(d) - 3:  # Need at least 3 days ahead
            if d.iloc[i]['Bullish_Breakout']:
                # Check if price drops back below resistance within 3 days
                future_prices = d.iloc[i+1:i+4]['Low']
                resistance = d.iloc[i]['Resistance_Level']
                if (future_prices < resistance).any():
                    d.loc[d.index[i], 'False_Breakout_Filter'] = False

            elif d.iloc[i]['Bearish_Breakout']:
                # Check if price rises back above support within 3 days
                future_prices = d.iloc[i+1:i+4]['High']
                support = d.iloc[i]['Support_Level']
                if (future_prices > support).any():
                    d.loc[d.index[i], 'False_Breakout_Filter'] = False

    # Volatility-adjusted breakout strength
    d['Breakout_Strength'] = np.where(
        d['Bullish_Breakout'],
        (d['Close'] - d['Resistance_Level']) / d['ATR20'],
        np.where(
            d['Bearish_Breakout'],
            (d['Support_Level'] - d['Close']) / d['ATR20'],
            0
        )
    )

    # Final signal with all filters
    d['Signal'] = np.where(
        (d['Bullish_Breakout'] & d['Trend_Up'] & d['Volume_Breakout'] & d['False_Breakout_Filter']), 1,
        np.where(
            (d['Bearish_Breakout'] & ~d['Trend_Up'] & d['Volume_Breakout'] & d['False_Breakout_Filter']), -1,
            0
        )
    )

    # Confidence based on breakout strength and trend
    d['Confidence'] = np.where(
        d['Signal'] != 0,
        (d['Breakout_Strength'].abs() * 0.4 +
         (d['Trend_Strength'] / 100) * 0.3 +
         (d['Volume_Breakout'].astype(int)) * 0.3).clip(0, 1),
        0
    )

    return d[['Signal', 'Confidence', 'Breakout_Strength', 'Resistance_Level',
              'Support_Level', 'False_Breakout_Filter', 'Lookback_Period']]