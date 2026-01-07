"""Enhanced Momentum Strategy with Trend Strength Weighting
Inspired by QuantConnect's 'Improved Momentum Strategy on Commodities Futures'
Adapted for NSE stocks with dynamic leverage and trend strength weighting.
"""

import pandas as pd
import numpy as np
from typing import Optional
from ..indicators import compute_all_indicators, momentum_score, trend_strength
from ..patterns import detect_weekly_patterns


def signal(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Enhanced momentum strategy with trend strength weighting and dynamic leverage.

    Features:
    - Multi-timeframe momentum scoring
    - Trend strength weighting
    - Dynamic leverage based on volatility
    - Risk-adjusted position sizing

    Args:
        df: Daily stock data
        benchmark_df: Benchmark index data for RS calculations

    Returns:
        DataFrame with momentum signals and components
    """
    # Compute all indicators
    d = compute_all_indicators(df, benchmark_df)

    # Multi-timeframe momentum calculation
    d['Momentum_1M'] = momentum_score(d, window=21)   # ~1 month
    d['Momentum_3M'] = momentum_score(d, window=63)   # ~3 months
    d['Momentum_6M'] = momentum_score(d, window=126)  # ~6 months
    d['Momentum_12M'] = momentum_score(d, window=252) # ~12 months

    # Trend strength indicators
    d['Trend_Strength'] = trend_strength(d, window=20)
    d['ADX_Trend'] = d.get('ADX', 25)  # Use ADX if available, else default

    # Composite momentum score with trend weighting
    # Weight recent momentum more heavily, but require trend strength
    momentum_weights = {
        'Momentum_1M': 0.4,
        'Momentum_3M': 0.3,
        'Momentum_6M': 0.2,
        'Momentum_12M': 0.1
    }

    d['Momentum_Composite'] = (
        d['Momentum_1M'] * momentum_weights['Momentum_1M'] +
        d['Momentum_3M'] * momentum_weights['Momentum_3M'] +
        d['Momentum_6M'] * momentum_weights['Momentum_6M'] +
        d['Momentum_12M'] * momentum_weights['Momentum_12M']
    )

    # Trend-strength weighted momentum
    d['Momentum_Weighted'] = d['Momentum_Composite'] * (d['Trend_Strength'] / 50).clip(0.5, 2.0)

    # Dynamic leverage based on volatility and trend strength
    d['Volatility_Regime'] = pd.cut(d['ATR20'] / d['Close'], bins=[0, 0.02, 0.05, 1], labels=['Low', 'Medium', 'High'])
    d['Leverage_Factor'] = np.where(
        d['Volatility_Regime'] == 'Low', 1.2,
        np.where(d['Volatility_Regime'] == 'Medium', 1.0, 0.8)
    )

    # Trend strength bonus
    d['Leverage_Factor'] *= np.where(d['ADX_Trend'] > 25, 1.1, 0.9)

    # Risk-adjusted momentum signal
    d['Momentum_Signal'] = np.where(
        (d['Momentum_Weighted'] > 0.7) & (d['Trend_Strength'] > 30), 1,
        np.where((d['Momentum_Weighted'] < -0.7) & (d['Trend_Strength'] > 30), -1, 0)
    )

    # Position sizing based on momentum strength and volatility
    d['Position_Size'] = d['Leverage_Factor'] * (d['Momentum_Weighted'].abs() / 2).clip(0.1, 1.0)

    # Risk management filters
    d['Volume_Filter'] = d['RVOL20'] > 0.8  # Minimum volume
    d['Volatility_Filter'] = d['ATR20'] / d['Close'] < 0.08  # Maximum volatility

    # Final signal with filters
    d['Signal'] = np.where(
        (d['Momentum_Signal'] != 0) &
        d['Volume_Filter'] &
        d['Volatility_Filter'], d['Momentum_Signal'], 0
    )

    # Confidence score based on momentum strength and trend
    d['Confidence'] = (d['Momentum_Weighted'].abs() * 0.6 +
                      (d['Trend_Strength'] / 100) * 0.4).clip(0, 1)

    return d[['Signal', 'Confidence', 'Momentum_Weighted', 'Trend_Strength',
              'Leverage_Factor', 'Position_Size', 'Momentum_Composite']]