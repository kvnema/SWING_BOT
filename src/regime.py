"""
Regime and Breadth Analysis
Computes market regime indicators and breadth measures
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def compute_index_regime(index_df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None,
                        ma_period: int = 200, rs_period: int = 60) -> pd.DataFrame:
    """
    Compute index regime indicators

    Args:
        index_df: Index daily data (NIFTY 50)
        benchmark_df: Benchmark index data (optional)
        ma_period: Moving average period for trend
        rs_period: Period for relative strength calculation

    Returns:
        DataFrame with regime indicators
    """
    df = index_df.copy()

    # Trend: Close > 200DMA
    df['SMA200'] = df['Close'].rolling(ma_period).mean()
    df['trend_up'] = df['Close'] > df['SMA200']

    # Relative Strength vs benchmark (if provided)
    if benchmark_df is not None and not benchmark_df.empty:
        # Merge benchmark data
        merged = df.merge(benchmark_df, left_index=True, right_index=True,
                         suffixes=('', '_bench'), how='left')

        # RS ratio
        merged['rs_ratio'] = merged['Close'] / merged['Close_bench']

        # RS slope (momentum)
        merged['rs_slope'] = merged['rs_ratio'].rolling(rs_period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )

        df['rs_rising'] = merged['rs_slope'] > 0
    else:
        df['rs_rising'] = True  # Default to True if no benchmark

    # Composite regime
    df['index_up_regime'] = df['trend_up'] & df['rs_rising']

    return df[['trend_up', 'rs_rising', 'index_up_regime', 'SMA200']]


def compute_market_breadth(stocks_df: pd.DataFrame, index_df: pd.DataFrame,
                          ma_period: int = 50, breadth_threshold: float = 0.6) -> pd.DataFrame:
    """
    Compute market breadth indicators

    Args:
        stocks_df: Individual stocks data
        index_df: Index data
        ma_period: MA period for breadth calculation
        breadth_threshold: Minimum breadth ratio for healthy market

    Returns:
        DataFrame with breadth indicators
    """
    # Group stocks by date and count above MA50
    breadth_daily = stocks_df.groupby('Date').agg({
        'Symbol': 'count',  # Total stocks
        'Close': lambda x: (x > stocks_df.loc[x.index, 'EMA50']).sum()  # Above EMA50
    }).rename(columns={'Close': 'above_ma50', 'Symbol': 'total_stocks'})

    breadth_daily['breadth_ratio'] = breadth_daily['above_ma50'] / breadth_daily['total_stocks']
    breadth_daily['breadth_healthy'] = breadth_daily['breadth_ratio'] >= breadth_threshold

    # Smooth breadth with MA
    breadth_daily['breadth_ratio_smooth'] = breadth_daily['breadth_ratio'].rolling(ma_period).mean()
    breadth_daily['breadth_healthy_smooth'] = breadth_daily['breadth_ratio_smooth'] >= breadth_threshold

    return breadth_daily[['breadth_ratio', 'breadth_healthy', 'breadth_ratio_smooth', 'breadth_healthy_smooth']]


def compute_volatility_regime(index_df: pd.DataFrame, atr_period: int = 14,
                             vol_threshold: float = 0.25) -> pd.DataFrame:
    """
    Compute volatility regime indicators

    Args:
        index_df: Index data with OHLC
        atr_period: ATR calculation period
        vol_threshold: Volatility threshold as fraction

    Returns:
        DataFrame with volatility regime
    """
    df = index_df.copy()

    # Calculate ATR
    high = df['High']
    low = df['Low']
    close = df['Close']

    df['tr'] = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )

    df['atr'] = df['tr'].rolling(atr_period).mean()
    df['atr_pct'] = df['atr'] / df['Close']

    # Volatility regimes
    df['low_vol_regime'] = df['atr_pct'] <= vol_threshold
    df['high_vol_regime'] = df['atr_pct'] > vol_threshold * 2

    return df[['atr', 'atr_pct', 'low_vol_regime', 'high_vol_regime']]


def get_composite_regime(index_df: pd.DataFrame, stocks_df: pd.DataFrame,
                        benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute composite regime score combining trend, breadth, and volatility

    Args:
        index_df: Index daily data
        stocks_df: Individual stocks data
        benchmark_df: Benchmark data

    Returns:
        DataFrame with composite regime indicators
    """
    # Index regime
    index_regime = compute_index_regime(index_df, benchmark_df)

    # Breadth
    breadth = compute_market_breadth(stocks_df, index_df)

    # Volatility
    vol_regime = compute_volatility_regime(index_df)

    # Combine all regime indicators
    regime_df = index_regime.join(breadth, how='left').join(vol_regime, how='left')

    # Composite regime score (0-1 scale)
    regime_df['regime_score'] = (
        regime_df['index_up_regime'].fillna(False).astype(int) * 0.4 +
        regime_df['breadth_healthy_smooth'].fillna(False).astype(int) * 0.4 +
        regime_df['low_vol_regime'].fillna(False).astype(int) * 0.2
    )

    # Regime categories
    regime_df['regime_category'] = pd.cut(
        regime_df['regime_score'],
        bins=[-0.1, 0.3, 0.7, 1.1],
        labels=['bear', 'neutral', 'bull']
    )

    # Final go/no-go signal
    regime_df['regime_go'] = (
        regime_df['index_up_regime'].fillna(False) &
        regime_df['breadth_healthy_smooth'].fillna(False) &
        ~regime_df['high_vol_regime'].fillna(True)  # Avoid high volatility periods
    )

    return regime_df


def get_current_regime_status(regime_df: pd.DataFrame) -> Dict:
    """
    Get current regime status summary

    Args:
        regime_df: DataFrame with regime indicators

    Returns:
        Dict with current regime status
    """
    if regime_df.empty:
        return {'status': 'unknown', 'score': 0, 'go': False}

    latest = regime_df.iloc[-1]

    return {
        'status': latest.get('regime_category', 'unknown'),
        'score': latest.get('regime_score', 0),
        'go': latest.get('regime_go', False),
        'trend_up': latest.get('index_up_regime', False),
        'breadth_healthy': latest.get('breadth_healthy_smooth', False),
        'low_volatility': latest.get('low_vol_regime', False),
        'high_volatility': latest.get('high_vol_regime', False),
        'breadth_ratio': latest.get('breadth_ratio_smooth', 0),
        'atr_pct': latest.get('atr_pct', 0)
    }


def apply_regime_filter(signals_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply regime filter to trading signals

    Args:
        signals_df: DataFrame with trading signals
        regime_df: DataFrame with regime indicators

    Returns:
        Filtered signals DataFrame
    """
    # Merge regime data
    merged = signals_df.merge(
        regime_df[['regime_go', 'regime_score', 'regime_category']],
        left_on='date',
        right_index=True,
        how='left'
    )

    # Apply filter
    merged['regime_filtered'] = merged.get('regime_go', True)  # Default to True if no regime data

    return merged


def detect_earnings_blackout(signals_df: pd.DataFrame, events_df: pd.DataFrame,
                           blackout_days: int = 2) -> pd.DataFrame:
    """
    Flag signals that occur during earnings blackout period

    Args:
        signals_df: Trading signals
        events_df: Earnings events
        blackout_days: Days to blackout before/after earnings

    Returns:
        DataFrame with blackout flags
    """
    df = signals_df.copy()
    df['earnings_blackout'] = False

    if events_df.empty or 'earnings_date' not in events_df.columns:
        return df

    # Convert earnings dates to datetime
    events_df = events_df.copy()
    events_df['earnings_date'] = pd.to_datetime(events_df['earnings_date'])

    for _, signal in df.iterrows():
        symbol = signal.get('ticker', signal.get('Symbol'))
        signal_date = pd.to_datetime(signal['date'])

        # Find earnings dates for this symbol
        symbol_earnings = events_df[
            (events_df['symbol'] == symbol) &
            (events_df['earnings_date'].notna())
        ]

        for _, earnings in symbol_earnings.iterrows():
            earnings_date = earnings['earnings_date']
            days_diff = abs((signal_date - earnings_date).days)

            if days_diff <= blackout_days:
                df.loc[df.index == signal.name, 'earnings_blackout'] = True
                break

    return df