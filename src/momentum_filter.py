"""
Momentum Stock Filter
Filters stocks based on momentum criteria for swing trading focus
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def filter_momentum_stocks(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None,
                          min_momentum_period: int = 252, min_rs_rank: float = 0.7,
                          require_volume: bool = True) -> pd.DataFrame:
    """
    Filter stocks based on momentum criteria

    Args:
        df: DataFrame with stock data and indicators
        benchmark_df: Benchmark data for relative strength calculations
        min_momentum_period: Minimum lookback period for momentum (default 252 trading days ~ 1 year)
        min_rs_rank: Minimum relative strength percentile rank (default 70th percentile)
        require_volume: Whether to require minimum volume criteria

    Returns:
        Filtered DataFrame with only momentum stocks
    """

    if df.empty:
        logger.warning("Empty dataframe provided to momentum filter")
        return df

    # Group by stock for filtering
    filtered_stocks = []

    for stock in df['Symbol'].unique():
        stock_data = df[df['Symbol'] == stock].copy()

        if len(stock_data) < min_momentum_period:
            continue  # Not enough data

        # Calculate momentum metrics
        latest_data = stock_data.iloc[-1:].copy()

        # 1. Time-series momentum (12-month return)
        ts_momentum = (latest_data['Close'].iloc[0] / stock_data['Close'].iloc[-min_momentum_period]) - 1

        # 2. Recent momentum (3-month return)
        recent_momentum = (latest_data['Close'].iloc[0] / stock_data['Close'].iloc[-63]) - 1

        # 3. Trend strength (EMA alignment)
        ema_alignment = (
            (latest_data['EMA20'].iloc[0] > latest_data['EMA50'].iloc[0]) and
            (latest_data['EMA50'].iloc[0] > latest_data['EMA200'].iloc[0])
        )

        # 4. RSI momentum (>50 indicates bullish momentum)
        rsi_momentum = latest_data['RSI14'].iloc[0] > 50

        # 5. MACD momentum (histogram rising)
        macd_momentum = latest_data['MACDHist'].iloc[0] > latest_data['MACDHist'].shift(1).iloc[0] if len(latest_data) > 1 else True

        # 6. Relative strength vs benchmark (if available)
        rs_ok = True
        if 'RS_vs_Index' in latest_data.columns:
            rs_value = latest_data['RS_vs_Index'].iloc[0]
            rs_ok = rs_value > 1.0  # Outperforming benchmark

        # 7. Volume criteria
        volume_ok = True
        if require_volume:
            avg_volume = stock_data['Volume'].tail(20).mean()
            volume_ok = avg_volume > 100000  # Minimum average volume

        # Composite momentum score
        momentum_score = 0
        momentum_score += 1 if ts_momentum > 0.1 else 0  # 10%+ 1-year return
        momentum_score += 1 if recent_momentum > 0.05 else 0  # 5%+ 3-month return
        momentum_score += 1 if ema_alignment else 0  # Strong trend
        momentum_score += 1 if rsi_momentum else 0  # RSI > 50
        momentum_score += 1 if macd_momentum else 0  # MACD rising
        momentum_score += 1 if rs_ok else 0  # RS > 1
        momentum_score += 1 if volume_ok else 0  # Volume OK

        # Must pass minimum criteria for momentum stock
        is_momentum_stock = (
            ts_momentum > 0 and  # Positive 1-year momentum
            recent_momentum > 0 and  # Positive recent momentum
            ema_alignment and  # Strong trend alignment
            rsi_momentum and  # Bullish RSI
            momentum_score >= 4  # At least 4 out of 7 criteria
        )

        if is_momentum_stock:
            # Add momentum metadata
            stock_data['momentum_score'] = momentum_score
            stock_data['ts_momentum'] = ts_momentum
            stock_data['recent_momentum'] = recent_momentum
            stock_data['is_momentum_stock'] = True
            filtered_stocks.append(stock_data)

    if filtered_stocks:
        result_df = pd.concat(filtered_stocks, ignore_index=True)
        momentum_stocks = result_df['Symbol'].unique()
        logger.info(f"Filtered to {len(momentum_stocks)} momentum stocks out of {len(df['Symbol'].unique())} total stocks")
        return result_df
    else:
        logger.warning("No momentum stocks found matching criteria")
        return pd.DataFrame()


def get_momentum_rankings(df: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Rank stocks by momentum strength

    Args:
        df: DataFrame with momentum-filtered stocks
        top_n: Number of top momentum stocks to return (None for all)

    Returns:
        Ranked DataFrame of momentum stocks
    """

    if df.empty:
        return df

    # Get latest data for each stock
    latest_data = []
    for stock in df['Symbol'].unique():
        stock_data = df[df['Symbol'] == stock]
        latest_row = stock_data.iloc[-1:].copy()

        # Calculate composite momentum score
        ts_momentum = latest_row['ts_momentum'].iloc[0] if 'ts_momentum' in latest_row.columns else 0
        recent_momentum = latest_row['recent_momentum'].iloc[0] if 'recent_momentum' in latest_row.columns else 0
        rsi_score = (latest_row['RSI14'].iloc[0] - 50) / 50  # Normalize RSI around 50
        volume_score = np.log(latest_row['Volume'].iloc[0]) / 20  # Log volume score

        composite_score = (
            ts_momentum * 0.4 +  # 40% weight on long-term momentum
            recent_momentum * 0.3 +  # 30% weight on recent momentum
            rsi_score * 0.2 +  # 20% weight on RSI
            volume_score * 0.1  # 10% weight on volume
        )

        latest_row['momentum_rank_score'] = composite_score
        latest_data.append(latest_row)

    if latest_data:
        rankings_df = pd.concat(latest_data, ignore_index=True)
        rankings_df = rankings_df.sort_values('momentum_rank_score', ascending=False)

        if top_n:
            rankings_df = rankings_df.head(top_n)

        return rankings_df

    return pd.DataFrame()