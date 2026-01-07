"""Enhanced Sector Momentum Strategy
Inspired by QuantConnect's 'Sector Momentum' strategy
Adapted for NSE with sector ETFs, momentum scoring, and dynamic allocation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from ..indicators import compute_all_indicators, momentum_score, relative_strength
from ..patterns import detect_weekly_patterns


# NSE Sector ETF mappings (simplified)
NSE_SECTOR_ETFS = {
    'NIFTY IT': ['INFY.NS', 'TCS.NS', 'WIPRO.NS', 'HCLTECH.NS'],
    'NIFTY BANK': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS'],
    'NIFTY PHARMA': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
    'NIFTY AUTO': ['MARUTI.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS'],
    'NIFTY FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'],
    'NIFTY METAL': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'COALINDIA.NS'],
    'NIFTY ENERGY': ['RELIANCE.NS', 'NTPC.NS', 'POWERGRID.NS', 'ONGC.NS']
}


def calculate_sector_momentum(df: pd.DataFrame, sector_stocks: List[str]) -> pd.DataFrame:
    """
    Calculate momentum for a sector based on constituent stocks.

    Args:
        df: DataFrame with stock data
        sector_stocks: List of stock symbols in the sector

    Returns:
        DataFrame with sector momentum metrics
    """
    sector_data = df[df['Symbol'].isin(sector_stocks)].copy()

    if sector_data.empty:
        return pd.DataFrame()

    # Group by date and calculate sector metrics
    sector_metrics = sector_data.groupby(sector_data.index).agg({
        'Close': ['mean', 'std'],
        'Volume': 'sum',
        'Returns': 'mean'
    }).fillna(method='ffill')

    # Flatten column names
    sector_metrics.columns = ['avg_close', 'volatility', 'total_volume', 'avg_returns']
    sector_metrics = sector_metrics.reset_index()

    # Calculate momentum scores
    sector_metrics['momentum_1m'] = momentum_score(sector_metrics, window=21, column='avg_close')
    sector_metrics['momentum_3m'] = momentum_score(sector_metrics, window=63, column='avg_close')
    sector_metrics['momentum_6m'] = momentum_score(sector_metrics, window=126, column='avg_close')

    # Composite momentum score
    sector_metrics['sector_momentum'] = (
        sector_metrics['momentum_1m'] * 0.5 +
        sector_metrics['momentum_3m'] * 0.3 +
        sector_metrics['momentum_6m'] * 0.2
    )

    # Risk-adjusted momentum
    sector_metrics['risk_adjusted_momentum'] = (
        sector_metrics['sector_momentum'] /
        (1 + sector_metrics['volatility'] / sector_metrics['avg_close'])
    )

    return sector_metrics


def signal(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Enhanced sector momentum strategy with dynamic sector allocation.

    Features:
    - Multi-sector momentum comparison
    - Risk-adjusted sector selection
    - Dynamic position sizing
    - Sector rotation based on momentum

    Args:
        df: Daily stock data (should include multiple stocks)
        benchmark_df: Benchmark index data

    Returns:
        DataFrame with sector momentum signals
    """
    # Initialize result dataframe
    result_df = pd.DataFrame(index=df.index.unique())
    result_df['Signal'] = 0
    result_df['Confidence'] = 0.0
    result_df['Top_Sector'] = 'NONE'
    result_df['Sector_Momentum'] = 0.0

    # Calculate momentum for each sector
    sector_momentum_scores = {}

    for sector_name, stocks in NSE_SECTOR_ETFS.items():
        sector_metrics = calculate_sector_momentum(df, stocks)

        if not sector_metrics.empty:
            # Get latest momentum score
            latest_score = sector_metrics['risk_adjusted_momentum'].iloc[-1]
            sector_momentum_scores[sector_name] = latest_score

            # Store sector data for this date
            for idx in sector_metrics.index:
                result_df.loc[idx, f'{sector_name}_momentum'] = sector_metrics.loc[idx, 'risk_adjusted_momentum']

    # Select top sectors based on momentum
    if sector_momentum_scores:
        # Sort sectors by momentum
        sorted_sectors = sorted(sector_momentum_scores.items(),
                               key=lambda x: x[1], reverse=True)

        # Top sector gets strongest signal
        top_sector, top_momentum = sorted_sectors[0]

        # Generate signals based on sector momentum ranking
        for idx in result_df.index:
            if pd.notna(result_df.loc[idx, f'{top_sector}_momentum']):
                result_df.loc[idx, 'Top_Sector'] = top_sector
                result_df.loc[idx, 'Sector_Momentum'] = top_momentum

                # Strong momentum = buy signal, weak = sell/avoid
                if top_momentum > 0.7:
                    result_df.loc[idx, 'Signal'] = 1
                    result_df.loc[idx, 'Confidence'] = min(top_momentum, 1.0)
                elif top_momentum < -0.7:
                    result_df.loc[idx, 'Signal'] = -1
                    result_df.loc[idx, 'Confidence'] = min(abs(top_momentum), 1.0)

    # Add sector diversification metrics
    sector_columns = [col for col in result_df.columns if col.endswith('_momentum')]
    if sector_columns:
        # Calculate sector momentum dispersion
        momentum_values = result_df[sector_columns].iloc[-1]
        result_df['momentum_dispersion'] = momentum_values.std()

        # Calculate sector concentration (Herfindahl index)
        total_momentum_sq = (momentum_values ** 2).sum()
        result_df['momentum_concentration'] = total_momentum_sq / len(momentum_values)

    return result_df[['Signal', 'Confidence', 'Top_Sector', 'Sector_Momentum',
                     'momentum_dispersion', 'momentum_concentration']]