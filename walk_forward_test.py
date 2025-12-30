#!/usr/bin/env python3
"""
Walk-Forward Backtesting for SWING_BOT Safety Validation

This script validates the safety enhancements by testing on out-of-sample data
using walk-forward analysis to ensure the system performs well beyond the
training period.

Usage:
    python walk_forward_test.py --symbol RELIANCE.NS --start-date 2023-01-01 --end-date 2025-12-01
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import SWING_BOT modules
from src.backtest import walk_forward_backtest
from src.data_fetch import fetch_market_index_data
from src.signals import compute_signals
from src.config import API_KEY, API_SECRET, ACCESS_TOKEN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def load_or_fetch_data(symbol: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Load data from cache or fetch from API."""
    cache_file = Path(f"data/{symbol.replace('.NS', '')}_{start_date}_{end_date}.csv")

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached data for {symbol}")
        df = pd.read_csv(cache_file, parse_dates=['Date'])
        return df

    logger.info(f"Fetching fresh data for {symbol} from {start_date} to {end_date}")

    # Fetch data using SWING_BOT's data fetcher
    # Convert date range to days
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = (end - start).days + 30  # Add buffer for safety
    
    df = fetch_market_index_data(
        symbol=symbol,
        days=days,
        broker='upstox'
    )
    
    # Filter to the requested date range
    df['Date'] = pd.to_datetime(df['Date'])
    # Convert to timezone-naive for comparison
    df['Date'] = df['Date'].dt.tz_localize(None) if df['Date'].dt.tz is not None else df['Date']
    df = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()
    df = df.sort_values('Date').reset_index(drop=True)

    if not df.empty:
        # Save to cache
        cache_file.parent.mkdir(exist_ok=True)
        df.to_csv(cache_file, index=False)
        logger.info(f"Saved {len(df)} records to cache")

    return df


def run_walk_forward_test(symbol: str, start_date: str, end_date: str):
    """Run walk-forward backtesting for safety validation."""

    logger.info(f"Starting walk-forward test for {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")

    # Load data
    df = load_or_fetch_data(symbol, start_date, end_date)

    if df.empty:
        logger.error("No data available for testing")
        return None

    logger.info(f"Loaded {len(df)} data points")

    # Compute signals with all safety features
    logger.info("Computing signals with safety enhancements...")
    df_with_signals = compute_signals(df)

    # Backtest configuration (conservative settings)
    cfg = {
        'risk': {
            'equity': 100000,
            'risk_per_trade_pct': 1.0,  # 1% risk per trade
            'stop_multiple_atr': 1.5    # 1.5 ATR stop
        },
        'backtest': {
            'transaction_cost_pct': 0.001  # 0.1% transaction costs
        }
    }

    # Run walk-forward backtest
    logger.info("Running walk-forward backtest...")
    wf_results = walk_forward_backtest(
        df=df_with_signals,
        flag_col='SEPA_Flag',  # Test with primary strategy
        cfg=cfg,
        train_years=1,    # 1 year training window
        test_months=3,    # 3 months test window
        start_date=start_date,
        end_date=end_date
    )

    if 'error' in wf_results:
        logger.error(f"Walk-forward test failed: {wf_results['error']}")
        return None

    # Display results
    print("\n" + "="*60)
    print("SWING_BOT SAFETY VALIDATION - WALK-FORWARD RESULTS")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Windows Tested: {wf_results['combined_kpi']['Windows']}")
    print(f"Total Trades: {wf_results['combined_kpi']['Total_Trades']}")
    print()

    print("COMBINED PERFORMANCE METRICS:")
    print("-" * 40)
    kpi = wf_results['combined_kpi']
    print(".2f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    # Safety validation checks
    print("\nSAFETY VALIDATION CHECKS:")
    print("-" * 40)

    safety_passed = True

    # Check drawdown limit
    if kpi['MaxDD'] > -0.30:  # Max 30% drawdown
        print("✅ Max Drawdown: PASSED (< 30%)")
    else:
        print("❌ Max Drawdown: FAILED (> 30%)")
        safety_passed = False

    # Check Sharpe ratio
    if kpi['Sharpe'] > 1.0:
        print("✅ Sharpe Ratio: PASSED (> 1.0)")
    else:
        print("❌ Sharpe Ratio: FAILED (< 1.0)")
        safety_passed = False

    # Check win rate
    if 50 <= kpi['Win_Rate_%'] <= 70:
        print("✅ Win Rate: PASSED (50-70%)")
    else:
        print("⚠️  Win Rate: WARNING (outside 50-70% range)")

    # Check positive expectancy
    if kpi['AvgR'] > 0:
        print("✅ Positive Expectancy: PASSED")
    else:
        print("❌ Positive Expectancy: FAILED")
        safety_passed = False

    print()
    if safety_passed:
        print("🎉 SAFETY VALIDATION: PASSED - System ready for live trading!")
    else:
        print("⚠️  SAFETY VALIDATION: ISSUES FOUND - Review before live deployment")

    # Generate plots if we have data
    if not wf_results['combined_equity'].empty:
        plot_results(wf_results, symbol, start_date, end_date)

    return wf_results


def plot_results(results: dict, symbol: str, start_date: str, end_date: str):
    """Generate performance plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'SWING_BOT Walk-Forward Backtest: {symbol} ({start_date} to {end_date})')

    # Equity curve
    if not results['combined_equity'].empty:
        equity_data = results['combined_equity'].sort_values('Date')
        equity_data['Date'] = pd.to_datetime(equity_data['Date'])
        equity_data.set_index('Date')['Equity'].plot(ax=axes[0,0], title='Equity Curve')
        axes[0,0].set_ylabel('Portfolio Value')

    # Drawdown
    if not results['combined_equity'].empty:
        equity_series = results['combined_equity'].groupby('Date')['Equity'].last()
        peak = equity_series.cummax()
        dd = (equity_series - peak) / peak
        dd.plot(ax=axes[0,1], title='Drawdown', color='red')
        axes[0,1].set_ylabel('Drawdown %')
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Trade R:R distribution
    if not results['combined_trades'].empty:
        r_values = results['combined_trades']['R']
        r_values.hist(ax=axes[1,0], bins=20, alpha=0.7, title='R:R Distribution')
        axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('R:R Ratio')
        axes[1,0].set_ylabel('Frequency')

    # Rolling Sharpe (if enough data)
    if not results['combined_equity'].empty and len(results['combined_equity']) > 60:
        equity_series = results['combined_equity'].groupby('Date')['Equity'].last()
        daily_ret = equity_series.pct_change().dropna()
        rolling_sharpe = daily_ret.rolling(60).mean() / daily_ret.rolling(60).std() * np.sqrt(252)
        rolling_sharpe.plot(ax=axes[1,1], title='Rolling Sharpe Ratio (60-day)')
        axes[1,1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
        axes[1,1].legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = f"outputs/walk_forward_{symbol.replace('.NS', '')}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Performance plots saved to: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Backtesting for SWING_BOT Safety Validation')
    parser.add_argument('--symbol', type=str, default='RELIANCE.NS',
                       help='Stock symbol to test (default: RELIANCE.NS)')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Start date for testing (default: 2023-01-01)')
    parser.add_argument('--end-date', type=str, default='2025-12-01',
                       help='End date for testing (default: 2025-12-01)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Force fresh data fetch instead of using cache')

    args = parser.parse_args()

    # Run the test
    results = run_walk_forward_test(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )

    if results:
        logger.info("Walk-forward testing completed successfully")
        return 0
    else:
        logger.error("Walk-forward testing failed")
        return 1


if __name__ == '__main__':
    exit(main())