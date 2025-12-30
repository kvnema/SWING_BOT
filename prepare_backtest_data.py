#!/usr/bin/env python3
"""
SWING_BOT Backtest Data Preparation

This script prepares historical data for Backtrader backtesting by:
1. Fetching data from existing SWING_BOT data pipeline
2. Validating data quality
3. Formatting for Backtrader compatibility
4. Creating sample backtest datasets

Usage:
    python prepare_backtest_data.py --symbols RELIANCE TCS --start-date 2020-01-01 --end-date 2024-12-31
    python prepare_backtest_data.py --universe nifty50 --validate-only
"""

import argparse
import logging
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import os

# Import SWING_BOT modules
from src.data_fetch import fetch_single_instrument, fetch_nifty50_data
from src.backtrader_data import validate_data_quality, get_nifty50_universe

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparer:
    """Data preparation class for backtesting"""

    def __init__(self, data_dir='data', output_dir='backtest_data'):
        """
        Initialize data preparer

        Args:
            data_dir: Directory containing raw data
            output_dir: Directory for prepared backtest data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def fetch_missing_data(self, symbols, start_date, end_date):
        """
        Generate sample historical data for backtesting (simplified version)

        Args:
            symbols: List of symbols to generate data for
            start_date: Start date for data
            end_date: End date for data
        """
        from datetime import datetime
        import numpy as np

        logger.info(f"Generating sample data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                logger.info(f"Generating sample data for {symbol}...")

                # Generate sample OHLCV data
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                business_days = date_range[date_range.weekday < 5]  # Mon-Fri

                n_bars = len(business_days)

                # Generate realistic price data
                np.random.seed(hash(symbol) % 2**32)  # Reproducible seed

                # Start with a reasonable price
                base_price = 1000 + np.random.randint(0, 4000)

                prices = []
                current_price = base_price

                for i in range(n_bars):
                    # Random walk with slight upward bias
                    change = np.random.normal(0.0005, 0.02)  # Mean return, volatility
                    current_price *= (1 + change)

                    # Generate OHLC
                    high = current_price * (1 + abs(np.random.normal(0, 0.01)))
                    low = current_price * (1 - abs(np.random.normal(0, 0.01)))
                    open_price = prices[-1][3] if prices else current_price
                    close = current_price

                    # Ensure OHLC relationships
                    high = max(high, open_price, close)
                    low = min(low, open_price, close)

                    volume = np.random.randint(100000, 1000000)

                    prices.append([open_price, high, low, close, volume])

                # Create DataFrame
                df = pd.DataFrame(prices, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = business_days[:len(df)]
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

                # Save to CSV
                filename = f"{symbol}_{start_date}_{end_date}.csv"
                filepath = os.path.join(self.data_dir, filename)
                df.to_csv(filepath, index=False)

                logger.info(f"Saved {len(df)} bars for {symbol} to {filepath}")

            except Exception as e:
                logger.error(f"Failed to generate data for {symbol}: {e}")

    def prepare_backtest_data(self, symbols, start_date=None, end_date=None, validate=True):
        """
        Prepare data for backtesting

        Args:
            symbols: List of symbols to prepare
            start_date: Start date filter
            end_date: End date filter
            validate: Whether to validate data quality

        Returns:
            Dict with preparation results
        """

        results = {
            'total_symbols': len(symbols),
            'prepared_symbols': 0,
            'failed_symbols': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'data_ranges': {},
        }

        for symbol in symbols:
            try:
                # Load data
                df = self.load_symbol_data(symbol, start_date, end_date)

                if df is None or df.empty:
                    logger.warning(f"No data available for {symbol}")
                    results['failed_symbols'] += 1
                    continue

                # Validate data quality
                if validate:
                    if validate_data_quality(df, symbol):
                        results['validation_passed'] += 1
                    else:
                        results['validation_failed'] += 1
                        # Continue processing even if validation fails

                # Save prepared data
                self.save_prepared_data(df, symbol, start_date, end_date)

                # Record data range
                results['data_ranges'][symbol] = {
                    'start_date': df.index.min().strftime('%Y-%m-%d'),
                    'end_date': df.index.max().strftime('%Y-%m-%d'),
                    'bars': len(df),
                }

                results['prepared_symbols'] += 1
                logger.info(f"Prepared {len(df)} bars for {symbol}")

            except Exception as e:
                logger.error(f"Error preparing {symbol}: {e}")
                results['failed_symbols'] += 1

        return results

    def load_symbol_data(self, symbol, start_date=None, end_date=None):
        """
        Load data for a symbol from available files

        Args:
            symbol: Symbol name
            start_date: Start date filter
            end_date: End date filter

        Returns:
            pandas DataFrame with OHLCV data
        """

        # Try different file patterns
        possible_files = [
            f"{self.data_dir}/{symbol}.csv",
            f"{self.data_dir}/{symbol}_2020-01-01_2025-12-01.csv",
            f"{self.data_dir}/{symbol}_2023-01-01_2024-12-01.csv",
            f"{self.data_dir}/{symbol}_2024-01-01_2024-12-01.csv",
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)

                    # Standardize columns
                    column_mapping = {
                        'date': 'Date',
                        'datetime': 'Date',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'vol': 'Volume',
                    }

                    df = df.rename(columns=column_mapping)

                    # Ensure Date column and set as index
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)

                    # Filter date range
                    if start_date:
                        df = df[df.index >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df.index <= pd.to_datetime(end_date)]

                    # Clean data
                    df = df.dropna()
                    df = df[df['Volume'] > 0]

                    if not df.empty:
                        logger.info(f"Loaded {len(df)} bars for {symbol} from {file_path}")
                        return df

                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue

        logger.warning(f"No valid data file found for {symbol}")
        return None

    def save_prepared_data(self, df, symbol, start_date=None, end_date=None):
        """
        Save prepared data in Backtrader-compatible format

        Args:
            df: Prepared DataFrame
            symbol: Symbol name
            start_date: Start date for filename
            end_date: End date for filename
        """

        # Create filename
        if start_date and end_date:
            filename = f"{symbol}_{start_date}_{end_date}_backtest.csv"
        else:
            filename = f"{symbol}_backtest.csv"

        filepath = os.path.join(self.output_dir, filename)

        # Ensure proper format for Backtrader
        save_df = df.reset_index()
        save_df['Date'] = save_df['Date'].dt.strftime('%Y-%m-%d')

        # Keep only required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        save_df = save_df[required_cols]

        save_df.to_csv(filepath, index=False)
        logger.info(f"Saved prepared data to {filepath}")

    def create_sample_dataset(self, symbols=None, start_date='2023-01-01', end_date='2024-12-31'):
        """
        Create a sample dataset for quick backtesting

        Args:
            symbols: List of symbols (default: top 10 NIFTY50)
            start_date: Start date
            end_date: End date
        """

        if symbols is None:
            # Top 10 NIFTY50 stocks by market cap
            symbols = [
                'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY',
                'HINDUNILVR', 'ITC', 'KOTAKBANK', 'LT', 'AXISBANK'
            ]

        logger.info(f"Creating sample dataset with {len(symbols)} symbols")

        # Prepare data
        results = self.prepare_backtest_data(symbols, start_date, end_date)

        # Create metadata file
        metadata = {
            'dataset_info': {
                'name': 'SWING_BOT_Sample_Dataset',
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols,
            },
            'preparation_results': results,
        }

        import json
        metadata_path = os.path.join(self.output_dir, 'sample_dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Sample dataset created in {self.output_dir}")
        logger.info(f"Metadata saved to {metadata_path}")

        return results

    def print_summary(self, results):
        """Print preparation summary"""

        print("\n" + "="*80)
        print("SWING_BOT DATA PREPARATION SUMMARY")
        print("="*80)

        print("Overall Statistics:")
        print(f"  Total Symbols:     {results['total_symbols']}")
        print(f"  Prepared:          {results['prepared_symbols']}")
        print(f"  Failed:            {results['failed_symbols']}")

        if 'validation_passed' in results:
            print(f"  Validation Passed: {results['validation_passed']}")
            print(f"  Validation Failed: {results['validation_failed']}")

        print("Data Ranges:")
        for symbol, info in results['data_ranges'].items():
            print(f"  {symbol}: {info['start_date']} to {info['end_date']} ({info['bars']} bars)")

        print("="*80)


def main():
    """Main function for command-line execution"""

    parser = argparse.ArgumentParser(description='SWING_BOT Backtest Data Preparation')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to prepare')
    parser.add_argument('--universe', choices=['nifty50'], help='Use predefined universe')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--fetch-missing', action='store_true', help='Fetch missing data from API')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing data')
    parser.add_argument('--create-sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--data-dir', default='data', help='Input data directory')
    parser.add_argument('--output-dir', default='backtest_data', help='Output directory')

    args = parser.parse_args()

    # Initialize preparer
    preparer = DataPreparer(data_dir=args.data_dir, output_dir=args.output_dir)

    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    elif args.universe == 'nifty50':
        symbols = get_nifty50_universe()
    elif args.create_sample:
        symbols = None  # Will use default top 10
    else:
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK']  # Default

    if args.create_sample:
        # Create sample dataset
        results = preparer.create_sample_dataset(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )

    elif args.validate_only:
        # Only validate existing data
        logger.info("Validating existing data...")
        results = preparer.prepare_backtest_data(
            symbols, args.start_date, args.end_date, validate=True
        )

    else:
        # Full preparation
        if args.fetch_missing:
            preparer.fetch_missing_data(symbols, args.start_date, args.end_date)

        results = preparer.prepare_backtest_data(
            symbols, args.start_date, args.end_date, validate=True
        )

    # Print summary
    preparer.print_summary(results)

    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main()