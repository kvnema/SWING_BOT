import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import random

# Import centralized configuration
from .config import (
    API_KEY, API_SECRET, ACCESS_TOKEN, INSTRUMENT_KEYS, ALL_INSTRUMENTS,
    API_RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_BACKOFF_FACTOR
)

logger = logging.getLogger(__name__)


class CandleCompletenessDiagnostic:
    """Diagnose Upstox API candle data completeness across multiple timeframes and options data."""

    def __init__(self):
        self.instrument_keys = self._load_instrument_keys()
        self.headers = self._get_headers()
        # F&O stocks (subset of NIFTY 200 that have derivatives)
        self.fo_stocks = self._get_fo_stocks()

    def _load_instrument_keys(self) -> Dict[str, str]:
        """Load instrument keys from artifacts or config."""
        universe_file = Path('artifacts/universe/instrument_keys.json')

        if universe_file.exists():
            try:
                with open(universe_file, 'r') as f:
                    universe_data = json.load(f)
                return {symbol: data.get('instrument_key', symbol)
                       for symbol, data in universe_data.items()}
            except Exception as e:
                logger.warning(f"Failed to load instrument keys: {e}")

        return INSTRUMENT_KEYS

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers with access token."""
        access_token = os.getenv('UPSTOX_ACCESS_TOKEN') or ACCESS_TOKEN
        if not access_token:
            raise ValueError("UPSTOX_ACCESS_TOKEN not found in environment or config")

        return {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }

    def _get_fo_stocks(self) -> List[str]:
        """Get list of F&O eligible stocks from NIFTY 200."""
        # Top F&O stocks by liquidity (subset of NIFTY 200)
        fo_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
            'MARUTI.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 'WIPRO.NS',
            'ADANIPORTS.NS', 'POWERGRID.NS', 'NTPC.NS', 'ONGC.NS', 'COALINDIA.NS'
        ]
        return [stock for stock in fo_stocks if stock in ALL_INSTRUMENTS]

    def _calculate_expected_bars(self, days: int, timeframe: str) -> int:
        """Calculate expected number of bars for given days and timeframe."""
        # Approximate trading days per year
        trading_days_per_year = 252

        if timeframe == '1d':
            return int(days * 0.7)  # ~70% of days are trading days
        elif timeframe == '1w':
            return int(days / 7 * 0.8)  # ~80% of weeks have trading days
        elif timeframe == '1mo':
            return int(days / 30.44 * 0.9)  # ~90% of months have trading days
        elif timeframe == '5m':
            # 5-minute bars: ~6.5 hours trading per day * 12 bars per hour * trading days
            trading_hours_per_day = 6.5
            bars_per_hour = 12
            return int(days * 0.7 * trading_hours_per_day * bars_per_hour)
        else:
            return int(days * 0.7)  # Default approximation

    def _fetch_candle_data(self, symbol: str, timeframe: str, days: int, mock: bool = False) -> Tuple[int, str, str, str, float]:
        """Fetch candle data for a single symbol and timeframe."""
        try:
            if mock:
                # Mock data for testing
                expected_bars = self._calculate_expected_bars(days, timeframe)
                returned_bars = random.randint(int(expected_bars * 0.5), expected_bars)
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                completeness = min(returned_bars / expected_bars * 100, 100.0)
                status = "FULL" if completeness >= 90.0 else "LIMITED"
                return returned_bars, start_date, end_date, status, completeness

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get instrument key
            clean_symbol = symbol.replace('.NS', '')
            instrument_key = self.instrument_keys.get(clean_symbol, clean_symbol)

            # Map timeframe to Upstox interval
            interval_map = {
                '1d': 'day',
                '1w': 'week',
                '1mo': 'month',
                '5m': '5minute'
            }
            interval = interval_map.get(timeframe, 'day')

            url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{end_date.strftime('%Y-%m-%d')}/{start_date.strftime('%Y-%m-%d')}"

            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code != 200:
                return 0, "", "", f"API_ERROR_{response.status_code}", 0.0

            data = response.json()
            candles = data.get('data', {}).get('candles', [])

            if not candles:
                return 0, "", "", "NO_DATA", 0.0

            returned_bars = len(candles)

            # Get actual date range from data
            if candles:
                # API returns candles in reverse chronological order (newest first)
                newest_date = pd.to_datetime(candles[0][0])
                oldest_date = pd.to_datetime(candles[-1][0])

                actual_start = oldest_date.strftime('%Y-%m-%d')
                actual_end = newest_date.strftime('%Y-%m-%d')

                # Calculate actual days covered
                actual_days = (newest_date - oldest_date).days + 1

                # Calculate expected bars for the actual period
                expected_bars = self._calculate_expected_bars(actual_days, timeframe)
                completeness = min(returned_bars / expected_bars * 100, 100.0)
            else:
                actual_start = actual_end = ""
                completeness = 0.0

            # Determine status
            if completeness >= 90.0:
                status = "FULL"
            elif completeness >= 50.0:
                status = "LIMITED"
            else:
                status = "MISSING"

            return returned_bars, actual_start, actual_end, status, completeness

        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {str(e)}")
            return 0, "", "", f"ERROR_{str(e)[:20]}", 0.0

    def _fetch_options_data(self, symbol: str, mock: bool = False) -> Tuple[int, str, str, float]:
        """Fetch options chain data for F&O stock."""
        try:
            if mock:
                # Mock options data
                strikes_count = random.randint(20, 50)
                expiry = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                completeness = random.uniform(80, 100)
                status = "FULL" if completeness >= 90.0 else "LIMITED"
                return strikes_count, expiry, status, completeness

            # Get instrument key for underlying
            clean_symbol = symbol.replace('.NS', '')
            instrument_key = self.instrument_keys.get(clean_symbol, clean_symbol)

            # Get current expiry (next monthly expiry)
            today = datetime.now()
            # Find next monthly expiry (last Thursday of month)
            next_month = today.replace(day=28) + timedelta(days=4)  # Go to next month
            expiry = next_month - timedelta(days=next_month.weekday() - 3)  # Last Thursday
            if expiry < today:
                expiry = expiry + timedelta(days=28)  # Next month if already passed

            expiry_str = expiry.strftime('%Y-%m-%d')

            # Fetch options chain
            url = f"https://api.upstox.com/v2/option/chain/{instrument_key}/{expiry_str}"

            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code != 200:
                return 0, "", f"API_ERROR_{response.status_code}", 0.0

            data = response.json()
            options = data.get('data', [])

            if not options:
                return 0, expiry_str, "NO_DATA", 0.0

            strikes_count = len(options)

            # Check completeness - should have both calls and puts for each strike
            call_count = sum(1 for opt in options if opt.get('instrument_type') == 'CE')
            put_count = sum(1 for opt in options if opt.get('instrument_type') == 'PE')

            # Expected: roughly equal calls and puts
            expected_strikes = min(call_count, put_count) * 2
            completeness = min(strikes_count / expected_strikes * 100, 100.0) if expected_strikes > 0 else 0.0

            # Determine status
            if completeness >= 90.0:
                status = "FULL"
            elif completeness >= 50.0:
                status = "LIMITED"
            else:
                status = "MISSING"

            return strikes_count, expiry_str, status, completeness

        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {str(e)}")
            return 0, "", f"ERROR_{str(e)[:20]}", 0.0

    def run_diagnostic(self, timeframes: List[str], days: int = 1000, sample_size: int = 50,
                      output_path: str = 'outputs/candle_completeness_report.csv',
                      rate_limit: float = 0.5, include_options: bool = False, mock: bool = False) -> Dict[str, Dict]:
        """Run completeness diagnostic for multiple timeframes and optionally options data."""
        print(f"Testing candle data completeness for {days} days across {len(timeframes)} timeframes...")

        # Get symbols to test
        if sample_size > 0 and sample_size < len(ALL_INSTRUMENTS):
            symbols_to_test = random.sample(ALL_INSTRUMENTS, min(sample_size, len(ALL_INSTRUMENTS)))
            print(f"Sampling {len(symbols_to_test)} symbols from {len(ALL_INSTRUMENTS)} total")
        else:
            symbols_to_test = ALL_INSTRUMENTS
            print(f"Testing full universe: {len(symbols_to_test)} symbols")

        results = {}
        total_tests = len(symbols_to_test) * len(timeframes)
        if include_options:
            total_tests += len([s for s in symbols_to_test if s in self.fo_stocks])

        print(f"Total tests to run: {total_tests}")

        test_count = 0
        for symbol in symbols_to_test:
            for timeframe in timeframes:
                test_count += 1
                if test_count % 10 == 0:
                    print(f"Progress: {test_count}/{total_tests} tests completed")

                returned_bars, start_date, end_date, status, completeness = self._fetch_candle_data(
                    symbol, timeframe, days, mock
                )

                test_key = f"{symbol}_{timeframe}"
                results[test_key] = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'requested_days': days,
                    'returned_bars': returned_bars,
                    'start_date': start_date,
                    'end_date': end_date,
                    'status': status,
                    'completeness_pct': round(completeness, 2),
                    'data_type': 'candle'
                }

                # Rate limiting
                if not mock and rate_limit > 0:
                    time.sleep(rate_limit)

            # Options data for F&O stocks
            if include_options and symbol in self.fo_stocks:
                test_count += 1
                strikes_count, expiry, status, completeness = self._fetch_options_data(symbol, mock)

                test_key = f"{symbol}_options"
                results[test_key] = {
                    'symbol': symbol,
                    'timeframe': 'options',
                    'requested_days': days,
                    'returned_bars': strikes_count,
                    'start_date': expiry,
                    'end_date': '',
                    'status': status,
                    'completeness_pct': round(completeness, 2),
                    'data_type': 'options'
                }

                if not mock and rate_limit > 0:
                    time.sleep(rate_limit)

        print(f"Completed {len(results)} tests")
        return results

    def generate_report(self, results: Dict[str, Dict], output_path: str):
        """Generate CSV report from diagnostic results."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame.from_dict(results, orient='index')

        # Sort by completeness percentage (worst first)
        df = df.sort_values('completeness_pct')

        # Save to CSV
        df.to_csv(output_file, index=False)

        print(f"Report saved to {output_file}")
        print(f"Total tests: {len(df)}")

        # Print summary statistics
        status_counts = df['status'].value_counts()
        print(f"Status breakdown: {dict(status_counts)}")

        if 'FULL' in status_counts:
            print(f"Tests with full data (≥90%): {status_counts['FULL']}")
        if 'LIMITED' in status_counts:
            print(f"Tests with limited data (<90%): {status_counts['LIMITED']}")
        if 'MISSING' in status_counts:
            print(f"Tests with missing data: {status_counts['MISSING']}")

        # Breakdown by data type
        candle_results = df[df['data_type'] == 'candle']
        if not candle_results.empty:
            avg_candle_completeness = candle_results[candle_results['status'] != 'MISSING']['completeness_pct'].mean()
            print(f"Average candle completeness: {avg_candle_completeness:.1f}%")

        options_results = df[df['data_type'] == 'options']
        if not options_results.empty:
            avg_options_completeness = options_results[options_results['status'] != 'MISSING']['completeness_pct'].mean()
            print(f"Average options completeness: {avg_options_completeness:.1f}%")

        # Show worst performers
        worst_10 = df.head(10)
        if not worst_10.empty:
            print("\nWorst 10 performers:")
            for _, row in worst_10.iterrows():
                if row['data_type'] == 'candle':
                    print(f"  {row['symbol']} {row['timeframe']}: {row['completeness_pct']:.1f}% ({row['returned_bars']} bars, {row['status']})")
                else:
                    print(f"  {row['symbol']} options: {row['completeness_pct']:.1f}% ({row['returned_bars']} strikes, {row['status']})")