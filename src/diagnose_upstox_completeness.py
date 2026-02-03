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


class UpstoxCompletenessDiagnostic:
    """Diagnose Upstox API data completeness for SWING_BOT universe."""

    def __init__(self):
        self.instrument_keys = self._load_instrument_keys()
        self.headers = self._get_headers()

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

    def _calculate_expected_trading_days(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate expected number of trading days between dates (excluding weekends)."""
        total_days = (end_date - start_date).days + 1
        # Rough approximation: ~70% of days are trading days (excluding weekends)
        # This is approximate since it doesn't account for holidays
        expected_trading_days = int(total_days * 0.7)
        return max(expected_trading_days, 1)

    def _fetch_symbol_data(self, symbol: str, days: int, mock: bool = False) -> Tuple[int, str, str, str, float]:
        """Fetch data for a single symbol and return completeness metrics."""
        try:
            if mock:
                # Mock data for testing
                returned_bars = random.randint(int(days * 0.5), days)
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                status = "FULL" if returned_bars >= days * 0.9 else "LIMITED"
                completeness = min(returned_bars / days * 100, 100.0)
                return returned_bars, start_date, end_date, status, completeness

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get instrument key
            clean_symbol = symbol.replace('.NS', '')
            instrument_key = self.instrument_keys.get(clean_symbol, clean_symbol)

            url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{end_date.strftime('%Y-%m-%d')}/{start_date.strftime('%Y-%m-%d')}"

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
                
                # Calculate actual days covered by the returned data
                actual_days = (newest_date - oldest_date).days + 1
                
                # Completeness is based on whether we got data covering the requested period
                # If we requested 500 days but only got data for 200 days, completeness is 200/500 = 40%
                completeness = min(actual_days / days * 100, 100.0)
            else:
                actual_start = actual_end = ""
                actual_days = 0
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
            logger.error(f"Error fetching {symbol}: {str(e)}")
            return 0, "", "", f"ERROR_{str(e)[:20]}", 0.0

    def run_diagnostic(self, days: int = 1000, sample_size: int = 50, output_path: str = 'outputs/upstox_completeness_report.csv',
                      rate_limit: float = 0.5, mock: bool = False) -> Dict[str, Dict]:
        """Run completeness diagnostic on symbol universe."""
        print(f"Testing data completeness for {days} days...")

        # Get symbols to test
        if sample_size > 0 and sample_size < len(ALL_INSTRUMENTS):
            symbols_to_test = random.sample(ALL_INSTRUMENTS, min(sample_size, len(ALL_INSTRUMENTS)))
            print(f"Sampling {len(symbols_to_test)} symbols from {len(ALL_INSTRUMENTS)} total")
        else:
            symbols_to_test = ALL_INSTRUMENTS
            print(f"Testing full universe: {len(symbols_to_test)} symbols")

        results = {}

        for i, symbol in enumerate(symbols_to_test):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(symbols_to_test)} symbols tested")

            returned_bars, start_date, end_date, status, completeness = self._fetch_symbol_data(symbol, days, mock)

            results[symbol] = {
                'symbol': symbol,
                'requested_days': days,
                'returned_bars': returned_bars,
                'start_date': start_date,
                'end_date': end_date,
                'status': status,
                'completeness_pct': round(completeness, 2)
            }

            # Rate limiting
            if not mock and rate_limit > 0:
                time.sleep(rate_limit)

        print(f"Completed testing {len(results)} symbols")
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
        print(f"Total symbols: {len(df)}")

        # Print summary statistics
        status_counts = df['status'].value_counts()
        print(f"Status breakdown: {dict(status_counts)}")

        if 'FULL' in status_counts:
            print(f"Symbols with full data (≥90%): {status_counts['FULL']}")
        if 'LIMITED' in status_counts:
            print(f"Symbols with limited data (<90%): {status_counts['LIMITED']}")
        if 'MISSING' in status_counts:
            print(f"Symbols with missing data: {status_counts['MISSING']}")

        avg_completeness = df[df['status'] != 'MISSING']['completeness_pct'].mean()
        print(f"Average completeness: {avg_completeness:.1f}%")

        # Show worst performers
        worst_10 = df.head(10)
        if not worst_10.empty:
            print("\nWorst 10 performers:")
            for _, row in worst_10.iterrows():
                print(f"  {row['symbol']}: {row['completeness_pct']:.1f}% ({row['returned_bars']} bars, {row['status']})")