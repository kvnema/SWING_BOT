#!/usr/bin/env python3
"""
ETF Universe Auto-Updater for SWING_BOT

Downloads Upstox BOD instruments (gzipped JSON), filters for NSE ETFs,
and auto-updates SWING_BOT's instrument universe.

Usage:
    python etf_universe.py --bod-url https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz
    python etf_universe.py --bod-path artifacts/bod/complete.json.gz --dry-run
"""

import argparse
import gzip
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.request import urlopen
from urllib.error import URLError


@dataclass
class Instrument:
    """Represents a financial instrument from Upstox BOD data."""
    segment: str
    name: str
    exchange: str
    isin: str
    instrument_type: str
    instrument_key: str
    trading_symbol: str
    short_name: str
    lot_size: int
    tick_size: float
    exchange_token: str
    security_type: str
    namespace: str = ""


class ETFUniverseUpdater:
    """Handles downloading, filtering, and merging ETF instruments into the universe."""

    def __init__(self, log_level: str = "INFO"):
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)

    def setup_logging(self, level: str):
        """Configure structured logging."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    def download_bod_data(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Download gzipped BOD data with retry logic."""
        self.logger.info(f"Downloading BOD data from: {url}")

        for attempt in range(max_retries):
            try:
                with urlopen(url) as response:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as tmp_file:
                        tmp_file.write(response.read())
                        self.logger.info(f"Downloaded {tmp_file.name}")
                        return tmp_file.name
            except URLError as e:
                wait_time = 2 ** attempt
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}, retrying in {wait_time}s")
                time.sleep(wait_time)

        self.logger.error("Failed to download BOD data after all retries")
        return None

    def load_bod_data(self, path: str) -> List[Dict]:
        """Load and parse gzipped BOD JSON data."""
        self.logger.info(f"Loading BOD data from: {path}")

        try:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("BOD data must be a JSON array")

            self.logger.info(f"Loaded {len(data)} instruments from BOD data")
            return data

        except (gzip.BadGzipFile, json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse BOD data: {e}")
            raise

    def filter_etfs(self, instruments: List[Dict], exchange: str, allowlist: Optional[Set[str]] = None) -> List[Instrument]:
        """Filter instruments for ETFs matching criteria."""
        self.logger.info(f"Filtering ETFs for exchange: {exchange}")
        if allowlist:
            self.logger.info(f"Using allowlist: {sorted(allowlist)}")

        etfs = []
        for item in instruments:
            # Check basic criteria
            if item.get('exchange') != exchange:
                continue

            trading_symbol = item.get('trading_symbol', '').upper()

            # Check if it's an ETF by symbol pattern or name
            is_etf = False
            if 'ETF' in trading_symbol or 'BEES' in trading_symbol or 'IETF' in trading_symbol:
                is_etf = True
            elif 'GOLD' in trading_symbol and ('ETF' in trading_symbol or 'BEES' in trading_symbol):
                is_etf = True
            elif any(etf_term in trading_symbol for etf_term in ['NIFTY', 'SENSEX', 'BANK', 'IT', 'PHARMA', 'AUTO', 'METAL', 'OIL', 'SILVER']):
                # Check if it has ETF-like characteristics
                if 'BEES' in trading_symbol or 'IETF' in trading_symbol or 'ETF' in trading_symbol:
                    is_etf = True

            if not is_etf:
                continue

            # Check allowlist if provided
            if allowlist and trading_symbol not in allowlist:
                continue

            # Create Instrument object
            instrument = Instrument(
                segment=item.get('segment', ''),
                name=item.get('name', ''),
                exchange=item.get('exchange', ''),
                isin=item.get('isin', ''),
                instrument_type='ETF',  # Override to ETF
                instrument_key=item.get('instrument_key', ''),
                trading_symbol=trading_symbol,
                short_name=item.get('short_name', ''),
                lot_size=int(item.get('lot_size', 1)),
                tick_size=float(item.get('tick_size', 0.01)),
                exchange_token=item.get('exchange_token', ''),
                security_type=item.get('security_type', ''),
                namespace=f"{exchange}_ETF"
            )

            etfs.append(instrument)

        self.logger.info(f"Found {len(etfs)} ETFs matching criteria")
        return etfs

    def load_existing_universe(self, path: str) -> Dict[str, Dict]:
        """Load existing instrument universe."""
        if not Path(path).exists():
            self.logger.warning(f"Existing universe file not found: {path}")
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                universe = json.load(f)
            self.logger.info(f"Loaded existing universe with {len(universe)} instruments")
            return universe
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load existing universe: {e}")
            raise

    def merge_universe(self, existing: Dict[str, Dict], etfs: List[Instrument]) -> Dict[str, Dict]:
        """Merge ETF instruments into existing universe."""
        merged = existing.copy()

        for etf in etfs:
            # Use trading_symbol as key, fallback to instrument_key
            key = etf.trading_symbol or etf.instrument_key

            # Convert dataclass to dict
            etf_dict = {
                'segment': etf.segment,
                'name': etf.name,
                'exchange': etf.exchange,
                'isin': etf.isin,
                'instrument_type': etf.instrument_type,
                'instrument_key': etf.instrument_key,
                'trading_symbol': etf.trading_symbol,
                'short_name': etf.short_name,
                'lot_size': etf.lot_size,
                'tick_size': etf.tick_size,
                'exchange_token': etf.exchange_token,
                'security_type': etf.security_type,
                'namespace': etf.namespace
            }

            if key in merged:
                self.logger.info(f"Updating existing ETF: {key}")
            else:
                self.logger.info(f"Adding new ETF: {key}")

            merged[key] = etf_dict

        self.logger.info(f"Merged universe now contains {len(merged)} instruments")
        return merged

    def save_universe(self, universe: Dict[str, Dict], output_path: str):
        """Save universe to JSON file atomically."""
        output_path = Path(output_path)

        # Create temp file
        temp_path = output_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(universe, f, indent=2, ensure_ascii=False)

            # Atomic move
            temp_path.replace(output_path)
            self.logger.info(f"Successfully saved universe to: {output_path}")

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def print_dry_run(self, etfs: List[Instrument]):
        """Print ETF preview for dry run."""
        print("\nETF Universe Update Preview:")
        print("=" * 50)

        for etf in sorted(etfs, key=lambda x: x.trading_symbol):
            print("12")

        print(f"\nTotal ETFs to add/update: {len(etfs)}")


def main():
    parser = argparse.ArgumentParser(
        description="ETF Universe Auto-Updater for SWING_BOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live update from Upstox BOD
  python etf_universe.py --bod-url https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz

  # Offline testing with local file
  python etf_universe.py --bod-path artifacts/bod/complete.json.gz --dry-run

  # Update with allowlist
  python etf_universe.py --allowlist NIFTYBEES,SETFNIF50,ICICINIFTY,KOTAKNIFTY
        """
    )

    parser.add_argument(
        '--bod-url',
        default='https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz',
        help='URL to download BOD instruments JSON.gz'
    )

    parser.add_argument(
        '--bod-path',
        help='Local path to BOD instruments JSON.gz file (offline mode)'
    )

    parser.add_argument(
        '--exchange',
        choices=['NSE', 'BSE', 'MCX'],
        default='NSE',
        help='Exchange to filter ETFs for'
    )

    parser.add_argument(
        '--allowlist',
        help='Comma-separated list of trading symbols to include (case-insensitive)'
    )

    parser.add_argument(
        '--existing-universe',
        default='artifacts/universe/instrument_keys.json',
        help='Path to existing instrument universe JSON'
    )

    parser.add_argument(
        '--output',
        default='artifacts/universe/instrument_keys.json',
        help='Path to write merged universe JSON'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print selected ETFs without writing changes'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Parse allowlist
    allowlist = None
    if args.allowlist:
        allowlist = {symbol.strip().upper() for symbol in args.allowlist.split(',')}

    # Initialize updater
    updater = ETFUniverseUpdater(args.log_level)

    try:
        # Load BOD data
        if args.bod_path:
            # Offline mode
            if not Path(args.bod_path).exists():
                updater.logger.error(f"BOD path does not exist: {args.bod_path}")
                return 1

            bod_data = updater.load_bod_data(args.bod_path)
        else:
            # Download mode
            downloaded_path = updater.download_bod_data(args.bod_url)
            if not downloaded_path:
                return 1

            try:
                bod_data = updater.load_bod_data(downloaded_path)
            finally:
                # Clean up downloaded file
                Path(downloaded_path).unlink()

        # Filter ETFs
        etfs = updater.filter_etfs(bod_data, args.exchange, allowlist)

        if not etfs:
            updater.logger.warning("No ETFs found matching criteria")
            return 1

        # Dry run or actual update
        if args.dry_run:
            updater.print_dry_run(etfs)
            return 0

        # Load existing universe
        existing_universe = updater.load_existing_universe(args.existing_universe)

        # Merge
        merged_universe = updater.merge_universe(existing_universe, etfs)

        # Save
        updater.save_universe(merged_universe, args.output)

        updater.logger.info("ETF universe update completed successfully")
        return 0

    except Exception as e:
        updater.logger.error(f"ETF universe update failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())