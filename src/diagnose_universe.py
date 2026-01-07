"""
Universe Diagnostic Tool for SWING_BOT

This module provides diagnostics for the SWING_BOT screening pipeline,
ensuring all NIFTY200 stocks + ETFs are properly considered.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from .data_io import load_dataset
from .signals import compute_signals
from .scoring import compute_composite_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniverseDiagnostic:
    """Diagnostic tool for SWING_BOT universe screening."""

    def __init__(self, data_path: str = "outputs/daily_data.csv"):
        self.data_path = Path(data_path)
        self.universe_path = Path("artifacts/universe/instrument_keys.json")
        self.results = []

    def load_universe(self) -> List[str]:
        """Load symbols from data file."""
        data = self.load_data()
        symbols = data['Symbol'].unique().tolist()
        logger.info(f"Loaded {len(symbols)} symbols from data file")
        return symbols

    def load_data(self) -> pd.DataFrame:
        """Load market data."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = load_dataset(str(self.data_path))
        logger.info(f"Loaded {len(df)} records from {self.data_path}")
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a stock DataFrame."""
        df = df.copy()
        
        # RSI14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI14'] = 100 - (100 / (1 + rs))
        
        # EMAs
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['EMA200'] = df['Close'].ewm(span=200).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        
        # ATR14
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14).mean()
        
        # Donchian
        df['DonchianH20'] = df['High'].rolling(20).max()
        df['DonchianL20'] = df['Low'].rolling(20).min()
        
        # RVOL20
        df['RVOL20'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Bollinger Bandwidth
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BandWidth'] = (std20 / sma20) * 100
        
        return df

    def run_diagnostics(self, max_symbols: int = None, verbose: bool = False) -> pd.DataFrame:
        """Run full diagnostic on universe."""
        # Load data
        data = self.load_data()
        universe = self.load_universe()

        if max_symbols:
            universe = universe[:max_symbols]

        logger.info(f"Running diagnostics on {len(universe)} symbols")

        results = []

        for symbol in universe:
            try:
                symbol_data = data[data['Symbol'] == symbol].copy()
                if len(symbol_data) == 0:
                    results.append({
                        'Symbol': symbol,
                        'Status': 'NO_DATA',
                        'Score': 0,
                        'Reason': 'No data available',
                        'Is_ETF': 'ETF' in symbol.upper() or symbol.endswith('ETF')
                    })
                    continue

                # Calculate indicators
                symbol_data = self.calculate_indicators(symbol_data)

                # Generate signals
                symbol_data = compute_signals(symbol_data)

                # Calculate composite score
                scores = compute_composite_score(symbol_data)
                score = scores.iloc[-1] if not scores.empty else 0

                # Get latest row
                latest = symbol_data.iloc[-1].copy()

                # Check screener criteria
                screener_result = self.check_screener_criteria(latest)

                results.append({
                    'Symbol': symbol,
                    'Status': 'PASS' if screener_result['passed'] else 'FAIL',
                    'Score': score,
                    'Reason': screener_result['reason'],
                    'RSI14': latest.get('RSI14', 0),
                    'MACD_CrossUp': latest.get('MACD_CrossUp', False),
                    'Trend_OK': latest.get('Trend_OK', False),
                    'Breakout': latest.get('Breakout', False),
                    'RVOL20': latest.get('RVOL20', 0),
                    'Close': latest.get('Close', 0),
                    'Is_ETF': 'ETF' in symbol.upper() or symbol.endswith('ETF')
                })

                if verbose and len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)}/{len(universe)} symbols")

            except Exception as e:
                results.append({
                    'Symbol': symbol,
                    'Status': 'ERROR',
                    'Score': 0,
                    'Reason': str(e),
                    'Is_ETF': 'ETF' in symbol.upper() or symbol.endswith('ETF')
                })

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('Score', ascending=False)

        logger.info(f"Diagnostics complete: {len(df_results)} results")
        return df_results

    def check_screener_criteria(self, row: pd.Series) -> Dict:
        """Check if symbol passes screener criteria."""
        reasons = []

        # RSI check
        rsi = row.get('RSI14', 50)
        if rsi < 50:
            reasons.append(f"RSI14 {rsi:.1f} < 50")

        # MACD cross
        if not row.get('MACD_CrossUp', False):
            reasons.append("No MACD cross up")

        # Trend OK
        if not row.get('Trend_OK', False):
            reasons.append("Trend not OK")

        # RVOL
        rvol = row.get('RVOL20', 1.0)
        if rvol < 1.0:
            reasons.append(f"RVOL20 {rvol:.2f} < 1.0")

        # Breakout signal
        if not row.get('Breakout', False):
            reasons.append("No breakout signal")

        passed = len(reasons) == 0
        return {
            'passed': passed,
            'reason': '; '.join(reasons) if reasons else 'All criteria met'
        }

    def generate_report(self, results: pd.DataFrame, output_path: str = "outputs/universe_diagnostic.csv"):
        """Generate diagnostic report."""
        results.to_csv(output_path, index=False)

        # Summary stats
        total = len(results)
        passed = len(results[results['Status'] == 'PASS'])
        etfs = len(results[results['Is_ETF'] == True])
        stocks = total - etfs

        passed_etfs = len(results[(results['Status'] == 'PASS') & (results['Is_ETF'] == True)])
        passed_stocks = len(results[(results['Status'] == 'PASS') & (results['Is_ETF'] == False)])

        logger.info("=== Universe Diagnostic Report ===")
        logger.info(f"Total symbols: {total}")
        logger.info(f"ETFs: {etfs}, Stocks: {stocks}")
        logger.info(f"Passed: {passed} ({passed/total*100:.1f}%)")
        logger.info(f"Passed ETFs: {passed_etfs} ({passed_etfs/etfs*100 if etfs > 0 else 0:.1f}%)")
        logger.info(f"Passed Stocks: {passed_stocks} ({passed_stocks/stocks*100 if stocks > 0 else 0:.1f}%)")

        # Top 20 by score
        logger.info("\nTop 20 symbols by score:")
        for _, row in results.head(20).iterrows():
            logger.info(f"  {row['Symbol']}: {row['Score']:.3f} ({row['Status']}) - {row['Reason']}")

        logger.info(f"\nReport saved to: {output_path}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SWING_BOT Universe Diagnostic Tool")
    parser.add_argument("--max-symbols", type=int, help="Limit number of symbols to test")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--output", default="outputs/universe_diagnostic.csv", help="Output CSV path")

    args = parser.parse_args()

    diagnostic = UniverseDiagnostic()
    results = diagnostic.run_diagnostics(max_symbols=args.max_symbols, verbose=args.verbose)
    diagnostic.generate_report(results, args.output)


if __name__ == "__main__":
    main()