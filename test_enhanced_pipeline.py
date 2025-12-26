#!/usr/bin/env python3
"""
Test script for the enhanced SWING_BOT pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_swing_pipeline
from src.data_io import load_dataset
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_pipeline():
    """Test the enhanced SWING_BOT pipeline with new strategies"""

    # Configuration for the enhanced pipeline
    config = {
        'rr_gate': {
            'min_ratio': 1.5,
            'max_ratio': 4.0,
            'max_risk_pct': 0.02
        },
        'portfolio': {
            'max_weight': 0.05,
            'max_sector_weight': 0.20,
            'min_sharpe': 0.5
        }
    }

    # Load existing data
    try:
        stocks_df = load_dataset('data/nifty50_data_today.csv')
        logger.info(f"Loaded stock data: {len(stocks_df)} records, {stocks_df['Symbol'].nunique()} symbols")

        # Ensure Date column is datetime
        stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])

        # Get unique symbols
        symbols = stocks_df['Symbol'].unique().tolist()[:10]  # Test with first 10 symbols
        logger.info(f"Testing with symbols: {symbols}")

        # Create mock index data (simplified)
        index_df = stocks_df[stocks_df['Symbol'].str.contains('NIFTY', case=False)].copy()
        if index_df.empty:
            # Use first symbol as proxy for index
            index_df = stocks_df[stocks_df['Symbol'] == stocks_df['Symbol'].iloc[0]].copy()
            index_df['Symbol'] = 'NIFTY_50'

        # Run the enhanced pipeline
        results = run_swing_pipeline(
            config=config,
            symbols=symbols,
            start_date='2024-01-01',
            end_date='2025-12-25'
        )

        logger.info(f"Pipeline completed with status: {results.get('status')}")

        if results.get('status') == 'completed':
            portfolio_result = results.get('portfolio_result', {})
            gtt_payloads = results.get('gtt_payloads', [])

            logger.info(f"Generated {len(gtt_payloads)} GTT payloads")
            logger.info(f"Portfolio result: {bool(portfolio_result)}")

            if portfolio_result:
                weights = portfolio_result.get('weights', pd.Series())
                logger.info(f"Portfolio positions: {len(weights)}")
                logger.info(f"Selected strategies: {portfolio_result.get('selected_strategies', [])}")

        return results

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

if __name__ == '__main__':
    print("🧪 Testing Enhanced SWING_BOT Pipeline...")
    results = test_enhanced_pipeline()
    print(f"✅ Test completed with status: {results.get('status')}")