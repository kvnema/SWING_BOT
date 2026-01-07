#!/usr/bin/env python3
"""
Test script to verify QuantConnect-inspired strategies integration
"""
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_fetch import fetch_nifty50_data
from indicators import compute_all_indicators
from signals import compute_signals

def test_new_strategies():
    """Test the new QuantConnect-inspired strategies"""
    print('🧪 Testing QuantConnect-inspired strategies integration...')

    # Fetch sample data
    print('📊 Fetching Nifty50 data...')
    df = fetch_nifty50_data()
    if df.empty:
        print('❌ No data fetched')
        return False

    # Take first 3 symbols for testing
    test_symbols = df['Symbol'].unique()[:3]
    df = df[df['Symbol'].isin(test_symbols)].copy()

    print(f'📈 Testing with symbols: {list(test_symbols)}')

    # Compute indicators
    print('🔧 Computing indicators...')
    df = compute_all_indicators(df)

    # Compute signals
    print('📊 Computing signals...')
    df = compute_signals(df)

    # Check if new signals are present
    new_signals = ['EnhancedMomentum_Signal', 'DynamicBreakout_Signal', 'SectorMomentum_Signal']
    success = True

    for signal in new_signals:
        if signal in df.columns:
            signal_count = df[signal].sum()
            print(f'✅ {signal}: {signal_count} signals generated')
        else:
            print(f'❌ {signal}: Column not found')
            success = False

    # Check for any errors in the data
    if df.isnull().any().any():
        print('⚠️  Warning: NaN values found in dataframe')
        nan_cols = df.columns[df.isnull().any()].tolist()
        print(f'   Columns with NaN: {nan_cols}')

    print('✅ Integration test completed')
    return success

if __name__ == '__main__':
    test_new_strategies()