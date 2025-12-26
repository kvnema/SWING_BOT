import pandas as pd
from src.backtest import backtest_strategy
from src.indicators import compute_all_indicators


def test_backtest_rsi_filter():
    """Test that --confirm-rsi filters entries to RSI_Above50 and not RSI_Overbought."""
    # Create synthetic data with varying RSI
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    closes = [100 + i * 0.1 for i in range(100)]
    df = pd.DataFrame({
        'Date': dates,
        'Symbol': 'TEST',
        'Close': closes,
        'High': [c + 1 for c in closes],
        'Low': [c - 1 for c in closes],
        'Volume': [1000] * 100
    })
    
    # Add indicators
    from src.indicators import compute_all_indicators
    df = compute_all_indicators(df)
    
    # Add strategy flag (simple: all true for testing filters)
    df['Test_Strategy'] = True
    
    # Backtest without filter
    result_no_filter = backtest_strategy(df, 'Test_Strategy', {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0}, 'backtest': {}}, False, False, False)
    trades_no_filter = result_no_filter['trades']
    
    # Backtest with RSI filter
    result_with_filter = backtest_strategy(df, 'Test_Strategy', {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0}, 'backtest': {}}, True, False, False)
    trades_with_filter = result_with_filter['trades']
    
    # Should have fewer or equal trades with filter
    assert len(trades_with_filter) <= len(trades_no_filter)
    
    # If there are filtered trades, check they meet RSI criteria
    if not trades_with_filter.empty:
        # Get the entry rows
        entry_dates = trades_with_filter['EntryDate']
        entry_rows = df[df['Date'].isin(entry_dates)]
        
        # All should have RSI_Above50 and not RSI_Overbought
        assert (entry_rows['RSI_Above50'] == True).all()
        assert (entry_rows['RSI_Overbought'] == False).all()


def test_backtest_macd_filter():
    """Test that --confirm-macd filters entries to MACD_CrossUp and MACD_AboveZero."""
    # Similar setup as above
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    closes = [100 + i * 0.1 for i in range(100)]
    df = pd.DataFrame({
        'Date': dates,
        'Symbol': 'TEST',
        'Close': closes,
        'High': [c + 1 for c in closes],
        'Low': [c - 1 for c in closes],
        'Volume': [1000] * 100
    })
    
    df = compute_all_indicators(df)
    df['Test_Strategy'] = True
    
    # Backtest without filter
    result_no_filter = backtest_strategy(df, 'Test_Strategy', {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0}, 'backtest': {}}, False, False, False)
    trades_no_filter = result_no_filter['trades']
    
    # Backtest with MACD filter
    result_with_filter = backtest_strategy(df, 'Test_Strategy', {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0}, 'backtest': {}}, False, True, False)
    trades_with_filter = result_with_filter['trades']
    
    # Should have fewer or equal trades
    assert len(trades_with_filter) <= len(trades_no_filter)
    
    # Check criteria
    if not trades_with_filter.empty:
        entry_dates = trades_with_filter['EntryDate']
        entry_rows = df[df['Date'].isin(entry_dates)]
        
        assert (entry_rows['MACD_CrossUp'] == True).all()
        assert (entry_rows['MACD_AboveZero'] == True).all()


def test_backtest_histogram_filter():
    """Test that --confirm-hist filters entries to MACD_Hist_Rising."""
    # Similar setup
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    closes = [100 + i * 0.1 for i in range(100)]
    df = pd.DataFrame({
        'Date': dates,
        'Symbol': 'TEST',
        'Close': closes,
        'High': [c + 1 for c in closes],
        'Low': [c - 1 for c in closes],
        'Volume': [1000] * 100
    })
    
    df = compute_all_indicators(df)
    df['Test_Strategy'] = True
    
    # Backtest without filter
    result_no_filter = backtest_strategy(df, 'Test_Strategy', {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0}, 'backtest': {}}, False, False, False)
    trades_no_filter = result_no_filter['trades']
    
    # Backtest with histogram filter
    result_with_filter = backtest_strategy(df, 'Test_Strategy', {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0}, 'backtest': {}}, False, False, True)
    trades_with_filter = result_with_filter['trades']
    
    # Should have fewer or equal trades
    assert len(trades_with_filter) <= len(trades_no_filter)
    
    # Check criteria
    if not trades_with_filter.empty:
        entry_dates = trades_with_filter['EntryDate']
        entry_rows = df[df['Date'].isin(entry_dates)]
        
        assert (entry_rows['MACD_Hist_Rising'] == True).all()