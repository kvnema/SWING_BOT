import pandas as pd
from src.success_model import make_buckets
from src.gtt_sizing import context_from_row


def test_make_buckets_macd_regime():
    """Test that make_buckets adds MACD_Regime and MACD_Cross_Status."""
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=10),
        'Symbol': 'TEST',
        'Strategy': 'TestStrat',
        'is_win': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'MACD_Line': [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5],
        'MACD_CrossUp': [True, False, True, False, True, False, True, False, True, False]
    })
    
    buckets_df = make_buckets(df, pd.Timestamp('2023-01-01'))
    
    # Should have MACD_Regime
    assert 'MACD_Regime' in buckets_df.columns
    assert 'AboveZero' in buckets_df['MACD_Regime'].values
    assert 'BelowZero' in buckets_df['MACD_Regime'].values
    
    # Should have MACD_Cross_Status
    assert 'MACD_Cross_Status' in buckets_df.columns
    assert 'CrossUp' in buckets_df['MACD_Cross_Status'].values
    assert 'NoCross' in buckets_df['MACD_Cross_Status'].values


def test_context_from_row_macd():
    """Test that context_from_row extracts MACD_Regime and MACD_Cross_Status."""
    row = pd.Series({
        'RSI14': 55,
        'MACD_Line': 0.2,
        'MACD_CrossUp': True,
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 1.2,
        'ATR14': 1.0,
        'Close': 100
    })
    
    context = context_from_row(row)
    
    assert 'MACD_Regime' in context
    assert context['MACD_Regime'] == 'AboveZero'
    assert 'MACD_Cross_Status' in context
    assert context['MACD_Cross_Status'] == 'CrossUp'


def test_aggregate_oos_includes_macd():
    """Test that aggregate_oos groups by MACD buckets."""
    from src.success_model import aggregate_oos
    
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=20),
        'Symbol': 'TEST',
        'Strategy': 'TestStrat',
        'is_win': [1] * 10 + [0] * 10,
        'R': [0.1] * 10 + [-0.1] * 10,  # Add R column
        'MACD_Line': [0.1] * 10 + [-0.1] * 10,
        'MACD_CrossUp': [True] * 10 + [False] * 10,
        'RSI14': [50] * 20,
        'GoldenBull_Flag': [1] * 20,
        'GoldenBear_Flag': [0] * 20,
        'Trend_OK': [1] * 20,
        'RVOL20': [1.2] * 20,
        'ATRpct': [1.5] * 20,
        'Sector': 'TestSector'
    })
    
    buckets_df = make_buckets(df, pd.Timestamp('2023-01-01'))
    aggregated = aggregate_oos(buckets_df)
    
    # Should have groups with different MACD regimes
    macd_regimes = aggregated['MACD_Regime'].unique()
    assert 'AboveZero' in macd_regimes
    assert 'BelowZero' in macd_regimes
    
    macd_crosses = aggregated['MACD_Cross_Status'].unique()
    assert 'CrossUp' in macd_crosses
    assert 'NoCross' in macd_crosses