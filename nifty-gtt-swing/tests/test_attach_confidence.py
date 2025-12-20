import pytest
import pandas as pd
import numpy as np

from src.gtt_sizing import context_from_row, compute_decision_confidence
from src.success_model import lookup_confidence


def test_context_from_row():
    """Test extracting context buckets from a plan row."""
    # Create sample row
    row = pd.Series({
        'RSI14': 25,  # Oversold
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 1.8,  # >1.5
        'ATR14': 15,
        'Close': 1000,  # ATRpct = 15/1000*100 = 1.5% -> 1–2%
        'ENTRY_trigger_price': 1000
    })
    
    context = context_from_row(row)
    
    assert context['RSI14_Status'] == 'Oversold'
    assert context['GoldenBull_Flag'] == 1
    assert context['GoldenBear_Flag'] == 0
    assert context['Trend_OK'] == 1
    assert context['RVOL20_bucket'] == '>1.5'
    assert context['ATRpct_bucket'] == '1–2%'


def test_compute_decision_confidence_with_ci():
    """Test confidence computation with CI using hierarchical lookup."""
    # Create sample model with CI columns
    model_data = pd.DataFrame({
        'Strategy': ['SEPA', 'SEPA', 'Donchian'],
        'Sector': ['IT', 'Finance', 'IT'],
        'Symbol': ['TCS', 'HDFC', 'INFY'],
        'RSI14_Status': ['Neutral', 'Neutral', 'Overbought'],
        'GoldenBull_Flag': [1, 0, 1],
        'GoldenBear_Flag': [0, 0, 0],
        'Trend_OK': [1, 1, 1],
        'RVOL20_bucket': ['1.0–1.5', '1.0–1.5', '>1.5'],
        'ATRpct_bucket': ['1–2%', '1–2%', '<1%'],
        'Trades_OOS': [50, 30, 25],
        'Weighted_Wins': [35, 18, 15],
        'OOS_WinRate_raw': [0.7, 0.6, 0.6],
        'CalibratedWinRate': [0.68, 0.62, 0.58],
        'CI_low': [0.55, 0.48, 0.42],
        'CI_high': [0.78, 0.74, 0.72],
        'OOS_ExpectancyR': [0.45, 0.35, 0.40],
        'Reliability': [1.0, 1.0, 0.83],
        'WFO_Efficiency': [1.0, 1.0, 1.0],
        'CoverageNote': ['ContextExact', 'SymbolPool', 'StrategyPrior']
    })
    
    # Test row with exact match
    row = pd.Series({
        'Strategy': 'SEPA',
        'Symbol': 'TCS',
        'RSI14': 55,  # Neutral
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 1.2,  # 1.0–1.5
        'ATR14': 15,
        'Close': 1000  # ATRpct = 1.5%
    })
    
    result = compute_decision_confidence(row, model_data)
    
    assert 'DecisionConfidence' in result
    assert 'CI_low' in result
    assert 'CI_high' in result
    assert 'Confidence_Reason' in result
    assert 'CoverageNote' in result
    
    # Check CI bounds
    assert result['CI_low'] < result['DecisionConfidence'] < result['CI_high']
    assert 0.05 <= result['DecisionConfidence'] <= 0.95
    
    # Check reason string contains key info
    reason = result['Confidence_Reason']
    assert 'SEPA' in reason
    assert 'Neutral' in reason
    assert 'GBull=1' in reason
    assert 'calibrated=' in reason


def test_compute_decision_confidence_composite_ensemble():
    """Test composite score ensemble confidence."""
    # Create model with multiple strategies
    model_data = pd.DataFrame({
        'Strategy': ['SEPA', 'Donchian', 'VCP'],
        'Sector': ['IT', 'IT', 'IT'],
        'Symbol': ['TCS', 'TCS', 'TCS'],
        'RSI14_Status': ['Neutral', 'Neutral', 'Neutral'],
        'GoldenBull_Flag': [1, 1, 1],
        'GoldenBear_Flag': [0, 0, 0],
        'Trend_OK': [1, 1, 1],
        'RVOL20_bucket': ['1.0–1.5', '1.0–1.5', '1.0–1.5'],
        'ATRpct_bucket': ['1–2%', '1–2%', '1–2%'],
        'Trades_OOS': [50, 40, 35],
        'Weighted_Wins': [35, 28, 21],
        'OOS_WinRate_raw': [0.7, 0.7, 0.6],
        'CalibratedWinRate': [0.68, 0.67, 0.59],
        'CI_low': [0.55, 0.53, 0.45],
        'CI_high': [0.78, 0.78, 0.71],
        'OOS_ExpectancyR': [0.45, 0.42, 0.38],
        'Reliability': [1.0, 1.0, 1.0],
        'WFO_Efficiency': [1.0, 1.0, 1.0],
        'CoverageNote': ['ContextExact', 'ContextExact', 'ContextExact']
    })
    
    # Test composite row
    row = pd.Series({
        'Strategy': 'CompositeScore',
        'Symbol': 'TCS',
        'RSI14': 55,
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 1.2,
        'ATR14': 15,
        'Close': 1000
    })
    
    result = compute_decision_confidence(row, model_data)
    
    assert 'DecisionConfidence' in result
    assert 'CI_low' in result
    assert 'CI_high' in result
    
    # Ensemble should blend the three strategies
    assert result['CI_low'] < result['DecisionConfidence'] < result['CI_high']
    
    # Check reason mentions ensemble
    reason = result['Confidence_Reason']
    assert 'ensemble' in reason.lower()
    assert '3 strategies' in reason


def test_compute_decision_confidence_backoff():
    """Test backoff logic when exact match not found."""
    # Model with limited data
    model_data = pd.DataFrame({
        'Strategy': ['SEPA'],
        'Sector': ['IT'],
        'Symbol': ['TCS'],
        'RSI14_Status': ['Neutral'],
        'GoldenBull_Flag': [1],
        'GoldenBear_Flag': [0],
        'Trend_OK': [1],
        'RVOL20_bucket': ['1.0–1.5'],
        'ATRpct_bucket': ['1–2%'],
        'Trades_OOS': [10],  # Small sample
        'Weighted_Wins': [6],
        'OOS_WinRate_raw': [0.6],
        'CalibratedWinRate': [0.58],
        'CI_low': [0.35],
        'CI_high': [0.78],
        'OOS_ExpectancyR': [0.35],
        'Reliability': [0.33],  # 10/30 = 0.33
        'WFO_Efficiency': [1.0],
        'CoverageNote': ['ContextExact']
    })
    
    # Row with different context (should back off)
    row = pd.Series({
        'Strategy': 'SEPA',
        'Symbol': 'INFY',  # Different symbol
        'RSI14': 75,  # Overbought (different from Neutral)
        'GoldenBull_Flag': 0,  # Different
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 0.8,  # <1.0 (different bucket)
        'ATR14': 25,
        'Close': 1000  # ATRpct = 2.5% -> >2%
    })
    
    result = compute_decision_confidence(row, model_data)
    
    # Should still return valid confidence (fallback to priors)
    assert 'DecisionConfidence' in result
    assert isinstance(result['DecisionConfidence'], (int, float))
    assert 0.05 <= result['DecisionConfidence'] <= 0.95


def test_compute_decision_confidence_empty_model():
    """Test confidence with empty model."""
    row = pd.Series({
        'Strategy': 'SEPA',
        'RSI14': 55,
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 1.2,
        'ATR14': 15,
        'Close': 1000
    })
    
    result = compute_decision_confidence(row, pd.DataFrame())
    
    # Should return defaults
    assert result['DecisionConfidence'] == 0.5
    assert result['CI_low'] == 0.4
    assert result['CI_high'] == 0.6
    assert result['OOS_WinRate'] == 0.5
    assert result['Trades_OOS'] == 0
    assert 'No OOS data' in result['Confidence_Reason']


def test_ci_bounds_validity():
    """Test that CI bounds are always valid."""
    # Create model with various scenarios
    model_data = pd.DataFrame({
        'Strategy': ['SEPA', 'SEPA'],
        'Sector': ['IT', 'IT'],
        'Symbol': ['TCS', 'INFY'],
        'RSI14_Status': ['Neutral', 'Overbought'],
        'GoldenBull_Flag': [1, 0],
        'GoldenBear_Flag': [0, 0],
        'Trend_OK': [1, 1],
        'RVOL20_bucket': ['1.0–1.5', '>1.5'],
        'ATRpct_bucket': ['1–2%', '<1%'],
        'Trades_OOS': [100, 5],  # High and low confidence
        'Weighted_Wins': [75, 3],
        'OOS_WinRate_raw': [0.75, 0.6],
        'CalibratedWinRate': [0.73, 0.55],
        'CI_low': [0.65, 0.25],
        'CI_high': [0.80, 0.80],
        'OOS_ExpectancyR': [0.50, 0.30],
        'Reliability': [1.0, 0.17],
        'WFO_Efficiency': [1.0, 1.0],
        'CoverageNote': ['ContextExact', 'GlobalPrior']
    })
    
    for _, model_row in model_data.iterrows():
        # CI should always be valid
        assert 0 <= model_row['CI_low'] <= 1
        assert 0 <= model_row['CI_high'] <= 1
        assert model_row['CI_low'] < model_row['CI_high']
        
        # For high confidence case
        if model_row['Trades_OOS'] == 100:
            assert model_row['CI_high'] - model_row['CI_low'] < 0.3  # Narrow CI
        
        # For low confidence case
        if model_row['Trades_OOS'] == 5:
            assert model_row['CI_high'] - model_row['CI_low'] > 0.4  # Wide CI
    model = pd.DataFrame(model_data)
    
    # Create matching row
    row = pd.Series({
        'Strategy': 'SEPA',  # Match the strategy in model
        'Sector': 'IT',  # Match the sector
        'Symbol': 'TCS',  # Match the symbol
        'RSI14': 55,
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 1.2,
        'ATR14': 15,
        'Close': 1000
    })
    
    result = compute_decision_confidence(row, model)
    
    # Should match first row: calibrated=0.73, reliability=1.0, wfo=1.0
    expected_conf = 0.73 * 1.0 * 1.0  # 0.73
    assert abs(result['DecisionConfidence'] - expected_conf) < 1e-3
    assert result['OOS_WinRate'] == 0.75
    assert result['OOS_ExpectancyR'] == 0.50
    assert result['Trades_OOS'] == 100
    assert 'Strategy=SEPA' in result['Confidence_Reason']


def test_compute_decision_confidence_backoff():
    """Test confidence computation with backoff to broader buckets."""
    model_data = {
        'Strategy': ['MR'],
        'RSI14_Status': ['Oversold'],
        'GoldenBull_Flag': [1],
        'GoldenBear_Flag': [0],
        'Trend_OK': [1],
        'RVOL20_bucket': ['1.0–1.5'],  # Different RVOL
        'ATRpct_bucket': ['<1%'],  # Different ATR
        'Trades_OOS': [40],
        'Wins_OOS': [28],
        'OOS_WinRate': [0.7],
        'OOS_ExpectancyR': [0.4],
        'CalibratedWinRate': [0.683],
        'Reliability': [1.0],
        'WFO_Efficiency': [1.0]
    }
    model = pd.DataFrame(model_data)
    
    # Row that doesn't match RVOL or ATR
    row = pd.Series({
        'Strategy': 'MR',
        'RSI14': 25,
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 1.8,  # >1.5, not 1.0–1.5
        'ATR14': 20,    # 2%, not <1%
        'Close': 1000
    })
    
    result = compute_decision_confidence(row, model)
    
    # Should back off and still match on RSI/Golden/Trend
    expected_conf = 0.683 * 1.0 * 1.0
    assert abs(result['DecisionConfidence'] - expected_conf) < 1e-3


def test_compute_decision_confidence_composite_ensemble():
    """Test ensemble confidence for CompositeScore strategy."""
    # Model with multiple strategies
    model_data = {
        'Strategy': ['MR', 'Donchian', 'SEPA'],
        'Sector': ['IT', 'IT', 'IT'],
        'Symbol': ['TCS', 'TCS', 'TCS'],
        'RSI14_Status': ['Neutral', 'Neutral', 'Neutral'],
        'GoldenBull_Flag': [1, 1, 1],
        'GoldenBear_Flag': [0, 0, 0],
        'Trend_OK': [1, 1, 1],
        'RVOL20_bucket': ['1.0–1.5', '1.0–1.5', '1.0–1.5'],
        'ATRpct_bucket': ['1–2%', '1–2%', '1–2%'],
        'Trades_OOS': [100, 80, 60],  # Different trade counts for weighting
        'Weighted_Wins': [70, 48, 48],
        'CalibratedWinRate': [0.7, 0.6, 0.8],
        'CI_low': [0.62, 0.52, 0.72],
        'CI_high': [0.78, 0.68, 0.88],
        'Reliability': [1.0, 1.0, 1.0],
        'WFO_Efficiency': [1.0, 1.0, 1.0],
        'OOS_WinRate_raw': [0.7, 0.6, 0.8],
        'OOS_ExpectancyR': [0.4, 0.3, 0.5],
        'CoverageNote': ['ContextExact', 'ContextExact', 'ContextExact']
    }
    model = pd.DataFrame(model_data)
    
    row = pd.Series({
        'Strategy': 'CompositeScore',
        'RSI14': 55,
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20': 1.2,
        'ATR14': 15,
        'Close': 1000
    })
    
    result = compute_decision_confidence(row, model)
    
    # Ensemble: weighted average by Trades_OOS
    # MR: 0.7 * (100/240) = 0.7 * 0.417 = 0.292
    # Donchian: 0.6 * (80/240) = 0.6 * 0.333 = 0.200
    # SEPA: 0.8 * (60/240) = 0.8 * 0.250 = 0.200
    # Total: 0.692
    expected_ensemble = (0.7 * 100 + 0.6 * 80 + 0.8 * 60) / (100 + 80 + 60)
    assert abs(result['DecisionConfidence'] - expected_ensemble) < 1e-3
    assert 'Composite ensemble of 3 strategies' in result['Confidence_Reason']


def test_compute_decision_confidence_bounds():
    """Test that confidence is clipped to [0.05, 0.95]."""
    model_data = {
        'Strategy': ['MR'],
        'Sector': ['IT'],
        'Symbol': ['TCS'],
        'RSI14_Status': ['Neutral'],
        'GoldenBull_Flag': [0],
        'GoldenBear_Flag': [0],
        'Trend_OK': [0],
        'RVOL20_bucket': ['<1.0'],
        'ATRpct_bucket': ['<1%'],
        'Trades_OOS': [5],  # Low trades
        'Weighted_Wins': [0.5],  # Beta smoothing
        'OOS_WinRate_raw': [0.0],
        'OOS_ExpectancyR': [-0.5],
        'CalibratedWinRate': [0.083],  # (0+0.5)/(5+1) = 0.5/6 ≈ 0.083
        'CI_low': [0.01],
        'CI_high': [0.25],
        'Reliability': [5/30],  # 0.167
        'WFO_Efficiency': [1.0],
        'CoverageNote': ['GlobalPrior']
    }
    model = pd.DataFrame(model_data)
    
    row = pd.Series({
        'Strategy': 'MR',
        'RSI14': 55,
        'GoldenBull_Flag': 0,
        'GoldenBear_Flag': 0,
        'Trend_OK': 0,
        'RVOL20': 0.8,
        'ATR14': 5,
        'Close': 1000
    })
    
    result = compute_decision_confidence(row, model)
    
    # 0.083 * 0.167 * 1.0 ≈ 0.014, should be clipped to 0.05
    assert result['DecisionConfidence'] == 0.05


def test_compute_decision_confidence_empty_model():
    """Test confidence computation with empty model."""
    model = pd.DataFrame()
    
    row = pd.Series({
        'Strategy': 'MR',
        'RSI14': 55
    })
    
    result = compute_decision_confidence(row, model)
    
    assert result['DecisionConfidence'] == 0.5
    assert result['OOS_WinRate'] == 0.5
    assert result['OOS_ExpectancyR'] == 0.0
    assert result['Trades_OOS'] == 0
    assert 'pooling=NoModel' in result['Confidence_Reason']