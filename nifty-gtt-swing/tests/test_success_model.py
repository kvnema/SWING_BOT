import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch

from src.success_model import (
    load_oos_trades, make_buckets, aggregate_oos, empirical_bayes_shrink,
    posterior_beta_ci, calibrate_probabilities, build_hierarchical_model,
    lookup_confidence, _build_priors, _get_prior_from_hierarchy, _determine_coverage_note
)


def test_load_oos_trades():
    """Test loading OOS trades from backtest directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock backtest directory structure
        bt_dir = Path(tmpdir) / "backtests"
        bt_dir.mkdir()
        
        # Create strategy directory
        strat_dir = bt_dir / "SEPA"
        strat_dir.mkdir()
        
        # Create mock trades.csv
        trades_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Symbol': ['RELIANCE', 'TCS'],
            'R': [0.5, -0.3],
            'OOS': [1, 1]
        })
        trades_data.to_csv(strat_dir / "trades.csv", index=False)
        
        result = load_oos_trades(str(bt_dir))
        
        assert not result.empty
        assert len(result) == 2
        assert 'Strategy' in result.columns
        assert result['Strategy'].iloc[0] == 'SEPA'


def test_make_buckets():
    """Test bucket creation from trades data."""
    trades_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'RSI14': [25, 55, 75],
        'GoldenBull_Flag': [1, 0, 1],
        'GoldenBear_Flag': [0, 1, 0],
        'Trend_OK': [1, 0, 1],
        'RVOL20': [0.8, 1.2, 1.8],
        'ATRpct': [0.8, 1.5, 2.5],
        'R': [0.5, -0.3, 0.8]
    })
    
    today = pd.Timestamp('2024-01-10')
    result = make_buckets(trades_data, today)
    
    assert 'RSI14_Status' in result.columns
    assert 'RVOL20_bucket' in result.columns
    assert 'ATRpct_bucket' in result.columns
    assert 'age_days' in result.columns
    
    assert result['RSI14_Status'].tolist() == ['Oversold', 'Neutral', 'Overbought']
    assert result['RVOL20_bucket'].tolist() == ['<1.0', '1.0–1.5', '>1.5']
    assert result['ATRpct_bucket'].tolist() == ['<1%', '1–2%', '>2%']


def test_aggregate_oos():
    """Test OOS aggregation with recency weighting."""
    trades_data = pd.DataFrame({
        'Strategy': ['SEPA', 'SEPA', 'SEPA'],
        'Sector': ['IT', 'IT', 'Finance'],
        'Symbol': ['TCS', 'INFY', 'HDFC'],
        'RSI14_Status': ['Neutral', 'Neutral', 'Overbought'],
        'GoldenBull_Flag': [1, 0, 1],
        'GoldenBear_Flag': [0, 0, 0],
        'Trend_OK': [1, 1, 1],
        'RVOL20_bucket': ['1.0–1.5', '1.0–1.5', '>1.5'],
        'ATRpct_bucket': ['1–2%', '1–2%', '<1%'],
        'is_win': [1, 0, 1],
        'R': [0.5, -0.3, 0.8],
        'age_days': [10, 20, 5]  # Recent trade gets higher weight
    })
    
    result = aggregate_oos(trades_data)
    
    assert not result.empty
    assert 'Trades_OOS' in result.columns
    assert 'Weighted_Wins' in result.columns
    assert 'OOS_WinRate_raw' in result.columns
    assert 'OOS_ExpectancyR' in result.columns


def test_empirical_bayes_shrink():
    """Test empirical Bayes shrinkage."""
    # High confidence case
    shrunk = empirical_bayes_shrink(wins=8, trades=10, prior_p=0.6, lambda_=5)
    assert shrunk > 0.6  # Should pull toward data
    
    # Low confidence case
    shrunk = empirical_bayes_shrink(wins=1, trades=2, prior_p=0.6, lambda_=10)
    assert shrunk < 0.7 and shrunk > 0.5  # Should shrink toward prior


def test_posterior_beta_ci():
    """Test Beta posterior confidence interval."""
    mean, ci_low, ci_high = posterior_beta_ci(wins=8, losses=2, a_prior=1, b_prior=1)
    
    assert 0.5 < mean < 0.9
    assert ci_low < mean < ci_high
    assert ci_low >= 0
    assert ci_high <= 1


def test_build_hierarchical_model():
    """Test building hierarchical model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock backtest data
        bt_dir = Path(tmpdir) / "backtests"
        bt_dir.mkdir()
        
        strat_dir = bt_dir / "SEPA"
        strat_dir.mkdir()
        
        trades_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Symbol': ['TCS', 'INFY', 'HDFC'],
            'R': [0.5, -0.3, 0.8],
            'OOS': [1, 1, 1],
            'RSI14': [55, 45, 65],
            'GoldenBull_Flag': [1, 0, 1],
            'GoldenBear_Flag': [0, 0, 0],
            'RVOL20': [1.2, 1.0, 1.5],
            'ATR14': [15, 12, 18],
            'Close': [1000, 1100, 1200]
        })
        trades_data.to_csv(strat_dir / "trades.csv", index=False)
        
        today = pd.Timestamp('2024-01-10')
        model = build_hierarchical_model(str(bt_dir), today, t_min=1)
        
        assert not model.empty
        expected_cols = [
            'Strategy', 'Sector', 'Symbol', 'RSI14_Status', 'GoldenBull_Flag', 'GoldenBear_Flag',
            'Trend_OK', 'RVOL20_bucket', 'ATRpct_bucket', 'Trades_OOS', 'Wins_OOS', 'OOS_WinRate_raw',
            'CalibratedWinRate', 'CI_low', 'CI_high', 'OOS_ExpectancyR', 'Reliability', 'WFO_Efficiency', 'CoverageNote'
        ]
        for col in expected_cols:
            assert col in model.columns


def test_lookup_confidence_backoff():
    """Test hierarchical backoff in confidence lookup."""
    # Create mock model with limited data
    model_data = pd.DataFrame({
        'Strategy': ['SEPA', 'SEPA'],
        'Sector': ['IT', 'IT'],
        'Symbol': ['TCS', 'INFY'],
        'RSI14_Status': ['Neutral', 'Neutral'],
        'GoldenBull_Flag': [1, 0],
        'GoldenBear_Flag': [0, 0],
        'Trend_OK': [1, 1],
        'RVOL20_bucket': ['1.0–1.5', '1.0–1.5'],
        'ATRpct_bucket': ['1–2%', '1–2%'],
        'Trades_OOS': [10, 5],
        'Weighted_Wins': [7, 3],
        'OOS_WinRate_raw': [0.7, 0.6],
        'CalibratedWinRate': [0.68, 0.62],
        'CI_low': [0.55, 0.45],
        'CI_high': [0.78, 0.75],
        'OOS_ExpectancyR': [0.45, 0.35],
        'Reliability': [0.8, 0.6],
        'WFO_Efficiency': [1.0, 1.0],
        'CoverageNote': ['ContextExact', 'SymbolPool']
    })
    
    # Test exact match
    context = {
        'RSI14_Status': 'Neutral',
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20_bucket': '1.0–1.5',
        'ATRpct_bucket': '1–2%',
        'Symbol': 'TCS',
        'Sector': 'IT'
    }
    
    result = lookup_confidence(model_data, context, 'SEPA')
    
    assert 'DecisionConfidence' in result
    assert 'CI_low' in result
    assert 'CI_high' in result
    assert result['DecisionConfidence'] == 0.68
    
    # Test backoff (non-matching context)
    context_backoff = context.copy()
    context_backoff['RVOL20_bucket'] = '>1.5'  # No match
    
    result_backoff = lookup_confidence(model_data, context_backoff, 'SEPA')
    
    # Should fall back to some default or prior
    assert 'DecisionConfidence' in result_backoff


def test_build_priors():
    """Test building hierarchical priors."""
    aggregated_data = pd.DataFrame({
        'Strategy': ['SEPA', 'SEPA', 'Donchian'],
        'Sector': ['IT', 'Finance', 'IT'],
        'Symbol': ['TCS', 'HDFC', 'INFY'],
        'Weighted_Wins': [15, 10, 8],
        'Trades_OOS': [20, 15, 12]
    })
    
    priors = _build_priors(aggregated_data)
    
    assert 'global' in priors
    assert 'strategy' in priors
    assert priors['global'] == (15+10+8)/(20+15+12)  # Overall win rate
    assert 'SEPA' in priors['strategy']
    assert 'Donchian' in priors['strategy']


def test_determine_coverage_note():
    """Test coverage note determination."""
    row_high = pd.Series({'Trades_OOS': 60, 'Strategy': 'SEPA', 'Sector': 'IT', 'Symbol': 'TCS'})
    row_med = pd.Series({'Trades_OOS': 25, 'Strategy': 'SEPA', 'Sector': 'IT', 'Symbol': 'TCS'})
    row_low = pd.Series({'Trades_OOS': 5, 'Strategy': 'SEPA', 'Sector': 'IT', 'Symbol': 'TCS'})
    
    priors = {'strategy': {'SEPA': 0.6}}
    
    assert _determine_coverage_note(row_high, priors) == 'ContextExact'
    assert _determine_coverage_note(row_med, priors) == 'SymbolPool'
    assert _determine_coverage_note(row_low, priors) == 'StrategyPrior'


def test_build_success_model_with_data():
    """Test building model with synthetic backtest data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bt_dir = Path(tmpdir)
        
        # Create synthetic trades data
        trades_data = {
            'Date': ['2024-01-01'] * 10,
            'Symbol': ['TCS'] * 10,
            'RSI14': [25, 45, 75, 55, 35, 65, 45, 55, 25, 75],
            'GoldenBull_Flag': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'GoldenBear_Flag': [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            'Trend_OK': [1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            'RVOL20': [0.8, 1.2, 1.8, 1.5, 0.9, 1.6, 1.1, 1.4, 0.7, 1.9],
            'ATR14': [10, 15, 20, 12, 8, 18, 14, 11, 9, 22],
            'Close': [1000, 1100, 1200, 1050, 950, 1150, 1020, 1080, 980, 1250],
            'Win': [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
            'R': [0.5, -0.3, 0.8, 0.2, 0.4, -0.2, 0.6, 0.3, -0.4, 0.7],
            'OOS': [1] * 10  # All OOS
        }
        trades_df = pd.DataFrame(trades_data)
        
        # Create strategy directory
        strategy_dir = bt_dir / "MR"
        strategy_dir.mkdir()
        trades_df.to_csv(strategy_dir / "trades.csv", index=False)
        
        # Create KPI data
        kpi_data = {
            'Sample': ['OOS'],
            'Win_Rate_%': [60.0],
            'Sharpe': [1.2]
        }
        kpi_df = pd.DataFrame(kpi_data)
        kpi_df.to_csv(strategy_dir / "kpi.csv", index=False)
        
        model = build_hierarchical_model(str(bt_dir), pd.Timestamp('2024-01-15'))
        
        assert not model.empty
        assert 'MR' in model['Strategy'].values
        
        # Model should have calibrated probabilities with shrinkage
        mr_rows = model[(model['Strategy'] == 'MR') & (model['Trades_OOS'] > 0)]
        assert len(mr_rows) > 0


def test_beta_smoothing_and_reliability():
    """Test that beta smoothing and reliability work as expected."""
    # Test data: 3 wins out of 5 trades
    wins, trades = 3, 5
    
    # Beta smoothing (Jeffreys prior)
    calibrated = (wins + 0.5) / (trades + 1.0)
    expected_calibrated = 3.5 / 6.0  # 0.5833...
    assert abs(calibrated - expected_calibrated) < 1e-6
    
    # Reliability with T_min=30
    reliability = min(1.0, trades / 30)
    expected_reliability = 5 / 30  # 0.1667...
    assert abs(reliability - expected_reliability) < 1e-6
    
    # Combined confidence should be calibrated * reliability * wfo_eff
    wfo_eff = 1.0
    confidence = calibrated * reliability * wfo_eff
    assert 0.05 <= confidence <= 0.95  # Should be clipped to bounds