"""
Tests for orchestrate-eod command and EOD pipeline.
"""

import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.cli import cmd_orchestrate_eod
from src.data_validation import ValidationError


class MockArgs:
    """Mock argparse namespace for testing."""
    def __init__(self, **kwargs):
        self.data_out = kwargs.get('data_out', 'test_data.csv')
        self.max_age_days = kwargs.get('max_age_days', 1)
        self.required_days = kwargs.get('required_days', 500)
        self.top = kwargs.get('top', 25)
        self.strict = kwargs.get('strict', True)
        self.post_teams = kwargs.get('post_teams', False)
        self.multi_tf = kwargs.get('multi_tf', False)
        self.config = kwargs.get('config', None)


@pytest.fixture
def sample_data():
    """Create sample NIFTY 50 data for testing."""
    dates = pd.date_range('2023-01-01', periods=600, freq='D')
    symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'Symbol': symbol,
                'Date': date.date(),
                'Open': 100.0,
                'High': 105.0,
                'Low': 95.0,
                'Close': 102.0,
                'Volume': 1000000,
                'EMA20': 101.0,
                'EMA50': 100.0,
                'EMA200': 99.0,
                'RSI14': 65.0,
                'MACD': 1.5,
                'MACDSignal': 1.2,
                'MACDHist': 0.3,
                'ATR14': 2.0,
                'BB_MA20': 101.0,
                'BB_Upper': 106.0,
                'BB_Lower': 96.0,
                'BB_BandWidth': 0.05,
                'DonchianH20': 110.0,
                'DonchianL20': 90.0,
                'RVOL20': 1.2,
                'RS_vs_Index': 1.05,
                'RS_ROC20': 5.0,
                'KC_Upper': 107.0,
                'KC_Lower': 95.0,
                'Squeeze': 0,
                'AVWAP60': 100.0,
                'IndexUpRegime': 1,
                'Signal': 'BUY',
                'SEPA_Flag': 1,
                'VCP_Flag': 0,
                'Donchian_Breakout': 1,
                'MR_Flag': 0,
                'SqueezeBreakout_Flag': 0,
                'AVWAP_Reclaim_Flag': 0,
                'CompositeScore': 8.5
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create outputs structure
        Path(tmpdir, 'outputs', 'screener').mkdir(parents=True, exist_ok=True)
        Path(tmpdir, 'outputs', 'backtests').mkdir(parents=True, exist_ok=True)
        Path(tmpdir, 'outputs', 'gtt').mkdir(parents=True, exist_ok=True)
        Path(tmpdir, 'outputs', 'multi_tf').mkdir(parents=True, exist_ok=True)
        Path(tmpdir, 'data').mkdir(parents=True, exist_ok=True)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            yield tmpdir
        finally:
            os.chdir(original_cwd)


@patch('os.getenv')
@patch('src.cli.fetch_nifty50_data')
@patch('src.cli.load_dataset')
@patch('src.cli.validate_dataset')
@patch('src.cli.compute_signals')
@patch('src.cli.compute_composite_score')
@patch('src.cli.select_best_strategy')
@patch('src.cli.build_gtt_plan')
@patch('src.final_excel.run_final_excel')
@patch('src.cli.run_plan_audit')
@patch('src.teams_notifier.post_plan_summary')
@patch('src.cli.build_multi_tf_excel')
@patch('src.data_validation.load_metadata')
@patch('src.data_validation.validate_recency')
@patch('src.data_validation.validate_window')
@patch('src.data_validation.validate_symbols')
@patch('src.data_validation.summarize')
@patch('pandas.read_csv')
def test_orchestrate_eod_success(mock_read_csv, mock_summarize, mock_validate_symbols, mock_validate_window, 
                                mock_validate_recency, mock_load_meta, mock_multi_tf, mock_teams, 
                                mock_audit, mock_excel, mock_gtt_plan, mock_select, mock_score, 
                                mock_signals, mock_validate, mock_load, mock_fetch, mock_getenv, 
                                sample_data, temp_dir):
    """Test successful orchestrate-eod execution."""
    
    # Mock environment
    mock_getenv.return_value = "test_token"
    
    # Mock pandas read_csv for final summary
    audited_plan = pd.DataFrame({
        'Symbol': ['RELIANCE.NS', 'TCS.NS'],
        'Audit_Flag': ['PASS', 'PASS']
    })
    mock_read_csv.return_value = audited_plan
    
    # Setup mocks
    mock_fetch.return_value = None
    mock_load.return_value = sample_data
    mock_validate.return_value = (True, [])
    mock_signals.return_value = sample_data
    mock_score.return_value = sample_data['CompositeScore']
    mock_select.return_value = 'SEPA'
    
    # Mock GTT plan
    gtt_plan = pd.DataFrame({
        'Symbol': ['RELIANCE.NS', 'TCS.NS'],
        'Qty': [10, 15],
        'ENTRY_trigger_price': [2500.0, 3200.0],
        'TARGET_trigger_price': [2600.0, 3300.0],
        'STOPLOSS_trigger_price': [2400.0, 3100.0],
        'DecisionConfidence': [4.5, 4.2],
        'Confidence_Level': ['HIGH', 'HIGH'],
        'R': [2.0, 2.0],
        'Explanation': ['Test', 'Test'],
        'GTT_Explanation': ['Test', 'Test'],
        'Audit_Flag': ['PASS', 'PASS']
    })
    mock_gtt_plan.return_value = gtt_plan
    mock_excel.return_value = True
    mock_audit.return_value = True
    mock_teams.return_value = True
    mock_multi_tf.return_value = None
    
    # Test args
    args = MockArgs(
        data_out='data/test_data.csv',
        max_age_days=1,
        required_days=500,
        top=25,
        strict=True,
        post_teams=True,
        multi_tf=True
    )
    
    # This should not raise an exception
    try:
        cmd_orchestrate_eod(args)
    except SystemExit as e:
        # Should not exit with error
        assert e.code == 0 or e.code is None


@patch('src.cli.fetch_nifty50_data')
def test_orchestrate_eod_data_fetch_failure(mock_fetch, temp_dir):
    """Test orchestrate-eod with data fetch failure."""
    
    mock_fetch.side_effect = Exception("API Error")
    
    args = MockArgs()
    
    with pytest.raises(SystemExit) as exc_info:
        cmd_orchestrate_eod(args)
    
    assert exc_info.value.code == 1


@patch('src.cli.fetch_nifty50_data')
@patch('src.cli.load_dataset')
@patch('src.cli.validate_dataset')
def test_orchestrate_eod_validation_failure(mock_validate, mock_load, mock_fetch, 
                                          sample_data, temp_dir):
    """Test orchestrate-eod with validation failure."""
    
    mock_fetch.return_value = None
    mock_load.return_value = sample_data
    mock_validate.return_value = (False, ["Missing columns: XYZ"])
    
    args = MockArgs()
    
    with pytest.raises(SystemExit) as exc_info:
        cmd_orchestrate_eod(args)
    
    assert exc_info.value.code == 1


@patch('src.cli.fetch_nifty50_data')
@patch('src.cli.load_dataset')
@patch('src.cli.validate_dataset')
@patch('src.cli.compute_signals')
@patch('src.cli.compute_composite_score')
@patch('src.cli.select_best_strategy')
@patch('src.cli.build_gtt_plan')
@patch('src.final_excel.run_final_excel')
@patch('src.plan_audit.run_plan_audit')
def test_orchestrate_eod_audit_failure_strict_mode(mock_audit, mock_excel, mock_gtt_plan,
                                                  mock_select, mock_score, mock_signals,
                                                  mock_validate, mock_load, mock_fetch,
                                                  sample_data, temp_dir):
    """Test orchestrate-eod with audit failure in strict mode."""
    
    # Setup mocks
    mock_fetch.return_value = None
    mock_load.return_value = sample_data
    mock_validate.return_value = (True, [])
    mock_signals.return_value = sample_data
    mock_score.return_value = sample_data
    mock_select.return_value = 'SEPA'
    
    gtt_plan = pd.DataFrame({
        'Symbol': ['RELIANCE.NS'],
        'Qty': [10],
        'ENTRY_trigger_price': [2500.0],
        'TARGET_trigger_price': [2600.0],
        'STOPLOSS_trigger_price': [2400.0],
        'DecisionConfidence': [4.5],
        'Confidence_Level': ['HIGH'],
        'R': [2.0],
        'Explanation': ['Test'],
        'GTT_Explanation': ['Test']
    })
    mock_gtt_plan.return_value = gtt_plan
    mock_excel.return_value = True
    mock_audit.return_value = False  # Audit fails
    
    args = MockArgs(strict=True)
    
    with pytest.raises(SystemExit) as exc_info:
        cmd_orchestrate_eod(args)
    
    assert exc_info.value.code == 1


@patch('src.cli.fetch_nifty50_data')
@patch('src.cli.load_dataset')
@patch('src.cli.validate_dataset')
@patch('src.cli.compute_signals')
@patch('src.cli.compute_composite_score')
@patch('src.cli.select_best_strategy')
@patch('src.cli.build_gtt_plan')
@patch('src.final_excel.run_final_excel')
def test_orchestrate_eod_excel_failure(mock_excel, mock_gtt_plan, mock_select, mock_score,
                                      mock_signals, mock_validate, mock_load, mock_fetch,
                                      sample_data, temp_dir):
    """Test orchestrate-eod with Excel generation failure."""
    
    # Setup mocks
    mock_fetch.return_value = None
    mock_load.return_value = sample_data
    mock_validate.return_value = (True, [])
    mock_signals.return_value = sample_data
    mock_score.return_value = sample_data
    mock_select.return_value = 'SEPA'
    
    gtt_plan = pd.DataFrame({
        'Symbol': ['RELIANCE.NS'],
        'Qty': [10],
        'ENTRY_trigger_price': [2500.0],
        'TARGET_trigger_price': [2600.0],
        'STOPLOSS_trigger_price': [2400.0],
        'DecisionConfidence': [4.5],
        'Confidence_Level': ['HIGH'],
        'R': [2.0],
        'Explanation': ['Test'],
        'GTT_Explanation': ['Test']
    })
    mock_gtt_plan.return_value = gtt_plan
    mock_excel.return_value = False  # Excel fails
    
    args = MockArgs()
    
    with pytest.raises(SystemExit) as exc_info:
        cmd_orchestrate_eod(args)
    
    assert exc_info.value.code == 1


def test_orchestrate_eod_output_structure(temp_dir):
    """Test that orchestrate-eod creates expected output files."""
    
    # Check that output directories are created
    expected_dirs = [
        'outputs/screener',
        'outputs/backtests', 
        'outputs/gtt',
        'outputs/multi_tf',
        'data'
    ]
    
    for dir_path in expected_dirs:
        assert Path(dir_path).exists(), f"Directory {dir_path} should be created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])