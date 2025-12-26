"""
Test Plan Audit functionality
============================

Tests for plan audit system ensuring GTT plans use correct prices.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.plan_audit import (
    AuditParams, PlanAuditError, round_tick, get_strategy_pivot,
    compute_canonical_prices, audit_plan_row, attach_audit, run_plan_audit
)


class TestPlanAudit:
    """Test plan audit functionality."""

    def test_round_tick(self):
        """Test tick rounding to NSE standards."""
        assert abs(round_tick(100.123, 0.05) - 100.10) < 1e-10
        assert abs(round_tick(100.128, 0.05) - 100.15) < 1e-10
        assert round_tick(100.00, 0.05) == 100.00
        assert round_tick(0, 0.05) == 0

    def test_get_strategy_pivot_mr(self):
        """Test MR strategy pivot selection."""
        data = pd.Series({
            'EMA20': 100.0,
            'EMA50': 98.0,
            'EMA200': 95.0,
            'Donchian_20_High': 105.0,
            'Close': 102.0
        })
        
        price, source = get_strategy_pivot(data, 'MR')
        assert price == 100.0
        assert source == 'EMA20'

    def test_get_strategy_pivot_breakout(self):
        """Test breakout strategy pivot selection."""
        data = pd.Series({
            'EMA20': 100.0,
            'DonchianH20': 105.0,
            'Close': 102.0
        })
        
        price, source = get_strategy_pivot(data, 'Donchian_Breakout')
        assert price == 105.0
        assert source == 'DonchianH20'

    def test_compute_canonical_prices_mr(self):
        """Test canonical price computation for MR strategy."""
        ap = AuditParams()
        data = pd.Series({
            'EMA20': 100.0,
            'ATR14': 2.0,
            'Close': 102.0
        })
        
        result = compute_canonical_prices(data, 'MR', ap)
        
        assert result['canonical_entry'] == 100.0  # Entry at EMA20
        assert result['canonical_stop'] == 97.0     # EMA20 - 1.5*ATR
        assert result['canonical_target'] == 106.0  # EMA20 + (EMA20-stop) * 2
        assert result['pivot_source'] == 'EMA20'

    def test_audit_plan_row_pass(self):
        """Test successful plan audit."""
        ap = AuditParams()
        
        plan_row = pd.Series({
            'Symbol': 'TEST',
            'Strategy': 'MR',
            'ENTRY_trigger_price': 100.0,
            'STOPLOSS_trigger_price': 97.0,
            'TARGET_trigger_price': 103.0
        })
        
        symbol_data = pd.Series({
            'EMA20': 100.0,
            'ATR14': 2.0,
            'Close': 102.0
        })
        
        latest_data = pd.Series({
            'Close': 102.0,
            'LastTradedPrice': 101.5
        })
        
        result = audit_plan_row(plan_row, symbol_data, latest_data, ap)
        
        assert result['Audit_Flag'] == 'PASS'
        assert result['Issues'] == ''
        assert result['Canonical_Entry'] == 100.0

    def test_audit_plan_row_fail_wrong_entry(self):
        """Test plan audit failure on wrong entry price."""
        ap = AuditParams(max_entry_pct_diff=0.01)  # 1% tolerance
        
        plan_row = pd.Series({
            'Symbol': 'TEST',
            'Strategy': 'MR',
            'ENTRY_trigger_price': 105.0,  # Wrong: should be 100
            'STOPLOSS_trigger_price': 97.0,
            'TARGET_trigger_price': 103.0
        })
        
        symbol_data = pd.Series({
            'EMA20': 100.0,
            'ATR14': 2.0,
            'Close': 102.0
        })
        
        latest_data = pd.Series({
            'Close': 102.0,
            'LastTradedPrice': 101.5
        })
        
        result = audit_plan_row(plan_row, symbol_data, latest_data, ap)
        
        assert result['Audit_Flag'] == 'FAIL'
        assert 'entry price' in result['Issues'].lower()
        assert '100.00' in result['Fix_Suggestion']

    def test_attach_audit_strict_mode(self):
        """Test strict mode audit failure."""
        ap = AuditParams(strict_mode=True, max_entry_pct_diff=0.01)
        
        # Create test dataframes
        plan_df = pd.DataFrame([{
            'Symbol': 'TEST',
            'Strategy': 'MR',
            'ENTRY_trigger_price': 105.0,  # Wrong price
            'STOPLOSS_trigger_price': 97.0,
            'TARGET_trigger_price': 103.0
        }])
        
        indicators_df = pd.DataFrame([{
            'Symbol': 'TEST',
            'EMA20': 100.0,
            'ATR14': 2.0,
            'Close': 102.0
        }])
        
        latest_df = pd.DataFrame([{
            'Symbol': 'TEST',
            'Close': 102.0,
            'LastTradedPrice': 101.5
        }])
        
        # Should raise exception in strict mode
        with pytest.raises(PlanAuditError):
            attach_audit(plan_df, indicators_df, latest_df, ap)

    def test_run_plan_audit_integration(self):
        """Test full plan audit pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            plan_path = os.path.join(tmpdir, 'plan.csv')
            indicators_path = os.path.join(tmpdir, 'indicators.csv')
            latest_path = os.path.join(tmpdir, 'latest.csv')
            output_path = os.path.join(tmpdir, 'audited.csv')
            
            # Create test data
            plan_df = pd.DataFrame([{
                'Symbol': 'TEST',
                'Strategy': 'MR',
                'ENTRY_trigger_price': 100.0,
                'STOPLOSS_trigger_price': 97.0,
                'TARGET_trigger_price': 103.0
            }])
            plan_df.to_csv(plan_path, index=False)
            
            indicators_df = pd.DataFrame([{
                'Symbol': 'TEST',
                'Date': '2024-01-01',
                'EMA20': 100.0,
                'ATR14': 2.0,
                'Close': 102.0
            }])
            indicators_df.to_csv(indicators_path, index=False)
            
            latest_df = pd.DataFrame([{
                'Symbol': 'TEST',
                'Close': 102.0,
                'LastTradedPrice': 101.5
            }])
            latest_df.to_csv(latest_path, index=False)
            
            # Run audit
            ap = AuditParams()
            success = run_plan_audit(plan_path, indicators_path, latest_path, output_path, ap)
            
            assert success
            assert os.path.exists(output_path)
            
            # Check output has audit columns
            result_df = pd.read_csv(output_path)
            assert 'Audit_Flag' in result_df.columns
            assert 'Canonical_Entry' in result_df.columns
            assert result_df['Audit_Flag'].iloc[0] == 'PASS'