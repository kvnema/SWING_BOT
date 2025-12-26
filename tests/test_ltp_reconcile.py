"""
Tests for LTP reconciliation module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.ltp_reconcile import (
    LTPParams,
    fetch_live_quotes,
    reconcile_entry_stop_target,
    reconcile_plan,
    round_to_tick,
    compute_stop_target_from_entry
)


class TestLTPReconcile:
    """Test LTP reconciliation functionality."""

    def test_round_to_tick(self):
        """Test tick rounding."""
        assert abs(round_to_tick(100.03) - 100.05) < 1e-10
        assert abs(round_to_tick(100.04) - 100.05) < 1e-10
        assert abs(round_to_tick(100.05) - 100.05) < 1e-10
        assert abs(round_to_tick(100.06) - 100.05) < 1e-10

    def test_compute_stop_target_from_entry(self):
        """Test stop/target computation."""
        stop, target = compute_stop_target_from_entry(100.0, 'MR')
        assert stop < 100.0
        assert target > 100.0

        stop, target = compute_stop_target_from_entry(100.0, 'Donchian')
        assert stop < 100.0
        assert target > 100.0

    @patch('src.ltp_reconcile.requests.get')
    def test_fetch_live_quotes_success(self, mock_get):
        """Test successful quote fetching."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'data': {
                'NSE_EQ:INFY': {
                    'symbol': 'INFY',
                    'last_price': 1500.25,
                    'timestamp': '2024-01-01T10:00:00Z',
                    'ohlc': {'open': 1490.0, 'high': 1510.0, 'low': 1485.0, 'close': 1500.25}
                }
            }
        }
        mock_get.return_value = mock_response

        result = fetch_live_quotes(['NSE_EQ:INFY'], 'test_token')

        assert len(result) == 1
        assert result.iloc[0]['instrument_token'] == 'NSE_EQ:INFY'
        assert result.iloc[0]['last_price'] == 1500.25

    @patch('src.ltp_reconcile.requests.get')
    def test_fetch_live_quotes_failure(self, mock_get):
        """Test quote fetching with API failure."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        result = fetch_live_quotes(['NSE_EQ:INFY'], 'test_token')
        
        assert len(result) == 0

    def test_reconcile_entry_stop_target_within_tolerance(self):
        """Test reconciliation when entry is within LTP tolerance."""
        plan_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'Symbol': ['INFY'],
            'Strategy': ['MR'],
            'ENTRY_trigger_price': [100.00],
            'STOPLOSS_trigger_price': [98.00],
            'TARGET_trigger_price': [104.00]
        })

        quotes_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'last_price': [100.10],  # Within 2% tolerance
            'ohlc': [{}]
        })

        params = LTPParams(max_entry_ltppct=0.02, adjust_mode='soft')
        result = reconcile_entry_stop_target(plan_df, quotes_df, params)

        assert len(result) == 1
        assert result.iloc[0]['Audit_Flag'] == 'PASS'
        assert result.iloc[0]['Reconciled_Entry'] == 100.00  # No change
        assert abs(result.iloc[0]['ltp_delta_pct']) < 0.02

    def test_reconcile_entry_stop_target_outside_tolerance_soft(self):
        """Test soft mode reconciliation when entry is outside LTP tolerance."""
        plan_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'Symbol': ['INFY'],
            'Strategy': ['MR'],
            'ENTRY_trigger_price': [100.00],
            'STOPLOSS_trigger_price': [98.00],
            'TARGET_trigger_price': [104.00]
        })

        quotes_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'last_price': [105.00],  # Outside 2% tolerance
            'ohlc': [{}]
        })

        params = LTPParams(max_entry_ltppct=0.02, adjust_mode='soft')
        result = reconcile_entry_stop_target(plan_df, quotes_df, params)

        assert len(result) == 1
        assert result.iloc[0]['Audit_Flag'] == 'PASS'
        assert result.iloc[0]['Reconciled_Entry'] != 100.00  # Adjusted
        assert abs(result.iloc[0]['ltp_delta_pct']) > 0.02

    def test_reconcile_entry_stop_target_outside_tolerance_strict(self):
        """Test strict mode reconciliation when entry is outside LTP tolerance."""
        plan_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'Symbol': ['INFY'],
            'Strategy': ['MR'],
            'ENTRY_trigger_price': [100.00],
            'STOPLOSS_trigger_price': [98.00],
            'TARGET_trigger_price': [104.00]
        })

        quotes_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'last_price': [105.00],  # Outside 2% tolerance
            'ohlc': [{}]
        })

        params = LTPParams(max_entry_ltppct=0.02, adjust_mode='strict')
        result = reconcile_entry_stop_target(plan_df, quotes_df, params)

        assert len(result) == 1
        assert result.iloc[0]['Audit_Flag'] == 'FAIL'
        assert 'Entry price' in result.iloc[0]['Issues']

    def test_reconcile_entry_stop_target_missing_quote(self):
        """Test reconciliation when quote fetch fails."""
        plan_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'Symbol': ['INFY'],
            'Strategy': ['MR'],
            'ENTRY_trigger_price': [100.00],
            'STOPLOSS_trigger_price': [98.00],
            'TARGET_trigger_price': [104.00]
        })

        quotes_df = pd.DataFrame(columns=['instrument_token', 'last_price', 'ohlc'])

        params = LTPParams(max_entry_ltppct=0.02, adjust_mode='soft')
        result = reconcile_entry_stop_target(plan_df, quotes_df, params)

        assert len(result) == 1
        assert result.iloc[0]['Audit_Flag'] == 'FAIL'
        assert 'Quote fetch failed' in result.iloc[0]['Issues']

    @patch('src.ltp_reconcile.fetch_live_quotes')
    @patch.dict(os.environ, {'UPSTOX_ACCESS_TOKEN': 'test_token'})
    def test_reconcile_plan_integration(self, mock_fetch_quotes):
        """Test full reconcile_plan function."""
        mock_fetch_quotes.return_value = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'last_price': [100.10],
            'ohlc': [{}]
        })

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_csv = Path(temp_dir) / 'plan.csv'
            out_csv = Path(temp_dir) / 'reconciled.csv'
            report_txt = Path(temp_dir) / 'report.txt'

            # Create sample plan
            plan_df = pd.DataFrame({
                'instrument_token': ['NSE_EQ:INFY'],
                'Symbol': ['INFY'],
                'Strategy': ['MR'],
                'ENTRY_trigger_price': [100.00],
                'STOPLOSS_trigger_price': [98.00],
                'TARGET_trigger_price': [104.00]
            })
            plan_df.to_csv(plan_csv, index=False)

            # Run reconciliation
            result = reconcile_plan(
                plan_csv=str(plan_csv),
                out_csv=str(out_csv),
                out_report=str(report_txt),
                adjust_mode='soft',
                max_entry_ltppct=0.02
            )

            assert len(result) == 1
            assert Path(out_csv).exists()
            assert Path(report_txt).exists()

            # Check report content
            with open(report_txt, 'r') as f:
                report_content = f.read()
                assert 'SWING_BOT LTP Reconciliation Report' in report_content

    def test_strategy_aware_adjustments_breakout(self):
        """Test strategy-aware adjustments for breakout strategies."""
        plan_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'Symbol': ['INFY'],
            'Strategy': ['Donchian'],
            'ENTRY_trigger_price': [100.00],
            'STOPLOSS_trigger_price': [98.00],
            'TARGET_trigger_price': [104.00],
            'DonchianH20': [99.00]  # Pivot below LTP
        })

        quotes_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'last_price': [105.00],  # Above pivot and outside tolerance
            'ohlc': [{}]
        })

        params = LTPParams(max_entry_ltppct=0.02, adjust_mode='soft')
        result = reconcile_entry_stop_target(plan_df, quotes_df, params)

        assert len(result) == 1
        # For breakout above pivot, should snap to LTP
        assert result.iloc[0]['Reconciled_Entry'] == 105.00

    def test_strategy_aware_adjustments_mr(self):
        """Test strategy-aware adjustments for mean reversion."""
        plan_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'Symbol': ['INFY'],
            'Strategy': ['MR'],
            'ENTRY_trigger_price': [100.00],
            'STOPLOSS_trigger_price': [98.00],
            'TARGET_trigger_price': [104.00]
        })

        quotes_df = pd.DataFrame({
            'instrument_token': ['NSE_EQ:INFY'],
            'last_price': [105.00],  # Far from entry
            'ohlc': [{}]
        })

        params = LTPParams(max_entry_ltppct=0.02, adjust_mode='soft')
        result = reconcile_entry_stop_target(plan_df, quotes_df, params)

        assert len(result) == 1
        # For MR, should snap to LTP
        assert result.iloc[0]['Reconciled_Entry'] == 105.00