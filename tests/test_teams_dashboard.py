"""
Tests for SWING_BOT Teams Dashboard
"""

import pytest
import pandas as pd
import json
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from src.dashboards.teams_dashboard import (
    build_daily_html,
    build_adaptive_card_summary,
    build_failure_card
)


class TestTeamsDashboard:
    """Test Teams dashboard functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        # Sample plan data
        plan_df = pd.DataFrame({
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
            'ENTRY_trigger_price': [2500.50, 3200.75, 1600.25],
            'STOPLOSS_trigger_price': [2450.00, 3150.00, 1575.00],
            'TARGET_trigger_price': [2600.00, 3300.00, 1650.00],
            'DecisionConfidence': [4.2, 3.8, 4.5]
        })

        # Sample audit data
        audit_df = pd.DataFrame({
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
            'ENTRY_trigger_price': [2500.50, 3200.75, 1600.25],
            'STOPLOSS_trigger_price': [2450.00, 3150.00, 1575.00],
            'TARGET_trigger_price': [2600.00, 3300.00, 1650.00],
            'DecisionConfidence': [4.2, 3.8, 4.5],
            'Audit_Flag': ['PASS', 'PASS', 'FAIL'],
            'Issues': ['', '', 'Entry price tolerance exceeded'],
            'Fix_Suggestion': ['', '', 'Check pivot calculation']
        })

        # Sample screener data
        screener_df = pd.DataFrame({
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
            'Date': ['2025-12-21', '2025-12-21', '2025-12-21']
        })

        return plan_df, audit_df, screener_df

    def test_build_daily_html(self, sample_data):
        """Test HTML dashboard generation."""
        plan_df, audit_df, screener_df = sample_data

        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = Path(temp_dir) / "test_dashboard.html"

            # Build dashboard
            build_daily_html(
                plan_df=plan_df,
                audit_df=audit_df,
                screener_df=screener_df,
                out_html=str(html_path)
            )

            # Verify file was created
            assert html_path.exists()

            # Read and verify content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Check for key elements
            assert "SWING_BOT Daily Dashboard" in html_content
            assert "RELIANCE.NS" in html_content
            assert "Audit Pass Rate" in html_content
            assert "Top Positions" in html_content
            assert "Audit Issues Summary" in html_content

            # Check for KPI values
            assert "2/3" in html_content  # 2 PASS out of 3 total
            assert "2500.50" in html_content  # Entry price

    def test_build_adaptive_card_summary(self, sample_data):
        """Test Adaptive Card summary generation."""
        _, audit_df, _ = sample_data

        top_rows = audit_df.head(2)

        card = build_adaptive_card_summary(
            latest_date="2025-12-21",
            pass_count=2,
            fail_count=1,
            top_rows_df=top_rows,
            links={
                'Excel': 'file://outputs/gtt/final.xlsx',
                'CSV': 'file://outputs/gtt/plan.csv'
            }
        )

        # Verify card structure
        assert card['type'] == 'AdaptiveCard'
        assert card['version'] == '1.4'
        assert len(card['body']) >= 3

        # Check title
        title_block = card['body'][0]
        assert "SWING_BOT Daily Summary" in title_block['text']

        # Check facts
        fact_set = card['body'][1]
        assert fact_set['type'] == 'FactSet'
        facts = fact_set['facts']
        assert len(facts) == 4  # date, pass, fail, total

        # Check actions
        assert 'actions' in card
        assert len(card['actions']) == 2

    def test_build_failure_card(self):
        """Test failure Adaptive Card generation."""
        card = build_failure_card(
            stage="plan-audit",
            error_msg="Audit validation failed",
            hints=["Check data freshness", "Verify price tolerances"],
            links={
                'Logs': 'file://outputs/logs/error.log',
                'Dashboard': 'file://outputs/dashboard/index.html'
            }
        )

        # Verify card structure
        assert card['type'] == 'AdaptiveCard'
        assert card['version'] == '1.4'

        # Check title
        title_block = card['body'][0]
        assert "SWING_BOT Pipeline Failure" in title_block['text']

        # Check stage info
        stage_block = card['body'][1]
        assert stage_block['type'] == 'TextBlock'
        assert "plan-audit" in stage_block['text']

        # Check error message
        error_block = card['body'][2]
        assert error_block['type'] == 'TextBlock'
        assert "Audit validation failed" in error_block['text']

        # Check hints
        hints_block = card['body'][3]
        assert hints_block['type'] == 'TextBlock'
        assert "Check data freshness" in hints_block['text']
        assert "Verify price tolerances" in hints_block['text']

        # Check actions
        assert 'actions' in card
        assert len(card['actions']) == 2

    def test_build_adaptive_card_json_validity(self, sample_data):
        """Test that generated Adaptive Cards are valid JSON."""
        _, audit_df, _ = sample_data

        # Test success card
        success_card = build_adaptive_card_summary(
            latest_date="2025-12-21",
            pass_count=2,
            fail_count=1,
            top_rows_df=audit_df.head(2),
            links={}
        )

        # Should not raise exception
        json_str = json.dumps(success_card)
        parsed = json.loads(json_str)
        assert parsed == success_card

        # Test failure card
        failure_card = build_failure_card(
            stage="test",
            error_msg="test error",
            hints=["hint 1"],
            links={}
        )

        json_str = json.dumps(failure_card)
        parsed = json.loads(json_str)
        assert parsed == failure_card

    @patch('src.dashboards.teams_dashboard.get_ist_now')
    def test_html_with_timezone(self, mock_ist_now, sample_data):
        """Test HTML generation with timezone handling."""
        from datetime import datetime
        import pytz

        # Mock IST time
        mock_time = datetime(2025, 12, 21, 16, 30, 0, tzinfo=pytz.timezone('Asia/Kolkata'))
        mock_ist_now.return_value = mock_time

        plan_df, audit_df, screener_df = sample_data

        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = Path(temp_dir) / "test_dashboard.html"

            build_daily_html(
                plan_df=plan_df,
                audit_df=audit_df,
                screener_df=screener_df,
                out_html=str(html_path)
            )

            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Check for formatted timestamp
            assert "Generated at 2025-12-21 16:30:00" in html_content

    def test_empty_data_handling(self):
        """Test dashboard generation with empty data."""
        empty_plan = pd.DataFrame()
        empty_audit = pd.DataFrame()
        empty_screener = pd.DataFrame()

        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = Path(temp_dir) / "empty_dashboard.html"

            # Should not raise exception
            build_daily_html(
                plan_df=empty_plan,
                audit_df=empty_audit,
                screener_df=empty_screener,
                out_html=str(html_path)
            )

            assert html_path.exists()

            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Should contain appropriate empty state messages
            assert "No positions available" in html_content
            assert "No audit issues found" in html_content