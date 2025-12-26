"""
Tests for SWING_BOT orchestrate-eod notifications integration
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import tempfile

from src.cli import cmd_orchestrate_eod


class TestOrchestrateEodNotifications:
    """Test orchestrate-eod command with notification integration."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        screener_data = pd.DataFrame({
            'symbol': ['RELIANCE', 'TCS', 'INFY'],
            'close': [2500.0, 3200.0, 1800.0],
            'volume': [1000000, 800000, 600000],
            'rsi': [65.0, 55.0, 70.0],
            'macd_signal': ['BUY', 'SELL', 'BUY']
        })

        gtt_data = pd.DataFrame({
            'symbol': ['RELIANCE', 'INFY'],
            'entry_price': [2480.0, 1780.0],
            'stop_loss': [2400.0, 1720.0],
            'target': [2600.0, 1900.0],
            'quantity': [10, 15]
        })

        backtest_data = {
            'selected_strategy': 'AVWAP',
            'kpi': {
                'total_return': 15.5,
                'sharpe_ratio': 1.8,
                'max_drawdown': -8.2,
                'win_rate': 0.65
            },
            'trades': [
                {'symbol': 'RELIANCE', 'pnl': 500.0, 'entry_date': '2024-01-01'},
                {'symbol': 'INFY', 'pnl': 300.0, 'entry_date': '2024-01-02'}
            ]
        }

        return {
            'screener': screener_data,
            'gtt': gtt_data,
            'backtest': backtest_data
        }

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create necessary directories
            outputs_dir = Path(temp_dir) / "outputs"
            outputs_dir.mkdir()

            # Create sample output files
            screener_file = outputs_dir / "screener_results.csv"
            gtt_file = outputs_dir / "gtt_plan.csv"
            backtest_file = outputs_dir / "backtest_results.json"

            yield {
                'temp_dir': temp_dir,
                'outputs_dir': outputs_dir,
                'screener_file': screener_file,
                'gtt_file': gtt_file,
                'backtest_file': backtest_file
            }

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    @patch('src.cli.run_backtest')
    @patch('src.cli.generate_gtt_plan')
    @patch('src.notifications_router.notify_eod_success')
    @patch('src.dashboards.teams_dashboard.build_daily_html')
    @patch('src.metrics_exporter.MetricsExporter')
    def test_orchestrate_eod_success_with_notifications_and_dashboard(
        self, mock_metrics, mock_html, mock_notify, mock_gtt, mock_backtest,
        mock_screener, mock_config, sample_data, temp_workspace
    ):
        """Test successful orchestrate-eod with notifications, dashboard, and metrics."""
        # Setup mocks
        mock_config.return_value = {
            "notifications": {"enabled": True},
            "dashboard": {"enabled": True},
            "metrics": {"enabled": True}
        }

        mock_screener.return_value = sample_data['screener']
        mock_backtest.return_value = sample_data['backtest']
        mock_gtt.return_value = sample_data['gtt']
        mock_notify.return_value = True
        mock_html.return_value = "<html>Dashboard</html>"

        mock_metrics_instance = MagicMock()
        mock_metrics.return_value = mock_metrics_instance

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            # Run orchestrate-eod with all flags
            result = orchestrate_eod(
                dashboard=True,
                metrics=True,
                notifications=True
            )

            assert result is True

            # Verify all components were called
            mock_screener.assert_called_once()
            mock_backtest.assert_called_once()
            mock_gtt.assert_called_once()
            mock_notify.assert_called_once()
            mock_html.assert_called_once()
            mock_metrics.assert_called_once()

            # Verify metrics recording
            mock_metrics_instance.record_runtime_metrics.assert_called_once()
            mock_metrics_instance.record_data_quality_metrics.assert_called_once()
            mock_metrics_instance.record_audit_results.assert_called_once()

        finally:
            os.chdir(original_cwd)

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    @patch('src.cli.run_backtest')
    @patch('src.cli.generate_gtt_plan')
    @patch('src.notifications_router.notify_eod_success')
    def test_orchestrate_eod_success_with_notifications_only(
        self, mock_notify, mock_gtt, mock_backtest, mock_screener, mock_config,
        sample_data, temp_workspace
    ):
        """Test successful orchestrate-eod with notifications only."""
        mock_config.return_value = {
            "notifications": {"enabled": True},
            "dashboard": {"enabled": False},
            "metrics": {"enabled": False}
        }

        mock_screener.return_value = sample_data['screener']
        mock_backtest.return_value = sample_data['backtest']
        mock_gtt.return_value = sample_data['gtt']
        mock_notify.return_value = True

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            result = orchestrate_eod(
                dashboard=False,
                metrics=False,
                notifications=True
            )

            assert result is True
            mock_notify.assert_called_once()

        finally:
            os.chdir(original_cwd)

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    @patch('src.cli.run_backtest')
    @patch('src.cli.generate_gtt_plan')
    @patch('src.notifications_router.notify_eod_failure')
    def test_orchestrate_eod_failure_with_notifications(
        self, mock_notify_failure, mock_gtt, mock_backtest, mock_screener, mock_config,
        sample_data, temp_workspace
    ):
        """Test orchestrate-eod failure with notifications."""
        mock_config.return_value = {
            "notifications": {"enabled": True}
        }

        # Simulate screener failure
        mock_screener.side_effect = Exception("Screener failed")
        mock_notify_failure.return_value = True

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            result = orchestrate_eod(
                dashboard=False,
                metrics=False,
                notifications=True
            )

            assert result is False
            mock_notify_failure.assert_called_once()

        finally:
            os.chdir(original_cwd)

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    @patch('src.cli.run_backtest')
    @patch('src.cli.generate_gtt_plan')
    @patch('src.cli.notify_eod_success')
    @patch('src.cli.build_daily_html')
    def test_orchestrate_eod_dashboard_failure_handling(
        self, mock_html, mock_notify, mock_gtt, mock_backtest, mock_screener, mock_config,
        sample_data, temp_workspace
    ):
        """Test orchestrate-eod handles dashboard generation failure gracefully."""
        mock_config.return_value = {
            "notifications": {"enabled": True},
            "dashboard": {"enabled": True}
        }

        mock_screener.return_value = sample_data['screener']
        mock_backtest.return_value = sample_data['backtest']
        mock_gtt.return_value = sample_data['gtt']
        mock_notify.return_value = True
        mock_html.side_effect = Exception("Dashboard generation failed")

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            # Should still succeed despite dashboard failure
            result = orchestrate_eod(
                dashboard=True,
                metrics=False,
                notifications=True
            )

            assert result is True
            mock_notify.assert_called_once()

        finally:
            os.chdir(original_cwd)

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    @patch('src.cli.run_backtest')
    @patch('src.cli.generate_gtt_plan')
    @patch('src.cli.notify_eod_success')
    @patch('src.cli.MetricsExporter')
    def test_orchestrate_eod_metrics_failure_handling(
        self, mock_metrics_class, mock_notify, mock_gtt, mock_backtest, mock_screener, mock_config,
        sample_data, temp_workspace
    ):
        """Test orchestrate-eod handles metrics export failure gracefully."""
        mock_config.return_value = {
            "notifications": {"enabled": True},
            "metrics": {"enabled": True}
        }

        mock_screener.return_value = sample_data['screener']
        mock_backtest.return_value = sample_data['backtest']
        mock_gtt.return_value = sample_data['gtt']
        mock_notify.return_value = True

        mock_metrics_instance = MagicMock()
        mock_metrics_instance.record_runtime_metrics.side_effect = Exception("Metrics failed")
        mock_metrics_class.return_value = mock_metrics_instance

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            # Should still succeed despite metrics failure
            result = orchestrate_eod(
                dashboard=False,
                metrics=True,
                notifications=True
            )

            assert result is True
            mock_notify.assert_called_once()

        finally:
            os.chdir(original_cwd)

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    @patch('src.cli.run_backtest')
    @patch('src.cli.generate_gtt_plan')
    @patch('src.cli.notify_eod_success')
    def test_orchestrate_eod_notification_failure_handling(
        self, mock_notify, mock_gtt, mock_backtest, mock_screener, mock_config,
        sample_data, temp_workspace
    ):
        """Test orchestrate-eod handles notification failure gracefully."""
        mock_config.return_value = {
            "notifications": {"enabled": True}
        }

        mock_screener.return_value = sample_data['screener']
        mock_backtest.return_value = sample_data['backtest']
        mock_gtt.return_value = sample_data['gtt']
        mock_notify.return_value = False  # Notification fails

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            # Should still return True as the core processing succeeded
            result = orchestrate_eod(
                dashboard=False,
                metrics=False,
                notifications=True
            )

            assert result is True
            mock_notify.assert_called_once()

        finally:
            os.chdir(original_cwd)

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    @patch('src.cli.run_backtest')
    @patch('src.cli.generate_gtt_plan')
    def test_orchestrate_eod_without_notifications(
        self, mock_gtt, mock_backtest, mock_screener, mock_config,
        sample_data, temp_workspace
    ):
        """Test orchestrate-eod runs successfully without notifications."""
        mock_config.return_value = {
            "notifications": {"enabled": False}
        }

        mock_screener.return_value = sample_data['screener']
        mock_backtest.return_value = sample_data['backtest']
        mock_gtt.return_value = sample_data['gtt']

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            result = orchestrate_eod(
                dashboard=False,
                metrics=False,
                notifications=False
            )

            assert result is True

        finally:
            os.chdir(original_cwd)

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    @patch('src.cli.run_backtest')
    @patch('src.cli.generate_gtt_plan')
    @patch('src.cli.notify_eod_success')
    @patch('src.cli.build_daily_html')
    @patch('src.cli.MetricsExporter')
    def test_orchestrate_eod_all_features_enabled(
        self, mock_metrics_class, mock_html, mock_notify, mock_gtt, mock_backtest,
        mock_screener, mock_config, sample_data, temp_workspace
    ):
        """Test orchestrate-eod with all features enabled."""
        mock_config.return_value = {
            "notifications": {"enabled": True},
            "dashboard": {"enabled": True},
            "metrics": {"enabled": True}
        }

        mock_screener.return_value = sample_data['screener']
        mock_backtest.return_value = sample_data['backtest']
        mock_gtt.return_value = sample_data['gtt']
        mock_notify.return_value = True
        mock_html.return_value = "<html>Full Dashboard</html>"

        mock_metrics_instance = MagicMock()
        mock_metrics_class.return_value = mock_metrics_instance

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            result = orchestrate_eod(
                dashboard=True,
                metrics=True,
                notifications=True
            )

            assert result is True

            # Verify all components were called with correct parameters
            mock_screener.assert_called_once()
            mock_backtest.assert_called_once()
            mock_gtt.assert_called_once()
            mock_notify.assert_called_once()
            mock_html.assert_called_once()
            mock_metrics_class.assert_called_once()

            # Verify metrics were recorded
            mock_metrics_instance.record_runtime_metrics.assert_called_once()
            mock_metrics_instance.record_data_quality_metrics.assert_called_once()
            mock_metrics_instance.record_audit_results.assert_called_once()

        finally:
            os.chdir(original_cwd)

    @patch('src.cli.get_config')
    @patch('src.cli.run_screener')
    def test_orchestrate_eod_early_screener_failure(
        self, mock_screener, mock_config, temp_workspace
    ):
        """Test orchestrate-eod handles early screener failure."""
        mock_config.return_value = {}
        mock_screener.side_effect = Exception("Critical screener failure")

        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_workspace['temp_dir'])

            result = orchestrate_eod(
                dashboard=False,
                metrics=False,
                notifications=False
            )

            assert result is False

        finally:
            os.chdir(original_cwd)