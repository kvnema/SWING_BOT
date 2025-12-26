"""
Tests for SWING_BOT Notifications Router
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.notifications_router import (
    notify_eod_success,
    notify_eod_failure,
    NotificationConfig
)


class TestNotificationRouter:
    """Test notification routing functionality."""

    @pytest.fixture
    def sample_success_data(self):
        """Create sample data for success notifications."""
        return {
            'webhook_url': "https://outlook.office.com/webhook/test",
            'latest_date': "2024-01-01",
            'pass_count': 5,
            'fail_count': 2,
            'top_rows_df': pd.DataFrame({'symbol': ['RELIANCE'], 'pnl': [1000.0]}),
            'file_links': {"report": "/path/to/report.csv"}
        }

    @pytest.fixture
    def sample_failure_data(self):
        """Create sample data for failure notifications."""
        return {
            'webhook_url': "https://outlook.office.com/webhook/test",
            'stage': "backtest",
            'error_msg': "Test error",
            'latest_date': "2024-01-01",
            'hints': ["Check configuration", "Verify data"],
            'file_links': {"report": "/path/to/report.csv"}
        }

    @patch('src.notifications_router.send_email_notification')
    @patch('src.notifications_router.build_adaptive_card_summary')
    @patch('src.notifications_router.requests.post')
    def test_notify_eod_success_teams_success(self, mock_requests_post, mock_build_card, mock_email, sample_success_data):
        """Test successful EOD success notification via Teams."""
        mock_requests_post.return_value.status_code = 200
        mock_build_card.return_value = {"type": "AdaptiveCard", "body": []}

        success = notify_eod_success(**sample_success_data)

        assert success is True
        mock_requests_post.assert_called_once()
        mock_email.assert_not_called()

    @patch('src.notifications_router.send_email_notification')
    @patch('src.notifications_router.build_adaptive_card_summary')
    @patch('src.notifications_router.requests.post')
    def test_notify_eod_success_teams_failure_email_fallback(self, mock_requests_post, mock_build_card, mock_email, sample_success_data):
        """Test EOD success notification with Teams failure and email fallback."""
        mock_requests_post.return_value.status_code = 400
        mock_build_card.return_value = {"type": "AdaptiveCard", "body": []}
        mock_email.return_value = True

        success = notify_eod_success(**sample_success_data)

        assert success is True
        mock_requests_post.assert_called_once()
        mock_email.assert_called_once()

    @patch('src.notifications_router.send_email_notification')
    @patch('src.notifications_router.build_adaptive_card_summary')
    @patch('src.notifications_router.requests.post')
    def test_notify_eod_success_all_channels_fail(self, mock_requests_post, mock_build_card, mock_email, sample_success_data):
        """Test EOD success notification when all channels fail."""
        mock_requests_post.return_value.status_code = 400
        mock_build_card.return_value = {"type": "AdaptiveCard", "body": []}
        mock_email.return_value = False

        success = notify_eod_success(**sample_success_data)

        assert success is False
        mock_requests_post.assert_called_once()
        mock_email.assert_called_once()

    @patch('src.notifications_router.send_email_notification')
    @patch('src.notifications_router.build_failure_card')
    @patch('src.notifications_router.requests.post')
    def test_notify_eod_failure_teams_success(self, mock_requests_post, mock_build_card, mock_email, sample_failure_data):
        """Test successful EOD failure notification via Teams."""
        mock_requests_post.return_value.status_code = 200
        mock_build_card.return_value = {"type": "AdaptiveCard", "body": []}

        success = notify_eod_failure(
            webhook_url=sample_failure_data['webhook_url'],
            stage=sample_failure_data['stage'],
            error_msg=sample_failure_data['error_msg'],
            hints=sample_failure_data['hints'],
            file_links=sample_failure_data['file_links'],
            email_config=None
        )

        assert success is True
        mock_requests_post.assert_called_once()
        mock_email.assert_not_called()

    @patch('src.notifications_router.send_email_notification')
    @patch('src.notifications_router.build_failure_card')
    @patch('src.notifications_router.requests.post')
    def test_notify_eod_failure_teams_failure_email_fallback(self, mock_requests_post, mock_build_card, mock_email, sample_failure_data):
        """Test EOD failure notification with Teams failure and email fallback."""
        mock_requests_post.return_value.status_code = 400
        mock_build_card.return_value = {"type": "AdaptiveCard", "body": []}
        mock_email.return_value = True

        success = notify_eod_failure(
            webhook_url=sample_failure_data['webhook_url'],
            stage=sample_failure_data['stage'],
            error_msg=sample_failure_data['error_msg'],
            hints=sample_failure_data['hints'],
            file_links=sample_failure_data['file_links'],
            email_config=None
        )

        assert success is True
        mock_requests_post.assert_called_once()
        mock_email.assert_called_once()

    @patch('src.notifications_router.send_email_notification')
    @patch('src.notifications_router.build_failure_card')
    @patch('src.notifications_router.requests.post')
    def test_notify_eod_failure_all_channels_fail(self, mock_requests_post, mock_build_card, mock_email, sample_failure_data):
        """Test EOD failure notification when all channels fail."""
        mock_requests_post.return_value.status_code = 400
        mock_build_card.return_value = {"type": "AdaptiveCard", "body": []}
        mock_email.return_value = False

        success = notify_eod_failure(
            webhook_url=sample_failure_data['webhook_url'],
            stage=sample_failure_data['stage'],
            error_msg=sample_failure_data['error_msg'],
            hints=sample_failure_data['hints'],
            file_links=sample_failure_data['file_links'],
            email_config=None
        )

        assert success is False
        mock_requests_post.assert_called_once()
        mock_email.assert_called_once()

    @patch('src.notifications_router.send_email_notification')
    def test_notify_eod_success_teams_disabled(self, mock_email, sample_success_data):
        """Test EOD success notification when Teams is disabled."""
        # Create config with teams disabled
        config = NotificationConfig()
        config.teams_enabled = False
        mock_email.return_value = True

        success = notify_eod_success(
            webhook_url=None,  # No webhook URL
            latest_date=sample_success_data['latest_date'],
            pass_count=sample_success_data['pass_count'],
            fail_count=sample_success_data['fail_count'],
            top_rows_df=sample_success_data['top_rows_df'],
            file_links=sample_success_data['file_links'],
            email_config=config
        )

        assert success is True
        mock_email.assert_called_once()

    @patch('src.notifications_router.send_email_notification')
    def test_notify_eod_success_email_disabled(self, mock_email, sample_success_data):
        """Test EOD success notification when email is disabled."""
        config = NotificationConfig()
        config.email_enabled = False

        success = notify_eod_success(
            webhook_url=sample_success_data['webhook_url'],
            latest_date=sample_success_data['latest_date'],
            pass_count=sample_success_data['pass_count'],
            fail_count=sample_success_data['fail_count'],
            top_rows_df=sample_success_data['top_rows_df'],
            file_links=sample_success_data['file_links'],
            email_config=config
        )

        assert success is False  # Teams fails, email disabled
        mock_email.assert_not_called()

    def test_notify_eod_success_all_channels_disabled(self, sample_success_data):
        """Test EOD success notification when all channels are disabled."""
        config = NotificationConfig()
        config.teams_enabled = False
        config.email_enabled = False

        success = notify_eod_success(
            webhook_url=None,
            latest_date=sample_success_data['latest_date'],
            pass_count=sample_success_data['pass_count'],
            fail_count=sample_success_data['fail_count'],
            top_rows_df=sample_success_data['top_rows_df'],
            file_links=sample_success_data['file_links'],
            email_config=config
        )

        assert success is False

    @patch('src.notifications_router.send_email_notification')
    def test_notify_eod_failure_teams_disabled(self, mock_email, sample_failure_data):
        """Test EOD failure notification when Teams is disabled."""
        config = NotificationConfig()
        config.teams_enabled = False
        mock_email.return_value = True

        success = notify_eod_failure(
            webhook_url=None,
            stage=sample_failure_data['stage'],
            error_msg=sample_failure_data['error_msg'],
            hints=sample_failure_data['hints'],
            file_links=sample_failure_data['file_links'],
            email_config=config
        )

        assert success is True
        mock_email.assert_called_once()

    @patch('src.notifications_router.send_email_notification')
    def test_notify_eod_failure_email_disabled(self, mock_email, sample_failure_data):
        """Test EOD failure notification when email is disabled."""
        config = NotificationConfig()
        config.email_enabled = False

        success = notify_eod_failure(
            webhook_url=sample_failure_data['webhook_url'],
            stage=sample_failure_data['stage'],
            error_msg=sample_failure_data['error_msg'],
            hints=sample_failure_data['hints'],
            file_links=sample_failure_data['file_links'],
            email_config=config
        )

        assert success is False  # Teams fails, email disabled
        mock_email.assert_not_called()

    def test_notify_eod_failure_all_channels_disabled(self, sample_failure_data):
        """Test EOD failure notification when all channels are disabled."""
        config = NotificationConfig()
        config.teams_enabled = False
        config.email_enabled = False

        success = notify_eod_failure(
            webhook_url=None,
            stage=sample_failure_data['stage'],
            error_msg=sample_failure_data['error_msg'],
            hints=sample_failure_data['hints'],
            file_links=sample_failure_data['file_links'],
            email_config=config
        )

        assert success is False

    def test_notification_config_class_initialization(self):
        """Test NotificationConfig class initialization."""
        config = NotificationConfig()

        assert hasattr(config, 'teams_enabled')
        assert hasattr(config, 'email_enabled')
        assert hasattr(config, 'fallback_enabled')
        assert hasattr(config, 'webhook_url')
        assert hasattr(config, 'email_provider')

    def test_notification_config_validate(self):
        """Test NotificationConfig validation."""
        config = NotificationConfig()
        config.teams_enabled = True
        config.webhook_url = "https://example.com/webhook"

        assert config.validate() is True

        # Test invalid config
        config.webhook_url = None
        assert config.validate() is False