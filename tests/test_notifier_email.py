import pytest
from unittest.mock import patch, MagicMock, mock_open
import smtplib
from pathlib import Path
import tempfile

from src.notifier_email import (
    EmailNotifier,
    get_email_notifier,
    send_email_notification,
    MSAL_AVAILABLE
)


class TestEmailNotifier:
    """Test email notification functionality."""

    @pytest.fixture
    def smtp_env_vars(self):
        """Setup SMTP environment variables."""
        env_vars = {
            "SMTP_SERVER": "smtp.gmail.com",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "test@example.com",
            "SMTP_PASSWORD": "test_password",
            "FROM_EMAIL": "test@example.com",
            "TO_EMAILS": "user1@example.com,user2@example.com"
        }

        with patch.dict('os.environ', env_vars):
            yield env_vars

    @pytest.fixture
    def graph_env_vars(self):
        """Setup Microsoft Graph environment variables."""
        env_vars = {
            "GRAPH_TENANT_ID": "test-tenant-id",
            "GRAPH_CLIENT_ID": "test-client-id",
            "GRAPH_CLIENT_SECRET": "test-client-secret",
            "MAILBOX_UPN": "test@example.com",
            "TO_EMAILS": "user1@example.com,user2@example.com"
        }

        with patch.dict('os.environ', env_vars):
            yield env_vars

    def test_smtp_notifier_initialization(self, smtp_env_vars):
        """Test SMTP notifier initialization."""
        notifier = EmailNotifier(provider="smtp")

        assert notifier.provider == "smtp"
        assert notifier.smtp_server == "smtp.gmail.com"
        assert notifier.smtp_port == 587
        assert notifier.from_email == "test@example.com"
        assert notifier.to_emails == ["user1@example.com", "user2@example.com"]

    @pytest.mark.skipif(not MSAL_AVAILABLE, reason="MSAL not available")
    def test_graph_notifier_initialization(self, graph_env_vars):
        """Test Microsoft Graph notifier initialization."""
        with patch('src.notifier_email.msal') as mock_msal:
            mock_app_instance = MagicMock()
            mock_msal.ConfidentialClientApplication.return_value = mock_app_instance

            notifier = EmailNotifier(provider="graph")

            assert notifier.provider == "graph"
            assert notifier.tenant_id == "test-tenant-id"
            mock_msal.ConfidentialClientApplication.assert_called_once()

    def test_invalid_provider(self):
        """Test invalid email provider."""
        with pytest.raises(ValueError, match="Unsupported email provider"):
            EmailNotifier(provider="invalid")

    def test_missing_smtp_config(self):
        """Test SMTP notifier with missing configuration."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="SMTP configuration incomplete"):
                EmailNotifier(provider="smtp")

    def test_missing_graph_config(self):
        """Test Graph notifier with missing configuration."""
        with patch('src.notifier_email.MSAL_AVAILABLE', True):
            with patch.dict('os.environ', {}, clear=True):
                with pytest.raises(ValueError, match="Microsoft Graph configuration incomplete"):
                    EmailNotifier(provider="graph")

    @patch('smtplib.SMTP')
    def test_send_smtp_email_success(self, mock_smtp_class, smtp_env_vars):
        """Test successful SMTP email sending."""
        # Setup mocks
        mock_smtp = MagicMock()
        mock_smtp_class.return_value = mock_smtp

        notifier = EmailNotifier(provider="smtp")

        # Create temporary attachment
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            attachment_path = f.name

        try:
            success = notifier.send_email(
                subject="Test Subject",
                html_body="<h1>Test</h1>",
                attachments=[attachment_path]
            )

            assert success is True
            mock_smtp.starttls.assert_called_once()
            mock_smtp.login.assert_called_once_with("test@example.com", "test_password")
            mock_smtp.sendmail.assert_called_once()

        finally:
            Path(attachment_path).unlink(missing_ok=True)

    @patch('smtplib.SMTP')
    def test_send_smtp_email_failure(self, mock_smtp_class, smtp_env_vars):
        """Test SMTP email sending failure."""
        mock_smtp = MagicMock()
        mock_smtp.starttls.side_effect = smtplib.SMTPException("Connection failed")
        mock_smtp_class.return_value = mock_smtp

        notifier = EmailNotifier(provider="smtp")

        success = notifier.send_email(
            subject="Test Subject",
            html_body="<h1>Test</h1>"
        )

        assert success is False

    @pytest.mark.skipif(not MSAL_AVAILABLE, reason="MSAL not available")
    def test_send_graph_email_success(self, mock_post, graph_env_vars):
        """Test successful Microsoft Graph email sending."""
        with patch('src.notifier_email.msal') as mock_msal, \
             patch('src.notifier_email.requests') as mock_requests:

            mock_app_instance = MagicMock()
            mock_token = {"access_token": "test_token"}
            mock_app_instance.acquire_token_for_client.return_value = mock_token
            mock_msal.ConfidentialClientApplication.return_value = mock_app_instance

            mock_response = MagicMock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response

            notifier = EmailNotifier(provider="graph")

            success = notifier.send_email(
                subject="Test Subject",
                html_body="<h1>Test</h1>"
            )

            assert success is True
            mock_requests.post.assert_called_once()

    @pytest.mark.skipif(not MSAL_AVAILABLE, reason="MSAL not available")
    def test_send_graph_email_failure(self, mock_post, graph_env_vars):
        """Test Microsoft Graph email sending failure."""
        with patch('src.notifier_email.msal') as mock_msal, \
             patch('src.notifier_email.requests') as mock_requests:

            mock_app_instance = MagicMock()
            mock_token = {"access_token": "test_token"}
            mock_app_instance.acquire_token_for_client.return_value = mock_token
            mock_msal.ConfidentialClientApplication.return_value = mock_app_instance

            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"
            mock_requests.post.return_value = mock_response

            notifier = EmailNotifier(provider="graph")

            success = notifier.send_email(
                subject="Test Subject",
                html_body="<h1>Test</h1>"
            )

            assert success is False

    def test_attachment_handling(self, smtp_env_vars):
        """Test attachment processing."""
        with patch('smtplib.SMTP') as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value = mock_smtp

            notifier = EmailNotifier(provider="smtp")

            # Test with non-existent attachment
            success = notifier.send_email(
                subject="Test",
                html_body="<h1>Test</h1>",
                attachments=["/nonexistent/file.txt"]
            )

            assert success is True  # Should succeed despite missing attachment

    def test_get_email_notifier_singleton(self, smtp_env_vars):
        """Test singleton pattern for email notifier."""
        notifier1 = get_email_notifier(provider="smtp")
        notifier2 = get_email_notifier(provider="smtp")

        assert notifier1 is notifier2

    def test_send_email_notification_convenience_function(self, smtp_env_vars):
        """Test convenience function for sending emails."""
        with patch('src.notifier_email.EmailNotifier.send_email') as mock_send:
            mock_send.return_value = True

            success = send_email_notification(
                subject="Test",
                html_body="<h1>Test</h1>",
                provider="smtp"
            )

            assert success is True
            mock_send.assert_called_once()

    def test_send_email_notification_failure(self):
        """Test convenience function with initialization failure."""
        with patch('src.notifier_email.EmailNotifier', side_effect=ValueError("Config error")):
            success = send_email_notification(
                subject="Test",
                html_body="<h1>Test</h1>",
                provider="smtp"
            )

            assert success is False

    @pytest.mark.skipif(not MSAL_AVAILABLE, reason="MSAL not available")
    def test_graph_attachment_processing(self, graph_env_vars):
        """Test Microsoft Graph attachment processing."""
        with patch('src.notifier_email.msal') as mock_msal, \
             patch('src.notifier_email.requests') as mock_requests:

            mock_app_instance = MagicMock()
            mock_token = {"access_token": "test_token"}
            mock_app_instance.acquire_token_for_client.return_value = mock_token
            mock_msal.ConfidentialClientApplication.return_value = mock_app_instance

            mock_response = MagicMock()
            mock_response.status_code = 202
            mock_requests.post.return_value = mock_response

            notifier = EmailNotifier(provider="graph")

            # Create temporary attachment
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("test attachment content")
                attachment_path = f.name

            try:
                success = notifier.send_email(
                    subject="Test with attachment",
                    html_body="<h1>Test</h1>",
                    attachments=[attachment_path]
                )

                assert success is True

            finally:
                Path(attachment_path).unlink(missing_ok=True)

    def test_msal_not_available(self, graph_env_vars):
        """Test graceful handling when MSAL is not available."""
        with patch('src.notifier_email.MSAL_AVAILABLE', False):
            with pytest.raises(ImportError, match="MSAL required for Microsoft Graph"):
                EmailNotifier(provider="graph")