"""
SWING_BOT Email Notifier

Provides SMTP and Microsoft Graph email fallback notifications.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import List, Optional
import mimetypes

try:
    import msal
    import requests
    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False
    print("MSAL not available. Microsoft Graph email disabled.")

from .logging_setup import get_logger

logger = get_logger(__name__)

class EmailNotifier:
    """Unified email notification system supporting SMTP and Microsoft Graph."""

    def __init__(self, provider: str = "smtp"):
        """
        Initialize email notifier.

        Args:
            provider: Email provider ('smtp' or 'graph')
        """
        self.provider = provider.lower()

        if self.provider == "smtp":
            self._setup_smtp()
        elif self.provider == "graph":
            self._setup_graph()
        else:
            raise ValueError(f"Unsupported email provider: {provider}")

    def _setup_smtp(self):
        """Setup SMTP configuration."""
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL")
        self.to_emails = os.getenv("TO_EMAILS", "").split(",")

        # Validate required config
        required = [self.smtp_server, self.smtp_username, self.smtp_password, self.from_email]
        if not all(required):
            raise ValueError("SMTP configuration incomplete. Check environment variables.")

        logger.info("SMTP email configuration loaded")

    def _setup_graph(self):
        """Setup Microsoft Graph configuration."""
        if not MSAL_AVAILABLE:
            raise ImportError("MSAL required for Microsoft Graph. Install with: pip install msal")

        self.tenant_id = os.getenv("GRAPH_TENANT_ID")
        self.client_id = os.getenv("GRAPH_CLIENT_ID")
        self.client_secret = os.getenv("GRAPH_CLIENT_SECRET")
        self.mailbox_upn = os.getenv("MAILBOX_UPN")
        self.to_emails = os.getenv("TO_EMAILS", "").split(",")

        # Validate required config
        required = [self.tenant_id, self.client_id, self.client_secret, self.mailbox_upn]
        if not all(required):
            raise ValueError("Microsoft Graph configuration incomplete. Check environment variables.")

        # Initialize MSAL app
        self.app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
            client_credential=self.client_secret
        )

        logger.info("Microsoft Graph email configuration loaded")

    def _get_graph_token(self) -> str:
        """Get Microsoft Graph access token."""
        scopes = ["https://graph.microsoft.com/.default"]
        result = self.app.acquire_token_silent(scopes, account=None)

        if not result:
            result = self.app.acquire_token_for_client(scopes=scopes)

        if "access_token" in result:
            return result["access_token"]
        else:
            raise Exception(f"Failed to acquire token: {result.get('error_description', 'Unknown error')}")

    def send_email(
        self,
        subject: str,
        html_body: str,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """
        Send email notification.

        Args:
            subject: Email subject
            html_body: HTML email body
            attachments: List of file paths to attach

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.provider == "smtp":
                return self._send_smtp_email(subject, html_body, attachments)
            elif self.provider == "graph":
                return self._send_graph_email(subject, html_body, attachments)
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                return False
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            return False

    def _send_smtp_email(
        self,
        subject: str,
        html_body: str,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """Send email via SMTP."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)

            # Add HTML body
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)

            # Add attachments
            if attachments:
                for attachment_path in attachments:
                    if os.path.exists(attachment_path):
                        self._add_attachment(msg, attachment_path)
                    else:
                        logger.warning(f"Attachment not found: {attachment_path}")

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()

            logger.info(f"SMTP email sent successfully to {len(self.to_emails)} recipients")
            return True

        except Exception as e:
            logger.error(f"SMTP email failed: {str(e)}")
            return False

    def _send_graph_email(
        self,
        subject: str,
        html_body: str,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """Send email via Microsoft Graph."""
        try:
            token = self._get_graph_token()

            # Prepare recipients
            to_recipients = [{"emailAddress": {"address": email.strip()}} for email in self.to_emails]

            # Prepare message
            message = {
                "message": {
                    "subject": subject,
                    "body": {
                        "contentType": "HTML",
                        "content": html_body
                    },
                    "toRecipients": to_recipients
                }
            }

            # Add attachments if provided
            if attachments:
                attachments_data = []
                for attachment_path in attachments:
                    if os.path.exists(attachment_path):
                        attachment_data = self._prepare_graph_attachment(attachment_path)
                        if attachment_data:
                            attachments_data.append(attachment_data)
                    else:
                        logger.warning(f"Attachment not found: {attachment_path}")

                if attachments_data:
                    message["message"]["attachments"] = attachments_data

            # Send email
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

            url = f"https://graph.microsoft.com/v1.0/users/{self.mailbox_upn}/sendMail"
            response = requests.post(url, headers=headers, json=message)

            if response.status_code == 202:
                logger.info(f"Microsoft Graph email sent successfully to {len(self.to_emails)} recipients")
                return True
            else:
                logger.error(f"Microsoft Graph email failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Microsoft Graph email failed: {str(e)}")
            return False

    def _add_attachment(self, msg: MIMEMultipart, file_path: str):
        """Add file attachment to SMTP message."""
        try:
            filename = Path(file_path).name

            # Guess MIME type
            ctype, encoding = mimetypes.guess_type(file_path)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'

            maintype, subtype = ctype.split('/', 1)

            # Read file
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # Create attachment
            attachment = MIMEBase(maintype, subtype)
            attachment.set_payload(file_data)
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)

            msg.attach(attachment)
            logger.debug(f"Attachment added: {filename}")

        except Exception as e:
            logger.error(f"Failed to add attachment {file_path}: {str(e)}")

    def _prepare_graph_attachment(self, file_path: str) -> Optional[Dict]:
        """Prepare attachment for Microsoft Graph."""
        try:
            filename = Path(file_path).name

            # Read file as base64
            import base64
            with open(file_path, 'rb') as f:
                file_content = base64.b64encode(f.read()).decode('utf-8')

            return {
                "@odata.type": "#microsoft.graph.fileAttachment",
                "name": filename,
                "contentType": mimetypes.guess_type(file_path)[0] or "application/octet-stream",
                "contentBytes": file_content
            }

        except Exception as e:
            logger.error(f"Failed to prepare Graph attachment {file_path}: {str(e)}")
            return None

# Global email notifier instance
_email_notifier = None

def get_email_notifier(provider: str = "smtp") -> Optional[EmailNotifier]:
    """Get or create global email notifier instance."""
    global _email_notifier
    if _email_notifier is None:
        try:
            _email_notifier = EmailNotifier(provider=provider)
        except Exception as e:
            logger.error(f"Failed to initialize email notifier: {str(e)}")
            return None
    return _email_notifier

def send_email_notification(
    subject: str,
    html_body: str,
    attachments: Optional[List[str]] = None,
    provider: str = "smtp"
) -> bool:
    """
    Send email notification with automatic provider setup.

    Args:
        subject: Email subject
        html_body: HTML email body
        attachments: List of file paths to attach
        provider: Email provider ('smtp' or 'graph')

    Returns:
        True if successful, False otherwise
    """
    notifier = get_email_notifier(provider=provider)
    if notifier:
        return notifier.send_email(subject, html_body, attachments)
    return False