"""
SWING_BOT Notifications Router

Orchestrates Teams and email fallback notifications.
"""

import os
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path

from .teams_notifier import post_plan_summary, post_error_notification
from .notifier_email import send_email_notification
from .dashboards.teams_dashboard import build_adaptive_card_summary, build_failure_card
from .logging_setup import get_logger

logger = get_logger(__name__)

class NotificationConfig:
    """Configuration for notifications."""

    def __init__(self):
        self.teams_enabled = True
        self.email_enabled = True
        self.fallback_enabled = True
        self.webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
        self.email_provider = "smtp"

    def validate(self) -> bool:
        """Validate notification configuration."""
        if self.teams_enabled and not self.webhook_url:
            logger.warning("Teams enabled but TEAMS_WEBHOOK_URL not set")
            return False
        return True

def notify_eod_success(
    webhook_url: Optional[str],
    latest_date: str,
    pass_count: int,
    fail_count: int,
    top_rows_df,
    file_links: Dict[str, str],
    email_config: Optional[NotificationConfig] = None
) -> bool:
    """
    Send success notification with Teams primary and email fallback.

    Args:
        webhook_url: Teams webhook URL
        latest_date: Latest trading date
        pass_count: Number of passing positions
        fail_count: Number of failing positions
        top_rows_df: Top positions DataFrame
        file_links: Dictionary of file links
        email_config: Email configuration

    Returns:
        True if at least one notification method succeeded
    """

    if email_config is None:
        email_config = NotificationConfig()

    success = False

    # Try Teams first (prefer explicit webhook_url to avoid env flakiness)
    if webhook_url:
        try:
            logger.info("Attempting Teams success notification")

            # Build Adaptive Card
            card = build_adaptive_card_summary(
                latest_date=latest_date,
                pass_count=pass_count,
                fail_count=fail_count,
                top_rows_df=top_rows_df,
                links=file_links
            )

            # Post to Teams
            payload = {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]}
            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info("Teams success notification sent successfully")
                success = True
            else:
                logger.warning(f"Teams notification failed: {response.status_code}")
                raise Exception(f"Teams API returned {response.status_code}")

        except Exception as e:
            logger.error(f"Teams success notification failed: {str(e)}")

            # Try email fallback
            if email_config.fallback_enabled and email_config.email_enabled:
                logger.info("Attempting email fallback for success notification")
                success = _send_success_email_fallback(
                    latest_date, pass_count, fail_count, top_rows_df, file_links, email_config
                )

    # If Teams disabled or failed without fallback, try email directly
    elif email_config.email_enabled:
        logger.info("Teams disabled, sending email success notification")
        success = _send_success_email_fallback(
            latest_date, pass_count, fail_count, top_rows_df, file_links, email_config
        )

    return success

def notify_eod_failure(
    webhook_url: Optional[str],
    stage: str,
    error_msg: str,
    hints: List[str],
    file_links: Dict[str, str],
    email_config: Optional[NotificationConfig] = None
) -> bool:
    """
    Send failure notification with Teams primary and email fallback.

    Args:
        webhook_url: Teams webhook URL
        stage: Pipeline stage where failure occurred
        error_msg: Error message
        hints: Troubleshooting hints
        file_links: Dictionary of file links
        email_config: Email configuration

    Returns:
        True if at least one notification method succeeded
    """

    if email_config is None:
        email_config = NotificationConfig()

    success = False

    # Try Teams first (prefer explicit webhook_url to avoid env flakiness)
    if webhook_url:
        try:
            logger.info("Attempting Teams failure notification")

            # Build failure card
            card = build_failure_card(
                stage=stage,
                error_msg=error_msg,
                hints=hints,
                links=file_links
            )

            # Post to Teams
            payload = {"type": "message", "attachments": [{"contentType": "application/vnd.microsoft.card.adaptive", "content": card}]}
            response = requests.post(webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info("Teams failure notification sent successfully")
                success = True
            else:
                logger.warning(f"Teams notification failed: {response.status_code}")
                raise Exception(f"Teams API returned {response.status_code}")

        except Exception as e:
            logger.error(f"Teams failure notification failed: {str(e)}")

            # Try email fallback
            if email_config.fallback_enabled and email_config.email_enabled:
                logger.info("Attempting email fallback for failure notification")
                success = _send_failure_email_fallback(
                    stage, error_msg, hints, file_links, email_config
                )

    # If Teams disabled or failed without fallback, try email directly
    elif email_config.email_enabled:
        logger.info("Teams disabled, sending email failure notification")
        success = _send_failure_email_fallback(
            stage, error_msg, hints, file_links, email_config
        )

    return success

def _send_success_email_fallback(
    latest_date: str,
    pass_count: int,
    fail_count: int,
    top_rows_df,
    file_links: Dict[str, str],
    email_config: NotificationConfig
) -> bool:
    """Send success notification via email."""
    try:
        # Build HTML email body
        html_body = f"""
        <html>
        <body>
            <h2>SWING_BOT Daily Summary - {latest_date}</h2>

            <h3>Results</h3>
            <ul>
                <li><strong>Pass Count:</strong> {pass_count}</li>
                <li><strong>Fail Count:</strong> {fail_count}</li>
                <li><strong>Total Positions:</strong> {pass_count + fail_count}</li>
            </ul>

            <h3>Top Positions</h3>
            <table border="1" style="border-collapse: collapse;">
                <tr>
                    <th>Symbol</th>
                    <th>Entry Price</th>
                    <th>Stop Loss</th>
                    <th>Target</th>
                    <th>Confidence</th>
                    <th>Audit Status</th>
                </tr>
        """

        if not top_rows_df.empty:
            for _, row in top_rows_df.head(5).iterrows():
                html_body += f"""
                <tr>
                    <td>{row.get('Symbol', 'N/A')}</td>
                    <td>₹{row.get('ENTRY_trigger_price', 0):.2f}</td>
                    <td>₹{row.get('STOPLOSS_trigger_price', 0):.2f}</td>
                    <td>₹{row.get('TARGET_trigger_price', 0):.2f}</td>
                    <td>{row.get('DecisionConfidence', 0):.3f}</td>
                    <td>{row.get('Audit_Flag', 'UNKNOWN')}</td>
                </tr>
                """
        else:
            html_body += "<tr><td colspan='6'>No positions available</td></tr>"

        html_body += """
            </table>

            <h3>Attachments</h3>
            <ul>
        """

        # Prepare attachments
        attachments = []
        for name, path in file_links.items():
            if Path(path).exists() and Path(path).stat().st_size < 10 * 1024 * 1024:  # < 10MB
                attachments.append(path)
                html_body += f"<li>{name}: {Path(path).name}</li>"
            else:
                html_body += f"<li>{name}: <a href='file://{path}'>View File</a></li>"

        html_body += """
            </ul>

            <p><em>This is an automated notification from SWING_BOT.</em></p>
        </body>
        </html>
        """

        subject = f"SWING_BOT Daily Summary - {latest_date}"

        return send_email_notification(
            subject=subject,
            html_body=html_body,
            attachments=attachments,
            provider=email_config.email_provider
        )

    except Exception as e:
        logger.error(f"Email success fallback failed: {str(e)}")
        return False

def _send_failure_email_fallback(
    stage: str,
    error_msg: str,
    hints: List[str],
    file_links: Dict[str, str],
    email_config: NotificationConfig
) -> bool:
    """Send failure notification via email."""
    try:
        # Build HTML email body
        hints_html = "<ul>" + "".join(f"<li>{hint}</li>" for hint in hints) + "</ul>"

        html_body = f"""
        <html>
        <body style="color: #721c24; background-color: #f8d7da; padding: 20px;">
            <h2 style="color: #721c24;">🚨 SWING_BOT Pipeline Failure</h2>

            <h3>Failure Details</h3>
            <ul>
                <li><strong>Stage:</strong> {stage}</li>
                <li><strong>Error:</strong> {error_msg}</li>
            </ul>

            <h3>Troubleshooting Hints</h3>
            {hints_html}

            <h3>Related Files</h3>
            <ul>
        """

        # Prepare attachments
        attachments = []
        for name, path in file_links.items():
            if Path(path).exists() and Path(path).stat().st_size < 10 * 1024 * 1024:  # < 10MB
                attachments.append(path)
                html_body += f"<li>{name}: {Path(path).name} (attached)</li>"
            else:
                html_body += f"<li>{name}: <a href='file://{path}'>View File</a></li>"

        html_body += """
            </ul>

            <p><em>This is an automated failure notification from SWING_BOT.</em></p>
        </body>
        </html>
        """

        subject = f"🚨 SWING_BOT Pipeline Failure - {stage}"

        return send_email_notification(
            subject=subject,
            html_body=html_body,
            attachments=attachments,
            provider=email_config.email_provider
        )

    except Exception as e:
        logger.error(f"Email failure fallback failed: {str(e)}")
        return False