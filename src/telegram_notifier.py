"""
SWING_BOT Telegram Notifier

Sends notifications to Telegram bot for SWING_BOT updates.
"""

import os
import requests
from typing import Optional, Dict, List
from pathlib import Path

from .logging_setup import get_logger

logger = get_logger(__name__)

class TelegramNotifier:
    """Telegram notification system for SWING_BOT."""

    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'

        if self.enabled and not all([self.bot_token, self.chat_id]):
            logger.warning("Telegram enabled but missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
            self.enabled = False

    def send_message(self, message: str, parse_mode: str = 'Markdown') -> bool:
        """Send a message to Telegram."""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }

            response = requests.post(url, data=data, timeout=10)

            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")
            return False

    def send_hourly_update(self, excel_path: str, trade_signals: List[Dict], screener_summary: Dict) -> bool:
        """Send trade signals update with Excel attachment and active signals."""
        if not self.enabled:
            return False

        try:
            # Create message
            message = "🎯 *SWING_BOT Trade Signals*\n\n"

            # Add screener summary
            total_stocks = screener_summary.get('total_stocks', 0)
            stocks_with_signals = screener_summary.get('stocks_with_signals', 0)
            trade_signals_count = screener_summary.get('trade_signals', 0)
            message += f"📊 *Market Scan:* {stocks_with_signals}/{total_stocks} stocks with signals\n"
            message += f"🎯 *Active Trade Signals:* {trade_signals_count}\n\n"

            if trade_signals:
                message += "*NEW TRADES:*\n"
                for i, signal in enumerate(trade_signals[:8], 1):  # Show top 8 signals
                    symbol = signal.get('Symbol', 'N/A')
                    strategy = signal.get('Strategy', 'N/A')
                    entry = signal.get('Entry_Price', 0)
                    target = signal.get('Target_Price', 0)
                    score = signal.get('CompositeScore', 0)

                    message += f"{i}. *{symbol}* ({strategy})\n"
                    message += f"   📈 Entry: ₹{entry}, Target: ₹{target}\n"
                    message += f"   📊 Score: {score:.2f}\n\n"

                if len(trade_signals) > 8:
                    message += f"... and {len(trade_signals) - 8} more signals\n\n"
            else:
                message += "📭 No active trade signals at this time.\n\n"

            message += f"📎 *Excel Report:* {Path(excel_path).name}"

            # Send message
            return self.send_message(message)

        except Exception as e:
            logger.error(f"Failed to send trade signals update: {str(e)}")
            return False

def send_telegram_notification(message: str) -> bool:
    """Convenience function to send a simple Telegram message."""
    notifier = TelegramNotifier()
    return notifier.send_message(message)