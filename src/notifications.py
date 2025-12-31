"""
SWING_BOT Notification System
Handles Telegram alerts for regime changes and trading signals
"""

import requests
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Telegram notification handler for SWING_BOT alerts."""

    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = TELEGRAM_ENABLED

        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram notifications disabled - missing BOT_TOKEN or CHAT_ID")

    def send_message(self, message: str, parse_mode: str = None) -> bool:
        """Send a message via Telegram bot."""
        if not self.enabled:
            logger.debug(f"Telegram disabled, would send: {message}")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "disable_web_page_preview": True
            }
            
            if parse_mode:
                payload["parse_mode"] = parse_mode

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            logger.info(f"Telegram message sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_regime_alert(self, regime_data: Dict[str, Any]) -> bool:
        """Send alert for market regime change."""
        status = regime_data.get('regime_status', 'UNKNOWN')
        symbol = regime_data.get('symbol', 'NIFTY')

        emoji = "🟢" if status == "ON" else "🔴"

        message = f"""{emoji} **MARKET REGIME ALERT**

**Symbol**: {symbol}
**Status**: {status}
**Close**: {regime_data.get('latest_close', 'N/A'):.1f}
**SMA200**: {regime_data.get('sma200', 'N/A'):.1f}
**ADX(14)**: {regime_data.get('adx14', 'N/A'):.2f}
**RSI(14)**: {regime_data.get('rsi14', 'N/A'):.2f}

**Condition**: {regime_data.get('regime_condition', 'NONE')}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}

**Action**: {'TRADING ENABLED' if status == 'ON' else 'HOLDING CASH'}"""

        return self.send_message(message)

    def send_signal_alert(self, signal_data: Dict[str, Any]) -> bool:
        """Send alert for trading signal."""
        symbol = signal_data.get('symbol', 'UNKNOWN')
        strategy = signal_data.get('strategy', 'UNKNOWN')
        action = signal_data.get('action', 'UNKNOWN')
        confidence = signal_data.get('confidence', 0)

        emoji = "📈" if action.upper() == "BUY" else "📉"

        message = f"""{emoji} **TRADING SIGNAL ALERT**

**Symbol**: {symbol}
**Strategy**: {strategy}
**Action**: {action.upper()}
**Confidence**: {confidence:.1f}/10

**Entry Price**: ₹{signal_data.get('entry_price', 'N/A'):.2f}
**Stop Loss**: ₹{signal_data.get('stop_loss', 'N/A'):.2f}
**Target**: ₹{signal_data.get('target', 'N/A'):.2f}

**Risk/Reward**: {signal_data.get('risk_reward', 'N/A'):.2f}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}

⚠️ *Paper trade first - validate signal quality*"""

        return self.send_message(message)

    def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily performance summary."""
        regime_status = summary_data.get('regime_status', 'UNKNOWN')
        signals_count = summary_data.get('signals_count', 0)
        market_close = summary_data.get('market_close', 'N/A')

        emoji = "📊"

        message = f"""{emoji} **DAILY SUMMARY - {datetime.now().strftime('%Y-%m-%d')}**

**Market Regime**: {regime_status}
**Nifty Close**: {market_close}
**Signals Generated**: {signals_count}

**System Status**: {'ACTIVE' if regime_status == 'ON' else 'STANDBY'}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"""

        return self.send_message(message)

    def send_error_alert(self, error_message: str, context: str = "") -> bool:
        """Send error alert."""
        message = f"""🚨 **SYSTEM ERROR ALERT**

**Error**: {error_message}
**Context**: {context}
**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}

Please check system logs and resolve."""

        return self.send_message(message)

# Global notifier instance
notifier = TelegramNotifier()

def send_regime_alert(regime_data: Dict[str, Any]) -> bool:
    """Convenience function to send regime alerts."""
    return notifier.send_regime_alert(regime_data)

def send_signal_alert(signal_data: Dict[str, Any]) -> bool:
    """Convenience function to send signal alerts."""
    return notifier.send_signal_alert(signal_data)

def send_daily_summary(summary_data: Dict[str, Any]) -> bool:
    """Convenience function to send daily summaries."""
    return notifier.send_daily_summary(summary_data)

def send_error_alert(error_message: str, context: str = "") -> bool:
    """Convenience function to send error alerts."""
    return notifier.send_error_alert(error_message, context)