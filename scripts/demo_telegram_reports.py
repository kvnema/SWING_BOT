#!/usr/bin/env python3
"""
Demo SWING_BOT Telegram Reports

Shows sample Telegram reports for the self-improvement system.
"""

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def send_telegram_self_improvement_report_demo(
    optimized_params,
    recent_performance,
    system_health,
    run_id=None,
    dry_run=True
):
    """Demo version of Telegram self-improvement report."""

    # Build message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    run_info = f" (Run: {run_id})" if run_id else ""

    message = f"🚀 *SWING_BOT Daily Self-Improvement Report* {run_info}\n"
    message += f"📅 {timestamp}\n\n"

    # Current Parameters
    if optimized_params:
        message += "📊 *Current Optimized Parameters:*\n"
        for key, value in optimized_params.items():
            if key not in ['last_updated', 'performance_baseline']:
                message += f"• {key}: {value}\n"
        message += f"• Performance Baseline: {optimized_params.get('performance_baseline', 'N/A')}\n"
        message += f"• Last Updated: {optimized_params.get('last_updated', 'Never')}\n\n"
    else:
        message += "📊 *Current Parameters:* No optimized parameters found\n\n"

    # Recent Performance
    if recent_performance:
        message += "📈 *Recent Performance (Last 7 Days):*\n"
        for perf in recent_performance[-3:]:  # Show last 3 entries
            date = perf.get('date', 'N/A')
            symbol = perf.get('symbol', 'N/A')
            strategy = perf.get('best_strategy', 'N/A')
            sharpe = perf.get('sharpe_ratio', 0)
            regime = perf.get('regime_hit_rate', 0)
            message += f"• {date}: {symbol} | Strategy: {strategy} | Sharpe: {sharpe:.2f} | Regime: {regime:.1f}%\n"
        message += "\n"
    else:
        message += "📈 *Recent Performance:* No recent data available\n\n"

    # System Health
    health_status = system_health.get('status', 'unknown')
    health_icon = "✅" if health_status == 'healthy' else "⚠️" if health_status == 'warning' else "❌"
    message += f"{health_icon} *System Health:* {health_status.title()}\n"

    if system_health.get('issues'):
        for issue in system_health['issues']:
            message += f"• {issue}\n"

    # Footer
    message += f"\n🔄 *Next Run:* Daily at 16:30 IST (weekdays)\n"
    message += f"📋 *Commands:*\n"
    message += f"• Manual run: `python scripts\\daily_self_improve.py`\n"
    message += f"• Check status: `python scripts\\status_dashboard.py`\n"
    message += f"• View logs: `type logs\\daily_self_improve_*.log`"

    if dry_run:
        logger.info("DRY RUN - Telegram Self-Improvement Report:")
        logger.info("=" * 50)
        for line in message.split('\n'):
            logger.info(line)
        logger.info("=" * 50)
        return True

    return True

def send_telegram_alert_demo(
    alert_type,
    message,
    details=None,
    run_id=None,
    priority="normal",
    dry_run=True
):
    """Demo version of Telegram alert."""

    # Priority indicators
    priority_icons = {
        "normal": "📢",
        "high": "⚠️",
        "critical": "🚨"
    }
    icon = priority_icons.get(priority, "📢")

    # Build message
    timestamp = datetime.now().strftime("%H:%M IST")
    run_info = f" (Run: {run_id})" if run_id else ""

    alert_message = f"{icon} *SWING_BOT Alert: {alert_type.replace('_', ' ').title()}* {run_info}\n"
    alert_message += f"🕐 {timestamp}\n\n"
    alert_message += f"{message}\n"

    # Add details if provided
    if details:
        alert_message += "\n📋 *Details:*\n"
        for key, value in details.items():
            alert_message += f"• {key}: {value}\n"

    if dry_run:
        logger.info(f"DRY RUN - Telegram Alert ({alert_type}):")
        logger.info("=" * 50)
        for line in alert_message.split('\n'):
            logger.info(line)
        logger.info("=" * 50)
        return True

    return True

def demo_telegram_reports():
    """Demo the Telegram reporting functionality."""

    print("🚀 SWING_BOT Telegram Reports Demo")
    print("=" * 50)

    # Sample optimized parameters
    optimized_params = {
        'rsi_min': 28.5,
        'rsi_max': 66.5,
        'adx_threshold': 21.0,
        'trail_multiplier': 1.575,
        'ensemble_count': 3.15,
        'atr_period': 13.3,
        'last_updated': '2025-12-29',
        'performance_baseline': 1.3493
    }

    # Sample recent performance
    recent_performance = [
        {
            'date': '2025-12-29',
            'symbol': 'RELIANCE.NS',
            'best_strategy': 'VCP',
            'sharpe_ratio': 1.2,
            'regime_hit_rate': 65.0
        },
        {
            'date': '2025-12-28',
            'symbol': 'TCS.NS',
            'best_strategy': 'SEPA',
            'sharpe_ratio': 1.1,
            'regime_hit_rate': 62.0
        }
    ]

    # Sample system health
    system_health = {
        'status': 'healthy',
        'issues': []
    }

    print("\n1. 📊 Daily Self-Improvement Report")
    print("-" * 40)

    # Send daily report (dry run)
    send_telegram_self_improvement_report_demo(
        optimized_params,
        recent_performance,
        system_health,
        run_id="20251229_1645",
        dry_run=True
    )

    print("\n2. ⚠️ Instant Alerts")
    print("-" * 40)

    # Parameter change alert
    send_telegram_alert_demo(
        "parameter_change",
        "🔄 Parameters updated with +12.3% improvement\n• Changes applied: 3 parameters",
        details={
            'rsi_min': '28.5',
            'adx_threshold': '21.0',
            'trail_multiplier': '1.575'
        },
        run_id="20251229_1645",
        priority="high",
        dry_run=True
    )

    print()

    # Test completion alert
    send_telegram_alert_demo(
        "test_complete",
        "✅ Auto-testing completed successfully\n• Symbol: RELIANCE.NS\n• Best Strategy: VCP\n• Regime Hit Rate: 65.0%",
        details={"sharpe_ratio": "1.2"},
        run_id="20251229_1645",
        dry_run=True
    )

    print()

    # Critical error alert
    send_telegram_alert_demo(
        "cycle_failure",
        "🚨 Daily self-improvement cycle failed: Data fetch timeout",
        priority="critical",
        run_id="20251229_1645",
        dry_run=True
    )

    print("\n✅ Demo completed! These messages would be sent to Telegram when configured.")
    print("\nTo enable Telegram reports:")
    print("1. Create a Telegram bot: Message @BotFather with /newbot")
    print("2. Get your chat ID: Message @userinfobot")
    print("3. Set environment variables:")
    print("   TELEGRAM_BOT_TOKEN=your_bot_token")
    print("   TELEGRAM_CHAT_ID=your_chat_id")

if __name__ == '__main__':
    demo_telegram_reports()