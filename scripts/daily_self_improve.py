#!/usr/bin/env python3
"""
SWING_BOT Daily Self-Improving Cycle

This script runs the complete daily self-improvement pipeline:
1. Auto-testing on recent market data
2. Self-optimization of strategy parameters
3. Logging and notifications

Usage:
    python scripts/daily_self_improve.py

Or schedule via cron/Task Scheduler:
    # Daily at 16:30 IST (post-market close)
    30 16 * * 1-5 /path/to/python /path/to/scripts/daily_self_improve.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.auto_test import run_daily_auto_test
from src.self_optimize import run_daily_self_optimization
from src.notifications_router import send_telegram_self_improvement_report, send_telegram_alert
from scripts.status_dashboard import load_optimized_params, load_test_history


def send_notification(message: str, title: str = "SWING_BOT Notification"):
    """Simple notification function."""
    logger = logging.getLogger(__name__)
    logger.info(f"{title}: {message}")
    return True  # TODO: Fix import


def setup_logging():
    """Setup logging for the daily cycle."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f'daily_self_improve_{datetime.now().strftime("%Y%m%d")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def main():
    """Run the daily self-improving cycle."""
    logger = setup_logging()
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    logger.info(f"🚀 Starting SWING_BOT Daily Self-Improvement Cycle (Run: {run_id})")

    start_time = datetime.now()
    success = True
    results = {}

    try:
        # Step 1: Auto-Testing
        logger.info("🧪 Step 1: Running daily auto-testing...")
        try:
            test_result = run_daily_auto_test()
            results['auto_test'] = test_result
            logger.info("✅ Auto-testing completed successfully")

            # Send test completion alert
            send_telegram_alert(
                "test_complete",
                f"✅ Auto-testing completed successfully\n• Symbol: {test_result.get('symbol', 'N/A')}\n• Best Strategy: {test_result.get('best_strategy', 'N/A')}\n• Regime Hit Rate: {test_result.get('regime_hit_rate', 0):.1f}%",
                details={"sharpe_ratio": f"{test_result.get('sharpe_ratio', 0):.2f}"},
                run_id=run_id
            )

        except Exception as e:
            logger.error(f"❌ Auto-testing failed: {str(e)}")
            results['auto_test'] = {'error': str(e)}
            success = False

            # Send test failure alert
            send_telegram_alert(
                "test_failure",
                f"❌ Auto-testing failed: {str(e)}",
                priority="high",
                run_id=run_id
            )

        # Step 2: Self-Optimization
        logger.info("🔧 Step 2: Running self-optimization...")
        try:
            optimize_result = run_daily_self_optimization()
            results['self_optimize'] = optimize_result
            logger.info("✅ Self-optimization completed successfully")

            # Check if parameters were updated
            if optimize_result.get('applied_changes'):
                changes = optimize_result['applied_changes']
                improvement = optimize_result.get('improvement_pct', 0)

                # Send parameter change alert
                change_details = {k: f"{v:.3f}" for k, v in changes.items()}
                send_telegram_alert(
                    "parameter_change",
                    f"🔄 Parameters updated with {improvement:+.1f}% improvement\n• Changes applied: {len(changes)} parameters",
                    details=change_details,
                    run_id=run_id,
                    priority="high"
                )

        except Exception as e:
            logger.error(f"❌ Self-optimization failed: {str(e)}")
            results['self_optimize'] = {'error': str(e)}
            success = False

            # Send optimization failure alert
            send_telegram_alert(
                "optimization_failure",
                f"❌ Self-optimization failed: {str(e)}",
                priority="critical",
                run_id=run_id
            )

        # Step 3: Send daily summary report
        logger.info("📢 Step 3: Sending daily summary report...")
        try:
            # Load current data for report
            optimized_params = load_optimized_params()
            recent_performance = load_test_history()

            # Simple system health check
            system_health = {"status": "healthy", "issues": []}
            if not success:
                system_health["status"] = "warning"
                system_health["issues"].append("Issues detected in daily cycle")

            # Send Telegram report
            report_success = send_telegram_self_improvement_report(
                optimized_params,
                recent_performance,
                system_health,
                run_id
            )

            if report_success:
                logger.info("✅ Daily Telegram report sent successfully")
            else:
                logger.warning("⚠️ Daily Telegram report failed to send")

        except Exception as e:
            logger.error(f"❌ Failed to send daily report: {str(e)}")

        # Legacy notification (keep for backward compatibility)
        duration = datetime.now() - start_time
        message = f"SWING_BOT Daily Self-Improvement Summary\n"
        message += f"Date: {datetime.now().date()}\n"
        message += f"Duration: {duration.total_seconds():.1f}s\n"
        message += f"Status: {'✅ Success' if success else '❌ Issues Detected'}\n\n"

        # Auto-test summary
        if 'auto_test' in results and 'error' not in results['auto_test']:
            test = results['auto_test']
            message += f"🧪 Auto-Test Results:\n"
            message += f"  Symbol: {test.get('symbol', 'N/A')}\n"
            message += f"  Window: {test.get('window_days', 0)} days\n"
            message += f"  Best Strategy: {test.get('best_strategy', 'N/A')}\n"
            message += f"  Regime Hit Rate: {test.get('regime_hit_rate', 0):.1f}%\n"
        else:
            message += f"🧪 Auto-Test: Failed\n"

        # Optimization summary
        if 'self_optimize' in results and 'error' not in results['self_optimize']:
            opt = results['self_optimize']
            message += f"\n🔧 Self-Optimization Results:\n"
            message += f"  Validation Passed: {opt.get('validation_passed', False)}\n"
            message += f"  Improvement: {opt.get('improvement_pct', 0):+.1f}%\n"
            if opt.get('applied_changes'):
                message += f"  Parameters Updated: {len(opt['applied_changes'])}\n"
        else:
            message += f"\n🔧 Self-Optimization: Failed\n"

        try:
            send_notification(message, "Daily Self-Improvement Summary")
            logger.info("✅ Legacy summary notification sent")
        except Exception as e:
            logger.error(f"❌ Failed to send legacy notification: {str(e)}")

    except Exception as e:
        logger.error(f"❌ Daily cycle failed: {str(e)}")
        success = False

        # Send critical failure alert
        send_telegram_alert(
            "cycle_failure",
            f"🚨 Daily self-improvement cycle failed: {str(e)}",
            priority="critical",
            run_id=run_id,
            dry_run=True
        )

        # Send failure notification
        try:
            error_message = f"SWING_BOT Daily Self-Improvement FAILED\n"
            error_message += f"Date: {datetime.now().date()}\n"
            error_message += f"Error: {str(e)}\n"
            send_notification(error_message, "Daily Self-Improvement FAILURE")
        except Exception:
            pass

    logger.info(f"🏁 Daily Self-Improvement Cycle {'completed successfully' if success else 'completed with issues'} (Run: {run_id})")
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)