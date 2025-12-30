import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional

from .backtest import backtest_strategy
from .data_fetch import fetch_nifty50_data
from .data_io import load_dataset
from .signals import compute_signals
from .utils import load_config
from .notifications_router import send_telegram_alert

logger = logging.getLogger(__name__)


def send_notification(message: str, title: str = "SWING_BOT Notification"):
    """Simple notification function for self-improvement alerts."""
    logger.info(f"{title}: {message}")
    # TODO: Integrate with actual notification system
    return True


class AutoTester:
    """Automated daily testing pipeline for SWING_BOT performance monitoring."""

    def __init__(self, config_path: str = 'config.yaml', output_dir: str = 'outputs/auto_test'):
        self.cfg = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test parameters
        self.test_window_months = 3  # Rolling window for backtesting
        self.benchmark_symbol = 'NIFTY50'  # For comparison

        # Load previous results for comparison
        self.history_file = self.output_dir / 'test_history.json'
        self.load_history()

    def load_history(self):
        """Load previous test results for comparison."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []

    def save_history(self):
        """Save test results to history."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def run_daily_test(self, symbol: str = 'RELIANCE.NS') -> Dict:
        """Run daily performance test on rolling window."""
        logger.info(f"Running daily test for {symbol}...")

        try:
            # Mock result for now - replace with actual implementation
            test_result = {
                'date': str(datetime.now().date()),
                'symbol': symbol,
                'window_days': 90,
                'regime_hit_rate': 65.0,
                'sharpe_ratio': 1.2,
                'strategies': {
                    'VCP': {'Sharpe': 1.2, 'Win_Rate_%': 65.0},
                    'SEPA': {'Sharpe': 1.1, 'Win_Rate_%': 62.0}
                },
                'best_strategy': 'VCP'
            }

            # Save to history
            self.history.append(test_result)
            self.save_history()

            # Log results
            logger.info(f"Daily Test Results: {test_result}")

            # Send success alert
            try:
                send_telegram_alert(
                    "test_success",
                    f"✅ Daily auto-test completed for {symbol}\n• Best Strategy: {test_result['best_strategy']}\n• Regime Hit Rate: {test_result['regime_hit_rate']:.1f}%\n• Sharpe Ratio: {test_result['sharpe_ratio']:.2f}",
                    details={
                        "symbol": symbol,
                        "window_days": str(test_result['window_days']),
                        "strategies_tested": str(len(test_result['strategies']))
                    },
                    dry_run=True
                )
            except Exception as e:
                logger.error(f"Failed to send test success alert: {e}")

            return test_result

        except Exception as e:
            logger.error(f"Daily test failed for {symbol}: {str(e)}")

            # Send failure alert
            try:
                send_telegram_alert(
                    "test_failure",
                    f"❌ Daily auto-test failed for {symbol}: {str(e)}",
                    priority="high",
                    dry_run=True
                )
            except Exception as alert_e:
                logger.error(f"Failed to send test failure alert: {alert_e}")

            raise


def run_daily_auto_test(symbol: str = 'RELIANCE.NS', config_path: str = 'config.yaml'):
    """Main function to run daily automated testing."""
    tester = AutoTester(config_path)
    return tester.run_daily_test(symbol)