#!/usr/bin/env python3
"""
SWING_BOT Market Monitoring & Alert System
Monitors market regime and sends Telegram alerts for status changes and signals
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.data_fetch import calculate_market_regime
from src.notifications import send_regime_alert, send_daily_summary, send_error_alert
from src.signals import compute_signals
from src.data_fetch import fetch_market_index_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketMonitor:
    """Monitors market regime and sends alerts."""

    def __init__(self, state_file: str = 'data/monitor_state.json'):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(exist_ok=True)
        self.previous_regime = self.load_previous_state()

    def load_previous_state(self) -> Dict[str, Any]:
        """Load previous monitoring state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
        return {}

    def save_state(self, state: Dict[str, Any]):
        """Save current monitoring state."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def check_regime_change(self) -> Optional[Dict[str, Any]]:
        """Check if market regime has changed."""
        try:
            regime_data = calculate_market_regime('NSE_INDEX|Nifty 50')
            current_status = regime_data.get('regime_status')

            previous_status = self.previous_regime.get('regime_status')

            if current_status != previous_status:
                logger.info(f"Regime change detected: {previous_status} -> {current_status}")
                return regime_data

            return None

        except Exception as e:
            logger.error(f"Failed to check regime: {e}")
            send_error_alert(f"Regime check failed: {str(e)}", "Market Monitor")
            return None

    def generate_signals_summary(self) -> Dict[str, Any]:
        """Generate summary of current signals."""
        try:
            # Get recent data for a sample stock
            df = fetch_market_index_data('RELIANCE.NS', 100)
            if df.empty:
                return {'signals_count': 0, 'error': 'No data available'}

            # Compute signals
            df_signals = compute_signals(df)

            # Count active signals (simplified - in practice you'd check specific flags)
            signal_flags = ['SEPA_Flag', 'VCP_Flag', 'Donchian_Flag', 'MR_Flag', 'AVWAP_Flag']
            active_signals = 0

            for flag in signal_flags:
                if flag in df_signals.columns:
                    # Count recent signals (last 5 days)
                    recent_signals = df_signals[df_signals[flag] == 1].tail(5)
                    active_signals += len(recent_signals)

            return {
                'signals_count': active_signals,
                'last_updated': datetime.now().isoformat(),
                'sample_stock': 'RELIANCE.NS'
            }

        except Exception as e:
            logger.error(f"Failed to generate signals summary: {e}")
            return {'signals_count': 0, 'error': str(e)}

    def send_daily_report(self):
        """Send daily market summary."""
        try:
            # Get current regime
            regime_data = calculate_market_regime('NSE_INDEX|Nifty 50')

            # Get signals summary
            signals_summary = self.generate_signals_summary()

            # Prepare summary data
            summary_data = {
                'regime_status': regime_data.get('regime_status', 'UNKNOWN'),
                'market_close': regime_data.get('latest_close', 'N/A'),
                'signals_count': signals_summary.get('signals_count', 0),
                'date': datetime.now().strftime('%Y-%m-%d')
            }

            # Send daily summary
            if send_daily_summary(summary_data):
                logger.info("Daily summary sent successfully")
            else:
                logger.warning("Failed to send daily summary")

        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
            send_error_alert(f"Daily report failed: {str(e)}", "Market Monitor")

    def run_monitoring_cycle(self):
        """Run one monitoring cycle."""
        logger.info("Starting monitoring cycle...")

        # Check for regime changes
        regime_change = self.check_regime_change()
        if regime_change:
            if send_regime_alert(regime_change):
                logger.info("Regime change alert sent")
                # Update stored state
                self.previous_regime = {
                    'regime_status': regime_change.get('regime_status'),
                    'last_updated': datetime.now().isoformat()
                }
                self.save_state(self.previous_regime)
            else:
                logger.error("Failed to send regime alert")

        # Update state with current info
        current_state = {
            'regime_status': calculate_market_regime('NSE_INDEX|Nifty 50').get('regime_status'),
            'last_check': datetime.now().isoformat(),
            'signals_summary': self.generate_signals_summary()
        }
        self.save_state(current_state)

        logger.info("Monitoring cycle completed")

def main():
    parser = argparse.ArgumentParser(description='SWING_BOT Market Monitor')
    parser.add_argument('--mode', choices=['once', 'daily', 'continuous'],
                       default='once', help='Monitoring mode')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Check interval in seconds (default: 1 hour)')
    parser.add_argument('--daily-report', action='store_true',
                       help='Send daily summary report')

    args = parser.parse_args()

    monitor = MarketMonitor()

    if args.mode == 'once':
        # Single check
        monitor.run_monitoring_cycle()
        if args.daily_report:
            monitor.send_daily_report()

    elif args.mode == 'daily':
        # Send daily report only
        monitor.send_daily_report()

    elif args.mode == 'continuous':
        # Continuous monitoring
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        while True:
            try:
                monitor.run_monitoring_cycle()
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")
                time.sleep(60)  # Wait 1 minute before retry

if __name__ == "__main__":
    main()