#!/usr/bin/env python3
"""
SWING_BOT Token Auto-Refresh Scheduler
Automatically checks and refreshes Upstox tokens before expiration.
"""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from token_manager import UpstoxTokenManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/token_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TokenScheduler:
    """Automated token refresh scheduler."""

    def __init__(self, check_interval_hours: int = 6):
        self.check_interval = check_interval_hours * 3600  # Convert to seconds
        self.manager = UpstoxTokenManager()
        self.status_file = Path('data/token_scheduler_status.json')

    def save_status(self, status: dict):
        """Save scheduler status."""
        try:
            self.status_file.parent.mkdir(exist_ok=True)
            import json
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save status: {e}")

    def load_status(self) -> dict:
        """Load scheduler status."""
        if self.status_file.exists():
            try:
                import json
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load status: {e}")
        return {}

    def should_check_token(self) -> bool:
        """Determine if token should be checked based on last check time."""
        status = self.load_status()
        last_check = status.get('last_check')

        if not last_check:
            return True

        try:
            last_check_time = datetime.fromisoformat(last_check)
            time_since_check = datetime.now() - last_check_time
            return time_since_check.total_seconds() >= self.check_interval
        except Exception as e:
            logger.error(f"Error checking last update time: {e}")
            return True

    def run_check(self) -> bool:
        """Run token check and refresh if needed."""
        logger.info("Running scheduled token check...")

        try:
            # Check if token needs attention
            token_valid = self.manager.check_and_refresh_token()

            # Update status
            status = {
                'last_check': datetime.now(),
                'token_valid': token_valid,
                'next_check': datetime.now() + timedelta(seconds=self.check_interval)
            }
            self.save_status(status)

            if token_valid:
                logger.info("✅ Token check completed successfully")
                return True
            else:
                logger.error("❌ Token check failed - manual intervention required")
                return False

        except Exception as e:
            logger.error(f"Token check error: {e}")
            return False

    def run_continuous(self):
        """Run continuous monitoring loop."""
        logger.info(f"Starting token scheduler (check every {self.check_interval/3600:.1f} hours)")

        while True:
            try:
                if self.should_check_token():
                    self.run_check()
                else:
                    logger.debug("Token check not due yet")

                # Sleep for check interval
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Token scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry

def main():
    """Main function for scheduler."""
    import argparse

    parser = argparse.ArgumentParser(description='Token Auto-Refresh Scheduler')
    parser.add_argument('--interval', type=int, default=6,
                       help='Check interval in hours (default: 6)')
    parser.add_argument('--once', action='store_true',
                       help='Run one check and exit')
    parser.add_argument('--status', action='store_true',
                       help='Show current status')

    args = parser.parse_args()

    scheduler = TokenScheduler(check_interval_hours=args.interval)

    if args.status:
        status = scheduler.load_status()
        token_info = scheduler.manager.get_token_info()

        print("🔍 Token Scheduler Status:")
        print(f"   Last Check: {status.get('last_check', 'Never')}")
        print(f"   Next Check: {status.get('next_check', 'Unknown')}")
        print(f"   Token Valid: {token_info.get('is_valid', False)}")
        print(f"   Token Expired: {token_info.get('is_expired', True)}")
        if token_info.get('hours_until_expiry'):
            print(f"   Hours Until Expiry: {token_info.get('hours_until_expiry'):.1f}")
    elif args.once:
        success = scheduler.run_check()
        exit(0 if success else 1)
    else:
        # Run continuous scheduler
        try:
            scheduler.run_continuous()
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
            exit(0)

if __name__ == "__main__":
    main()