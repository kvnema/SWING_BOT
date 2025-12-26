#!/usr/bin/env python3
"""
SWING_BOT Teams Failure Notification Script

Posts failure notification to Teams with error details.
Usage: python scripts/post_teams_failure.py [--error-category CATEGORY] [--error-message MESSAGE]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from teams_notifier import post_error_notification

def main():
    """Post failure notification to Teams."""

    # Setup logging
    log_dir = Path('outputs/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    import logging
    logging.basicConfig(
        filename=log_dir / f'teams_failure_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Post failure notification to Teams')
    parser.add_argument('--error-category', choices=['DATA', 'SIGNAL', 'GTT', 'VALIDATION', 'SYSTEM'],
                       default='SYSTEM', help='Error category')
    parser.add_argument('--error-message', default='Unknown error occurred',
                       help='Error message details')
    parser.add_argument('--retry-count', type=int, default=0,
                       help='Number of retries attempted')

    args = parser.parse_args()

    try:
        logger.info(f"Starting Teams failure notification: {args.error_category} - {args.error_message}")

        # Check environment
        webhook_url = os.getenv('TEAMS_WEBHOOK_URL')
        if not webhook_url:
            logger.error("TEAMS_WEBHOOK_URL not set")
            print("❌ TEAMS_WEBHOOK_URL not set")
            sys.exit(1)

        # Get latest log file for context
        log_files = list(log_dir.glob('*.log'))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            with open(latest_log, 'r') as f:
                recent_logs = f.readlines()[-10:]  # Last 10 lines
        else:
            recent_logs = ["No recent logs available"]

        # Get system info
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'environment': os.getenv('ENVIRONMENT', 'production')
        }

        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Posting failure to Teams (attempt {attempt + 1}/{max_retries})")
                success = post_error_notification(
                    webhook_url=webhook_url,
                    error_message=f"{args.error_category}: {args.error_message}",
                    stage="EOD Pipeline"
                )

                if success:
                    logger.info("Teams failure notification posted successfully")
                    print("✅ Teams failure notification posted")
                    return
                else:
                    logger.warning(f"Teams posting failed (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"Teams posting error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

        logger.error("All Teams posting attempts failed")
        print("❌ Teams notification failed after all retries")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()