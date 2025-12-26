#!/usr/bin/env python3
"""
SWING_BOT Teams Success Notification Script

Posts success notification to Teams with EOD results.
Usage: python scripts/post_teams_success.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from teams_notifier import post_plan_summary
from utils import get_ist_now

def main():
    """Post success notification to Teams."""

    # Setup logging
    log_dir = Path('outputs/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    import logging
    logging.basicConfig(
        filename=log_dir / f'teams_success_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting Teams success notification")

        # Check environment
        webhook_url = os.getenv('TEAMS_WEBHOOK_URL')
        if not webhook_url:
            logger.error("TEAMS_WEBHOOK_URL not set")
            print("❌ TEAMS_WEBHOOK_URL not set")
            sys.exit(1)

        # Load audited plan
        audit_file = Path('outputs/gtt/gtt_plan_audited.csv')
        if not audit_file.exists():
            logger.error(f"Audited plan file not found: {audit_file}")
            print(f"❌ Audited plan file not found: {audit_file}")
            sys.exit(1)

        # Load and analyze results
        df = pd.read_csv(audit_file)
        pass_count = (df['Audit_Flag'] == 'PASS').sum()
        fail_count = (df['Audit_Flag'] == 'FAIL').sum()
        total_positions = len(df)

        logger.info(f"Loaded plan: {total_positions} positions, {pass_count} PASS, {fail_count} FAIL")

        # Get latest date from data
        data_file = Path('data/indicators_500d.parquet')
        if data_file.exists():
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(data_file, columns=['Date'])
                latest_date = pd.to_datetime(table['Date']).max().date()
            except Exception as e:
                logger.warning(f"Could not read latest date from data: {e}")
                latest_date = get_ist_now().date()
        else:
            latest_date = get_today_ist().date()

        # Get top 5 positions
        top_positions = df.head(5)[['Symbol', 'ENTRY_trigger_price', 'STOPLOSS_trigger_price',
                                   'TARGET_trigger_price', 'DecisionConfidence', 'Audit_Flag']]

        # Excel file path
        excel_file = Path('outputs/gtt/GTT_Delivery_Final.xlsx')

        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Posting to Teams (attempt {attempt + 1}/{max_retries})")
                success = post_plan_summary(
                    webhook_url=webhook_url,
                    latest_date=str(latest_date),
                    pass_count=pass_count,
                    fail_count=fail_count,
                    excel_path=str(excel_file),
                    top_positions=top_positions
                )

                if success:
                    logger.info("Teams notification posted successfully")
                    print("✅ Teams success notification posted")
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