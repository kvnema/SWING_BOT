import os
import time
import logging
from datetime import datetime, time as dt_time
from scheduled_gtt_monitor import GTTMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gtt_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalGTTScheduler:
    """Local scheduler for GTT monitoring on Windows."""

    def __init__(self):
        self.monitor = GTTMonitor()
        self.schedule_times = [
            dt_time(8, 15),  # 8:15 AM
            dt_time(9, 15),  # 9:15 AM
            dt_time(10, 15), # 10:15 AM
            dt_time(11, 15), # 11:15 AM
            dt_time(12, 15), # 12:15 PM
            dt_time(13, 15), # 1:15 PM
            dt_time(14, 15), # 2:15 PM
            dt_time(15, 15), # 3:15 PM
            dt_time(16, 30), # 4:30 PM
        ]
        self.last_run_date = None

    def should_run_now(self) -> bool:
        """Check if it's time to run the monitoring cycle."""
        now = datetime.now()
        current_time = now.time()
        current_date = now.date()

        # Don't run on weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if we've already run today for this time slot
        if self.last_run_date == current_date:
            return False

        # Check if current time is within 5 minutes of any scheduled time
        for schedule_time in self.schedule_times:
            time_diff = abs((datetime.combine(current_date, current_time) -
                           datetime.combine(current_date, schedule_time)).total_seconds())
            if time_diff <= 300:  # 5 minutes
                return True

        return False

    def run_scheduler(self):
        """Main scheduler loop."""
        logger.info("Starting SWING_BOT GTT Scheduler (Local Mode)")
        logger.info("Schedule: 8:15 AM, 9:15 AM - 3:15 PM (hourly), 4:30 PM IST")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                if self.should_run_now():
                    logger.info("Scheduled time reached - starting monitoring cycle")
                    try:
                        result = self.monitor.run_monitoring_cycle()
                        if result['success']:
                            logger.info("Monitoring cycle completed successfully")
                            self.last_run_date = datetime.now().date()
                        else:
                            logger.error(f"Monitoring cycle failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"Exception during monitoring cycle: {str(e)}")

                # Sleep for 1 minute before checking again
                time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler crashed: {str(e)}")
            raise

def main():
    """Entry point for local GTT scheduler."""
    # Check environment variables
    required_vars = ['UPSTOX_ACCESS_TOKEN', 'UPSTOX_API_KEY', 'UPSTOX_API_SECRET']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set them in your .env file or environment")
        return

    # Optional notification check
    has_notifications = any([
        os.getenv('TEAMS_WEBHOOK_URL'),
        os.getenv('TELEGRAM_BOT_TOKEN'),
        os.getenv('SMTP_USERNAME')
    ])

    if not has_notifications:
        logger.warning("No notification channels configured (Teams/Telegram/Email)")
        logger.warning("Consider setting TEAMS_WEBHOOK_URL or TELEGRAM_BOT_TOKEN for alerts")

    # Start scheduler
    scheduler = LocalGTTScheduler()
    scheduler.run_scheduler()

if __name__ == '__main__':
    main()