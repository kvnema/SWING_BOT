#!/usr/bin/env python3
"""
SWING_BOT Task Monitor

Checks the status of scheduled tasks and provides monitoring alerts.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_task_status(task_name):
    """Check if a scheduled task ran successfully today."""
    try:
        # Get task info
        result = subprocess.run(
            ['schtasks', '/query', '/tn', task_name, '/fo', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return False, f"Task query failed: {result.stderr}"

        output = result.stdout
        if not output:
            return False, "No task data returned"

        # Parse the output for key information
        lines = output.split('\n')
        last_run_time = None
        last_result = None

        for line in lines:
            line = line.strip()
            if line.startswith('Last Run Time:'):
                last_run_time = line.split(':', 1)[1].strip()
            elif line.startswith('Last Result:'):
                last_result = line.split(':', 1)[1].strip()

        if not last_run_time or last_run_time == 'Never':
            return False, "Task has never run"

        # Parse last run time
        try:
            # Handle different date formats
            if last_run_time == 'Never':
                return False, "Task has never run"
            last_run = datetime.strptime(last_run_time, '%m/%d/%Y %I:%M:%S %p')
        except ValueError:
            return False, f"Could not parse last run time: {last_run_time}"

        # Check if it ran today
        today = datetime.now().date()
        if last_run.date() != today:
            return False, f"Last run was {last_run.date()}, not today"

        # Check result code (0 = success)
        if last_result != '0':
            return False, f"Last run failed with code: {last_result}"

        return True, f"Successfully ran at {last_run_time}"

    except Exception as e:
        return False, f"Error checking task: {str(e)}"

def main():
    """Monitor scheduled tasks."""
    logger = setup_logging()
    logger.info("🔍 Checking SWING_BOT scheduled tasks...")

    tasks = [
        "SWING_BOT_Daily_Self_Improve",
        "SWING_BOT_EOD_Full"
    ]

    all_good = True
    status_report = []

    for task in tasks:
        success, message = check_task_status(task)
        if success:
            logger.info(f"✅ {task}: {message}")
            status_report.append(f"✅ {task}: {message}")
        else:
            logger.warning(f"❌ {task}: {message}")
            status_report.append(f"❌ {task}: {message}")
            all_good = False

    # Summary
    if all_good:
        logger.info("🎉 All scheduled tasks ran successfully today!")
    else:
        logger.warning("⚠️ Some tasks may need attention")

    return all_good

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)