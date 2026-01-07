#!/usr/bin/env python3
"""
SWING_BOT Dashboard Diagnostic Tool

Diagnoses dashboard data loading issues and identifies missing or stale data sources.
Run this to troubleshoot why the dashboard isn't showing current live trading data.
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.logging_setup import setup_logging

# Setup logging
logger = setup_logging("diagnose_dashboard")

# Output directories
OUTPUTS_DIR = Path('outputs')
GTT_DIR = OUTPUTS_DIR / 'gtt'

def diagnose_data_sources():
    """Diagnose all dashboard data sources and their freshness."""

    print("[SEARCH] SWING_BOT Dashboard Diagnostic Tool")
    print("=" * 60)

    issues = []
    recommendations = []

    # 1. Check GTT Plan Files
    print("\n[DATA] 1. GTT Plan Files Analysis:")
    print("-" * 40)

    gtt_files = {
        'gtt_plan_audited.csv': 'Legacy audited plan (dashboard currently uses this)',
        'gtt_plan_live_audited.csv': 'Live orchestration audited plan',
        'gtt_plan_live_reconciled.csv': 'Live reconciled plan',
        'gtt_plan_live.csv': 'Live generated plan'
    }

    latest_dates = {}
    latest_file = None
    latest_date = None

    for filename, description in gtt_files.items():
        file_path = GTT_DIR / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if not df.empty and 'Date' in df.columns:
                    # Get the most recent date in the file
                    dates = pd.to_datetime(df['Date'], errors='coerce')
                    max_date = dates.max()
                    latest_dates[filename] = max_date

                    print(f"[OK] {filename}: {len(df)} positions, latest date: {max_date.strftime('%Y-%m-%d')}")

                    if latest_date is None or max_date > latest_date:
                        latest_date = max_date
                        latest_file = filename
                else:
                    print(f"[WARN] {filename}: File exists but empty or missing Date column")
            except Exception as e:
                print(f"[ERROR] {filename}: Error reading file - {e}")
        else:
            print(f"[ERROR] {filename}: File not found")

    if latest_file:
        print(f"\n[TARGET] Latest data is in: {latest_file} (date: {latest_date.strftime('%Y-%m-%d')})")

        # Check if dashboard is using the right file
        dashboard_file = GTT_DIR / 'gtt_plan_audited.csv'
        if dashboard_file.exists():
            try:
                dashboard_df = pd.read_csv(dashboard_file)
                if not dashboard_df.empty and 'Date' in dashboard_df.columns:
                    dashboard_dates = pd.to_datetime(dashboard_df['Date'], errors='coerce')
                    dashboard_max_date = dashboard_dates.max()
                    print(f"[CHART] Dashboard currently shows: {dashboard_max_date.strftime('%Y-%m-%d')}")

                    if dashboard_max_date < latest_date:
                        days_diff = (latest_date - dashboard_max_date).days
                        issues.append(f"Dashboard data is {days_diff} days stale")
                        recommendations.append(f"Update dashboard to use {latest_file} instead of gtt_plan_audited.csv")
                else:
                    issues.append("Dashboard GTT file exists but is empty or malformed")
            except Exception as e:
                issues.append(f"Dashboard GTT file cannot be read: {e}")
        else:
            issues.append("Dashboard GTT file (gtt_plan_audited.csv) does not exist")
            recommendations.append(f"Create symlink or copy from {latest_file}")

    # 2. Check Audit Status
    print("\n[SEARCH] 2. Audit Status Analysis:")
    print("-" * 40)

    for filename in ['gtt_plan_audited.csv', 'gtt_plan_live_audited.csv']:
        file_path = GTT_DIR / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if 'Audit_Flag' in df.columns:
                    pass_count = len(df[df['Audit_Flag'] == 'PASS'])
                    fail_count = len(df[df['Audit_Flag'] == 'FAIL'])
                    total = len(df)

                    print(f"{filename}:")
                    print(f"  [OK] PASS: {pass_count}/{total} ({pass_count/total*100:.1f}%)")
                    print(f"  [ERROR] FAIL: {fail_count}/{total} ({fail_count/total*100:.1f}%)")

                    if fail_count > 0:
                        issues.append(f"{filename} has {fail_count} failed audits")
                else:
                    print(f"[WARN] {filename}: No Audit_Flag column found")
            except Exception as e:
                print(f"[ERROR] {filename}: Error analyzing audit status - {e}")

    # 3. Check Live Positions
    print("\n[CHART] 3. Live Positions Check:")
    print("-" * 40)

    live_positions_file = OUTPUTS_DIR / 'live_positions.json'
    if live_positions_file.exists():
        try:
            with open(live_positions_file, 'r') as f:
                positions = json.load(f)

            print(f"[OK] Live positions file exists: {len(positions)} positions")

            if positions:
                for symbol, pos_data in list(positions.items())[:3]:  # Show first 3
                    entry_time = pos_data.get('entry_time', 'Unknown')
                    print(f"  - {symbol}: Entry at {entry_time}")
                if len(positions) > 3:
                    print(f"  ... and {len(positions)-3} more positions")
        except Exception as e:
            print(f"[ERROR] Error reading live positions: {e}")
            issues.append(f"Live positions file corrupted: {e}")
    else:
        print("[ERROR] Live positions file not found")
        issues.append("No live positions data available")

    # 4. Check Recent Trading Activity
    print("\n[TREND] 4. Recent Trading Activity:")
    print("-" * 40)

    # Check for recent logs
    logs_dir = OUTPUTS_DIR / 'logs'
    if logs_dir.exists():
        log_files = list(logs_dir.glob('*.log'))
        if log_files:
            # Get most recent log
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            log_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
            days_since_log = (datetime.now() - log_time).days

            print(f"[OK] Latest log: {latest_log.name} ({log_time.strftime('%Y-%m-%d %H:%M')})")

            if days_since_log > 1:
                issues.append(f"Logs are {days_since_log} days old - system may not be running")
        else:
            print("[ERROR] No log files found")
            issues.append("No recent log files - system may not be running")
    else:
        print("[ERROR] Logs directory not found")

    # Check for recent GTT orders
    gtt_monitor_file = GTT_DIR / 'gtt_monitor_state.json'
    if gtt_monitor_file.exists():
        try:
            with open(gtt_monitor_file, 'r') as f:
                monitor_data = json.load(f)

            last_check = monitor_data.get('last_check')
            if last_check:
                last_check_time = pd.to_datetime(last_check)
                hours_since_check = (datetime.now() - last_check_time).total_seconds() / 3600

                print(f"[OK] GTT monitor last checked: {last_check_time.strftime('%Y-%m-%d %H:%M')} ({hours_since_check:.1f} hours ago)")

                if hours_since_check > 24:
                    issues.append(f"GTT monitoring is {hours_since_check:.1f} hours stale")
            else:
                print(f"[WARN] GTT monitor state exists but no last_check timestamp")
        except Exception as e:
            print(f"[ERROR] Error reading GTT monitor state: {e}")
    else:
        print("[ERROR] GTT monitor state not found")

    # 5. Summary and Recommendations
    print("\n[DATA] 5. Diagnostic Summary:")
    print("-" * 40)

    if issues:
        print("[ERROR] Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("[OK] No major issues detected")

    if recommendations:
        print("\n[IDEA] Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    # Specific dashboard fix
    if latest_file and latest_file != 'gtt_plan_audited.csv':
        print(f"\n[FIX] IMMEDIATE FIX NEEDED:")
        print(f"  The dashboard is hardcoded to read 'gtt_plan_audited.csv'")
        print(f"  But the live system creates '{latest_file}'")
        print(f"  Update dashboard.py to check for live files first")

    return {
        'issues': issues,
        'recommendations': recommendations,
        'latest_file': latest_file,
        'latest_date': latest_date
    }

def main():
    """Main diagnostic function."""
    try:
        results = diagnose_data_sources()

        # Save diagnostic report
        report_file = OUTPUTS_DIR / 'dashboard_diagnostic_report.json'
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2, default=str)

        print(f"\n[SAVE] Diagnostic report saved to: {report_file}")

    except Exception as e:
        print(f"[ERROR] Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()