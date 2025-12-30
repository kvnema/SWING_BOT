#!/usr/bin/env python3
"""
SWING_BOT Self-Improvement Status Dashboard

Shows current optimization parameters, recent performance, and system health.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

def load_optimized_params():
    """Load current optimized parameters."""
    params_file = Path('outputs/self_optimize/optimized_params.json')
    if params_file.exists():
        with open(params_file, 'r') as f:
            return json.load(f)
    return None

def load_test_history():
    """Load recent test history."""
    history_file = Path('outputs/auto_test/test_history.json')
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
            # Return last 7 days
            recent = [h for h in history if datetime.fromisoformat(h['date']) >= datetime.now() - timedelta(days=7)]
            return recent[-7:]  # Last 7 entries
    return []

def format_params(params):
    """Format parameters for display."""
    if not params:
        return "No optimized parameters found"

    lines = ["📊 Current Optimized Parameters:"]
    for key, value in params.items():
        if key not in ['last_updated', 'performance_baseline']:
            lines.append(f"  {key}: {value}")
    lines.append(f"  Last Updated: {params.get('last_updated', 'Never')}")
    lines.append(f"  Performance Baseline: {params.get('performance_baseline', 'N/A')}")
    return "\n".join(lines)

def format_history(history):
    """Format test history for display."""
    if not history:
        return "No recent test history found"

    lines = ["📈 Recent Performance (Last 7 Days):"]
    for entry in history[-5:]:  # Show last 5
        date = entry['date']
        symbol = entry.get('symbol', 'N/A')
        best_strategy = entry.get('best_strategy', 'N/A')
        sharpe = entry['strategies'].get(best_strategy, {}).get('Sharpe', 'N/A')
        regime_rate = entry.get('regime_hit_rate', 'N/A')
        lines.append(f"  {date}: {symbol} | Strategy: {best_strategy} | Sharpe: {sharpe} | Regime: {regime_rate}%")

    return "\n".join(lines)

def check_system_health():
    """Check basic system health."""
    issues = []

    # Check required directories
    required_dirs = ['outputs/auto_test', 'outputs/self_optimize', 'logs', 'data']
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            issues.append(f"Missing directory: {dir_path}")

    # Check virtual environment
    if not Path('.venv').exists():
        issues.append("Virtual environment not found")

    # Check config
    if not Path('config.yaml').exists():
        issues.append("Configuration file missing")

    if issues:
        return "❌ System Health Issues:\n" + "\n".join(f"  - {issue}" for issue in issues)
    else:
        return "✅ System Health: All checks passed"

def main():
    """Display self-improvement status dashboard."""
    print("🚀 SWING_BOT Self-Improvement Status Dashboard")
    print("=" * 60)

    # Current parameters
    params = load_optimized_params()
    print(format_params(params))
    print()

    # Recent performance
    history = load_test_history()
    print(format_history(history))
    print()

    # System health
    print(check_system_health())
    print()

    # Next scheduled run
    print("⏰ Next Scheduled Run: Daily at 16:30 IST (11:00 UTC) on weekdays")
    print("📋 Tasks: SWING_BOT_Daily_Self_Improve, SWING_BOT_EOD_Full")
    print()

    print("📝 Commands:")
    print("  Manual run: python scripts\\daily_self_improve.py")
    print("  Check tasks: python scripts\\task_monitor.py")
    print("  View logs: type logs\\daily_self_improve_*.log")

if __name__ == '__main__':
    main()