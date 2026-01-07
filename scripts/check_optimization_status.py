#!/usr/bin/env python3
"""
SWING_BOT Self-Optimization Status Checker
Shows current optimization cycle and parameter status
"""

import json
import os
from pathlib import Path

def check_optimization_status():
    """Check the current self-optimization status"""

    base_dir = Path(__file__).parent.parent
    self_enhancement_dir = base_dir / "outputs" / "self_enhancement"

    print("🚀 SWING_BOT Self-Optimization Status")
    print("=" * 50)

    # Check if optimization has run
    if not self_enhancement_dir.exists():
        print("❌ No self-optimization cycles completed yet")
        print("   Waiting for first realized trade outcomes...")
        return

    # Check latest cycle
    cycle_files = list(self_enhancement_dir.glob("cycle_*_results.json"))
    if not cycle_files:
        print("❌ No optimization results found")
        return

    # Get latest cycle
    latest_cycle = max(cycle_files, key=lambda x: int(x.stem.split('_')[1]))

    with open(latest_cycle, 'r') as f:
        results = json.load(f)

    cycle_num = results['cycle']
    timestamp = results['timestamp']

    print(f"✅ Latest Cycle: #{cycle_num}")
    print(f"📅 Completed: {timestamp}")
    print(f"🎯 Trigger: {results['trigger']}")
    print()

    print("📊 Performance Improvement:")
    perf = results['performance_comparison']
    print(f"   Win Rate: {perf['baseline']['win_rate_pct']:.1f}% → {perf['optimized']['win_rate_pct']:.1f}% ({perf['improvement']['win_rate_pct']:+.1f}%)")
    print(f"   Expectancy: {perf['baseline']['expectancy_r']:.2f}R → {perf['optimized']['expectancy_r']:.2f}R ({perf['improvement']['expectancy_r']:+.0f}%)")
    print(f"   Sharpe: {perf['baseline']['sharpe_ratio']:.2f} → {perf['optimized']['sharpe_ratio']:.2f} ({perf['improvement']['sharpe_ratio']:+.0f}%)")
    print(f"   Max DD: {perf['baseline']['max_drawdown']:.1f}% → {perf['optimized']['max_drawdown']:.1f}% ({perf['improvement']['max_drawdown']:+.0f}% reduction)")
    print(f"   Trades/Year: {perf['baseline']['total_trades_per_year']} → {perf['optimized']['total_trades_per_year']} ({perf['improvement']['total_trades_per_year']:+.0f}%)")
    print()

    print("🔧 Optimized Parameters:")
    base_params = results['baseline_parameters']
    opt_params = results['optimized_parameters']

    for param in base_params:
        if param in opt_params:
            old_val = base_params[param]
            new_val = opt_params[param]
            if old_val != new_val:
                print(f"   {param}: {old_val} → {new_val}")
    print()

    print("💡 Key Insights:")
    for insight in results['insights']:
        print(f"   • {insight}")
    print()

    print("🎯 Next Actions:")
    for action in results['next_actions']:
        print(f"   • {action}")
    print()

    print("🧠 SWING_BOT is evolving... Intelligence compounding activated! 🚀")

if __name__ == "__main__":
    check_optimization_status()