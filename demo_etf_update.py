#!/usr/bin/env python3
"""
Offline Demo Harness for ETF Universe Auto-Updater

Uses synthetic local gz file to test merging 4 popular ETFs into the universe.
No network required - perfect for development and testing.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the ETF universe updater with synthetic data."""

    # Paths
    bod_path = "artifacts/bod/complete.json.gz"
    universe_path = "artifacts/universe/instrument_keys.json"

    # Ensure files exist
    if not Path(bod_path).exists():
        print(f"Synthetic BOD file not found: {bod_path}")
        print("Run 'python create_synthetic_bod.py' first")
        return 1

    if not Path(universe_path).exists():
        print(f"Universe file not found: {universe_path}")
        return 1

    print("ETF Universe Auto-Updater - Offline Demo")
    print("=" * 50)
    print(f"BOD Path: {bod_path}")
    print(f"Universe: {universe_path}")
    print()

    # Run the updater in dry-run mode first
    print("Step 1: Dry run preview...")
    cmd_dry = [
        sys.executable, "etf_universe.py",
        "--bod-path", bod_path,
        "--exchange", "NSE",
        "--allowlist", "NIFTYBEES,SETFNIF50,ICICINIFTY,KOTAKNIFTY",
        "--dry-run",
        "--log-level", "INFO"
    ]

    result_dry = subprocess.run(cmd_dry, capture_output=True, text=True)
    if result_dry.returncode != 0:
        print("Dry run failed:")
        print(result_dry.stderr)
        return 1

    print(result_dry.stdout)

    # Now run the actual update
    print("Step 2: Applying ETF updates...")
    cmd_update = [
        sys.executable, "etf_universe.py",
        "--bod-path", bod_path,
        "--exchange", "NSE",
        "--allowlist", "NIFTYBEES,SETFNIF50,ICICINIFTY,KOTAKNIFTY",
        "--existing-universe", universe_path,
        "--output", universe_path,
        "--log-level", "INFO"
    ]

    result_update = subprocess.run(cmd_update, capture_output=True, text=True)
    if result_update.returncode != 0:
        print("Update failed:")
        print(result_update.stderr)
        return 1

    print(result_update.stdout)

    # Verify the results
    print("Step 3: Verification...")
    try:
        import json
        with open(universe_path, 'r') as f:
            universe = json.load(f)

        print(f"Universe now contains {len(universe)} instruments:")

        # Show ETFs
        etfs = {k: v for k, v in universe.items() if v.get('namespace') == 'NSE_ETF'}
        print(f"  ETFs: {len(etfs)}")

        for symbol, data in sorted(etfs.items()):
            instrument_key = data.get('instrument_key', 'N/A')
            print(f"    - {symbol} -> {instrument_key}")

        # Show equities
        equities = {k: v for k, v in universe.items() if v.get('namespace') == 'NSE_EQ'}
        print(f"  Equities: {len(equities)}")

        for symbol in sorted(equities.keys()):
            print(f"    - {symbol}")

        print("\nDemo completed successfully!")
        print("The universe now includes 4 popular ETFs alongside existing equities")

    except Exception as e:
        print(f"Verification failed: {e}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())