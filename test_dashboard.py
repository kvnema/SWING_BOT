#!/usr/bin/env python3
"""
SWING_BOT Dashboard Test Script

This script tests the dashboard functionality by checking data loading
and basic operations without starting the full Streamlit server.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_loading():
    """Test loading data from outputs directory."""
    print("🧪 Testing data loading functionality...")

    outputs_dir = Path("outputs")
    success_count = 0
    total_tests = 3

    # Test loading live positions
    positions_file = outputs_dir / "live_positions.json"
    try:
        if positions_file.exists():
            with open(positions_file, 'r') as f:
                positions = json.load(f)
            print(f"✅ Live positions loaded: {len(positions)} positions")
            success_count += 1
        else:
            print("⚠️  Live positions file not found")
    except Exception as e:
        print(f"❌ Error loading live positions: {e}")

    # Test loading backtest results
    backtest_file = outputs_dir / "backtest_results_today.json" / "selected_strategy.json"
    try:
        if backtest_file.exists():
            with open(backtest_file, 'r') as f:
                backtest_data = json.load(f)
            print(f"✅ Backtest results loaded: {backtest_data.get('selected', 'Unknown')} strategy selected")
            success_count += 1
        else:
            print("⚠️  Backtest results file not found")
    except Exception as e:
        print(f"❌ Error loading backtest results: {e}")

    # Test loading screener results
    screener_file = outputs_dir / "screener_results.csv"
    try:
        if screener_file.exists():
            df = pd.read_csv(screener_file)
            print(f"✅ Screener results loaded: {len(df)} stocks screened")
            success_count += 1
        else:
            print("⚠️  Screener results file not found")
    except Exception as e:
        print(f"❌ Error loading screener results: {e}")

    print()
    return success_count == total_tests

def test_dashboard_imports():
    """Test importing dashboard components."""
    print("🧪 Testing dashboard imports...")

    try:
        import streamlit as st
        print(f"✅ Streamlit imported: v{st.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

    try:
        import plotly
        print(f"✅ Plotly imported: v{plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False

    try:
        import psutil
        print(f"✅ Psutil imported: v{psutil.__version__}")
    except ImportError as e:
        print(f"❌ Psutil import failed: {e}")
        return False

    print()
    return True

def test_dashboard_functions():
    """Test dashboard utility functions."""
    print("🧪 Testing dashboard utility functions...")

    # Import dashboard functions
    try:
        from dashboard import load_data, get_system_status
        print("✅ Dashboard functions imported successfully")
    except ImportError as e:
        print(f"❌ Dashboard import failed: {e}")
        return False

    # Test load_data function
    try:
        # Test with existing file
        result = load_data(Path("outputs/live_positions.json"))
        print(f"✅ load_data function works: returned {type(result)}")

        # Test with non-existent file
        result = load_data(Path("outputs/non_existent.json"), default="test")
        print(f"✅ load_data with default works: returned {result}")

    except Exception as e:
        print(f"❌ load_data function failed: {e}")
        return False

    # Test get_system_status function
    try:
        status = get_system_status()
        print(f"✅ get_system_status works: {len(status)} status items")
        for key, value in status.items():
            print(f"   - {key}: {value}")
    except Exception as e:
        print(f"❌ get_system_status failed: {e}")
        return False

    print()
    return True

def main():
    """Run all tests."""
    print("🚀 SWING_BOT Dashboard Test Suite")
    print("=" * 50)

    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Run tests
    imports_ok = test_dashboard_imports()
    if not imports_ok:
        print("❌ Critical: Missing dependencies. Run: pip install -r requirements.txt")
        sys.exit(1)

    data_ok = test_data_loading()
    functions_ok = test_dashboard_functions()

    print("=" * 50)
    if data_ok and functions_ok:
        print("✅ All tests passed! Dashboard is ready to run.")
        print("\nTo start the dashboard:")
        print("  python run_dashboard.py")
        print("  # or")
        print("  streamlit run dashboard.py")
        print("\nDashboard will be available at: http://localhost:8501")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("The dashboard may still work but with limited functionality.")

if __name__ == "__main__":
    main()