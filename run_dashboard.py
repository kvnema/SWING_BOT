#!/usr/bin/env python3
"""
SWING_BOT Dashboard Launcher

This script launches the SWING_BOT web dashboard using Streamlit.

Usage:
    python run_dashboard.py

The dashboard will be available at http://localhost:8501
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    print("🚀 Starting SWING_BOT Dashboard...")

    # Get the directory of this script
    script_dir = Path(__file__).parent

    # Path to the dashboard file
    dashboard_file = script_dir / "dashboard.py"

    if not dashboard_file.exists():
        print(f"❌ Error: Dashboard file not found at {dashboard_file}")
        sys.exit(1)

    # Launch Streamlit
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_file),
            "--server.headless", "true",
            "--server.port", "8501",
            "--theme.base", "light"
        ]

        print("📊 Dashboard will be available at: http://localhost:8501")
        print("🔄 Starting Streamlit server...")

        # Run the command
        subprocess.run(cmd, cwd=script_dir)

    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()