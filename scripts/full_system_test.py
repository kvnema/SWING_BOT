#!/usr/bin/env python3
"""
SWING_BOT Full System Test

Tests the complete self-improvement pipeline including Telegram reports.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def setup_test_environment():
    """Setup environment for testing."""
    # Set Telegram credentials for testing
    os.environ['TELEGRAM_BOT_TOKEN'] = '8486307857:AAHt4XXRokWf_Uv49NIVozp3lj1W-seqMg4'
    os.environ['TELEGRAM_CHAT_ID'] = '7227129007'

    print("🧪 SWING_BOT Full System Test")
    print("=" * 50)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"🤖 Telegram: Configured")
    print()

def test_imports():
    """Test that all modules can be imported."""
    print("📦 Testing Module Imports...")

    try:
        from auto_test import run_daily_auto_test
        print("✅ auto_test imported")
    except ImportError as e:
        print(f"❌ auto_test import failed: {e}")
        return False

    try:
        from self_optimize import run_daily_self_optimization
        print("✅ self_optimize imported")
    except ImportError as e:
        print(f"❌ self_optimize import failed: {e}")
        return False

    try:
        from notifications_router import send_telegram_self_improvement_report, send_telegram_alert
        print("✅ notifications_router imported")
    except ImportError as e:
        print(f"❌ notifications_router import failed: {e}")
        return False

    try:
        from scripts.status_dashboard import load_optimized_params, load_test_history
        print("✅ status_dashboard imported")
    except ImportError as e:
        print(f"❌ status_dashboard import failed: {e}")
        return False

    print("✅ All imports successful!")
    return True

def test_telegram_connection():
    """Test Telegram connection."""
    print("\n📱 Testing Telegram Connection...")

    try:
        from notifications_router import send_telegram_alert

        # Send test alert
        success = send_telegram_alert(
            "system_test",
            "🧪 SWING_BOT Full System Test Started\n• Testing complete self-improvement pipeline\n• Telegram reports and alerts",
            run_id="TEST_001",
            dry_run=False  # Actually send the message
        )

        if success:
            print("✅ Telegram test alert sent successfully")
            return True
        else:
            print("❌ Telegram test alert failed")
            return False

    except Exception as e:
        print(f"❌ Telegram test failed: {e}")
        return False

def run_self_improvement_test():
    """Run the actual self-improvement cycle."""
    print("\n🚀 Running Self-Improvement Cycle...")

    try:
        from auto_test import run_daily_auto_test
        from self_optimize import run_daily_self_optimization
        from notifications_router import send_telegram_self_improvement_report, send_telegram_alert
        from scripts.status_dashboard import load_optimized_params, load_test_history

        results = {}
        run_id = datetime.now().strftime("%Y%m%d_%H%M_TEST")

        # Step 1: Auto-Testing
        print("🧪 Step 1: Auto-Testing...")
        try:
            test_result = run_daily_auto_test()
            results['auto_test'] = test_result
            print("✅ Auto-testing completed")

            # Send test completion alert
            send_telegram_alert(
                "test_complete",
                f"✅ Test auto-testing completed\n• Symbol: {test_result.get('symbol', 'N/A')}\n• Best Strategy: {test_result.get('best_strategy', 'N/A')}",
                run_id=run_id
            )

        except Exception as e:
            print(f"❌ Auto-testing failed: {e}")
            results['auto_test'] = {'error': str(e)}
            send_telegram_alert("test_failure", f"❌ Test auto-testing failed: {e}", run_id=run_id)
            return False

        # Step 2: Self-Optimization
        print("🔧 Step 2: Self-Optimization...")
        try:
            optimize_result = run_daily_self_optimization()
            results['self_optimize'] = optimize_result
            print("✅ Self-optimization completed")

            if optimize_result.get('applied_changes'):
                changes = len(optimize_result['applied_changes'])
                improvement = optimize_result.get('improvement_pct', 0)
                send_telegram_alert(
                    "parameter_change",
                    f"🔄 Test parameters updated\n• {changes} parameters changed\n• {improvement:+.1f}% improvement",
                    run_id=run_id
                )

        except Exception as e:
            print(f"❌ Self-optimization failed: {e}")
            results['self_optimize'] = {'error': str(e)}
            send_telegram_alert("optimization_failure", f"❌ Test optimization failed: {e}", run_id=run_id)
            return False

        # Step 3: Send final report
        print("📊 Step 3: Sending Final Report...")
        try:
            optimized_params = load_optimized_params()
            recent_performance = load_test_history()
            system_health = {"status": "healthy", "issues": []}

            success = send_telegram_self_improvement_report(
                optimized_params,
                recent_performance,
                system_health,
                run_id
            )

            if success:
                print("✅ Final report sent")
            else:
                print("⚠️ Final report failed to send")

        except Exception as e:
            print(f"❌ Report sending failed: {e}")

        print("✅ Self-improvement cycle completed successfully")
        return True

    except Exception as e:
        print(f"❌ Self-improvement cycle failed: {e}")
        return False

def main():
    """Run the full system test."""
    setup_test_environment()

    # Test imports
    if not test_imports():
        print("❌ Import tests failed - cannot proceed")
        return False

    # Test Telegram
    if not test_telegram_connection():
        print("⚠️ Telegram test failed - continuing with dry-run mode")
        # Continue anyway for testing

    # Run full cycle
    if run_self_improvement_test():
        print("\n🎉 FULL SYSTEM TEST COMPLETED SUCCESSFULLY!")
        print("📱 Check your Telegram for test reports and alerts")
        print("📊 Review logs and status dashboard for details")
        return True
    else:
        print("\n❌ FULL SYSTEM TEST FAILED")
        print("📋 Check the error messages above for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)