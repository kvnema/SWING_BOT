#!/usr/bin/env python3
"""
SWING_BOT E2E Test Runner

Sample script to run comprehensive end-to-end tests for SWING_BOT.
This script demonstrates how to run the full E2E test suite with different configurations.

Usage:
    python run_e2e_tests.py [--output-dir OUTPUT_DIR] [--regime REGIME] [--verbose]

Author: SWING_BOT Development Team
Date: January 1, 2026
"""

import argparse
import sys
import time
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tests.e2e_test import run_e2e_test_suite


def run_comprehensive_e2e_tests(output_base_dir: str = "outputs/e2e_tests", verbose: bool = False):
    """
    Run comprehensive E2E tests across different scenarios.

    Args:
        output_base_dir: Base directory for test outputs
        verbose: Enable verbose output
    """

    print("🚀 SWING_BOT Comprehensive E2E Testing")
    print("=" * 50)

    test_scenarios = [
        {
            'name': 'Full_System_Test',
            'description': 'Complete E2E test with all components',
            'output_dir': f"{output_base_dir}/full_system"
        },
        {
            'name': 'Bullish_Regime_Test',
            'description': 'Test system performance in bullish market conditions',
            'output_dir': f"{output_base_dir}/bullish_regime"
        },
        {
            'name': 'Bearish_Regime_Test',
            'description': 'Test system performance in bearish market conditions',
            'output_dir': f"{output_base_dir}/bearish_regime"
        },
        {
            'name': 'Performance_Benchmark',
            'description': 'Performance benchmarking and timing tests',
            'output_dir': f"{output_base_dir}/performance"
        }
    ]

    overall_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scenarios': [],
        'summary': {}
    }

    total_start_time = time.time()

    for scenario in test_scenarios:
        print(f"\n🧪 Running: {scenario['name']}")
        print(f"   {scenario['description']}")
        print("-" * 40)

        scenario_start_time = time.time()

        try:
            # Run the test suite for this scenario
            results = run_e2e_test_suite(
                output_dir=scenario['output_dir'],
                verbose=verbose
            )

            scenario_duration = time.time() - scenario_start_time

            scenario_result = {
                'name': scenario['name'],
                'description': scenario['description'],
                'success': results['success'],
                'duration_seconds': round(scenario_duration, 2),
                'output_dir': results['output_dir'],
                'return_code': results['return_code']
            }

            overall_results['scenarios'].append(scenario_result)

            if results['success']:
                print(f"✅ {scenario['name']}: PASSED ({scenario_duration:.2f}s)")
            else:
                print(f"❌ {scenario['name']}: FAILED ({scenario_duration:.2f}s)")
                if results['stderr']:
                    print(f"   Error: {results['stderr'][:200]}...")

        except Exception as e:
            scenario_duration = time.time() - scenario_start_time
            print(f"❌ {scenario['name']}: ERROR - {str(e)} ({scenario_duration:.2f}s)")

            scenario_result = {
                'name': scenario['name'],
                'description': scenario['description'],
                'success': False,
                'duration_seconds': round(scenario_duration, 2),
                'error': str(e)
            }
            overall_results['scenarios'].append(scenario_result)

    # Calculate overall summary
    total_duration = time.time() - total_start_time
    successful_scenarios = sum(1 for s in overall_results['scenarios'] if s['success'])
    total_scenarios = len(overall_results['scenarios'])

    overall_results['summary'] = {
        'total_scenarios': total_scenarios,
        'successful_scenarios': successful_scenarios,
        'failed_scenarios': total_scenarios - successful_scenarios,
        'success_rate': round(successful_scenarios / total_scenarios * 100, 1) if total_scenarios > 0 else 0,
        'total_duration_seconds': round(total_duration, 2),
        'average_scenario_duration': round(total_duration / total_scenarios, 2) if total_scenarios > 0 else 0
    }

    # Save overall results
    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)

    summary_file = output_base_path / 'comprehensive_e2e_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(overall_results, f, indent=2)

    # Print final summary
    print("\n" + "=" * 50)
    print("📊 COMPREHENSIVE E2E TEST SUMMARY")
    print("=" * 50)
    print(f"Total Scenarios: {total_scenarios}")
    print(f"Successful: {successful_scenarios}")
    print(f"Failed: {total_scenarios - successful_scenarios}")
    print(f"Success Rate: {overall_results['summary']['success_rate']}%")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Average per Scenario: {overall_results['summary']['average_scenario_duration']:.2f} seconds")
    print(f"Results saved to: {summary_file}")

    if successful_scenarios == total_scenarios:
        print("\n🎉 ALL E2E TESTS PASSED! SWING_BOT is production-ready.")
        return True
    else:
        print(f"\n⚠️  {total_scenarios - successful_scenarios} E2E test(s) failed. Check detailed reports for issues.")
        return False


def run_quick_e2e_test(output_dir: str = "outputs/e2e_tests/quick", verbose: bool = False):
    """
    Run a quick E2E test focusing on critical components.

    Args:
        output_dir: Output directory for test results
        verbose: Enable verbose output
    """

    print("⚡ SWING_BOT Quick E2E Test")
    print("-" * 30)

    start_time = time.time()

    try:
        results = run_e2e_test_suite(
            output_dir=output_dir,
            verbose=verbose
        )

        duration = time.time() - start_time

        print(f"\n📊 Quick E2E Test Results:")
        print(f"   Status: {'✅ PASSED' if results['success'] else '❌ FAILED'}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Results: {results['output_dir']}")

        return results['success']

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Quick E2E test failed: {str(e)} ({duration:.2f}s)")
        return False


def main():
    """Main entry point for E2E test runner."""
    parser = argparse.ArgumentParser(
        description='SWING_BOT E2E Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive E2E tests
  python run_e2e_tests.py --comprehensive

  # Run quick E2E test
  python run_e2e_tests.py --quick

  # Run with custom output directory
  python run_e2e_tests.py --comprehensive --output-dir outputs/my_e2e_tests

  # Verbose output
  python run_e2e_tests.py --quick --verbose
        """
    )

    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help='Run comprehensive E2E tests across multiple scenarios'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick E2E test focusing on critical components'
    )

    parser.add_argument(
        '--output-dir',
        default='outputs/e2e_tests',
        help='Base output directory for test results (default: outputs/e2e_tests)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose test output'
    )

    args = parser.parse_args()

    # Default to comprehensive if no mode specified
    if not args.comprehensive and not args.quick:
        args.comprehensive = True

    success = False

    if args.comprehensive:
        success = run_comprehensive_e2e_tests(args.output_dir, args.verbose)
    elif args.quick:
        success = run_quick_e2e_test(args.output_dir, args.verbose)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()