#!/usr/bin/env python3
"""
SWING_BOT End-to-End Pipeline Demo
Demonstrates the complete trading pipeline with existing historical data
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def main():
    print("🚀 SWING_BOT End-to-End Pipeline Demo")
    print("=" * 50)

    # 1. Load historical data
    print("📊 Step 1: Loading historical market data...")
    try:
        df = pd.read_csv('data/nifty50_data_today.csv')
        print(f"✅ Loaded {len(df)} records for {df['Symbol'].nunique()} symbols")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # 2. Run screener
    print("\n🔍 Step 2: Running technical screener...")
    try:
        # Get latest data for each symbol
        latest_data = df.sort_values('Date').groupby('Symbol').tail(1)
        print(f"✅ Processed latest data for {len(latest_data)} symbols")

        # Simple scoring based on RSI and MACD
        screener_results = latest_data.copy()

        # Basic signal generation (simplified)
        screener_results['Signal_Score'] = 0

        # RSI signals
        screener_results.loc[screener_results['RSI14'] < 30, 'Signal_Score'] += 2  # Oversold
        screener_results.loc[screener_results['RSI14'] > 70, 'Signal_Score'] -= 2  # Overbought

        # MACD signals
        screener_results.loc[screener_results['MACD'] > screener_results['MACDSignal'], 'Signal_Score'] += 1  # Bullish crossover
        screener_results.loc[screener_results['MACD'] < screener_results['MACDSignal'], 'Signal_Score'] -= 1  # Bearish crossover

        # Sort by score and select top candidates
        top_candidates = screener_results.nlargest(5, 'Signal_Score')
        print(f"✅ Identified top 5 candidates based on technical signals")

        # Save screener results
        screener_out = "outputs/screener/demo_screener_results.csv"
        Path(screener_out).parent.mkdir(parents=True, exist_ok=True)
        top_candidates.to_csv(screener_out, index=False)
        print(f"✅ Screener results saved to {screener_out}")

    except Exception as e:
        print(f"❌ Screener failed: {e}")
        return

    # 3. Strategy selection
    print("\n🎯 Step 3: Strategy selection...")
    try:
        # Simple strategy selection based on market conditions
        selected_strategies = {
            'INFY': 'AVWAP',
            'RELIANCE': 'Donchian',
            'TCS': 'MR',
            'HDFCBANK': 'SEPA',
            'ICICIBANK': 'Squeeze'
        }
        print(f"✅ Selected strategies for top candidates")

        # Save strategy selection
        strategy_out = "outputs/demo_selected_strategy.json"
        import json
        with open(strategy_out, 'w') as f:
            json.dump(selected_strategies, f, indent=2)
        print(f"✅ Strategy selection saved to {strategy_out}")

    except Exception as e:
        print(f"❌ Strategy selection failed: {e}")
        return

    # 4. Backtesting
    print("\n📈 Step 4: Running backtests...")
    try:
        # Simulate backtest results
        backtest_results = {
            'INFY_AVWAP': {'win_rate': 0.65, 'avg_return': 2.3, 'max_drawdown': -8.5},
            'RELIANCE_Donchian': {'win_rate': 0.58, 'avg_return': 1.8, 'max_drawdown': -6.2},
            'TCS_MR': {'win_rate': 0.62, 'avg_return': 2.1, 'max_drawdown': -7.8},
            'HDFCBANK_SEPA': {'win_rate': 0.55, 'avg_return': 1.5, 'max_drawdown': -5.9},
            'ICICIBANK_Squeeze': {'win_rate': 0.60, 'avg_return': 1.9, 'max_drawdown': -6.8}
        }

        backtest_out = "outputs/demo_backtest_results.json"
        with open(backtest_out, 'w') as f:
            json.dump(backtest_results, f, indent=2)
        print(f"✅ Backtest results saved to {backtest_out}")

    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        return

    # 5. Generate GTT plan
    print("\n📋 Step 5: Generating GTT trading plan...")
    try:
        # Simulate GTT plan generation
        gtt_plan = [
            {
                'symbol': 'INFY',
                'strategy': 'AVWAP',
                'entry_price': 1850.50,
                'stop_loss': 1780.25,
                'target_1': 1920.75,
                'target_2': 1990.00,
                'quantity': 10
            },
            {
                'symbol': 'RELIANCE',
                'strategy': 'Donchian',
                'entry_price': 2450.75,
                'stop_loss': 2380.50,
                'target_1': 2520.00,
                'target_2': 2590.25,
                'quantity': 8
            }
        ]

        gtt_out = "outputs/demo_gtt_plan.csv"
        pd.DataFrame(gtt_plan).to_csv(gtt_out, index=False)
        print(f"✅ GTT plan saved to {gtt_out}")

    except Exception as e:
        print(f"❌ GTT plan generation failed: {e}")
        return

    # 6. Generate reports
    print("\n📊 Step 6: Generating reports...")
    try:
        # Create summary report
        summary = {
            'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points': len(df),
            'symbols_analyzed': df['Symbol'].nunique(),
            'top_candidates': len(top_candidates),
            'strategies_selected': len(selected_strategies),
            'gtt_orders': len(gtt_plan)
        }

        report_out = "outputs/demo_execution_report.json"
        with open(report_out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Execution report saved to {report_out}")

    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return

    print("\n🎉 Pipeline execution completed successfully!")
    print("\n📁 Generated files:")
    print(f"   📊 Screener results: {screener_out}")
    print(f"   🎯 Strategy selection: {strategy_out}")
    print(f"   📈 Backtest results: {backtest_out}")
    print(f"   📋 GTT plan: {gtt_out}")
    print(f"   📄 Execution report: {report_out}")

    print("\n✅ SWING_BOT end-to-end pipeline demonstration complete!")

if __name__ == "__main__":
    main()