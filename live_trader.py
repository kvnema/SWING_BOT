#!/usr/bin/env python3
"""
SWING_BOT Live Trading System
Automated execution with sector analysis and risk management
"""

import argparse
import logging
import time
from datetime import datetime

from src.data_fetch import calculate_market_regime
from src.signals import compute_signals
from src.data_fetch import fetch_market_index_data
from src.live_trading import live_trader, check_live_exits
from src.sector_analysis import SectorAnalyzer, get_sector_analysis, print_sector_analysis
from src.notifications import send_daily_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """Complete live trading system with all safety features."""

    def __init__(self, symbols: list = None, capital: float = 100000,
                 max_positions: int = 3, risk_per_trade_pct: float = 0.01,
                 max_sector_pct: float = 0.25, require_confirmation: bool = True):
        self.symbols = symbols or [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS'
        ]

        self.capital = capital
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_sector_pct = max_sector_pct
        self.require_confirmation = require_confirmation

        self.sector_analyzer = SectorAnalyzer()
        self.last_regime_check = None
        self.current_regime = None

        logger.info("Live trading system initialized")

    def check_market_regime(self) -> str:
        """Check current market regime."""
        try:
            regime_data = calculate_market_regime('NSE_INDEX|Nifty 50')
            regime_status = regime_data.get('regime_status', 'UNKNOWN')

            # Check if regime changed
            if self.current_regime != regime_status:
                logger.info(f"Regime change detected: {self.current_regime} -> {regime_status}")
                self.current_regime = regime_status

            self.last_regime_check = datetime.now()
            return regime_status

        except Exception as e:
            logger.error(f"Failed to check market regime: {e}")
            return 'UNKNOWN'

    def scan_signals(self) -> list:
        """Scan for trading signals across all symbols."""
        signals = []

        for symbol in self.symbols:
            try:
                # Get recent data
                df = fetch_market_index_data(symbol, 100)
                if df.empty or len(df) < 50:
                    continue

                # Compute signals
                df_signals = compute_signals(df)

                # Check for active signals (simplified - check last row)
                latest = df_signals.iloc[-1]

                # Check each strategy flag
                strategy_flags = {
                    'SEPA': 'SEPA_Flag',
                    'VCP': 'VCP_Flag',
                    'Donchian': 'Donchian_Flag',
                    'MR': 'MR_Flag',
                    'AVWAP': 'AVWAP_Flag'
                }

                for strategy, flag_col in strategy_flags.items():
                    if flag_col in df_signals.columns and latest.get(flag_col, 0) == 1:
                        # Calculate entry/exit levels
                        entry_price = latest['Close']
                        atr = latest.get('ATR14', entry_price * 0.02)
                        stop_loss = entry_price - (atr * 1.5)
                        target = entry_price + (atr * 3)

                        signal = {
                            'symbol': symbol,
                            'strategy': strategy,
                            'action': 'BUY',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'confidence': latest.get('Composite_Score', 5.0),
                            'timestamp': datetime.now().isoformat(),
                            'atr': atr
                        }

                        signals.append(signal)
                        break  # Take first signal per symbol

            except Exception as e:
                logger.warning(f"Failed to scan {symbol}: {e}")
                continue

        return signals

    def filter_signals_by_sector(self, signals: list) -> list:
        """Filter signals based on sector diversification."""
        # Get current positions (mock for now - in real implementation, get from live_trader)
        current_positions = getattr(live_trader, 'positions', {})

        # Filter by sector limits
        filtered_signals = self.sector_analyzer.filter_by_sector_limits(
            signals, current_positions, self.max_sector_pct
        )

        return filtered_signals

    def execute_signals(self, signals: list):
        """Execute filtered signals."""
        for signal in signals:
            try:
                # Calculate position size
                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']

                # Use live trader's position sizing
                quantity = live_trader.executor.calculate_order_quantity(
                    self.capital, entry_price, self.risk_per_trade_pct
                )

                if quantity == 0:
                    logger.info(f"Position size too small for {signal['symbol']}")
                    continue

                signal['quantity'] = quantity

                # Execute the signal
                success = live_trader.execute_signal(signal)

                if success:
                    logger.info(f"Signal executed: {signal['symbol']} ({signal['strategy']})")
                else:
                    logger.info(f"Signal rejected: {signal['symbol']} ({signal['strategy']})")

            except Exception as e:
                logger.error(f"Failed to execute signal for {signal['symbol']}: {e}")

    def run_trading_cycle(self):
        """Run one complete trading cycle."""
        logger.info("Starting trading cycle...")

        # 1. Check market regime
        regime = self.check_market_regime()
        logger.info(f"Current market regime: {regime}")

        if regime != 'ON':
            logger.info("Market regime OFF - scanning but not executing trades")
            # Still scan for signals but don't execute
            signals = self.scan_signals()
            if signals:
                logger.info(f"Found {len(signals)} signals (not executed due to regime)")
                for signal in signals:
                    logger.info(f"  Signal: {signal['symbol']} ({signal['strategy']})")
            return

        # 2. Scan for signals
        signals = self.scan_signals()
        logger.info(f"Found {len(signals)} raw signals")

        if not signals:
            logger.info("No signals found")
            return

        # 3. Filter by sector diversification
        filtered_signals = self.filter_signals_by_sector(signals)
        logger.info(f"After sector filtering: {len(filtered_signals)} signals")

        # 4. Execute signals
        if filtered_signals:
            self.execute_signals(filtered_signals)

        # 5. Check for exits
        check_live_exits()

        logger.info("Trading cycle completed")

    def get_status_report(self) -> dict:
        """Get comprehensive status report."""
        # Get live trading status
        live_status = live_trader.get_status()

        # Get sector analysis
        sector_analysis = get_sector_analysis(self.symbols)

        # Get regime status
        regime = self.check_market_regime()

        return {
            'regime_status': regime,
            'live_trading': live_status,
            'sector_analysis': sector_analysis,
            'last_update': datetime.now().isoformat(),
            'system_status': 'ACTIVE' if live_trader.executor.enabled else 'SIMULATION'
        }

    def print_status_report(self):
        """Print formatted status report."""
        status = self.get_status_report()

        print("\n" + "="*70)
        print("SWING_BOT LIVE TRADING STATUS REPORT")
        print("="*70)

        print(f"📊 Market Regime: {status['regime_status']}")
        print(f"🤖 System Status: {status['system_status']}")
        print(f"⏰ Last Update: {status['last_update'][:19]}")

        live = status['live_trading']
        print("\n💰 Account Status:")
        print(f"  Capital: ₹{live['capital']:,.0f}")
        print(f"  Positions: {live['positions_count']}")
        print(f"  Position Value: ₹{live['positions_value']:,.0f}")

        if live['positions']:
            print("  Open Positions:")
            for symbol in live['positions']:
                print(f"    • {symbol}")

        # Sector analysis summary
        sector = status['sector_analysis']
        print("\n📈 Sector Analysis:")
        strength = sector['sector_strength']
        if strength:
            top_sector = list(strength.keys())[0]
            top_strength = strength[top_sector]
            print(f"  Top Sector: {top_sector} ({top_strength:+.1f}%)")
        else:
            print("  No sector data available")

        rotation = sector['rotation_signals']
        if rotation['leading_sectors']:
            print(f"  Leading: {', '.join(rotation['leading_sectors'][:2])}")

        print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description='SWING_BOT Live Trading System')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    parser.add_argument('--capital', type=float, default=100000, help='Trading capital')
    parser.add_argument('--max-positions', type=int, default=3, help='Max concurrent positions')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade as decimal (0.01 = 1%%)')
    parser.add_argument('--max-sector-pct', type=float, default=0.25, help='Max sector exposure as decimal (0.25 = 25%%)')
    parser.add_argument('--no-confirmation', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--mode', choices=['once', 'continuous', 'status', 'sector-analysis'],
                       default='once', help='Operating mode')
    parser.add_argument('--interval', type=int, default=3600, help='Check interval in seconds')

    args = parser.parse_args()

    # Initialize system
    system = LiveTradingSystem(
        symbols=args.symbols,
        capital=args.capital,
        max_positions=args.max_positions,
        risk_per_trade_pct=args.risk_per_trade,
        max_sector_pct=args.max_sector_pct,
        require_confirmation=not args.no_confirmation
    )

    if args.mode == 'once':
        # Single trading cycle
        system.run_trading_cycle()

    elif args.mode == 'status':
        # Status report only
        system.print_status_report()

    elif args.mode == 'sector-analysis':
        # Sector analysis only
        analysis = get_sector_analysis(system.symbols)
        print_sector_analysis(analysis)

    elif args.mode == 'continuous':
        # Continuous trading
        logger.info(f"Starting continuous trading (interval: {args.interval}s)")

        while True:
            try:
                system.run_trading_cycle()

                # Send daily summary at end of day
                now = datetime.now()
                if now.hour == 15 and 30 <= now.minute <= 35:  # End of market hours
                    status = system.get_status_report()
                    send_daily_summary({
                        'regime_status': status['regime_status'],
                        'market_close': 'N/A',  # Would need to fetch
                        'signals_count': 0,  # Would need to track
                        'date': now.strftime('%Y-%m-%d')
                    })

                time.sleep(args.interval)

            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Trading cycle failed: {e}")
                time.sleep(60)  # Wait before retry

if __name__ == "__main__":
    main()