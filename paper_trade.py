#!/usr/bin/env python3
"""
SWING_BOT Paper Trading Simulator
Simulates trading signals and tracks performance without real capital
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json

from src.data_fetch import fetch_market_index_data, calculate_market_regime
from src.signals import compute_signals
from src.notifications import send_signal_alert

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperTrader:
    """Paper trading simulator for SWING_BOT signals."""

    def __init__(self, capital: float = 100000, max_positions: int = 5,
                 risk_per_trade: float = 0.01, state_file: str = 'data/paper_trading_state.json'):
        self.initial_capital = capital
        self.current_capital = capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade  # 1% per trade
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(exist_ok=True)

        # Load or initialize state
        self.positions = self.load_positions()
        self.trade_history = self.load_trade_history()

        logger.info(f"Paper trader initialized with ₹{capital:,.0f} capital")

    def load_positions(self) -> Dict[str, Dict]:
        """Load current positions from state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('positions', {})
            except Exception as e:
                logger.warning(f"Failed to load positions: {e}")
        return {}

    def load_trade_history(self) -> List[Dict]:
        """Load trade history from state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('trade_history', [])
            except Exception as e:
                logger.warning(f"Failed to load trade history: {e}")
        return []

    def save_state(self):
        """Save current state to file."""
        state = {
            'capital': self.current_capital,
            'positions': self.positions,
            'trade_history': self.trade_history,
            'last_updated': datetime.now().isoformat()
        }

        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management."""
        risk_amount = self.current_capital * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            return 0

        position_size = int(risk_amount / risk_per_share)
        return min(position_size, 1000)  # Max 1000 shares for liquidity

    def scan_signals(self, symbols: List[str]) -> List[Dict]:
        """Scan for trading signals across symbols."""
        signals = []

        for symbol in symbols:
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
                        # Calculate entry/exit levels (simplified)
                        entry_price = latest['Close']
                        atr = latest.get('ATR14', entry_price * 0.02)  # Fallback ATR
                        stop_loss = entry_price - (atr * 1.5)  # 1.5 ATR stop
                        target = entry_price + (atr * 3)  # 3:1 reward

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

    def execute_signal(self, signal: Dict) -> bool:
        """Execute a paper trade signal."""
        symbol = signal['symbol']

        # Check if already in position
        if symbol in self.positions:
            logger.info(f"Already in position for {symbol}, skipping")
            return False

        # Check position limit
        if len(self.positions) >= self.max_positions:
            logger.info("Max positions reached, skipping signal")
            return False

        # Calculate position size
        quantity = self.calculate_position_size(signal['entry_price'], signal['stop_loss'])

        if quantity == 0:
            logger.warning(f"Invalid position size for {symbol}")
            return False

        # Calculate costs
        position_value = quantity * signal['entry_price']
        transaction_cost = position_value * 0.001  # 0.1% broker fee

        if position_value + transaction_cost > self.current_capital:
            logger.warning(f"Insufficient capital for {symbol}")
            return False

        # Execute trade
        position = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'target': signal['target'],
            'entry_time': signal['timestamp'],
            'entry_cost': transaction_cost,
            'strategy': signal['strategy']
        }

        self.positions[symbol] = position
        self.current_capital -= (position_value + transaction_cost)

        # Record trade
        trade_record = {
            'type': 'ENTRY',
            'symbol': symbol,
            'quantity': quantity,
            'price': signal['entry_price'],
            'value': position_value,
            'cost': transaction_cost,
            'timestamp': signal['timestamp'],
            'strategy': signal['strategy']
        }

        self.trade_history.append(trade_record)

        logger.info(f"Paper trade executed: {symbol} {quantity} shares @ ₹{signal['entry_price']:.2f}")
        self.save_state()

        # Send alert
        alert_data = {
            'symbol': symbol,
            'strategy': signal['strategy'],
            'action': 'BUY',
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'target': signal['target'],
            'confidence': signal['confidence'],
            'quantity': quantity,
            'risk_reward': (signal['target'] - signal['entry_price']) / (signal['entry_price'] - signal['stop_loss'])
        }
        send_signal_alert(alert_data)

        return True

    def check_exits(self):
        """Check for exit conditions on open positions."""
        symbols_to_close = []

        for symbol, position in self.positions.items():
            try:
                # Get latest price
                df = fetch_market_index_data(symbol, 5)  # Last 5 days
                if df.empty:
                    continue

                latest_price = df.iloc[-1]['Close']
                entry_price = position['entry_price']
                stop_loss = position['stop_loss']
                target = position['target']

                exit_reason = None
                exit_price = latest_price

                # Check exit conditions
                if latest_price <= stop_loss:
                    exit_reason = 'STOP_LOSS'
                    exit_price = stop_loss
                elif latest_price >= target:
                    exit_reason = 'TARGET'
                    exit_price = target
                elif (datetime.now() - datetime.fromisoformat(position['entry_time'])).days > 30:
                    exit_reason = 'TIME_EXIT'
                    exit_price = latest_price

                if exit_reason:
                    symbols_to_close.append((symbol, exit_price, exit_reason))

            except Exception as e:
                logger.warning(f"Failed to check exit for {symbol}: {e}")
                continue

        # Execute exits
        for symbol, exit_price, exit_reason in symbols_to_close:
            self.exit_position(symbol, exit_price, exit_reason)

    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        quantity = position['quantity']
        entry_price = position['entry_price']

        # Calculate P&L
        exit_value = quantity * exit_price
        entry_value = quantity * entry_price
        transaction_cost = exit_value * 0.001  # Exit cost
        total_cost = position['entry_cost'] + transaction_cost

        gross_pnl = exit_value - entry_value
        net_pnl = gross_pnl - total_cost

        # Update capital
        self.current_capital += exit_value - transaction_cost

        # Record trade
        trade_record = {
            'type': 'EXIT',
            'symbol': symbol,
            'quantity': quantity,
            'price': exit_price,
            'value': exit_value,
            'cost': transaction_cost,
            'pnl': net_pnl,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'holding_period': (datetime.now() - datetime.fromisoformat(position['entry_time'])).days
        }

        self.trade_history.append(trade_record)

        # Remove position
        del self.positions[symbol]

        logger.info(f"Position closed: {symbol} | P&L: ₹{net_pnl:,.0f} | Reason: {reason}")
        self.save_state()

    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'capital_return_pct': 0,
                'current_capital': self.current_capital,
                'open_positions': len(self.positions)
            }

        closed_trades = [t for t in self.trade_history if t['type'] == 'EXIT']
        total_trades = len(closed_trades)

        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'capital_return_pct': 0,
                'current_capital': self.current_capital,
                'open_positions': len(self.positions)
            }

        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / total_trades

        total_pnl = sum(t['pnl'] for t in closed_trades)
        capital_return = (total_pnl / self.initial_capital) * 100

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'capital_return_pct': capital_return,
            'current_capital': self.current_capital,
            'open_positions': len(self.positions)
        }

    def print_status(self):
        """Print current status."""
        stats = self.get_performance_stats()

        print("\n" + "="*50)
        print("SWING_BOT PAPER TRADING STATUS")
        print("="*50)
        print(f"Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"Current Capital: ₹{self.current_capital:,.0f}")
        print(f"Open Positions: {stats['open_positions']}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1%}")
        print(f"Total P&L: ₹{stats['total_pnl']:,.0f}")
        print(f"Return: {stats['capital_return_pct']:.1f}%")

        if self.positions:
            print("\nOpen Positions:")
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos['quantity']} shares @ ₹{pos['entry_price']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='SWING_BOT Paper Trading Simulator')
    parser.add_argument('--symbols', nargs='+', default=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'],
                       help='Symbols to monitor')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Starting capital')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan for signals, do not execute')
    parser.add_argument('--status', action='store_true',
                       help='Show current status only')

    args = parser.parse_args()

    trader = PaperTrader(capital=args.capital)

    if args.status:
        trader.print_status()
        return

    # Check market regime first
    regime = calculate_market_regime('NSE_INDEX|Nifty 50')
    regime_status = regime.get('regime_status', 'UNKNOWN')

    print(f"Market Regime: {regime_status}")

    if regime_status != 'ON':
        print("Market regime is OFF - holding cash (no paper trades)")
        trader.print_status()
        return

    # Check for exits first
    trader.check_exits()

    # Scan for signals
    signals = trader.scan_signals(args.symbols)

    if signals:
        print(f"Found {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal['symbol']} - {signal['strategy']} (Confidence: {signal['confidence']:.1f})")

        if not args.scan_only:
            # Execute signals
            executed = 0
            for signal in signals:
                if trader.execute_signal(signal):
                    executed += 1

            print(f"Executed {executed} paper trades")
        else:
            print("Scan-only mode - no trades executed")
    else:
        print("No signals found")

    trader.print_status()

if __name__ == "__main__":
    main()