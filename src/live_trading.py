"""
SWING_BOT Live Order Execution
Automated order placement via Zerodha/Kite API with safety confirmations
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

from .config import (
    KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN, KITE_PUBLIC_TOKEN, KITE_ENABLED
)
from .notifications import send_signal_alert, send_error_alert

logger = logging.getLogger(__name__)

class KiteOrderExecutor:
    """Kite/Zerodha API order execution with safety features."""

    def __init__(self):
        self.enabled = KITE_ENABLED
        self.kite = None

        if self.enabled:
            try:
                from kiteconnect import KiteConnect
                self.kite = KiteConnect(api_key=KITE_API_KEY)

                # Set access token
                if KITE_ACCESS_TOKEN:
                    self.kite.set_access_token(KITE_ACCESS_TOKEN)

                logger.info("Kite API initialized successfully")
            except ImportError:
                logger.warning("kiteconnect package not installed. Install with: pip install kiteconnect")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Kite API: {e}")
                self.enabled = False
        else:
            logger.warning("Kite API not configured - live trading disabled")

    def get_instrument_token(self, symbol: str) -> Optional[str]:
        """Get instrument token for a symbol."""
        if not self.kite:
            return None

        try:
            # Get instrument details
            instruments = self.kite.instruments(exchange="NSE")
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol.replace('.NS', ''):
                    return instrument['instrument_token']
        except Exception as e:
            logger.error(f"Failed to get instrument token for {symbol}: {e}")

        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        if not self.kite:
            return None

        try:
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                return None

            quote = self.kite.quote(f"NSE:{instrument_token}")
            return quote[f"NSE:{instrument_token}"]['last_price']
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None

    def calculate_order_quantity(self, capital: float, entry_price: float,
                               risk_per_trade_pct: float = 0.01,
                               stop_loss_pct: float = 0.015) -> int:
        """Calculate safe order quantity based on risk management."""
        risk_amount = capital * risk_per_trade_pct
        risk_per_share = entry_price * stop_loss_pct
        quantity = int(risk_amount / risk_per_share)

        # Ensure minimum quantity and round to appropriate lot size
        quantity = max(1, quantity)

        # For large cap stocks, ensure reasonable position size
        max_quantity = min(1000, int(capital * 0.02 / entry_price))  # Max 2% of capital
        quantity = min(quantity, max_quantity)

        return quantity

    def place_buy_order(self, symbol: str, quantity: int, price: float,
                       order_type: str = "MARKET") -> Tuple[bool, Optional[Dict]]:
        """Place a buy order with safety checks."""
        if not self.enabled or not self.kite:
            logger.warning("Kite API not available - cannot place live orders")
            return False, None

        try:
            # Get instrument token
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                error_msg = f"Could not find instrument token for {symbol}"
                send_error_alert(error_msg, "Order Execution")
                return False, None

            # Prepare order parameters
            order_params = {
                "tradingsymbol": symbol.replace('.NS', ''),
                "exchange": "NSE",
                "transaction_type": "BUY",
                "order_type": order_type,
                "quantity": quantity,
                "product": "MIS",  # Intraday
                "variety": "regular"
            }

            if order_type == "LIMIT":
                order_params["price"] = price

            # Place the order
            order_id = self.kite.place_order(**order_params)

            logger.info(f"Buy order placed: {symbol} {quantity} shares @ {order_type}")

            # Get order details
            order_details = self.kite.order_history(order_id=order_id)

            return True, order_details

        except Exception as e:
            error_msg = f"Failed to place buy order for {symbol}: {str(e)}"
            logger.error(error_msg)
            send_error_alert(error_msg, "Order Execution")
            return False, None

    def place_sell_order(self, symbol: str, quantity: int, price: float = None,
                        order_type: str = "MARKET") -> Tuple[bool, Optional[Dict]]:
        """Place a sell order to exit position."""
        if not self.enabled or not self.kite:
            logger.warning("Kite API not available - cannot place live orders")
            return False, None

        try:
            # Get instrument token
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                error_msg = f"Could not find instrument token for {symbol}"
                send_error_alert(error_msg, "Order Execution")
                return False, None

            # Prepare order parameters
            order_params = {
                "tradingsymbol": symbol.replace('.NS', ''),
                "exchange": "NSE",
                "transaction_type": "SELL",
                "order_type": order_type,
                "quantity": quantity,
                "product": "MIS",  # Intraday
                "variety": "regular"
            }

            if order_type == "LIMIT" and price:
                order_params["price"] = price

            # Place the order
            order_id = self.kite.place_order(**order_params)

            logger.info(f"Sell order placed: {symbol} {quantity} shares @ {order_type}")

            # Get order details
            order_details = self.kite.order_history(order_id=order_id)

            return True, order_details

        except Exception as e:
            error_msg = f"Failed to place sell order for {symbol}: {str(e)}"
            logger.error(error_msg)
            send_error_alert(error_msg, "Order Execution")
            return False, None

    def get_positions(self) -> Optional[Dict]:
        """Get current positions."""
        if not self.kite:
            return None

        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return None

    def get_orders(self, order_id: str = None) -> Optional[Dict]:
        """Get order details."""
        if not self.kite:
            return None

        try:
            if order_id:
                return self.kite.order_history(order_id=order_id)
            else:
                return self.kite.orders()
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return None

class LiveTrader:
    """Live trading integration with safety confirmations."""

    def __init__(self, capital: float = 100000, max_positions: int = 3,
                 risk_per_trade_pct: float = 0.01, require_confirmation: bool = True):
        self.capital = capital
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        self.require_confirmation = require_confirmation

        self.executor = KiteOrderExecutor()
        self.positions = {}

        logger.info(f"Live trader initialized with ₹{capital:,.0f} capital")

    def execute_signal(self, signal: Dict) -> bool:
        """Execute a live trading signal with safety checks."""
        symbol = signal['symbol']
        strategy = signal.get('strategy', 'UNKNOWN')
        action = signal.get('action', 'BUY')

        # Safety checks
        if not self.executor.enabled:
            logger.warning("Live trading disabled - Kite API not configured")
            return False

        # Check position limits
        if action.upper() == "BUY" and len(self.positions) >= self.max_positions:
            logger.info("Max positions reached - skipping signal")
            return False

        # Check if already in position
        if symbol in self.positions and action.upper() == "BUY":
            logger.info(f"Already in position for {symbol}")
            return False

        # Calculate position size
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)

        if entry_price <= 0:
            logger.error(f"Invalid entry price for {symbol}")
            return False

        quantity = self.executor.calculate_order_quantity(
            self.capital, entry_price, self.risk_per_trade_pct
        )

        if quantity == 0:
            logger.warning(f"Position size too small for {symbol}")
            return False

        # Confirmation prompt (if required)
        if self.require_confirmation:
            print(f"\n🚨 LIVE TRADING SIGNAL 🚨")
            print(f"Symbol: {symbol}")
            print(f"Strategy: {strategy}")
            print(f"Action: {action.upper()}")
            print(f"Quantity: {quantity}")
            print(f"Entry Price: ₹{entry_price:.2f}")
            print(f"Stop Loss: ₹{stop_loss:.2f}")
            print(f"Risk Amount: ₹{quantity * (entry_price - stop_loss):.2f}")

            confirmation = input("\nExecute this live trade? (yes/no): ").lower().strip()
            if confirmation not in ['yes', 'y']:
                logger.info("Trade cancelled by user")
                return False

        # Execute the trade
        if action.upper() == "BUY":
            success, order_details = self.executor.place_buy_order(symbol, quantity, entry_price)

            if success:
                # Record position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': signal.get('target'),
                    'entry_time': datetime.now(),
                    'strategy': strategy
                }

                # Send alert
                alert_data = {
                    'symbol': symbol,
                    'strategy': strategy,
                    'action': 'BUY (LIVE)',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': signal.get('target'),
                    'confidence': signal.get('confidence', 0),
                    'quantity': quantity,
                    'risk_reward': (signal.get('target', 0) - entry_price) / (entry_price - stop_loss) if stop_loss < entry_price else 0
                }
                send_signal_alert(alert_data)

                logger.info(f"Live trade executed: {symbol} {quantity} shares")
                return True

        return False

    def check_exits(self):
        """Check for exit conditions on live positions."""
        symbols_to_close = []

        for symbol, position in self.positions.items():
            try:
                current_price = self.executor.get_current_price(symbol)
                if not current_price:
                    continue

                entry_price = position['entry_price']
                stop_loss = position['stop_loss']
                target = position['target']

                exit_reason = None
                exit_price = current_price

                # Check exit conditions
                if current_price <= stop_loss:
                    exit_reason = 'STOP_LOSS'
                    exit_price = stop_loss
                elif target and current_price >= target:
                    exit_reason = 'TARGET'
                    exit_price = target
                elif (datetime.now() - position['entry_time']).days > 30:
                    exit_reason = 'TIME_EXIT'

                if exit_reason:
                    symbols_to_close.append((symbol, exit_price, exit_reason))

            except Exception as e:
                logger.warning(f"Failed to check exit for {symbol}: {e}")

        # Execute exits
        for symbol, exit_price, exit_reason in symbols_to_close:
            self.exit_position(symbol, exit_price, exit_reason)

    def exit_position(self, symbol: str, exit_price: float, reason: str):
        """Exit a live position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        quantity = position['quantity']

        # Confirmation for exits (if required)
        if self.require_confirmation:
            print(f"\n📉 EXIT SIGNAL 📉")
            print(f"Symbol: {symbol}")
            print(f"Exit Price: ₹{exit_price:.2f}")
            print(f"Reason: {reason}")
            print(f"P&L: ₹{(exit_price - position['entry_price']) * quantity:,.0f}")

            confirmation = input("\nExecute exit? (yes/no): ").lower().strip()
            if confirmation not in ['yes', 'y']:
                logger.info("Exit cancelled by user")
                return

        # Execute sell order
        success, order_details = self.executor.place_sell_order(symbol, quantity, exit_price)

        if success:
            # Calculate P&L
            entry_price = position['entry_price']
            pnl = (exit_price - entry_price) * quantity

            # Send alert
            alert_data = {
                'symbol': symbol,
                'strategy': position['strategy'],
                'action': f'EXIT ({reason})',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'quantity': quantity
            }
            send_signal_alert(alert_data)

            # Remove position
            del self.positions[symbol]

            logger.info(f"Position closed: {symbol} | P&L: ₹{pnl:,.0f} | Reason: {reason}")

    def get_status(self) -> Dict:
        """Get current trading status."""
        positions_value = 0
        for symbol, pos in self.positions.items():
            current_price = self.executor.get_current_price(symbol)
            if current_price:
                positions_value += current_price * pos['quantity']

        return {
            'capital': self.capital,
            'positions_count': len(self.positions),
            'positions_value': positions_value,
            'positions': list(self.positions.keys()),
            'live_trading_enabled': self.executor.enabled
        }

# Global live trader instance
live_trader = LiveTrader()

def execute_live_signal(signal: Dict) -> bool:
    """Convenience function to execute live signals."""
    return live_trader.execute_signal(signal)

def check_live_exits():
    """Convenience function to check live exits."""
    live_trader.check_exits()

def get_live_status() -> Dict:
    """Convenience function to get live trading status."""
    return live_trader.get_status()