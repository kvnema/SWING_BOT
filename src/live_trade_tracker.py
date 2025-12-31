"""
SWING_BOT Live Trade Tracker

Tracks live GTT order executions, monitors positions, and logs completed trades
for success model updates.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .upstox_gtt import get_all_gtt_orders, get_gtt_order
from .data_fetch import fetch_live_quotes
from .notifications_router import send_telegram_alert

logger = logging.getLogger(__name__)

class LiveTradeTracker:
    """Track live GTT orders and log completed trades."""

    def __init__(self, trades_db: str = 'outputs/live_trades.jsonl',
                 positions_db: str = 'outputs/live_positions.json'):
        self.trades_db = Path(trades_db)
        self.positions_db = Path(positions_db)
        self.trades_db.parent.mkdir(parents=True, exist_ok=True)

        # Load existing positions
        self.positions = self._load_positions()

    def _load_positions(self) -> Dict[str, Dict]:
        """Load current open positions."""
        if self.positions_db.exists():
            try:
                with open(self.positions_db, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load positions: {e}")
        return {}

    def _save_positions(self):
        """Save current positions."""
        try:
            with open(self.positions_db, 'w') as f:
                json.dump(self.positions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save positions: {e}")

    def _log_trade(self, trade: Dict):
        """Log a completed trade to the database."""
        try:
            with open(self.trades_db, 'a') as f:
                json.dump(trade, f, default=str)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def scan_gtt_orders(self, access_token: str) -> Dict[str, List]:
        """Scan all GTT orders and detect executions/changes."""
        updates = {
            'new_entries': [],
            'exits': [],
            'modifications': []
        }

        # Get all GTT orders
        response = get_all_gtt_orders(access_token)
        if response.get('status_code') != 200:
            logger.error(f"Failed to fetch GTT orders: {response}")
            return updates

        orders_data = response.get('body', {}).get('data', [])

        for order in orders_data:
            gtt_id = order.get('gtt_order_id')
            status = order.get('status')
            instrument_token = order.get('instrument_token')
            symbol = order.get('trading_symbol', '')

            # Check if this is a new entry execution
            if status == 'triggered' and instrument_token not in self.positions:
                # New position opened
                entry_details = self._extract_entry_details(order)
                if entry_details:
                    self.positions[instrument_token] = {
                        'symbol': symbol,
                        'gtt_id': gtt_id,
                        'entry_price': entry_details['price'],
                        'entry_time': entry_details['time'],
                        'quantity': order.get('quantity', 0),
                        'stop_price': entry_details.get('stop_price'),
                        'target_price': entry_details.get('target_price'),
                        'status': 'open'
                    }
                    updates['new_entries'].append(self.positions[instrument_token])

                    # Telegram alert
                    try:
                        message = f"🎯 SWING_BOT: Position Opened\n• Symbol: {symbol}\n• Entry: ₹{entry_details['price']:.2f}\n• Quantity: {order.get('quantity', 0)}\n• GTT ID: {gtt_id}"
                        send_telegram_alert("position_opened", message)
                    except Exception as e:
                        logger.warning(f"Telegram alert failed: {e}")

            # Check if position was closed
            elif instrument_token in self.positions and status in ['cancelled', 'expired']:
                position = self.positions[instrument_token]
                exit_details = self._extract_exit_details(order)

                if exit_details:
                    # Log completed trade
                    trade = {
                        'symbol': position['symbol'],
                        'entry_date': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_date': exit_details['time'],
                        'exit_price': exit_details['price'],
                        'quantity': position['quantity'],
                        'pnl': (exit_details['price'] - position['entry_price']) * position['quantity'],
                        'strategy': 'Live_GTT',
                        'gtt_id': gtt_id,
                        'exit_reason': exit_details['reason']
                    }

                    # Calculate R multiple
                    risk_per_share = position['entry_price'] - position['stop_price']
                    if risk_per_share > 0:
                        trade['R'] = (exit_details['price'] - position['entry_price']) / risk_per_share
                    else:
                        trade['R'] = 0

                    self._log_trade(trade)
                    updates['exits'].append(trade)

                    # Telegram alert
                    try:
                        pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                        message = f"{pnl_emoji} SWING_BOT: Position Closed\n• Symbol: {position['symbol']}\n• P&L: ₹{trade['pnl']:.2f} ({trade['R']:.2f}R)\n• Exit: ₹{exit_details['price']:.2f}\n• Reason: {exit_details['reason']}"
                        send_telegram_alert("position_closed", message)
                    except Exception as e:
                        logger.warning(f"Telegram alert failed: {e}")

                # Remove from positions
                del self.positions[instrument_token]

        self._save_positions()
        return updates

    def _extract_entry_details(self, order: Dict) -> Optional[Dict]:
        """Extract entry execution details from order."""
        # Check order history or rules to determine entry price/time
        # This is a simplified implementation - in reality you'd check order history
        rules = order.get('rules', [])
        for rule in rules:
            if rule.get('strategy') == 'ENTRY' and rule.get('status') == 'executed':
                return {
                    'price': rule.get('trigger_price'),
                    'time': datetime.now(),  # Should get from order history
                    'stop_price': next((r.get('trigger_price') for r in rules if r.get('strategy') == 'STOPLOSS'), None),
                    'target_price': next((r.get('trigger_price') for r in rules if r.get('strategy') == 'TARGET'), None)
                }
        return None

    def _extract_exit_details(self, order: Dict) -> Optional[Dict]:
        """Extract exit execution details from order."""
        rules = order.get('rules', [])
        exit_time = datetime.now()

        # Determine exit reason and price
        for rule in rules:
            if rule.get('status') == 'executed':
                if rule.get('strategy') == 'STOPLOSS':
                    return {
                        'price': rule.get('trigger_price'),
                        'time': exit_time,
                        'reason': 'Stop Loss'
                    }
                elif rule.get('strategy') == 'TARGET':
                    return {
                        'price': rule.get('trigger_price'),
                        'time': exit_time,
                        'reason': 'Target Hit'
                    }

        # If no specific rule executed, assume manual or expired
        return {
            'price': order.get('last_price', 0),  # Should get actual exit price
            'time': exit_time,
            'reason': 'Manual/Expired'
        }

    def get_open_positions(self) -> List[Dict]:
        """Get current open positions."""
        return list(self.positions.values())

    def get_daily_pnl_summary(self, days: int = 1) -> Dict:
        """Get P&L summary for recent days."""
        if not self.trades_db.exists():
            return {'total_pnl': 0, 'winning_trades': 0, 'total_trades': 0}

        # Read recent trades
        trades = []
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            with open(self.trades_db, 'r') as f:
                for line in f:
                    trade = json.loads(line)
                    trade_date = pd.to_datetime(trade.get('exit_date'))
                    if trade_date >= cutoff_date:
                        trades.append(trade)
        except Exception as e:
            logger.error(f"Failed to read trades: {e}")
            return {'total_pnl': 0, 'winning_trades': 0, 'total_trades': 0}

        total_pnl = sum(t.get('pnl', 0) for t in trades)
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total_trades = len(trades)

        return {
            'total_pnl': total_pnl,
            'winning_trades': winning_trades,
            'total_trades': total_trades,
            'win_rate': winning_trades / max(total_trades, 1)
        }

# Global instance
live_trade_tracker = LiveTradeTracker()

def scan_live_trades(access_token: str) -> Dict[str, List]:
    """Scan for live trade updates."""
    return live_trade_tracker.scan_gtt_orders(access_token)

def get_live_positions() -> List[Dict]:
    """Get current open positions."""
    return live_trade_tracker.get_open_positions()

def get_daily_pnl(days: int = 1) -> Dict:
    """Get daily P&L summary."""
    return live_trade_tracker.get_daily_pnl_summary(days)