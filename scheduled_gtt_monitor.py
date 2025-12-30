import json
import os
import pandas as pd
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.upstox_gtt import get_all_gtt_orders, modify_gtt_order, place_gtt_order_multi
from src.notifications_router import notify_gtt_changes
from src.cli import orchestrate_eod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GTTMonitor:
    """Monitor and manage GTT orders with automated updates."""

    def __init__(self, state_file: str = 'outputs/gtt/gtt_monitor_state.json'):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load previous state of GTT orders."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
        return {'orders': {}, 'last_run': None}

    def _save_state(self):
        """Save current state of GTT orders."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    def get_current_gtt_orders(self, access_token: str) -> Dict[str, Dict]:
        """Fetch current GTT orders from Upstox."""
        response = get_all_gtt_orders(access_token)
        if response.get('status_code') == 200:
            orders_data = response.get('body', {}).get('data', [])
            current_orders = {}
            for order in orders_data:
                if order.get('status') == 'active':
                    instrument_token = order.get('instrument_token')
                    current_orders[instrument_token] = {
                        'gtt_id': order.get('gtt_order_id'),
                        'rules': order.get('rules', []),
                        'quantity': order.get('quantity'),
                        'symbol': order.get('trading_symbol', ''),
                        'status': order.get('status')
                    }
            return current_orders
        else:
            logger.error(f"Failed to fetch GTT orders: {response}")
            return {}

    def detect_changes(self, new_plan: pd.DataFrame, current_orders: Dict[str, Dict]) -> Dict[str, List]:
        """Detect changes between new plan and current orders."""
        changes = {
            'new_orders': [],
            'modify_orders': [],
            'cancel_orders': [],
            'unchanged': []
        }

        # Convert new plan to dict by instrument_token
        new_positions = {}
        for _, row in new_plan.iterrows():
            if pd.notna(row.get('InstrumentToken_x')):
                # Use full instrument key as returned by API
                full_key = str(row['InstrumentToken_x'])
                new_positions[full_key] = {
                    'symbol': row['Symbol'],
                    'instrument_token': full_key,  # Add this for placing orders
                    'entry_price': round(row['Reconciled_Entry'], 2),
                    'stop_price': round(row['Reconciled_Stop'], 2),
                    'target_price': round(row['Reconciled_Target'], 2),
                    'quantity': int(row['Qty'])
                }

        # Check for new/modified positions
        for token, new_pos in new_positions.items():
            if token not in current_orders:
                changes['new_orders'].append(new_pos)
            else:
                current_order = current_orders[token]
                current_rules = {rule['strategy']: rule['trigger_price'] for rule in current_order['rules']}

                # Check if prices changed significantly (>0.5% or 0.05 rupees, whichever is larger)
                entry_change = abs(current_rules.get('ENTRY', 0) - new_pos['entry_price']) / max(current_rules.get('ENTRY', 1), 0.01)
                stop_change = abs(current_rules.get('STOPLOSS', 0) - new_pos['stop_price']) / max(current_rules.get('STOPLOSS', 1), 0.01)
                target_change = abs(current_rules.get('TARGET', 0) - new_pos['target_price']) / max(current_rules.get('TARGET', 1), 0.01)

                threshold = max(0.005, 0.05 / max(current_rules.get('ENTRY', 1), 1))  # 0.5% or 5 paise

                if entry_change > threshold or stop_change > threshold or target_change > threshold:
                    changes['modify_orders'].append({
                        'gtt_id': current_order['gtt_id'],
                        'current': current_rules,
                        'new': new_pos
                    })
                else:
                    changes['unchanged'].append(new_pos)

        # Check for positions to cancel (in current orders but not in new plan)
        for token, order in current_orders.items():
            if token not in new_positions:
                changes['cancel_orders'].append(order)

        return changes

    def execute_changes(self, changes: Dict[str, List], access_token: str) -> Dict[str, List]:
        """Execute the detected changes."""
        results = {'placed': [], 'modified': [], 'cancelled': [], 'errors': []}

        # Place new orders
        for new_pos in changes['new_orders']:
            try:
                rules = [
                    {'strategy': 'ENTRY', 'trigger_type': 'ABOVE', 'trigger_price': new_pos['entry_price']},
                    {'strategy': 'STOPLOSS', 'trigger_type': 'BELOW', 'trigger_price': new_pos['stop_price']},
                    {'strategy': 'TARGET', 'trigger_type': 'ABOVE', 'trigger_price': new_pos['target_price']}
                ]

                result = place_gtt_order_multi(
                    instrument_token=new_pos['instrument_token'],
                    quantity=new_pos['quantity'],
                    product='D',
                    rules=rules,
                    transaction_type='BUY',
                    access_token=access_token,
                    dry_run=False,
                    retries=3
                )

                if result.get('status_code') in (200, 201, 202):
                    results['placed'].append({
                        'symbol': new_pos['symbol'],
                        'gtt_id': result.get('order_id', 'UNKNOWN')
                    })
                else:
                    results['errors'].append(f"Failed to place {new_pos['symbol']}: {result}")

            except Exception as e:
                results['errors'].append(f"Exception placing {new_pos['symbol']}: {str(e)}")

        # Modify existing orders
        for mod_order in changes['modify_orders']:
            try:
                payload = {
                    'gtt_order_id': mod_order['gtt_id'],
                    'rules': [
                        {'strategy': 'ENTRY', 'trigger_type': 'ABOVE', 'trigger_price': mod_order['new']['entry_price']},
                        {'strategy': 'STOPLOSS', 'trigger_type': 'BELOW', 'trigger_price': mod_order['new']['stop_price']},
                        {'strategy': 'TARGET', 'trigger_type': 'ABOVE', 'trigger_price': mod_order['new']['target_price']}
                    ]
                }

                result = modify_gtt_order(access_token, payload)
                if result.get('status_code') in (200, 201, 202):
                    results['modified'].append({
                        'symbol': mod_order['new']['symbol'],
                        'gtt_id': mod_order['gtt_id']
                    })
                else:
                    results['errors'].append(f"Failed to modify {mod_order['new']['symbol']}: {result}")

            except Exception as e:
                results['errors'].append(f"Exception modifying {mod_order['new']['symbol']}: {str(e)}")

        # Note: Cancelling orders would require additional API calls - for now, we'll just log them
        for cancel_order in changes['cancel_orders']:
            logger.info(f"Order to cancel: {cancel_order['symbol']} (GTT ID: {cancel_order['gtt_id']})")
            results['cancelled'].append(cancel_order)

        return results

    def run_monitoring_cycle(self) -> Dict[str, any]:
        """Run the complete monitoring and update cycle."""
        logger.info("Starting GTT monitoring cycle")

        # Get access token
        access_token = os.environ.get('UPSTOX_ACCESS_TOKEN')
        if not access_token:
            raise ValueError("UPSTOX_ACCESS_TOKEN not found")

        # Run the full SWING_BOT pipeline
        logger.info("Running SWING_BOT pipeline")
        try:
            success = orchestrate_eod(dashboard=False, metrics=False, notifications=False)
            if not success:
                logger.error("SWING_BOT pipeline failed")
                return {'success': False, 'error': 'Pipeline failed'}
        except Exception as e:
            logger.error(f"Pipeline exception: {e}")
            return {'success': False, 'error': str(e)}

        # Load the latest reconciled plan
        reconciled_file = Path('outputs/gtt/gtt_plan_reconciled.csv')
        if not reconciled_file.exists():
            logger.warning("No reconciled plan found")
            return {'success': False, 'error': 'No reconciled plan'}

        try:
            new_plan = pd.read_csv(reconciled_file)
        except Exception as e:
            logger.error(f"Failed to load reconciled plan: {e}")
            return {'success': False, 'error': str(e)}

        # Get current GTT orders
        current_orders = self.get_current_gtt_orders(access_token)

        # Detect changes
        changes = self.detect_changes(new_plan, current_orders)
        logger.info(f"Detected changes: {sum(len(v) for v in changes.values())} total changes")

        # Execute changes
        results = self.execute_changes(changes, access_token)

        # Update state
        self.state['orders'] = current_orders
        self.state['last_run'] = datetime.now().isoformat()
        self._save_state()

        # Send notifications
        self._send_notifications(changes, results)

        logger.info("GTT monitoring cycle completed")
        return {
            'success': True,
            'changes': changes,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    def _send_notifications(self, changes: Dict[str, List], results: Dict[str, List]):
        """Send notifications about changes."""
        try:
            # Prepare notification message
            message = f"🤖 SWING_BOT GTT Update ({datetime.now().strftime('%H:%M %d/%m')})\n\n"

            if changes['new_orders']:
                message += f"🆕 New Orders: {len(changes['new_orders'])}\n"
                for order in changes['new_orders'][:3]:  # Show first 3
                    message += f"  • {order['symbol']}\n"

            if changes['modify_orders']:
                message += f"📝 Modified Orders: {len(changes['modify_orders'])}\n"
                for order in changes['modify_orders'][:3]:
                    message += f"  • {order['new']['symbol']}\n"

            if changes['cancel_orders']:
                message += f"❌ Orders to Cancel: {len(changes['cancel_orders'])}\n"

            if results['placed']:
                message += f"✅ Successfully Placed: {len(results['placed'])}\n"

            if results['modified']:
                message += f"✅ Successfully Modified: {len(results['modified'])}\n"

            if results['errors']:
                message += f"❌ Errors: {len(results['errors'])}\n"

            # Send notification
            notify_gtt_changes(message)

        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")

def main():
    """Main entry point for scheduled GTT monitoring."""
    monitor = GTTMonitor()
    result = monitor.run_monitoring_cycle()

    if result['success']:
        print("✅ GTT monitoring cycle completed successfully")
        print(f"Changes detected: {result['changes']}")
        print(f"Actions taken: {result['results']}")
    else:
        print(f"❌ GTT monitoring failed: {result.get('error', 'Unknown error')}")
        exit(1)

if __name__ == '__main__':
    main()