"""
SWING_BOT GTT Order Reconciliation & Lifecycle Management

Handles the complete lifecycle of GTT orders to ensure the live GTT book
always matches the current strategy output - no duplicates, no orphans.

Key Features:
- Fetch all active GTT orders from Upstox
- Compare against new daily GTT plan
- Intelligent lifecycle actions: modify, cancel, place new
- Safety checks and comprehensive logging
- Telegram alerts for all changes
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import time
import json
from datetime import datetime

from .upstox_gtt import get_all_gtt_orders, cancel_gtt_order, modify_gtt_order, place_gtt_bulk, place_gtt_with_retries
from .notifications_router import send_telegram_alert
from .gtt_sizing import build_upstox_payload

logger = logging.getLogger(__name__)


@dataclass
class GTTOrder:
    """Represents a GTT order from Upstox API."""
    order_id: str
    symbol: str
    transaction_type: str  # 'BUY' or 'SELL'
    quantity: int
    product: str  # 'D' for delivery
    instrument_token: str
    rules: List[Dict]  # GTT rules (ENTRY, STOPLOSS, TARGET)
    status: str = 'active'

    @property
    def entry_price(self) -> Optional[float]:
        """Get entry trigger price."""
        for rule in self.rules:
            if rule.get('strategy') == 'ENTRY':
                return rule.get('trigger_price')
        return None

    @property
    def stoploss_price(self) -> Optional[float]:
        """Get stoploss trigger price."""
        for rule in self.rules:
            if rule.get('strategy') == 'STOPLOSS':
                return rule.get('trigger_price')
        return None

    @property
    def target_price(self) -> Optional[float]:
        """Get target trigger price."""
        for rule in self.rules:
            if rule.get('strategy') == 'TARGET':
                return rule.get('trigger_price')
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame operations."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'transaction_type': self.transaction_type,
            'quantity': self.quantity,
            'product': self.product,
            'instrument_token': self.instrument_token,
            'entry_price': self.entry_price,
            'stoploss_price': self.stoploss_price,
            'target_price': self.target_price,
            'status': self.status
        }


@dataclass
class ReconciliationAction:
    """Represents an action to be taken during reconciliation."""
    action_type: str  # 'cancel', 'modify', 'place_new', 'no_change'
    symbol: str
    order_id: Optional[str] = None
    old_params: Optional[Dict] = None
    new_params: Optional[Dict] = None
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            'action_type': self.action_type,
            'symbol': self.symbol,
            'order_id': self.order_id,
            'old_params': self.old_params,
            'new_params': self.new_params,
            'reason': self.reason
        }


class GTTReconciler:
    """
    Handles GTT order reconciliation and lifecycle management.

    Ensures the live GTT book always matches the current strategy output.
    """

    def __init__(self, access_token: str, config: Optional[Dict] = None):
        """
        Initialize the GTT reconciler.

        Args:
            access_token: Upstox API access token
            config: Configuration dictionary
        """
        self.access_token = access_token
        self.config = config or {}
        self.dry_run = self.config.get('dry_run', True)
        self.log_path = self.config.get('log_path', 'outputs/logs/gtt_reconcile.log')

        # Ensure log directory exists
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for reconciliation operations."""
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def fetch_active_gtts(self) -> List[GTTOrder]:
        """
        Fetch all active GTT orders from Upstox.

        Returns:
            List of active GTTOrder objects
        """
        logger.info("📥 Fetching active GTT orders from Upstox...")

        try:
            response = get_all_gtt_orders(self.access_token)

            if response['status_code'] != 200:
                logger.error(f"Failed to fetch GTT orders: {response}")
                return []

            orders_data = response['body'].get('data', [])
            active_orders = []

            for order_data in orders_data:
                # Only process active orders
                if order_data.get('status') != 'active':
                    continue

                try:
                    order = GTTOrder(
                        order_id=order_data['order_id'],
                        symbol=order_data.get('symbol', 'UNKNOWN'),
                        transaction_type=order_data['transaction_type'],
                        quantity=order_data['quantity'],
                        product=order_data['product'],
                        instrument_token=order_data['instrument_token'],
                        rules=order_data.get('rules', []),
                        status='active'
                    )
                    active_orders.append(order)
                except KeyError as e:
                    logger.warning(f"Skipping malformed GTT order: {e}")
                    continue

            logger.info(f"✅ Found {len(active_orders)} active GTT orders")
            return active_orders

        except Exception as e:
            logger.error(f"Error fetching active GTTs: {e}")
            return []

    def parse_gtt_plan(self, plan_df: pd.DataFrame) -> List[Dict]:
        """
        Parse GTT plan DataFrame into standardized format.

        Args:
            plan_df: DataFrame with GTT plan data

        Returns:
            List of plan entries in standardized format
        """
        plan_entries = []

        for _, row in plan_df.iterrows():
            entry = {
                'symbol': row['Symbol'],
                'transaction_type': 'BUY',  # Assuming all are BUY orders for now
                'quantity': row.get('Quantity', 1),
                'product': 'D',  # Delivery
                'instrument_token': row.get('InstrumentToken'),
                'entry_price': row.get('ENTRY_trigger_price'),
                'stoploss_price': row.get('STOPLOSS_trigger_price'),
                'target_price': row.get('TARGET_trigger_price'),
                'strategy': row.get('Strategy', 'UNKNOWN'),
                'confidence': row.get('DecisionConfidence', 0.0)
            }
            plan_entries.append(entry)

        return plan_entries

    def create_matching_key(self, entry: Dict) -> str:
        """
        Create a unique matching key for GTT order comparison.

        Args:
            entry: Either GTTOrder dict or plan entry dict

        Returns:
            Unique key for matching
        """
        symbol = entry.get('symbol', entry.get('Symbol', 'UNKNOWN'))
        transaction_type = entry.get('transaction_type', 'BUY')
        return f"{symbol}_{transaction_type}"

    def compare_plans(self, active_gtts: List[GTTOrder], new_plan: List[Dict]) -> List[ReconciliationAction]:
        """
        Compare active GTTs against new plan and determine required actions.

        Args:
            active_gtts: List of currently active GTT orders
            new_plan: List of entries from new GTT plan

        Returns:
            List of ReconciliationAction objects
        """
        logger.info("🔍 Comparing active GTTs against new plan...")

        actions = []

        # Convert active GTTs to dict format for easier processing
        active_dict = {self.create_matching_key(gtt.to_dict()): gtt for gtt in active_gtts}
        plan_dict = {self.create_matching_key(entry): entry for entry in new_plan}

        # Find GTTs to cancel (in active but not in new plan)
        for key, gtt in active_dict.items():
            if key not in plan_dict:
                actions.append(ReconciliationAction(
                    action_type='cancel',
                    symbol=gtt.symbol,
                    order_id=gtt.order_id,
                    old_params=gtt.to_dict(),
                    reason="No longer in current strategy plan"
                ))

        # Find GTTs to modify or keep (in both active and new plan)
        for key, plan_entry in plan_dict.items():
            if key in active_dict:
                # Exists in both - check if parameters changed
                active_gtt = active_dict[key]

                # Compare key parameters
                params_changed = (
                    abs(active_gtt.entry_price - plan_entry.get('entry_price', 0)) > 0.01 or
                    abs(active_gtt.stoploss_price - plan_entry.get('stoploss_price', 0)) > 0.01 or
                    abs(active_gtt.target_price - plan_entry.get('target_price', 0)) > 0.01 or
                    active_gtt.quantity != plan_entry.get('quantity', 1)
                )

                if params_changed:
                    actions.append(ReconciliationAction(
                        action_type='modify',
                        symbol=plan_entry['symbol'],
                        order_id=active_gtt.order_id,
                        old_params=active_gtt.to_dict(),
                        new_params=plan_entry,
                        reason="Parameters changed in new plan"
                    ))
                else:
                    # No change needed
                    actions.append(ReconciliationAction(
                        action_type='no_change',
                        symbol=plan_entry['symbol'],
                        order_id=active_gtt.order_id,
                        reason="Parameters unchanged"
                    ))
            else:
                # New entry - needs to be placed
                actions.append(ReconciliationAction(
                    action_type='place_new',
                    symbol=plan_entry['symbol'],
                    new_params=plan_entry,
                    reason="New position in strategy plan"
                ))

        logger.info(f"📋 Generated {len(actions)} reconciliation actions")
        return actions

    def execute_actions(self, actions: List[ReconciliationAction]) -> Dict[str, Any]:
        """
        Execute the reconciliation actions.

        Args:
            actions: List of actions to execute

        Returns:
            Summary of execution results
        """
        logger.info("⚡ Executing reconciliation actions...")

        results = {
            'cancelled': [],
            'modified': [],
            'placed': [],
            'failed': [],
            'skipped': []
        }

        for action in actions:
            try:
                if action.action_type == 'cancel':
                    result = self._cancel_gtt(action)
                    if result['success']:
                        results['cancelled'].append(result)
                    else:
                        results['failed'].append(result)

                elif action.action_type == 'modify':
                    result = self._modify_gtt(action)
                    if result['success']:
                        results['modified'].append(result)
                    else:
                        results['failed'].append(result)

                elif action.action_type == 'place_new':
                    result = self._place_new_gtt(action)
                    if result['success']:
                        results['placed'].append(result)
                    else:
                        results['failed'].append(result)

                elif action.action_type == 'no_change':
                    results['skipped'].append({
                        'symbol': action.symbol,
                        'order_id': action.order_id,
                        'reason': action.reason
                    })

            except Exception as e:
                logger.error(f"Error executing action {action.action_type} for {action.symbol}: {e}")
                results['failed'].append({
                    'action': action.to_dict(),
                    'error': str(e)
                })

            # Rate limiting
            time.sleep(0.5)

        logger.info(f"✅ Execution complete: {len(results['cancelled'])} cancelled, {len(results['modified'])} modified, {len(results['placed'])} placed, {len(results['failed'])} failed")
        return results

    def _cancel_gtt(self, action: ReconciliationAction) -> Dict:
        """Cancel a GTT order."""
        logger.info(f"[CANCEL] Cancelling GTT for {action.symbol} (ID: {action.order_id})")

        if self.dry_run:
            logger.info(f"   [DRY RUN] Would cancel GTT {action.order_id}")
            return {
                'success': True,
                'action': 'cancel',
                'symbol': action.symbol,
                'order_id': action.order_id,
                'dry_run': True
            }

        try:
            response = cancel_gtt_order(self.access_token, action.order_id)

            success = response['status_code'] in (200, 202)
            result = {
                'success': success,
                'action': 'cancel',
                'symbol': action.symbol,
                'order_id': action.order_id,
                'response': response
            }

            if success:
                logger.info(f"Cancelled GTT {action.order_id} for {action.symbol}")
                self._send_notification(f"[CANCEL] Cancelled GTT for {action.symbol}", f"Order ID: {action.order_id}")
            else:
                logger.error(f"Failed to cancel GTT {action.order_id}: {response}")

            return result

        except Exception as e:
            logger.error(f"Exception cancelling GTT {action.order_id}: {e}")
            return {
                'success': False,
                'action': 'cancel',
                'symbol': action.symbol,
                'order_id': action.order_id,
                'error': str(e)
            }

    def _modify_gtt(self, action: ReconciliationAction) -> Dict:
        """Modify a GTT order."""
        logger.info(f"[MODIFY] Modifying GTT for {action.symbol} (ID: {action.order_id})")

        if self.dry_run:
            logger.info(f"   [DRY RUN] Would modify GTT {action.order_id}")
            return {
                'success': True,
                'action': 'modify',
                'symbol': action.symbol,
                'order_id': action.order_id,
                'dry_run': True
            }

        try:
            # Build modify payload
            payload = {
                'order_id': action.order_id,
                'rules': self._build_rules_from_params(action.new_params)
            }

            response = modify_gtt_order(self.access_token, payload)

            success = response['status_code'] in (200, 202)
            result = {
                'success': success,
                'action': 'modify',
                'symbol': action.symbol,
                'order_id': action.order_id,
                'old_params': action.old_params,
                'new_params': action.new_params,
                'response': response
            }

            if success:
                old_entry = action.old_params.get('entry_price', 0) if action.old_params else 0
                new_entry = action.new_params.get('entry_price', 0) if action.new_params else 0
                logger.info(f"Modified GTT {action.order_id} for {action.symbol}: Entry {old_entry} → {new_entry}")
                self._send_notification(
                    f"[MODIFY] Modified GTT for {action.symbol}",
                    f"Order ID: {action.order_id}\nEntry: ₹{old_entry} → ₹{new_entry}"
                )
            else:
                logger.error(f"Failed to modify GTT {action.order_id}: {response}")

            return result

        except Exception as e:
            logger.error(f"Exception modifying GTT {action.order_id}: {e}")
            return {
                'success': False,
                'action': 'modify',
                'symbol': action.symbol,
                'order_id': action.order_id,
                'error': str(e)
            }

    def _place_new_gtt(self, action: ReconciliationAction) -> Dict:
        """Place a new GTT order."""
        logger.info(f"[PLACE] Placing new GTT for {action.symbol}")

        if self.dry_run:
            logger.info(f"   [DRY RUN] Would place new GTT for {action.symbol}")
            return {
                'success': True,
                'action': 'place_new',
                'symbol': action.symbol,
                'dry_run': True
            }

        try:
            # Build payload for new GTT
            payload = {
                'instrument_token': action.new_params['instrument_token'],
                'quantity': action.new_params['quantity'],
                'product': action.new_params['product'],
                'transaction_type': action.new_params['transaction_type'],
                'rules': self._build_rules_from_params(action.new_params)
            }

            # Use existing placement function
            response = place_gtt_with_retries(
                self.access_token,
                payload,
                dry_run=False,
                retries=3,
                backoff=1.0,
                log_path=self.log_path
            )

            success = response['status_code'] in (200, 201, 202)
            result = {
                'success': success,
                'action': 'place_new',
                'symbol': action.symbol,
                'params': action.new_params,
                'response': response
            }

            if success:
                order_id = response.get('order_id', response.get('body', {}).get('order_id', 'UNKNOWN'))
                logger.info(f"Placed new GTT for {action.symbol} (ID: {order_id})")
                self._send_notification(
                    f"[PLACE] Placed new GTT for {action.symbol}",
                    f"Entry: ₹{action.new_params.get('entry_price')}\nOrder ID: {order_id}"
                )
            else:
                logger.error(f"Failed to place new GTT for {action.symbol}: {response}")

            return result

        except Exception as e:
            logger.error(f"Exception placing new GTT for {action.symbol}: {e}")
            return {
                'success': False,
                'action': 'place_new',
                'symbol': action.symbol,
                'error': str(e)
            }

    def _build_rules_from_params(self, params: Dict) -> List[Dict]:
        """Build GTT rules from parameter dict."""
        rules = []

        if params.get('entry_price'):
            rules.append({
                'strategy': 'ENTRY',
                'trigger_type': 'ABOVE',
                'trigger_price': params['entry_price']
            })

        if params.get('stoploss_price'):
            rules.append({
                'strategy': 'STOPLOSS',
                'trigger_type': 'BELOW',
                'trigger_price': params['stoploss_price']
            })

        if params.get('target_price'):
            rules.append({
                'strategy': 'TARGET',
                'trigger_type': 'ABOVE',
                'trigger_price': params['target_price']
            })

        return rules

    def _send_notification(self, title: str, message: str):
        """Send notification about GTT changes."""
        try:
            # Combine title and message for telegram alert
            full_message = f"{title}\n{message}"
            send_telegram_alert(
                alert_type="gtt_change",
                message=full_message,
                priority='normal'
            )
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

    def reconcile_gtt_orders(self, new_plan_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main reconciliation function.

        Args:
            new_plan_df: DataFrame with new GTT plan

        Returns:
            Complete reconciliation results
        """
        logger.info("🚀 Starting GTT order reconciliation...")

        start_time = datetime.now()
        results = {
            'start_time': start_time.isoformat(),
            'active_gtts_count': 0,
            'plan_entries_count': 0,
            'actions': [],
            'execution_results': {},
            'duration_seconds': 0,
            'success': False
        }

        try:
            # 1. Fetch active GTTs
            active_gtts = self.fetch_active_gtts()
            results['active_gtts_count'] = len(active_gtts)

            # 2. Parse new plan
            new_plan = self.parse_gtt_plan(new_plan_df)
            results['plan_entries_count'] = len(new_plan)

            # 3. Compare and determine actions
            actions = self.compare_plans(active_gtts, new_plan)
            results['actions'] = [action.to_dict() for action in actions]

            # 4. Execute actions
            execution_results = self.execute_actions(actions)
            results['execution_results'] = execution_results

            # 5. Calculate summary
            results['duration_seconds'] = (datetime.now() - start_time).total_seconds()
            results['success'] = len(execution_results['failed']) == 0

            logger.info(f"✅ GTT reconciliation completed in {results['duration_seconds']:.1f}s")
            logger.info(f"   Active GTTs: {results['active_gtts_count']}")
            logger.info(f"   Plan entries: {results['plan_entries_count']}")
            logger.info(f"   Actions taken: {len(execution_results['cancelled'])} cancelled, {len(execution_results['modified'])} modified, {len(execution_results['placed'])} placed")

            return results

        except Exception as e:
            logger.error(f"GTT reconciliation failed: {e}")
            results['error'] = str(e)
            results['duration_seconds'] = (datetime.now() - start_time).total_seconds()
            return results

    def get_reconciliation_report(self, results: Dict) -> str:
        """Generate a human-readable reconciliation report."""
        report = []
        report.append("📊 GTT Order Reconciliation Report")
        report.append("=" * 50)
        report.append(f"Start Time: {results['start_time']}")
        report.append(f"Duration: {results['duration_seconds']:.1f} seconds")
        report.append(f"Status: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}")
        report.append("")

        exec_results = results['execution_results']
        report.append("📋 Summary:")
        report.append(f"  Active GTTs found: {results['active_gtts_count']}")
        report.append(f"  Plan entries: {results['plan_entries_count']}")
        report.append(f"  Cancelled: {len(exec_results.get('cancelled', []))}")
        report.append(f"  Modified: {len(exec_results.get('modified', []))}")
        report.append(f"  Placed: {len(exec_results.get('placed', []))}")
        report.append(f"  Failed: {len(exec_results.get('failed', []))}")
        report.append("")

        if exec_results.get('cancelled'):
            report.append("🗑️  Cancelled GTTs:")
            for item in exec_results['cancelled']:
                report.append(f"  • {item['symbol']} (ID: {item['order_id']})")
            report.append("")

        if exec_results.get('modified'):
            report.append("✏️  Modified GTTs:")
            for item in exec_results['modified']:
                old_entry = item.get('old_params', {}).get('entry_price', 0)
                new_entry = item.get('new_params', {}).get('entry_price', 0)
                report.append(f"  • {item['symbol']}: ₹{old_entry} → ₹{new_entry}")
            report.append("")

        if exec_results.get('placed'):
            report.append("📤 Placed New GTTs:")
            for item in exec_results['placed']:
                entry_price = item.get('params', {}).get('entry_price', 0)
                report.append(f"  • {item['symbol']}: Entry ₹{entry_price}")
            report.append("")

        if exec_results.get('failed'):
            report.append("❌ Failed Actions:")
            for item in exec_results['failed']:
                report.append(f"  • {item.get('action', {}).get('symbol', 'UNKNOWN')}: {item.get('error', 'Unknown error')}")
            report.append("")

        return "\n".join(report)


def reconcile_gtt_orders(plan_df: pd.DataFrame, access_token: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to reconcile GTT orders.

    Args:
        plan_df: DataFrame with new GTT plan
        access_token: Upstox API access token
        config: Optional configuration

    Returns:
        Reconciliation results
    """
    reconciler = GTTReconciler(access_token, config)
    return reconciler.reconcile_gtt_orders(plan_df)