import pandas as pd
import os
from src.upstox_gtt import place_gtt_order_multi, get_all_gtt_orders

def place_orders_from_reconciled_plan(plan_csv_path: str) -> list:
    '''Place GTT orders for positions that have successful LTP reconciliation'''

    # Load reconciled plan
    plan_df = pd.read_csv(plan_csv_path)
    access_token = os.environ.get('UPSTOX_ACCESS_TOKEN')

    if not access_token:
        print('❌ UPSTOX_ACCESS_TOKEN not found')
        return []

    placed_orders = []

    # Get existing GTT orders to check for duplicates
    existing_gtt_response = get_all_gtt_orders(access_token)
    if existing_gtt_response.get('status_code') == 200:
        existing_orders = existing_gtt_response.get('body', {}).get('data', [])
        active_symbols = {order.get('instrument_token') for order in existing_orders if order.get('status') == 'active'}
    else:
        print('⚠️ Could not fetch existing GTT orders; proceeding without duplicate check')
        active_symbols = set()

    # Filter for positions with successful LTP reconciliation (have LTP data)
    reconciled_positions = plan_df[plan_df['LTP'].notna() & (plan_df['LTP'] > 0)]

    print(f'📤 Placing GTT orders for {len(reconciled_positions)} positions with LTP data...')

    for _, row in reconciled_positions.iterrows():
        try:
            symbol = row['Symbol']
            entry_price = round(row['Reconciled_Entry'], 2)
            stop_price = round(row['Reconciled_Stop'], 2)
            target_price = round(row['Reconciled_Target'], 2)
            instrument_token = str(row['InstrumentToken_x'])
            qty = int(row['Qty'])

            print(f'📋 {symbol}: Entry={entry_price}, Stop={stop_price}, Target={target_price}, Qty={qty}')

            # Check if GTT already exists for this instrument
            if instrument_token in active_symbols:
                print(f'⏭️ {symbol}: Skipping - active GTT already exists')
                continue

            # Build GTT rules
            rules = [
                {
                    'strategy': 'ENTRY',
                    'trigger_type': row['ENTRY_trigger_type'],
                    'trigger_price': entry_price
                },
                {
                    'strategy': 'STOPLOSS',
                    'trigger_type': 'IMMEDIATE',
                    'trigger_price': stop_price
                },
                {
                    'strategy': 'TARGET',
                    'trigger_type': 'IMMEDIATE',
                    'trigger_price': target_price
                }
            ]

            # Place the order
            result = place_gtt_order_multi(
                instrument_token=instrument_token,
                quantity=qty,
                product='D',
                rules=rules,
                transaction_type='BUY',
                access_token=access_token,
                dry_run=False,
                retries=3,
                backoff=1.0
            )
            
            if result.get('status_code') in (200, 201, 202):
                order_record = {
                    'Symbol': symbol,
                    'order_id': result.get('order_id', 'UNKNOWN'),
                    'Entry': entry_price,
                    'Stop': stop_price,
                    'Target': target_price,
                    'Quantity': qty,
                    'status': 'PLACED'
                }
                placed_orders.append(order_record)
                order_id = result.get('order_id', 'UNKNOWN')
                print(f'✅ {symbol}: GTT placed (ID: {order_id})')
            else:
                error_body = result.get('body', {})
                print(f'❌ {symbol}: GTT failed - {error_body}')

        except Exception as e:
            print(f'❌ {symbol}: Exception - {str(e)}')
            continue

    return placed_orders

if __name__ == '__main__':
    # Place orders
    orders = place_orders_from_reconciled_plan('outputs/gtt/gtt_plan_reconciled.csv')
    print(f'\n📊 Order placement complete. {len(orders)} orders placed.')
    for order in orders:
        print(f'  {order["Symbol"]}: {order["status"]} (ID: {order["order_id"]})')