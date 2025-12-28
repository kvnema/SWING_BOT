import pandas as pd
import numpy as np

# Read the screener data
df = pd.read_csv('outputs/screener/screener_latest.csv')

# Filter for our top stocks
top_stocks = ['TITAN', 'LT', 'ASIANPAINT', 'TATAMOTORS', 'BAJAJFINSV', 'UPL']
analysis_df = df[df['Symbol'].isin(top_stocks)].copy()

# Calculate risk/reward setups
results = []
for _, row in analysis_df.iterrows():
    symbol = row['Symbol']
    current_price = row['Close']
    atr = row['ATR14']

    # Entry triggers based on technicals
    if symbol == 'TITAN':
        entry_trigger = f'Close above {row["DonchianH20"]:.1f} (Donchian high) with volume confirmation'
        stop_level = row['DonchianL20'] - (atr * 0.5)
        # Better target: measured move based on Donchian range
        donchian_range = row['DonchianH20'] - row['DonchianL20']
        initial_target = current_price + donchian_range
        risk = current_price - stop_level
        reward = initial_target - current_price
    elif symbol == 'LT':
        entry_trigger = f'BB squeeze breakout above {current_price + (atr * 0.5):.1f} with MR confirmation'
        stop_level = row['DonchianL20'] - atr
        # Better target: measured move based on Donchian range
        donchian_range = row['DonchianH20'] - row['DonchianL20']
        initial_target = current_price + donchian_range
        risk = current_price - stop_level
        reward = initial_target - current_price
    elif symbol == 'ASIANPAINT':
        entry_trigger = f'EMA20 reclaim above {row["EMA20"]:.1f} with volume > 200% RVOL'
        stop_level = row['DonchianL20'] - (atr * 0.3)
        initial_target = row['DonchianH20'] + (row['DonchianH20'] - row['DonchianL20'])
        risk = current_price - stop_level
        reward = initial_target - current_price
    elif symbol == 'TATAMOTORS':
        entry_trigger = f'EMA20 reclaim above {row["EMA20"]:.1f} after volume dry-up resolves'
        stop_level = row['DonchianL20'] - (atr * 1.5)
        initial_target = row['DonchianH20']
        risk = current_price - stop_level
        reward = initial_target - current_price
    elif symbol == 'BAJAJFINSV':
        entry_trigger = f'BB/KC squeeze resolution above {current_price + atr:.1f}'
        stop_level = row['DonchianL20'] - atr
        initial_target = row['DonchianH20'] + (row['DonchianH20'] - row['DonchianL20'])
        risk = current_price - stop_level
        reward = initial_target - current_price
    elif symbol == 'UPL':
        entry_trigger = f'Pivot break above recent high {current_price + (atr * 0.7):.1f}'
        stop_level = row['DonchianL20'] - atr
        initial_target = row['DonchianH20']
        risk = current_price - stop_level
        reward = initial_target - current_price

    rr_ratio = reward / risk if risk > 0 else 0

    # Position sizing (assuming ₹5 lakh portfolio, 1% risk)
    portfolio = 500000
    risk_amount = portfolio * 0.01
    position_size = int(risk_amount / risk) if risk > 0 else 0
    position_value = position_size * current_price

    results.append({
        'Symbol': symbol,
        'Current_Price': current_price,
        'Entry_Trigger': entry_trigger,
        'Stop_Level': f'{stop_level:.2f}',
        'Initial_Target': f'{initial_target:.2f}',
        'Risk_per_Share': f'{risk:.2f}',
        'Reward_per_Share': f'{reward:.2f}',
        'RR_Ratio': f'{rr_ratio:.2f}:1',
        'Position_Size': position_size,
        'Position_Value': f'Rs.{position_value:,.0f}',
        'Risk_Amount': f'Rs.{risk_amount:,.0f}'
    })

# Print results
for result in results:
    print(f'\n=== {result["Symbol"]} Risk/Reward Setup ===')
    for key, value in result.items():
        if key != 'Symbol':
            print(f'{key}: {value}')