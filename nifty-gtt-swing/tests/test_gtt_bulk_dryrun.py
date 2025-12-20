import pandas as pd
from src.upstox_gtt import place_gtt_bulk


def test_bulk_dryrun_plan():
    plan = [
        {'Symbol': 'RELIANCE', 'InstrumentToken': 'NSE_EQ|INE002A01018', 'Qty': 1, 'ENTRY_trigger_type': 'ABOVE', 'ENTRY_trigger_price': 2500.0, 'STOPLOSS_trigger_price': 2485.0, 'TARGET_trigger_price': 2540.0, 'Strategy': 'Donchian_Breakout'},
        {'Symbol': 'INFY', 'InstrumentToken': 'NSE_EQ|INE009A01021', 'Qty': 1, 'ENTRY_trigger_type': 'ABOVE', 'ENTRY_trigger_price': 1500.0, 'STOPLOSS_trigger_price': 1490.0, 'TARGET_trigger_price': 1520.0, 'Strategy': 'Donchian_Breakout'},
    ]
    cfg = {'gtt': {'default_product': 'D', 'trailing_sl_enable': False}}
    res = place_gtt_bulk('DUMMY', plan, cfg, dry_run=True, rate_limit_sleep=0.0, per_symbol_retries=1, backoff=0.01)
    assert len(res) == 2
    assert res[0]['response']['status_code'] == 0 or res[0].get('status_code') == 0
