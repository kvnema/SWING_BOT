import pandas as pd
from src.gtt_sizing import build_gtt_plan, build_upstox_payload
from src.upstox_gtt import place_gtt_order_payload


def test_build_and_dryrun_payload():
    df = pd.DataFrame([{ 'Symbol': 'RELIANCE', 'Close': 2500.0, 'DonchianH20': 2520.0, 'EMA20': 2490.0, 'ATR14': 10.0, 'Qty': 10, 'InstrumentToken':'NSE_EQ|INE002A01018', 'ENTRY_trigger_type':'ABOVE', 'ENTRY_trigger_price':2520.0, 'STOPLOSS_trigger_price':2505.0, 'TARGET_trigger_price':2540.0 }])
    row = df.iloc[0].to_dict()
    payload = build_upstox_payload(row, {'gtt': {'default_product': 'D', 'trailing_sl_enable': False}})
    assert 'type' in payload  # Type field required for multi-leg GTT
    assert payload['type'] == 'MULTIPLE'
    resp = place_gtt_order_payload('DUMMY', payload, dry_run=True)
    assert resp['status_code'] == 0
    assert resp['body']['instrument_token'] == 'NSE_EQ|INE002A01018'
