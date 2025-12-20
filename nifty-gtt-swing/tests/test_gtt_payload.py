from src.gtt_sizing import build_gtt_plan
import pandas as pd


def test_gtt_plan_builds():
    df = pd.DataFrame([{ 'Symbol': 'RELIANCE', 'Close': 2500.0, 'DonchianH20': 2520.0, 'EMA20': 2490.0, 'ATR14': 10.0 }])
    plan = build_gtt_plan(df, 'Donchian_Breakout', {'risk': {'equity':100000, 'risk_per_trade_pct':1.0, 'stop_multiple_atr':1.5}}, {'RELIANCE':'NSE_EQ|INE002A01018'})
    assert not plan.empty
