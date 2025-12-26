import pandas as pd
from src.gtt_sizing import build_gtt_plan
from src.gtt_sizing import AuditParams


def test_ensemble_vote_places_gtt_when_conditions_met():
    # Prepare a latest_df row where SEPA and Donchian are true, regime OK and RS leader
    latest = pd.DataFrame([{
        'Symbol': 'ABC',
        'Close': 100.0,
        'DonchianH20': 101.0,
        'ATR14': 1.0,
        'RSI14': 50,
        'RSI14_Status': 'Neutral',
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'GoldenBull_Date': '',
        'SEPA_Flag': 1,
        'VCP_Flag': 0,
        'Donchian_Breakout': 1,
        'BBKC_Squeeze_Flag': 0,
        'SqueezeBreakout_Flag': 0,
        'TS_Momentum_Flag': 1,
        'Trend_OK': 1,
        'RS_Leader_Flag': 1,
        'RSI_MACD_Confirm_D': True  # Added for confirmation requirement
    }])

    cfg = {'risk': {'stop_multiple_atr': 1.5, 'equity': 100000, 'risk_per_trade_pct': 1.0},
           'decision': {'min_confidence': 0.7, 'whitelist': []}}

    # success_model stub: include expected bucket columns to avoid lookup KeyErrors
    # include all expected aggregate columns to satisfy lookup_confidence's backoff aggregates
    success_model = pd.DataFrame([{
        'Strategy': 'Donchian_Breakout',
        'RSI14_Status': 'Neutral',
        'GoldenBull_Flag': 1,
        'GoldenBear_Flag': 0,
        'Trend_OK': 1,
        'RVOL20_bucket': '>1.5',
        'ATRpct_bucket': '1–2%',
        'CalibratedWinRate': 0.9,
        'CI_low': 0.8,
        'CI_high': 0.95,
        'OOS_WinRate_raw': 0.9,
        'OOS_ExpectancyR': 0.5,
        'Trades_OOS': 50,
        'Wins_OOS': 45,
        'Weighted_Wins': 45.0,
        'CoverageNote': 'SymbolExact',
        'Reliability': 1.0
    }])

    instrument_map = {'ABC': '12345'}
    ap = AuditParams()

    plan = build_gtt_plan(latest, 'Donchian_Breakout', cfg, instrument_map, success_model=success_model, indicators_df=latest, audit_params=ap)
    assert not plan.empty
    row = plan.iloc[0]
    assert bool(row['PlaceGTT']) is True, f"Expected PlaceGTT True, got {row['PlaceGTT']}, reasons: {row.get('NoTradeReason')}"
