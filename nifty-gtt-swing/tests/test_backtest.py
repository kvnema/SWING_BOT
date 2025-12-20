import pandas as pd
from src.backtest import backtest_strategy


def make_fixture():
    d = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=50),
        'Close': list(range(100,150)),
        'ATR14': [1.0]*50,
        'SEPA_Flag': [0]*49 + [1]
    })
    return d


def test_backtest_basic():
    df = make_fixture()
    cfg = {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0, 'stop_multiple_atr': 1.5}, 'backtest': {'transaction_cost_pct': 0.001}}
    res = backtest_strategy(df, 'SEPA_Flag', cfg)
    assert 'kpi' in res
