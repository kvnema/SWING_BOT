import pandas as pd
import numpy as np
from src.backtest import backtest_strategy, walk_forward_backtest


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


def test_walk_forward_basic():
    # Create longer test data
    dates = pd.date_range('2023-01-01', periods=400, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Close': np.sin(np.arange(400) * 0.1) * 10 + 100,  # Some price movement
        'ATR14': [1.0] * 400,
        'SEPA_Flag': [0] * 395 + [1] * 5  # Few signals at the end
    })

    cfg = {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0, 'stop_multiple_atr': 1.5}, 'backtest': {'transaction_cost_pct': 0.001}}

    res = walk_forward_backtest(df, 'SEPA_Flag', cfg, train_years=1, test_months=3)

    assert 'combined_kpi' in res
    assert 'window_results' in res
    assert isinstance(res['combined_kpi']['Total_Trades'], int)
