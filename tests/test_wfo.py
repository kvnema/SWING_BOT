import pytest
import pandas as pd
from src.wfo import walk_forward_optimization


def test_wfo_rolling_vs_anchored():
    # Mock data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Symbol': 'TEST',
        'Close': [100 + i * 0.1 for i in range(1000)],
        'High': [101 + i * 0.1 for i in range(1000)],
        'Low': [99 + i * 0.1 for i in range(1000)],
        'EMA20': [100 + i * 0.1 for i in range(1000)],
        'EMA50': [100 + i * 0.1 for i in range(1000)],
        'EMA200': [100 + i * 0.1 for i in range(1000)],
        'RSI14': 50,
        'ATR14': 1,
        'RVOL20': 1.5,
        'IndexUpRegime': 1,
        'SEPA_Flag': 1,
        'BB_BandWidth': 0.1,
        'MACD': 0,
        'MACDSignal': 0,
        'MACDHist': 0,
        'BB_MA20': 100,
        'BB_Upper': 101,
        'BB_Lower': 99,
        'DonchianH20': 105,
        'DonchianL20': 95,
        'RS_vs_Index': 1,
        'RS_ROC20': 5,
        'KC_Upper': 102,
        'KC_Lower': 98,
        'Squeeze': 0,
        'AVWAP60': 100,
        'Volume': 1000
    })
    cfg = {'risk': {'equity': 100000, 'risk_per_trade_pct': 1.0, 'stop_multiple_atr': 1.5}}

    # Rolling
    result_rolling = walk_forward_optimization(df, 'SEPA', 'SEPA_Flag', cfg, cycles=2, mode='rolling')
    assert 'oos_aggregate' in result_rolling
    assert result_rolling['mode'] == 'rolling'

    # Anchored
    result_anchored = walk_forward_optimization(df, 'SEPA', 'SEPA_Flag', cfg, cycles=2, mode='anchored')
    assert result_anchored['mode'] == 'anchored'
    assert len(result_anchored['oos_curve']) == 2