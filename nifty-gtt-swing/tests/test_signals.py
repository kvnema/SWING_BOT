import pandas as pd
from src.signals import compute_signals


def make_fixture():
    d = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=10),
        'Symbol': ['AAA']*10,
        'Open': range(10,20),
        'High': range(11,21),
        'Low': range(9,19),
        'Close': range(10,20),
        'Volume': [100]*10,
        'EMA20': [12]*10,
        'EMA50': [11]*10,
        'EMA200': [10]*10,
        'RSI14': [30]*10,
        'MACD': [0]*10,'MACDSignal':[0]*10,'MACDHist':[0]*10,
        'ATR14': [1]*10,
        'BB_MA20':[12]*10,'BB_Upper':[13]*10,'BB_Lower':[11]*10,'BB_BandWidth':[0.1]*10,
        'RVOL20':[1.6]*10,'DonchianH20':[15]*10,'DonchianL20':[5]*10,'RS_vs_Index':[1]*10,'RS_ROC20':[5]*10,
        'KC_Upper':[14]*10,'KC_Lower':[8]*10,'Squeeze':[1]*10,'AVWAP60':[11]*10,'IndexUpRegime':[1]*10
    })
    return d


def test_vcp_structural():
    df = make_fixture()
    # Set up VCP conditions
    df['BB_BandWidth'] = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]  # contracting
    df['Low'] = [9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5]  # higher lows
    df['Volume'] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]  # dry-up
    df['EMA200'] = [8] * 10  # pivot
    out = compute_signals(df)
    assert 'VCP_Flag' in out.columns
    # Check if flag is set (depends on exact conditions)


def test_bb_kc_squeeze():
    df = make_fixture()
    df['BB_Upper'] = [12.5] * 10
    df['BB_Lower'] = [11.5] * 10
    df['KC_Upper'] = [13] * 10
    df['KC_Lower'] = [11] * 10  # BB inside KC
    out = compute_signals(df)
    assert 'BBKC_Squeeze_Flag' in out.columns


def test_rsi_status():
    from src.signals import rsi_status
    rsi = pd.Series([25, 50, 85])
    status = rsi_status(rsi)
    assert status.iloc[0] == "Oversold (≤30)"
    assert status.iloc[1] == "Neutral (30–70)"
    assert status.iloc[2] == "Overbought (≥70)"


def test_golden_crossover():
    df = make_fixture()
    # Set EMA50 crossing above EMA200
    df['EMA50'] = [9, 9.5, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14]  # crosses at index 2
    df['EMA200'] = [10] * 10
    out = compute_signals(df)
    assert 'GoldenBull_Flag' in out.columns
    assert 'GoldenBear_Flag' in out.columns
    assert 'GoldenBull_Date' in out.columns
    assert 'GoldenBear_Date' in out.columns
    # Check flag at index 2 (0-based, but since shift(1), at t=2 if EMA50[2] >= EMA200[2] and EMA50[1] < EMA200[1]
    # EMA50[1]=9.5 <10, EMA50[2]=10.5 >=10, so flag at 2
    assert out['GoldenBull_Flag'].iloc[2] == 1
    # Date should be the latest, which is 2025-01-03
    assert out['GoldenBull_Date'].iloc[0] == '2025-01-03'
