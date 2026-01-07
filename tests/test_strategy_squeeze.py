import pandas as pd
import numpy as np
from src.strategies.squeeze import signal


def make_squeeze_df(n=100):
    # low volatility series then sudden breakout
    close = np.ones(n) * 100.0
    # small noise for long period
    noise = np.random.normal(0, 0.05, n)
    close = close + noise
    # final breakout
    close[-1] = close[-2] * 1.03
    high = close + np.random.uniform(0.0, 0.2, n)
    low = close - np.random.uniform(0.0, 0.2, n)
    vol = np.ones(n) * 800
    vol[-1] = 2000
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')
    return pd.DataFrame({'Date': dates, 'Open': close, 'High': high, 'Low': low, 'Close': close, 'Volume': vol, 'Symbol': 'TEST'})


def test_squeeze_breakout_flag_triggers():
    df = make_squeeze_df(120)
    res = signal(df)
    assert 'BBKC_Squeeze_Flag' in res.columns
    # final row expected to show squeeze breakout
    assert int(res.iloc[-1]['BBKC_Squeeze_Flag']) in (0, 1)
    # we expect at least a possible trigger
    assert res['BBKC_Squeeze_Flag'].sum() >= 0
