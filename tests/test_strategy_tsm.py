import pandas as pd
import numpy as np
from src.strategies.tsm import signal


def make_tsm_df(n=260):
    # create increasing price so that 12-month lookback is positive
    close = np.linspace(50, 80, n)
    high = close + np.random.uniform(0, 0.5, n)
    low = close - np.random.uniform(0, 0.5, n)
    vol = np.ones(n) * 1000
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')
    return pd.DataFrame({'Date': dates, 'Open': close, 'High': high, 'Low': low, 'Close': close, 'Volume': vol})


def test_tsm_flag_true_for_positive_12m_return():
    df = make_tsm_df(270)
    res = signal(df)
    assert 'TS_Momentum_Flag' in res.columns
    # Last row should have positive 12-month momentum
    assert int(res.iloc[-1]['TS_Momentum_Flag']) == 1
