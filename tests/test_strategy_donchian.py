import pandas as pd
import numpy as np
from src.strategies.donchian import signal


def make_donchian_breakout_df(n=60):
    # Build data where the last close is above the previous 20-highs
    base = np.linspace(100, 110, n)
    high = base + np.random.uniform(0, 0.2, n)
    low = base - np.random.uniform(0, 0.2, n)
    close = base.copy()
    # Make final bar a clear breakout
    close[-1] = close[-2] * 1.03
    vol = np.ones(n) * 1000
    vol[-1] = 2000  # volume spike
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')
    return pd.DataFrame({'Date': dates, 'Open': close, 'High': high, 'Low': low, 'Close': close, 'Volume': vol})


def test_donchian_breakout_flag_triggers():
    df = make_donchian_breakout_df(80)
    res = signal(df)
    assert 'Donchian_Breakout' in res.columns
    assert int(res.iloc[-1]['Donchian_Breakout']) == 1
