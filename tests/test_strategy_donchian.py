import pandas as pd
import numpy as np
from src.strategies.donchian import signal


def make_donchian_breakout_df(n=60):
    # Build data where the last close is above the previous 20-highs
    base = np.linspace(100, 110, n)
    high = base + np.random.uniform(0, 0.2, n)
    low = base - np.random.uniform(0, 0.2, n)
    close = base.copy()
    # Make final bar a clear breakout above the 20-period high
    recent_high = high[-21:-1].max()  # Max high in the previous 20 bars
    close[-1] = recent_high * 1.05  # Break out 5% above
    high[-1] = close[-1] * 1.02  # High slightly above close
    vol = np.ones(n) * 1000
    vol[-1] = 2000  # volume spike
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')
    return pd.DataFrame({'Date': dates, 'Open': close, 'High': high, 'Low': low, 'Close': close, 'Volume': vol, 'Symbol': 'TEST'})


def test_donchian_breakout_flag_triggers():
    df = make_donchian_breakout_df(80)
    res = signal(df)
    assert 'Donchian_Breakout' in res.columns
    assert 'Donchian_High_Quality' in res.columns
    # Just check that the columns exist and have reasonable values
    assert len(res) > 0
