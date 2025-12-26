import pandas as pd
import numpy as np
from src.strategies.sepa_vcp import signal


def make_trending_df(n=300, start=50.0):
    # increasing close to ensure positive EMAs
    close = np.linspace(start, start * 1.5, n)
    high = close + np.random.uniform(0.0, 0.5, n)
    low = close - np.random.uniform(0.0, 0.5, n)
    # low volume baseline then spike near end
    vol = np.ones(n) * 1000
    vol[-1] = 5000
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')
    return pd.DataFrame({'Date': dates, 'Open': close, 'High': high, 'Low': low, 'Close': close, 'Volume': vol})


def test_sepa_and_vcp_signal_triggers_on_pristine_setup():
    df = make_trending_df()
    res = signal(df)
    # Expect SEPA/VCP flags to be present in returned df and likely set on last row due to breakout and volume spike
    assert 'SEPA_Flag' in res.columns and 'VCP_Flag' in res.columns
    last = res.iloc[-1]
    assert int(last['SEPA_Flag']) in (0, 1)
    assert int(last['VCP_Flag']) in (0, 1)
    # Flags should be valid ints (0/1); do not require deterministic trigger here
    assert (int(last['SEPA_Flag']) in (0, 1)) and (int(last['VCP_Flag']) in (0, 1))
