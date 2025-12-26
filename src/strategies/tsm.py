"""Time-Series Momentum filter (TSM)."""
import pandas as pd
from ..indicators import compute_all_indicators


def signal(df: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    d = compute_all_indicators(df)
    # approx trading days per year 252; months ~ 21 trading days
    lookback_days = int(months * 21)
    # Momentum: positive total return over lookback
    d['TS_Momentum'] = (d['Close'] / d['Close'].shift(lookback_days) - 1).fillna(0)
    d['TS_Momentum_Flag'] = (d['TS_Momentum'] > 0).astype(int)
    return d[['TS_Momentum', 'TS_Momentum_Flag']]
