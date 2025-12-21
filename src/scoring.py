import pandas as pd
import numpy as np


def clipped_zscore(series: pd.Series, clip: float = 3.0) -> pd.Series:
    s = (series - series.mean()) / (series.std(ddof=0) + 1e-9)
    return s.clip(-clip, clip)


def compute_composite_score(df: pd.DataFrame) -> pd.Series:
    # weights per spec (partial)
    rs = clipped_zscore(df['RS_ROC20'].fillna(0)) * 25
    rvol = clipped_zscore(df['RVOL20'].fillna(0)) * 20
    trend = df['Trend_OK'].fillna(0) * 15
    breakout = df['Donchian_Breakout'].fillna(0) * 15
    # combine
    score = rs + rvol + trend + breakout
    # normalize to 0-100
    mn, mx = score.min(), score.max()
    if mx - mn < 1e-6:
        return pd.Series(50, index=df.index)
    scaled = 100 * (score - mn) / (mx - mn)
    return scaled.round(2)
