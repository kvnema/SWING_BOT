import pandas as pd
import numpy as np


def rolling_pctile(series: pd.Series, window: int, p: float) -> pd.Series:
    return series.rolling(window).apply(lambda x: np.nanpercentile(x, p), raw=True)


def slope(series: pd.Series, period: int) -> pd.Series:
    return series.diff(period)


def higher_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    # simple check: last low > min low of previous lookback-1 bars
    lows = df['Low']
    return lows > lows.shift(1).rolling(lookback-1).min()


def rsi_status(rsi: pd.Series) -> pd.Series:
    """Return RSI status bucket."""
    return pd.cut(rsi, bins=[-np.inf, 30, 70, np.inf], labels=["Oversold", "Neutral", "Overbought"], right=True)


def minervini_trend_template(df: pd.DataFrame) -> pd.Series:
    """8-point Minervini trend template compliance."""
    close = df['Close']
    ema150 = close.ewm(span=150, adjust=False).mean()
    ema200 = df['EMA200']
    sma200 = close.rolling(200).mean()
    return (
        (close > ema150) &
        (ema150 > ema200) &
        (ema200 > sma200) &
        (close > sma200) &
        (close > close.rolling(50).max().shift(1)) &  # 52-week high
        (close > close.rolling(10).mean()) &  # above 10MA
        (df['RSI14'] > 40) &  # RSI > 40
        (df['RSI14'] < 70)    # RSI < 70
    ).astype(int)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strategy flags per spec. Assumes df has required columns and sorted by Date."""
    from .indicators import compute_all_indicators  # Import here to avoid circular import
    d = df.copy()
    # ensure datetime
    d['Date'] = pd.to_datetime(d['Date'])

    # Compute all indicators if not present
    d = compute_all_indicators(d)

    # Trend_OK
    d['Trend_OK'] = ((d['Close'] > d['EMA20']) & (d['EMA20'] > d['EMA50']) & (d['EMA50'] > d['EMA200'])).astype(int)

    # Minervini trend template
    d['Minervini_Trend'] = minervini_trend_template(d)

    # rolling60 p20 of BB_BandWidth
    d['BB_bw_p20_60'] = rolling_pctile(d['BB_BandWidth'], 60, 20)

    # Enhanced SEPA_Flag: 8-point trend template + tight base + breakout
    d['SEPA_Flag'] = (
        (d['Minervini_Trend'] == 1) &
        (d['BB_BandWidth'] <= d['BB_bw_p20_60']) &
        (d['Close'] > d['DonchianH20']) &
        (d['RVOL20'] >= 1.5)
    ).astype(int)

    # Enhanced VCP_Flag: progressive contraction + volume dry-up + pivot break
    d['BB_slope20'] = slope(d['BB_BandWidth'], 1).rolling(20).mean()
    d['HL_5'] = higher_lows(d, lookback=5)
    d['Volume_Dry_Up'] = d['Volume'] < d['Volume'].rolling(20).mean() * 0.8  # volume below 80% of 20MA
    d['Pivot_Break'] = d['Close'] > d['EMA200']  # simple pivot as EMA200
    d['VCP_Flag'] = (
        (d['BB_slope20'] < 0) &  # contracting
        (d['HL_5']) &  # higher lows
        (d['Volume_Dry_Up']) &  # volume dry-up
        (d['Pivot_Break']) &  # breakout above pivot
        (d['RVOL20'] >= 1.5)  # volume confirmation
    ).astype(int)

    # Parameterized Donchian_Breakout: window=20, optional volume, middle-line pullback
    donchian_window = 20
    d['DonchianH20'] = d['High'].rolling(donchian_window).max()
    d['DonchianL20'] = d['Low'].rolling(donchian_window).min()
    d['Donchian_Mid'] = (d['DonchianH20'] + d['DonchianL20']) / 2
    d['Pullback_to_Mid'] = (d['Close'] < d['Donchian_Mid']) & (d['Close'] > d['DonchianL20'])
    d['Donchian_Breakout'] = (
        ((d['Close'] > d['DonchianH20']) | (d['Pullback_to_Mid'] & (d['Close'] > d['Donchian_Mid']))) &
        (d['RVOL20'] >= 1.5)
    ).astype(int)

    # MR_Flag (mean reversion in uptrend)
    d['MR_Flag'] = ((d['Trend_OK'] == 1) & (d['RSI14'] <= 35) & ((d['Close'] - d['EMA20']).abs() / d['Close'] <= 0.03)).astype(int)

    # BBKC_Squeeze_Flag: Bollinger inside Keltner -> breakout
    d['BB_Inside_KC'] = (d['BB_Upper'] < d['KC_Upper']) & (d['BB_Lower'] > d['KC_Lower'])
    d['BBKC_Squeeze_Flag'] = (
        (d['BB_Inside_KC']) &
        (d['Close'] > d['KC_Upper']) &  # breakout above KC
        (d['RVOL20'] >= 1.5)
    ).astype(int)

    # SqueezeBreakout_Flag (alias for BBKC)
    d['SqueezeBreakout_Flag'] = d['BBKC_Squeeze_Flag']

    # AVWAP_Reclaim_Flag
    d['AVWAP_Reclaim_Flag'] = ((d['Trend_OK'] == 1) & (d['Close'] > d['AVWAP60'])).astype(int)

    # RS_Leader_Flag: RS_ROC20 > 0 & >= rolling60 p80
    d['RS_p80_60'] = rolling_pctile(d['RS_ROC20'].fillna(0), 60, 80)
    d['RS_Leader_Flag'] = ((d['RS_ROC20'] > 0) & (d['RS_ROC20'] >= d['RS_p80_60'])).astype(int)

    # TS_Momentum: 12-month lookback, monthly rebalance
    d['TS_Momentum'] = (d['Close'] / d['Close'].shift(252) - 1).fillna(0)  # approx 252 trading days
    d['TS_Momentum_Flag'] = (d['TS_Momentum'] > 0).astype(int)  # simple positive momentum

    # RSI/MACD confirmation pack (per TF)
    d['RSIConfirm_D'] = d['RSI_Above50'] & ~d['RSI_Overbought']
    d['MACDConfirm_D'] = d['MACD_CrossUp'] & d['MACD_AboveZero']
    d['MomentumConfirm_D'] = d['MACD_Hist_Rising']
    # For H4, relax to: RSI_Above50_H4 OR MACD_CrossUp_H4
    d['RSIConfirm_H4'] = d.get('RSI_Above50_H4', d['RSI_Above50'])  # fallback to D if H4 not available
    d['MACDConfirm_H4'] = d.get('MACD_CrossUp_H4', d['MACD_CrossUp'])  # fallback
    d['RSI_MACD_Confirm_H4'] = d['RSIConfirm_H4'] | d['MACDConfirm_H4']
    # Overall confirmation: D strict + H4 relaxed
    d['RSI_MACD_Confirm_D'] = d['RSIConfirm_D'] & d['MACDConfirm_D'] & d['MomentumConfirm_D']
    d['RSI_MACD_Confirm_OK'] = d['RSI_MACD_Confirm_D'] & d['RSI_MACD_Confirm_H4']

    # Regime gate: zero out long flags when IndexUpRegime==0
    long_flags = ['SEPA_Flag', 'VCP_Flag', 'Donchian_Breakout', 'MR_Flag', 'BBKC_Squeeze_Flag', 'SqueezeBreakout_Flag', 'AVWAP_Reclaim_Flag', 'TS_Momentum_Flag']
    if 'IndexUpRegime' in d.columns:
        for f in long_flags:
            d.loc[d['IndexUpRegime'] != 1, f] = 0

    # Add RSI status
    d['RSI14_Status'] = rsi_status(d['RSI14'])

    # Golden Crossover flags
    d['GoldenBull_Flag'] = ((d['EMA50'] >= d['EMA200']) & (d['EMA50'].shift(1) < d['EMA200'].shift(1))).astype(int)
    d['GoldenBear_Flag'] = ((d['EMA50'] <= d['EMA200']) & (d['EMA50'].shift(1) > d['EMA200'].shift(1))).astype(int)

    # Latest crossover dates per symbol
    bull_dates = d[d['GoldenBull_Flag'] == 1].groupby('Symbol')['Date'].max().dt.strftime('%Y-%m-%d')
    bear_dates = d[d['GoldenBear_Flag'] == 1].groupby('Symbol')['Date'].max().dt.strftime('%Y-%m-%d')
    d = d.merge(bull_dates.rename('GoldenBull_Date'), left_on='Symbol', right_index=True, how='left').fillna({'GoldenBull_Date': ''})
    d = d.merge(bear_dates.rename('GoldenBear_Date'), left_on='Symbol', right_index=True, how='left').fillna({'GoldenBear_Date': ''})

    # return only flags appended
    return d
