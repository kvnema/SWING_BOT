"""Indicator computations used across strategies.

Provides: ATR, RVOL, Donchian, Bollinger, Keltner, BB bandwidth, BB/KC squeeze flag,
AVWAP (moving), RS_ROC, and simple breadth/regime helpers.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def atr(df: pd.DataFrame, period: int = 14, price_high='High', price_low='Low', price_close='Close') -> pd.Series:
    high = df[price_high]
    low = df[price_low]
    close = df[price_close]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    a = atr(df, period)
    return (a / df['Close']) * 100


def rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Relative volume vs rolling mean volume."""
    return df['Volume'] / (df['Volume'].rolling(window).mean().replace(0, np.nan))


def donchian(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    high = df['High'].rolling(window).max()
    low = df['Low'].rolling(window).min()
    return high, low


def bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    bw = (upper - lower) / sma.replace(0, np.nan)
    return upper, lower, bw


def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 20, atr_mult: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
    atr_series = atr(df, period=atr_period)
    upper = ema + atr_mult * atr_series
    lower = ema - atr_mult * atr_series
    return upper, lower


def bb_inside_kc(df: pd.DataFrame, bb_window: int = 20, bb_nstd: float = 2.0, kc_ema: int = 20, kc_atr: int = 20, kc_mult: float = 1.5) -> pd.Series:
    bb_u, bb_l, _ = bollinger_bands(df, bb_window, bb_nstd)
    kc_u, kc_l = keltner_channels(df, kc_ema, kc_atr, kc_mult)
    return (bb_u < kc_u) & (bb_l > kc_l)


def avwap(df: pd.DataFrame, window: int = 60) -> pd.Series:
    pv = df['Close'] * df['Volume']
    return pv.rolling(window).sum() / df['Volume'].rolling(window).sum()


def rs_roc(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df['Close'].pct_change(window) * 100


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's RSI calculation."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD components: macd_line, signal_line, histogram."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX) calculation."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = pd.Series(np.where((high - high.shift(1)) > (low.shift(1) - low),
                                np.maximum(high - high.shift(1), 0), 0), index=df.index)
    dm_minus = pd.Series(np.where((low.shift(1) - low) > (high - high.shift(1)),
                                 np.maximum(low.shift(1) - low, 0), 0), index=df.index)

    # Smoothed averages
    atr = tr.rolling(period).mean()
    di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(period).mean() / atr)

    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(period).mean()

    return adx


def compute_rsi_macd_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI and MACD gates for the given dataframe."""
    d = df.copy()
    # RSI(14)
    d['RSI14'] = rsi(d, 14)
    d['RSI_Overbought'] = d['RSI14'] >= 70
    d['RSI_Oversold'] = d['RSI14'] <= 30
    d['RSI_Above50'] = d['RSI14'] >= 50
    # MACD(12,26,9)
    d['MACD_Line'], d['MACD_Signal'], d['MACD_Hist'] = macd(d, 12, 26, 9)
    d['MACD_CrossUp'] = (d['MACD_Line'] > d['MACD_Signal']) & (d['MACD_Line'].shift(1) <= d['MACD_Signal'].shift(1))
    d['MACD_AboveZero'] = d['MACD_Line'] > 0
    d['MACD_Hist_Rising'] = d['MACD_Hist'] > d['MACD_Hist'].shift(1)
    return d


def mansfield_rs(df: pd.DataFrame, benchmark_df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Calculate Mansfield Relative Strength vs benchmark

    Args:
        df: Stock data
        benchmark_df: Benchmark index data
        window: Lookback period for RS calculation

    Returns:
        Mansfield RS series
    """
    # Merge stock and benchmark data
    merged = df.merge(benchmark_df, left_index=True, right_index=True,
                     suffixes=('', '_bench'), how='left')

    # Calculate RS ratio
    merged['rs_ratio'] = merged['Close'] / merged['Close_bench']

    # Mansfield RS: smoothed RS ratio
    mansfield_rs = merged['rs_ratio'].rolling(window).mean()

    return mansfield_rs


def enhanced_donchian(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Enhanced Donchian Channel with midline and width

    Args:
        df: OHLC data
        window: Channel period

    Returns:
        Tuple of (high, low, midline, width)
    """
    high = df['High'].rolling(window).max()
    low = df['Low'].rolling(window).min()
    midline = (high + low) / 2
    width = (high - low) / midline.replace(0, np.nan) * 100  # Width as percentage

    return high, low, midline, width


def donchian_breakout_signal(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Donchian breakout signal

    Args:
        df: OHLC data
        window: Channel period

    Returns:
        Breakout signal series (1 for bullish breakout, -1 for bearish)
    """
    high, low, midline, width = enhanced_donchian(df, window)

    # Bullish breakout: close breaks above previous high
    bullish_breakout = (df['Close'] > high.shift(1)) & (df['Close'].shift(1) <= high.shift(1))

    # Bearish breakout: close breaks below previous low
    bearish_breakout = (df['Close'] < low.shift(1)) & (df['Close'].shift(1) >= low.shift(1))

    signal = pd.Series(0, index=df.index)
    signal[bullish_breakout] = 1
    signal[bearish_breakout] = -1

    return signal


def compute_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility-based indicators

    Args:
        df: OHLC data

    Returns:
        DataFrame with volatility indicators
    """
    result_df = df.copy()

    # ATR percent (already exists as atr_pct)
    result_df['ATR_pct'] = atr_pct(df, 14)

    # RVOL (already exists as rvol)
    result_df['RVOL'] = rvol(df, 20)

    # Volatility regime
    result_df['high_vol'] = result_df['ATR_pct'] > result_df['ATR_pct'].rolling(50).mean()
    result_df['low_vol'] = result_df['ATR_pct'] <= result_df['ATR_pct'].rolling(50).mean()

    # Volume confirmation
    result_df['volume_confirm'] = result_df['Volume'] > result_df['Volume'].rolling(20).mean()

    return result_df


def compute_relative_strength(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative strength indicators

    Args:
        df: Stock data
        benchmark_df: Benchmark data

    Returns:
        DataFrame with RS indicators
    """
    result_df = df.copy()

    # Mansfield RS
    result_df['Mansfield_RS'] = mansfield_rs(df, benchmark_df, 60)

    # RS momentum (slope)
    result_df['RS_momentum'] = result_df['Mansfield_RS'].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )

    # RS strength categories
    result_df['RS_strong'] = result_df['Mansfield_RS'] > 1.0
    result_df['RS_weak'] = result_df['Mansfield_RS'] < 1.0
    result_df['RS_rising'] = result_df['RS_momentum'] > 0

    return result_df


def compute_enhanced_donchian_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute enhanced Donchian channels for multiple periods

    Args:
        df: OHLC data

    Returns:
        DataFrame with Donchian indicators
    """
    result_df = df.copy()

    # Multiple Donchian periods
    periods = [20, 55, 89]

    for period in periods:
        high, low, midline, width = enhanced_donchian(df, period)
        result_df[f'Donchian_H_{period}'] = high
        result_df[f'Donchian_L_{period}'] = low
        result_df[f'Donchian_M_{period}'] = midline
        result_df[f'Donchian_W_{period}'] = width

        # Breakout signals
        result_df[f'Donchian_Breakout_{period}'] = donchian_breakout_signal(df, period)

    # Position within channel
    result_df['Donchian_Position'] = (df['Close'] - result_df['Donchian_L_20']) / (result_df['Donchian_H_20'] - result_df['Donchian_L_20'])

    # Channel expansion/contraction
    result_df['Donchian_Expanding'] = result_df['Donchian_W_20'] > result_df['Donchian_W_20'].shift(5)
    result_df['Donchian_Contracting'] = result_df['Donchian_W_20'] < result_df['Donchian_W_20'].shift(5)

    return result_df
def compute_all_indicators(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute all technical indicators

    Args:
        df: Stock data
        benchmark_df: Optional benchmark data for RS calculations

    Returns:
        DataFrame with all indicators
    """
    d = df.copy()

    # Basic indicators
    d['ATR14'] = atr(d, 14)
    d['ATRpct'] = atr_pct(d, 14)
    d['RVOL20'] = rvol(d, 20)
    d['DonchianH20'], d['DonchianL20'] = donchian(d, 20)
    d['DonchianH55'], d['DonchianL55'] = donchian(d, 55)
    d['BB_Upper'], d['BB_Lower'], d['BB_BandWidth'] = bollinger_bands(d, 20, 2.0)
    d['KC_Upper'], d['KC_Lower'] = keltner_channels(d, 20, 20, 1.5)
    d['BB_inside_KC'] = bb_inside_kc(d)
    d['AVWAP60'] = avwap(d, 60)
    d['RS_ROC20'] = rs_roc(d, 20)

    # RSI and MACD gates
    d = compute_rsi_macd_gates(d)

    # Moving averages for regime
    d['EMA20'] = d['Close'].ewm(span=20, adjust=False).mean()
    d['EMA50'] = d['Close'].ewm(span=50, adjust=False).mean()
    d['EMA200'] = d['Close'].ewm(span=200, adjust=False).mean()
    d['SMA200'] = d['Close'].rolling(200).mean()

    # ADX for trend strength
    d['ADX14'] = adx(d, 14)

    # STRONG MARKET REGIME FILTER (Mandatory for safety)
    # Only allow long entries when BOTH conditions met:
    # 1. Index > EMA200/SMA200 (major trend up)
    # 2. ADX > 20 (confirmed trending environment, not sideways)
    # Use comprehensive market data for accurate calculation
    try:
        from .data_fetch import calculate_market_regime
        regime_data = calculate_market_regime('NSE_INDEX|Nifty 50', 400)

        if regime_data['regime_status'] == 'ON':
            d['IndexUpRegime'] = 1
            logger.info("Market regime ON: Trading allowed")
        else:
            d['IndexUpRegime'] = 0
            logger.info(f"Market regime OFF: Holding cash. Reason: {regime_data.get('reason', 'Conditions not met')}")

        # Store regime data for reference
        d['Regime_Status'] = regime_data['regime_status']
        d['Regime_ADX'] = regime_data.get('adx14', 0)
        d['Regime_SMA200'] = regime_data.get('sma200', 0)

    except Exception as e:
        logger.warning(f"Failed to calculate market regime: {e}. Defaulting to OFF for safety.")
        d['IndexUpRegime'] = 0  # Default to OFF for safety
        d['Regime_Status'] = 'ERROR'
        d['Regime_ADX'] = 0
        d['Regime_SMA200'] = 0

    # Enhanced indicators
    d = compute_volatility_indicators(d)
    d = compute_enhanced_donchian_channels(d)

    # Relative strength if benchmark provided
    if benchmark_df is not None and not benchmark_df.empty:
        d = compute_relative_strength(d, benchmark_df)

    # H4 indicators approximation (using daily data)
    h4_frames = []
    for symbol in d['Symbol'].unique():
        df_sym = d[d['Symbol'] == symbol].set_index('Date')

        if df_sym.empty:
            continue

        # Copy daily indicators as H4 approximation
        df_h4 = df_sym.copy()
        df_h4 = compute_rsi_macd_gates(df_h4.reset_index())

        # Take latest bar's indicators
        latest_h4 = df_h4.tail(1)[[
            'RSI14', 'RSI_Overbought', 'RSI_Oversold', 'RSI_Above50',
            'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'MACD_CrossUp',
            'MACD_AboveZero', 'MACD_Hist_Rising'
        ]].add_suffix('_H4')

        latest_h4['Symbol'] = symbol
        latest_h4['Date'] = df_sym.index[-1]
        h4_frames.append(latest_h4)

    if h4_frames:
        h4_df = pd.concat(h4_frames, ignore_index=True)
        d = d.merge(h4_df, on=['Symbol', 'Date'], how='left')

    return d
