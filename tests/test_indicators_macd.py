import pandas as pd
import numpy as np
from src.indicators import macd, compute_rsi_macd_gates


def test_macd_components():
    """Test MACD components: macd_line, signal_line, histogram."""
    # Create trending data
    closes = [100 + i * 0.5 for i in range(50)]
    df = pd.DataFrame({'Close': closes})
    
    macd_line, signal_line, histogram = macd(df, 12, 26, 9)
    
    # MACD line should be EMA12 - EMA26
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    expected_macd = ema12 - ema26
    pd.testing.assert_series_equal(macd_line, expected_macd, check_names=False)
    
    # Signal should be EMA9 of MACD
    expected_signal = macd_line.ewm(span=9, adjust=False).mean()
    pd.testing.assert_series_equal(signal_line, expected_signal, check_names=False)
    
    # Histogram should be MACD - Signal
    expected_hist = macd_line - signal_line
    pd.testing.assert_series_equal(histogram, expected_hist, check_names=False)


def test_macd_cross_up_flag():
    """Test MACD cross up flag: macd_line crosses above signal_line."""
    # Create data where MACD crosses up
    closes = [100] * 30 + [100 + i * 0.1 for i in range(20)]  # Flat then up
    df = pd.DataFrame({'Close': closes})
    df_gates = compute_rsi_macd_gates(df)
    
    # Should have at least one cross up
    cross_ups = df_gates['MACD_CrossUp']
    assert cross_ups.sum() > 0
    
    # Cross up should be where macd > signal and previous macd <= previous signal
    macd_line = df_gates['MACD_Line']
    signal_line = df_gates['MACD_Signal']
    expected_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    pd.testing.assert_series_equal(cross_ups, expected_cross, check_names=False)


def test_macd_above_zero_and_histogram_rising():
    """Test MACD above zero and histogram rising flags."""
    # Create oscillating data
    closes = [100 + 20 * np.sin(i * 0.3) for i in range(50)]
    df = pd.DataFrame({'Close': closes})
    df_gates = compute_rsi_macd_gates(df)
    
    macd_line = df_gates['MACD_Line']
    histogram = df_gates['MACD_Hist']
    
    # Above zero flag
    above_zero = df_gates['MACD_AboveZero']
    expected_above = macd_line > 0
    pd.testing.assert_series_equal(above_zero, expected_above, check_names=False)
    
    # Histogram rising
    hist_rising = df_gates['MACD_Hist_Rising']
    expected_rising = histogram > histogram.shift(1)
    pd.testing.assert_series_equal(hist_rising, expected_rising, check_names=False)
    
    # Should have both true and false values
    assert above_zero.sum() > 0
    assert (above_zero == False).sum() > 0
    assert hist_rising.sum() > 0
    assert (hist_rising == False).sum() > 0