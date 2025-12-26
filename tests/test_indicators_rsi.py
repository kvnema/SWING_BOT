import pandas as pd
import numpy as np
from src.indicators import rsi, compute_rsi_macd_gates


def test_rsi_wilder_calculation():
    """Test RSI(14) Wilder’s method with known series."""
    # Create test data with known RSI values
    data = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.32, 46.32, 46.32]
    df = pd.DataFrame({'Close': data})
    
    rsi_series = rsi(df, 14)
    # RSI for the last point should be approximately 72.78 (calculated value)
    last_rsi = rsi_series.iloc[-1]
    assert 70.0 <= last_rsi <= 75.0, f"Expected RSI ~72.78, got {last_rsi}"


def test_rsi_thresholds():
    """Test RSI gates: Overbought >=70, Oversold <=30, Above50 >=50."""
    df = pd.DataFrame({
        'Close': [100, 110, 90, 105, 115, 85, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185]
    })
    df_gates = compute_rsi_macd_gates(df)
    
    # Check that gates are boolean
    assert df_gates['RSI_Overbought'].dtype == bool
    assert df_gates['RSI_Oversold'].dtype == bool
    assert df_gates['RSI_Above50'].dtype == bool
    
    # Check some values
    assert df_gates['RSI_Overbought'].iloc[-1] == True  # High RSI
    assert df_gates['RSI_Oversold'].iloc[0] == False  # Not low
    assert df_gates['RSI_Above50'].iloc[-1] == True  # High RSI


def test_rsi_midpoint_bias():
    """Test RSI midpoint bias: >=50 indicates bullish momentum."""
    # Create oscillating data
    closes = []
    for i in range(50):
        closes.append(100 + 10 * np.sin(i * 0.2))
    df = pd.DataFrame({'Close': closes})
    df_gates = compute_rsi_macd_gates(df)
    
    # RSI should fluctuate around 50
    rsi_vals = df_gates['RSI14']
    above_50_count = (rsi_vals >= 50).sum()
    below_50_count = (rsi_vals < 50).sum()
    
    # Should have both above and below 50
    assert above_50_count > 0
    assert below_50_count > 0
    
    # Gates should match
    assert (df_gates['RSI_Above50'] == (rsi_vals >= 50)).all()