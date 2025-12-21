import pandas as pd
from src.cli import cmd_rsi_golden
import tempfile
import os


def make_fixture():
    d = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=5),
        'Symbol': ['AAA', 'BBB', 'AAA', 'BBB', 'AAA'],
        'Open': range(10,15),
        'High': range(11,16),
        'Low': range(9,14),
        'Close': range(10,15),
        'Volume': [100]*5,
        'EMA20': [12]*5,
        'EMA50': [11]*5,
        'EMA200': [10]*5,
        'RSI14': [25, 50, 75, 85, 30],
        'MACD': [0]*5,'MACDSignal':[0]*5,'MACDHist':[0]*5,
        'ATR14': [1]*5,
        'BB_MA20':[12]*5,'BB_Upper':[13]*5,'BB_Lower':[11]*5,'BB_BandWidth':[0.1]*5,
        'RVOL20':[1.6]*5,'DonchianH20':[15]*5,'DonchianL20':[5]*5,'RS_vs_Index':[1]*5,'RS_ROC20':[5]*5,
        'KC_Upper':[14]*5,'KC_Lower':[8]*5,'Squeeze':[1]*5,'AVWAP60':[11]*5,'IndexUpRegime':[1]*5
    })
    return d


def test_rsi_golden_cli():
    df = make_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, 'input.csv')
        output_path = os.path.join(tmp, 'output.csv')
        df.to_csv(input_path, index=False)
        
        # Mock args
        class Args:
            path = input_path
            out = output_path
        
        cmd_rsi_golden(Args())
        
        # Check output
        out_df = pd.read_csv(output_path)
        assert len(out_df) == 2  # Two symbols
        assert 'Symbol' in out_df.columns
        assert 'RSI14_Status' in out_df.columns
        assert 'GoldenBull_Flag' in out_df.columns
        # Check one value
        aaa_row = out_df[out_df['Symbol'] == 'AAA']
        assert not aaa_row.empty
        assert aaa_row['RSI14'].iloc[0] == 30.0  # Latest for AAA