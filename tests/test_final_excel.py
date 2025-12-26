import pytest
import pandas as pd
import os
from src.final_excel import run_final_excel


def test_build_final_excel(tmp_path):
    # Create mock plan CSV
    plan_data = {
        'Date': ['2025-12-16', '2025-12-16'],
        'Symbol': ['RELIANCE', 'TCS'],
        'Strategy': ['MR', 'Breakout'],
        'InstrumentToken': ['NSE_EQ|INE002A01018', 'NSE_EQ|INE467B01029'],
        'Qty': [12, 10],
        'ENTRY_trigger_type': ['BELOW', 'ABOVE'],
        'ENTRY_trigger_price': [1522.00, 4200.50],
        'STOPLOSS_trigger_price': [1499.00, 4150.00],
        'TARGET_trigger_price': [1558.00, 4280.00],
        'R': [2.0, 2.0],
        'ATR14': [12.5, 25.0],
        'RSI14': [45.0, 55.0],
        'RSI14_Status': ['Neutral', 'Neutral'],
        'GoldenBull_Flag': [0, 1],
        'GoldenBear_Flag': [0, 0],
        'GoldenBull_Date': ['', '2025-12-15'],
        'GoldenBear_Date': ['', ''],
        'Notes': ['MR dip near EMA20', 'RVOL=1.8; Donchian breakout'],
        'Explanation': ['Selected via MR strategy. RSI: 45.0 (Neutral). No Golden Crossover. ATR×1.5=18.75 for stoploss. Strategy template: MR dip near EMA20.', 'Selected via Donchian strategy. RSI: 55.0 (Neutral). Golden Bull on 2025-12-15. ATR×1.5=37.5 for stoploss. Strategy template: RVOL=1.8; Donchian breakout.'],
        'DecisionConfidence': [0.65, 0.72],
        'CI_low': [0.52, 0.58],
        'CI_high': [0.76, 0.83],
        'OOS_WinRate': [0.68, 0.71],
        'OOS_ExpectancyR': [0.42, 0.38],
        'Trades_OOS': [45, 52],
        'Confidence_Reason': ['Strategy=MR, bucket=Neutral/GBull=0/GBear=0/RVOL=1.0–1.5/ATR=1–2%, OOS trades=45, win=68.0%, expR=0.42, CI=[52.0%, 76.0%], calibrated=65.0%, pooling=ContextExact.', 'Strategy=Breakout, bucket=Neutral/GBull=1/GBear=0/RVOL>1.5/ATR>2%, OOS trades=52, win=71.0%, expR=0.38, CI=[58.0%, 83.0%], calibrated=72.0%, pooling=SymbolPool.']
    }
    plan_df = pd.DataFrame(plan_data)
    plan_csv = tmp_path / "plan.csv"
    plan_df.to_csv(plan_csv, index=False)
    
    # Output Excel
    out_xlsx = tmp_path / "final.xlsx"
    
    # Call function
    run_final_excel(str(plan_csv), str(out_xlsx))
    
    # Check file exists
    assert out_xlsx.exists()
    
    # Read back and check columns
    df_read = pd.read_excel(out_xlsx, sheet_name='GTT-Delivery-Plan', header=1)
    expected_cols = ['Symbol', 'Qty', 'ENTRY_trigger_price', 'TARGET_trigger_price', 'STOPLOSS_trigger_price', 'DecisionConfidence', 'Confidence_Level', 'R', 'Explanation', 'GTT_Explanation', 'RSI14', 'RSI_Above50', 'RSI_Overbought', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'MACD_CrossUp', 'MACD_AboveZero', 'RSI_MACD_Confirmations_OK', 'RSI_MACD_Notes', 'Audit_Flag', 'Issues', 'Fix_Suggestion', 'Pivot_Source', 'Entry_Logic', 'Stop_Logic', 'Target_Logic', 'Latest_Close', 'Latest_LTP', 'Canonical_Entry', 'Canonical_Stop', 'Canonical_Target']
    assert list(df_read.columns) == expected_cols
    # Check data rows (ignore footer)
    data_rows = df_read.dropna(subset=['ENTRY_trigger_price'])
    assert len(data_rows) == 2
    # Check that we have the expected symbols
    symbols = set(data_rows['Symbol'].str.replace('.NS', ''))  # Remove .NS suffix if present
    assert 'RELIANCE' in symbols or 'RELIANCE.NS' in symbols
    assert 'TCS' in symbols or 'TCS.NS' in symbols