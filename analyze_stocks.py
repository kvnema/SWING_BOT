import pandas as pd

# Read the screener results
df = pd.read_csv('outputs/screener_results_live.csv')

print('TOP SCORING STOCKS FOR SWING TRADING:')
print('=' * 50)
high_score_stocks = df[
    (df['CompositeScore'] > 20) &
    (df['Trend_OK'] == 1) &
    (df['RSI_MACD_Confirm_OK'] == True)
].sort_values('CompositeScore', ascending=False)

for idx, row in high_score_stocks.head(10).iterrows():
    trend = '✓' if row['Trend_OK'] else '✗'
    momentum = '✓' if row['TS_Momentum_Flag'] else '✗'
    print(f"{row['Symbol']:<12} | Score: {row['CompositeScore']:>5.1f} | RSI: {row['RSI14']:>5.1f} | Trend: {trend} | Momentum: {momentum}")

print('\nSTOCKS WITH GOLDEN BULL SIGNALS:')
print('=' * 35)
golden_bull = df[df['GoldenBull_Flag'] == 1]
for idx, row in golden_bull.iterrows():
    print(f"{row['Symbol']:<12} | Bull Date: {row['GoldenBull_Date']}")

print('\nSTOCKS WITH STRONG MOMENTUM:')
print('=' * 30)
momentum_stocks = df[
    (df['TS_Momentum_Flag'] == 1) &
    (df['RS_Leader_Flag'] == 1)
].sort_values('TS_Momentum', ascending=False)

for idx, row in momentum_stocks.head(5).iterrows():
    print(f"{row['Symbol']:<12} | Momentum: {row['TS_Momentum']:>+6.2f} | RS: {row['RS_vs_Index']:>+5.3f}")

print('\nSTOCKS WITH MINERVINI TREND TEMPLATE:')
print('=' * 40)
minervini_stocks = df[df['Minervini_Trend'] == 1]
for idx, row in minervini_stocks.iterrows():
    print(f"{row['Symbol']:<12} | Score: {row['CompositeScore']:>5.1f} | Volume: {row['RVOL20']:>4.1f}x")