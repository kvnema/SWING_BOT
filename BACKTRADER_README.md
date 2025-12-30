# SWING_BOT Backtrader Integration

This directory contains the Backtrader integration for advanced backtesting and validation of the SWING_BOT momentum strategy.

## Overview

The Backtrader integration provides:
- **Complete strategy port**: All SWING_BOT indicators, signals, and logic
- **Realistic simulation**: ATR-based position sizing, trailing stops, commissions
- **Multi-stock testing**: NIFTY50/200 universe with sector constraints
- **Walk-forward optimization**: In-sample/out-of-sample validation
- **Comprehensive analytics**: Sharpe, Sortino, Calmar, drawdown analysis
- **Interactive plotting**: Performance visualization

## Quick Start

### 1. Install Dependencies
```bash
pip install backtrader ta-lib
```

### 2. Prepare Data
```bash
# Create sample dataset (top 10 NIFTY50 stocks)
python prepare_backtest_data.py --create-sample

# Or prepare specific symbols
python prepare_backtest_data.py --symbols RELIANCE TCS HDFCBANK --start-date 2023-01-01 --end-date 2024-12-31
```

### 3. Run Backtest
```bash
# Basic backtest
python backtest_swing_bot.py --symbols RELIANCE TCS HDFCBANK --start-date 2023-01-01 --end-date 2024-12-31

# Full NIFTY50 universe
python backtest_swing_bot.py --universe nifty50 --start-date 2023-01-01 --end-date 2024-12-31

# With walk-forward optimization
python backtest_swing_bot.py --universe nifty50 --walk-forward --plot
```

## Architecture

### Core Components

#### `src/backtrader_strategy.py`
- **SwingBotStrategy**: Main strategy implementation
- All SWING_BOT indicators and signals ported to Backtrader
- Realistic position sizing and risk management
- Multi-stock portfolio management with sector limits

#### `src/backtrader_data.py`
- **SwingBotData**: Custom data feed for SWING_BOT format
- Data loading and validation utilities
- Benchmark data integration for regime filtering

#### `backtest_swing_bot.py`
- **BacktestRunner**: Main backtesting orchestrator
- Command-line interface for various backtest scenarios
- Results analysis and reporting

#### `prepare_backtest_data.py`
- Data preparation and validation pipeline
- Integration with existing SWING_BOT data fetching
- Quality checks and formatting

## Strategy Implementation

### Indicators Implemented
- **Price Indicators**: EMA(20,50,200), SMA(200)
- **Trend Indicators**: Trend OK composite
- **RSI & MACD**: Full signal generation
- **Bollinger Bands**: Bandwidth, squeeze detection
- **Keltner Channels**: ATR-based volatility bands
- **Donchian Channels**: Breakout detection
- **Volume Indicators**: RVOL, volume dry-up patterns
- **Relative Strength**: RS vs benchmark (simplified)

### Signals Implemented
- **SEPA Flag**: Minervini trend + tight base + breakout + volume
- **VCP Flag**: Contracting BB + higher lows + volume dry-up + breakout
- **Donchian Breakout**: Channel breakout with volume confirmation
- **MR Flag**: Mean reversion in uptrend
- **BBKC Squeeze**: Bollinger-Keltner squeeze breakout
- **AVWAP Reclaim**: Trend continuation signal
- **TS Momentum**: 12-month momentum filter

### Risk Management
- **Position Sizing**: 1% risk per trade, ATR-based stops
- **Initial Stops**: 1.5x ATR below entry
- **Trailing Stops**: ATR-based, percentage, or Parabolic SAR
- **Profit Taking**: 50% position exit at 2:1 reward:risk
- **Portfolio Limits**: Max 10 positions, sector exposure caps

### Regime Filtering
- **Nifty > SMA200**: Long-term uptrend confirmation
- **ADX > 20 OR RSI > 50**: Trend strength or momentum confirmation
- **Automatic OFF**: No new positions when regime fails

## Usage Examples

### Basic Backtest
```bash
python backtest_swing_bot.py \
    --symbols RELIANCE TCS HDFCBANK INFY ICICIBANK \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --initial-cash 1000000 \
    --risk-per-trade 1.0
```

### Walk-Forward Optimization
```bash
python backtest_swing_bot.py \
    --universe nifty50 \
    --walk-forward \
    --start-date 2020-01-01 \
    --end-date 2024-12-31
```

### Custom Strategy Parameters
```bash
python backtest_swing_bot.py \
    --symbols RELIANCE TCS \
    --max-positions 5 \
    --risk-per-trade 0.5 \
    --trail-type percentage
```

## Output & Analysis

### Performance Metrics
- **Total Return**: Absolute and annualized
- **Risk Metrics**: Sharpe, Sortino, Calmar ratios
- **Drawdown Analysis**: Max drawdown, recovery time
- **Trade Statistics**: Win rate, profit factor, trade count

### Sample Output
```
================================================================================
SWING_BOT BACKTEST RESULTS
================================================================================
Portfolio Performance:
  Initial Cash:     ₹1,000,000
  Final Value:      ₹1,450,230
  Total Return:     45.02%
  Annualized Return: 18.45%
  Test Period:      2023-01-01 to 2024-12-31
  Duration:         730 days
  Symbols Tested:   50

Risk Metrics:
  Sharpe Ratio:     1.23
  Max Drawdown:     12.45%
  Calmar Ratio:     1.48
  Sortino Ratio:    1.67

Trading Metrics:
  Total Trades:     127
  Win Rate:         62.1%
  Won/Lost:         79/48
  Profit Factor:    1.85
================================================================================
```

### Export Files
- `backtest_trade_log.csv`: Detailed trade log
- `backtest_summary.csv`: Performance summary
- `backtest.log`: Execution log
- `*_plot_*.png`: Performance charts (if --plot enabled)

## Data Preparation

### Using Existing SWING_BOT Data
```bash
# Validate existing data
python prepare_backtest_data.py --validate-only --universe nifty50

# Prepare specific date range
python prepare_backtest_data.py \
    --symbols RELIANCE TCS HDFCBANK \
    --start-date 2020-01-01 \
    --end-date 2024-12-31
```

### Fetching New Data
```bash
# Fetch missing data via API
python prepare_backtest_data.py \
    --symbols RELIANCE TCS \
    --fetch-missing \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

### Data Quality Checks
- **Sufficient History**: Minimum 252 trading days
- **No Missing Values**: Complete OHLCV data
- **Valid Price Relationships**: High >= Low, Close within range
- **Volume Data**: Positive volume values
- **Date Continuity**: Proper chronological ordering

## Advanced Features

### Parameter Optimization
```python
# Example parameter ranges for optimization
param_ranges = {
    'risk_per_trade': [0.5, 1.0, 1.5, 2.0],
    'atr_stop_mult': [1.0, 1.5, 2.0, 2.5],
    'min_signals': [1, 2, 3],
    'trail_type': ['atr', 'percentage']
}
```

### Custom Analyzers
The backtest includes standard Backtrader analyzers:
- **SharpeRatio**: Risk-adjusted returns
- **DrawDown**: Peak-to-valley decline
- **TradeAnalyzer**: Trade-level statistics
- **SQN**: System Quality Number
- **Returns**: Return calculations
- **Calmar**: Drawdown-adjusted returns
- **Sortino**: Downside deviation-adjusted returns
- **VWR**: Variability-Weighted Return

### Benchmark Comparison
```python
# Add benchmark for comparison
benchmark_feed = SwingBotData.from_dataframe(nifty_df, name='NIFTY50')
cerebro.adddata(benchmark_feed)

# Add benchmark analyzer
cerebro.addanalyzer(BenchmarkAnalyzer, _name='benchmark')
```

## Integration with Live Trading

### Seamless Transition
- **Same Indicators**: Live and backtest use identical calculations
- **Shared Signals**: Strategy logic matches live implementation
- **Risk Management**: Consistent position sizing and stops
- **Regime Filter**: Same market condition checks

### Validation Workflow
1. **Backtest Strategy**: Run comprehensive backtests
2. **Parameter Optimization**: Fine-tune via walk-forward analysis
3. **Paper Trading**: Validate in real-time paper environment
4. **Live Deployment**: Implement with confidence

## Troubleshooting

### Common Issues

#### "No signals generated"
- Check regime filter: Ensure Nifty > SMA200 and (ADX > 20 or RSI > 50)
- Verify data quality: Run data validation
- Check signal thresholds: May need adjustment for different markets

#### "Memory errors with large universe"
- Reduce universe size or use sampling
- Increase system memory or use chunked processing
- Use --max-positions to limit concurrent positions

#### "Poor performance metrics"
- Check data quality and preprocessing
- Validate indicator calculations
- Review strategy parameters and market conditions
- Consider different time periods or market regimes

#### "Import errors"
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Check TA-Lib installation
python -c "import talib; print('TA-Lib OK')"
```

### Performance Optimization
- **Data Loading**: Use HDF5 or Parquet for large datasets
- **Indicator Calculation**: Cache expensive computations
- **Memory Usage**: Process in batches for large universes
- **Parallel Processing**: Use multiple cores for optimization

## Future Enhancements

### Planned Features
- **Streamlit Dashboard**: Interactive backtest visualization
- **Monte Carlo Analysis**: Drawdown stress testing
- **Machine Learning**: Feature importance and signal weighting
- **Multi-timeframe**: Higher timeframe regime confirmation
- **Options Integration**: Covered calls and protective puts
- **Portfolio Optimization**: Modern portfolio theory integration

### Research Areas
- **Alternative Risk Measures**: CVaR, Expected Shortfall
- **Transaction Cost Analysis**: Market impact modeling
- **Behavioral Finance**: Loss aversion and disposition effects
- **Market Microstructure**: Order flow and liquidity analysis

## Support & Documentation

### Getting Help
1. **Check Logs**: Review `backtest.log` for detailed execution info
2. **Validate Data**: Use `prepare_backtest_data.py --validate-only`
3. **Test Components**: Run individual strategy components
4. **Compare Results**: Verify against known good backtests

### Documentation Links
- [Backtrader Documentation](https://www.backtrader.com/docu/)
- [TA-Lib Reference](https://ta-lib.org/)
- [SWING_BOT Strategy Guide](../docs/STRATEGY.md)
- [Data Pipeline Documentation](../docs/DATA_PIPELINE.md)

---

*This Backtrader integration provides professional-grade backtesting capabilities while maintaining full compatibility with the live SWING_BOT trading system.*