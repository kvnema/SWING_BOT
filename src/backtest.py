import pandas as pd
import numpy as np
from math import floor
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


def _compute_drawdown(equity_series: pd.Series) -> pd.Series:
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return dd


def backtest_strategy(df: pd.DataFrame, flag_col: str, cfg: dict, confirm_rsi: bool = False, confirm_macd: bool = False, confirm_hist: bool = False) -> Dict:
    """Event-driven backtest (simplified) returning KPIs, trades and equity curve.

    Notes: This is a simplified engine for quick evaluation. It uses a time-stop of 20 bars
    and uses Close prices for entry/exit as configured in the original spec.
    """
    equity = cfg.get('risk', {}).get('equity', 100000)
    risk_pct = cfg.get('risk', {}).get('risk_per_trade_pct', 1.0) / 100.0
    stop_mult = cfg.get('risk', {}).get('stop_multiple_atr', 1.5)
    tx_cost = cfg.get('backtest', {}).get('transaction_cost_pct', 0.0)

    trades = []
    equity_curve = []
    eq = float(equity)

    # iterate rows
    for i, row in df.reset_index(drop=True).iterrows():
        equity_curve.append({'Date': row['Date'], 'Equity': eq})
        if int(row.get(flag_col, 0)) != 1:
            continue
        
        # Apply RSI/MACD filters
        if confirm_rsi and not (row.get('RSI_Above50', False) and not row.get('RSI_Overbought', False)):
            continue
        if confirm_macd and not (row.get('MACD_CrossUp', False) and row.get('MACD_AboveZero', False)):
            continue
        if confirm_hist and not row.get('MACD_Hist_Rising', False):
            continue
        
        entry_price = float(row['Close'])
        atr = float(row.get('ATR14', 0.0))
        stop_price = entry_price - stop_mult * atr
        if stop_price <= 0 or atr <= 0:
            continue
        risk_amount = equity * risk_pct
        qty = floor(risk_amount / max(1e-6, (entry_price - stop_price)))
        if qty <= 0:
            continue

        # simulate time-stop at +20 bars (or last)
        exit_idx = min(len(df) - 1, i + 20)
        exit_row = df.iloc[exit_idx]
        exit_price = float(exit_row['Close'])

        R = (exit_price - entry_price) / max(1e-9, (entry_price - stop_price))
        pnl = qty * (exit_price - entry_price)
        cost = abs(qty * entry_price) * tx_cost
        eq += pnl - cost

        trades.append({
            'EntryDate': row['Date'],
            'EntryPrice': entry_price,
            'ExitDate': exit_row['Date'],
            'ExitPrice': exit_price,
            'R': R,
            'Qty': qty,
            'Pnl': pnl - cost,
        })

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve)

    rs = trades_df['R'].tolist() if not trades_df.empty else []
    n = len(rs)
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    win_rate = (len(wins) / n * 100) if n > 0 else 0.0
    avgR = (sum(rs) / n) if n > 0 else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    expectancy = ( (len(wins)/n) * avg_win + (len(losses)/n) * avg_loss ) if n > 0 else 0.0
    stdR = np.std(rs, ddof=1) if n > 1 else 0.0
    sqn = (avgR / stdR * np.sqrt(n)) if stdR > 0 else 0.0

    # Equity returns for Sharpe/Sortino
    if not eq_df.empty and len(eq_df) > 1:
        eq_series = eq_df['Equity'].astype(float)
        daily_ret = eq_series.pct_change().dropna()
        mean_ret = daily_ret.mean() if not daily_ret.empty else 0.0
        vol = daily_ret.std(ddof=0) if not daily_ret.empty else 0.0
        sharpe = (mean_ret / vol * (252**0.5)) if vol > 0 else 0.0
        downside = daily_ret[daily_ret < 0]
        downside_std = downside.std(ddof=0) if not downside.empty else 0.0
        sortino = (mean_ret / downside_std * (252**0.5)) if downside_std > 0 else 0.0

        # Max Drawdown
        dd = _compute_drawdown(eq_series)
        maxdd = dd.min()

        # CAGR approximation
        days = len(df['Date'].unique())
        years = max(1/252, days / 252)
        start_val = eq_series.iloc[0]
        end_val = eq_series.iloc[-1]
        total_return = (end_val - start_val) / start_val if start_val != 0 else 0.0
        cagr = ( (end_val / start_val) ** (1 / years) - 1 ) if start_val > 0 else 0.0
    else:
        sharpe = sortino = maxdd = cagr = total_return = 0.0

    kpi = {
        'Win_Rate_%': round(win_rate, 2),
        'AvgR': round(avgR, 4),
        'ExpectancyR': round(expectancy, 4),
        'SQN': round(sqn, 4),
        'MaxDD': round(maxdd, 4) if isinstance(maxdd, float) else 0.0,
        'Sharpe': round(sharpe, 4),
        'Sortino': round(sortino, 4),
        'CAGR': round(cagr, 4),
        'TotalReturn': round(total_return, 4),
        'Total_Trades': n,
    }

    return {'kpi': kpi, 'trades': trades_df, 'equity_curve': eq_df}


def walk_forward_backtest(df: pd.DataFrame, flag_col: str, cfg: dict,
                         train_years: int = 1, test_months: int = 3,
                         start_date: str = None, end_date: str = None) -> Dict:
    """
    Walk-forward backtesting to validate strategy robustness.

    Args:
        df: DataFrame with OHLCV and signals
        flag_col: Column name for entry signals
        cfg: Backtest configuration
        train_years: Years for in-sample training (optimization)
        test_months: Months for out-of-sample testing
        start_date: Start date for testing period
        end_date: End date for testing period

    Returns:
        Dictionary with walk-forward results and KPIs
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    if df.empty:
        return {'error': 'No data in specified date range'}

    # Split data into walk-forward windows
    results = []
    current_date = df['Date'].min()

    while current_date < df['Date'].max():
        # Define training window (lookback for regime context)
        train_end = current_date + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)

        # Ensure we have enough data
        if test_end > df['Date'].max():
            test_end = df['Date'].max()

        train_data = df[df['Date'] < train_end]
        test_data = df[(df['Date'] >= train_end) & (df['Date'] <= test_end)]

        if train_data.empty or test_data.empty:
            break

        # Run backtest on test period
        result = backtest_strategy(test_data, flag_col, cfg,
                                 confirm_rsi=True, confirm_macd=True, confirm_hist=True)

        # Add window info
        result['window'] = {
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'test_trades': len(result['trades']) if not result['trades'].empty else 0
        }

        results.append(result)

        # Move to next window
        current_date = test_end

    # Aggregate results
    if not results:
        return {'error': 'No valid walk-forward windows found'}

    # Combine all trades and equity curves
    all_trades = []
    all_equity = []

    for i, result in enumerate(results):
        if not result['trades'].empty:
            window_trades = result['trades'].copy()
            window_trades['window'] = i
            all_trades.append(window_trades)

        if not result['equity_curve'].empty:
            window_equity = result['equity_curve'].copy()
            window_equity['window'] = i
            all_equity.append(window_equity)

    combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    combined_equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()

    # Calculate combined KPIs
    if not combined_trades.empty:
        rs = combined_trades['R'].tolist()
        n = len(rs)
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r <= 0]
        win_rate = (len(wins) / n * 100) if n > 0 else 0.0
        avgR = (sum(rs) / n) if n > 0 else 0.0

        # Combined equity metrics
        if not combined_equity.empty:
            combined_equity = combined_equity.sort_values('Date')
            eq_series = combined_equity.groupby('Date')['Equity'].last()
            daily_ret = eq_series.pct_change().dropna()

            mean_ret = daily_ret.mean() if not daily_ret.empty else 0.0
            vol = daily_ret.std(ddof=0) if not daily_ret.empty else 0.0
            sharpe = (mean_ret / vol * (252**0.5)) if vol > 0 else 0.0

            dd = _compute_drawdown(eq_series)
            maxdd = dd.min()

            # CAGR
            days = (eq_series.index.max() - eq_series.index.min()).days
            years = max(1/252, days / 365.25)
            start_val = eq_series.iloc[0]
            end_val = eq_series.iloc[-1]
            cagr = ((end_val / start_val) ** (1 / years) - 1) if start_val > 0 else 0.0
        else:
            sharpe = maxdd = cagr = 0.0

        combined_kpi = {
            'Win_Rate_%': round(win_rate, 2),
            'AvgR': round(avgR, 4),
            'MaxDD': round(maxdd, 4) if isinstance(maxdd, float) else 0.0,
            'Sharpe': round(sharpe, 4),
            'CAGR': round(cagr, 4),
            'Total_Trades': n,
            'Windows': len(results)
        }
    else:
        combined_kpi = {
            'Win_Rate_%': 0.0,
            'AvgR': 0.0,
            'MaxDD': 0.0,
            'Sharpe': 0.0,
            'CAGR': 0.0,
            'Total_Trades': 0,
            'Windows': len(results)
        }

    return {
        'combined_kpi': combined_kpi,
        'window_results': results,
        'combined_trades': combined_trades,
        'combined_equity': combined_equity
    }
