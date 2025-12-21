import pandas as pd
import numpy as np
from math import floor
from typing import Dict, List


def _compute_drawdown(equity_series: pd.Series) -> pd.Series:
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return dd


def backtest_strategy(df: pd.DataFrame, flag_col: str, cfg: dict) -> Dict:
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
