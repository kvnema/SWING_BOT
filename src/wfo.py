import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .backtest import backtest_strategy
from .signals import compute_signals


def walk_forward_optimization(df: pd.DataFrame, strategy_name: str, flag_col: str, cfg: dict,
                              cycles: int = 8, mode: str = 'rolling', is_window: int = 252, oos_window: int = 63,
                              confirm_rsi: bool = False, confirm_macd: bool = False, confirm_hist: bool = False) -> Dict:
    """
    Perform Walk-Forward Optimization for a strategy.

    - mode: 'rolling' or 'anchored'
    - cycles: number of WFO cycles
    - is_window: in-sample days
    - oos_window: out-of-sample days

    Returns dict with OOS performance metrics, efficiency, etc.
    """
    total_days = len(df)
    step = (total_days - is_window - oos_window) // (cycles - 1) if cycles > 1 else 0

    oos_results = []
    is_results = []

    for i in range(cycles):
        if mode == 'rolling':
            start_idx = i * step
        else:  # anchored
            start_idx = 0

        is_end = start_idx + is_window
        oos_end = is_end + oos_window

        if oos_end > total_days:
            break

        is_df = df.iloc[start_idx:is_end].copy()
        oos_df = df.iloc[is_end:oos_end].copy()

        # Compute signals on IS for parameter tuning (placeholder: use fixed for now)
        is_df = compute_signals(is_df)
        oos_df = compute_signals(oos_df)

        # Backtest IS
        is_res = backtest_strategy(is_df, flag_col, cfg, confirm_rsi, confirm_macd, confirm_hist)
        is_results.append(is_res['kpi'])

        # Backtest OOS
        oos_res = backtest_strategy(oos_df, flag_col, cfg, confirm_rsi, confirm_macd, confirm_hist)
        oos_results.append(oos_res['kpi'])

    # Aggregate OOS metrics
    oos_df = pd.DataFrame(oos_results)
    agg_oos = {
        'Win_Rate_%': oos_df['Win_Rate_%'].mean(),
        'AvgR': oos_df['AvgR'].mean(),
        'ExpectancyR': oos_df['ExpectancyR'].mean(),
        'MaxDD': oos_df['MaxDD'].max(),
        'Sharpe': oos_df['Sharpe'].mean(),
        'Sortino': oos_df['Sortino'].mean(),
        'CAGR': oos_df['CAGR'].mean(),
        'TotalReturn': oos_df['TotalReturn'].mean(),
        'Total_Trades': oos_df['Total_Trades'].sum(),
    }

    # Convert numpy types to Python types for JSON serialization
    agg_oos = {k: v.item() if hasattr(v, 'item') else v for k, v in agg_oos.items()}

    # WFO Efficiency: OOS Sharpe / IS Sharpe
    is_sharpe = pd.DataFrame(is_results)['Sharpe'].mean()
    efficiency = agg_oos['Sharpe'] / is_sharpe if is_sharpe != 0 else 0

    return {
        'strategy': strategy_name,
        'mode': mode,
        'cycles': int(len(oos_results)),
        'oos_aggregate': agg_oos,
        'efficiency': float(efficiency),
        'oos_curve': oos_results,
        'is_curve': is_results
    }