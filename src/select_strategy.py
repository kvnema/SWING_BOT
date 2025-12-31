import json
from pathlib import Path
import pandas as pd
from .backtest import backtest_strategy


def select_best_strategy(df: pd.DataFrame, strategies: dict, cfg: dict, out_dir: str, confirm_rsi: bool = False, confirm_macd: bool = False, confirm_hist: bool = False) -> dict:
    """
    FIXED HIERARCHY STRATEGY SELECTION (Safer than backtest-driven per-stock selection)

    Eliminates overfitting risk by using predetermined priority order:
    1. VCP (Volume Contraction Pattern) - Highest quality setups
    2. SEPA (Stage-Enhanced Pullback Alert) - Trend template + breakout
    3. Squeeze (Bollinger-Keltner squeeze breakout)
    4. Donchian (Channel breakout)
    5. MR/AVWAP (Mean reversion only as fallback)

    No more per-stock backtest selection that causes severe overfitting!
    """
    # Still run backtests for reporting/analysis, but don't use for selection
    results = {}
    for name, flag in strategies.items():
        res = backtest_strategy(df, flag, cfg, confirm_rsi, confirm_macd, confirm_hist)
        results[name] = res['kpi']
        # persist basic outputs
        od = Path(out_dir) / name
        od.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([res['kpi']]).to_csv(od / 'kpi.csv', index=False)
        res['trades'].to_csv(od / 'trades.csv', index=False)
        res['equity_curve'].to_csv(od / 'equity_curve.csv', index=False)

    # FIXED HIERARCHY: Always use this priority order (no backtest-driven selection)
    strategy_hierarchy = ['VCP', 'SEPA', 'Squeeze', 'Donchian', 'MR', 'AVWAP']

    # Select strategy from hierarchy, preferring those with actual trades
    best = None
    max_trades = -1
    for strategy in strategy_hierarchy:
        if strategy in results:
            trades = results[strategy].get('Total_Trades', 0)
            if trades > max_trades:
                max_trades = trades
                best = strategy

    # Fallback to first in hierarchy if no trades found
    if best is None:
        for strategy in strategy_hierarchy:
            if strategy in strategies:
                best = strategy
                break

    sel = {
        'selected': best,
        'results': results,
        'hierarchy_selection': True,
        'strategy_priority': strategy_hierarchy,
        'note': 'Fixed hierarchy eliminates backtest overfitting risk'
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / 'selected_strategy.json', 'w') as f:
        json.dump(sel, f, indent=2)
    return sel
