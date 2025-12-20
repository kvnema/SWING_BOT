import json
from pathlib import Path
import pandas as pd
from .backtest import backtest_strategy


def select_best_strategy(df: pd.DataFrame, strategies: dict, cfg: dict, out_dir: str) -> dict:
    results = {}
    for name, flag in strategies.items():
        res = backtest_strategy(df, flag, cfg)
        results[name] = res['kpi']
        # persist basic outputs
        od = Path(out_dir) / name
        od.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([res['kpi']]).to_csv(od / 'kpi.csv', index=False)
        res['trades'].to_csv(od / 'trades.csv', index=False)
        res['equity_curve'].to_csv(od / 'equity_curve.csv', index=False)

    # choose by Sharpe, tie-breaker ExpectancyR
    best = max(results.items(), key=lambda kv: (kv[1].get('Sharpe', 0), kv[1].get('ExpectancyR', 0)))[0]
    sel = {'selected': best, 'results': results}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / 'selected_strategy.json', 'w') as f:
        json.dump(sel, f, indent=2)
    return sel
