import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
import warnings


def load_oos_trades(bt_root: str) -> pd.DataFrame:
    """
    Load OOS trades for all strategies from backtest directories.
    """
    bt_path = Path(bt_root)
    all_trades = []
    
    if not bt_path.exists():
        return pd.DataFrame()
    
    for strategy_dir in bt_path.glob("*/"):
        if not strategy_dir.is_dir():
            continue
        
        strategy = strategy_dir.name
        trades_file = strategy_dir / "trades.csv"
        
        if trades_file.exists():
            try:
                trades_df = pd.read_csv(trades_file)
                # Filter to OOS trades
                if 'OOS' in trades_df.columns:
                    oos_trades = trades_df[trades_df['OOS'] == 1].copy()
                else:
                    oos_trades = trades_df.copy()
                
                if not oos_trades.empty:
                    oos_trades['Strategy'] = strategy
                    # Ensure required columns exist
                    required_cols = ['Date', 'Symbol', 'R']
                    for col in required_cols:
                        if col not in oos_trades.columns:
                            # Map alternative column names
                            if col == 'Date' and 'EntryDate' in oos_trades.columns:
                                oos_trades['Date'] = oos_trades['EntryDate']
                            else:
                                oos_trades[col] = None
                    
                    # Add is_win column
                    oos_trades['is_win'] = (oos_trades['R'] > 0).astype(int)
                    all_trades.append(oos_trades)
                    
            except Exception as e:
                print(f"Warning: Could not read trades for {strategy}: {e}")
    
    if not all_trades:
        return pd.DataFrame()
    
    return pd.concat(all_trades, ignore_index=True)


def make_buckets(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """
    Map each trade to context buckets and add recency info.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # RSI14_Status
    rsi_series = df.get('RSI14')
    if rsi_series is not None and isinstance(rsi_series, pd.Series):
        df['RSI14_Status'] = pd.cut(rsi_series.values,
                                    bins=[-np.inf, 30, 70, np.inf],
                                    labels=['Oversold', 'Neutral', 'Overbought'])
    else:
        df['RSI14_Status'] = 'Neutral'
    
    # MACD buckets
    macd_line = df.get('MACD_Line')
    if macd_line is not None and isinstance(macd_line, pd.Series):
        df['MACD_Regime'] = pd.cut(macd_line.values,
                                   bins=[-np.inf, 0, np.inf],
                                   labels=['BelowZero', 'AboveZero'])
    else:
        df['MACD_Regime'] = 'AboveZero'
    
    macd_cross = df.get('MACD_CrossUp')
    if macd_cross is not None and isinstance(macd_cross, pd.Series):
        df['MACD_Cross_Status'] = macd_cross.map({True: 'CrossUp', False: 'NoCross'})
    else:
        df['MACD_Cross_Status'] = 'NoCross'
    
    # Golden flags
    if 'GoldenBull_Flag' in df.columns:
        df['GoldenBull_Flag'] = df['GoldenBull_Flag'].astype(int)
    else:
        df['GoldenBull_Flag'] = 0
    if 'GoldenBear_Flag' in df.columns:
        df['GoldenBear_Flag'] = df['GoldenBear_Flag'].astype(int)
    else:
        df['GoldenBear_Flag'] = 0
    
    # Trend_OK
    if 'Trend_OK' in df.columns:
        df['Trend_OK'] = df['Trend_OK'].astype(int)
    else:
        df['Trend_OK'] = 0
    
    # RVOL20_bucket
    rvol_series = df.get('RVOL20')
    if rvol_series is not None and isinstance(rvol_series, pd.Series):
        df['RVOL20_bucket'] = pd.cut(rvol_series,
                                     bins=[-np.inf, 1.0, 1.5, np.inf],
                                     labels=['<1.0', '1.0–1.5', '>1.5'])
    else:
        df['RVOL20_bucket'] = '1.0–1.5'
    
    # ATRpct_bucket
    if 'ATRpct' not in df.columns and 'ATR14' in df.columns and 'Close' in df.columns:
        df['ATRpct'] = (df['ATR14'] / df['Close']) * 100
    
    atrpct_series = df.get('ATRpct')
    if atrpct_series is not None and isinstance(atrpct_series, pd.Series):
        df['ATRpct_bucket'] = pd.cut(atrpct_series,
                                     bins=[-np.inf, 1.0, 2.0, np.inf],
                                     labels=['<1%', '1–2%', '>2%'])
    else:
        df['ATRpct_bucket'] = '1–2%'
    
    # Sector (placeholder - assume not available or add logic)
    df['Sector'] = df.get('Sector', 'Unknown')
    
    # Recency: age in days
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['age_days'] = (today - df['Date']).dt.days.fillna(365)
    
    return df


def aggregate_oos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate wins/losses by hierarchical buckets with recency weighting.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Define grouping columns
    group_cols = ['Strategy', 'Sector', 'Symbol', 'RSI14_Status', 'MACD_Regime', 'MACD_Cross_Status',
                  'GoldenBull_Flag', 'GoldenBear_Flag', 'Trend_OK', 'RVOL20_bucket', 'ATRpct_bucket']
    
    # Recency weights (exponential decay, tau=120 days)
    tau = 120
    df['recency_weight'] = np.exp(-df['age_days'] / tau)
    
    # Aggregate with weighted counts
    agg_dict = {
        'is_win': ['count', 'sum', lambda x: (x * df.loc[x.index, 'recency_weight']).sum()],  # total trades, raw wins, weighted wins
        'R': ['mean', 'count'],  # expectancy and total trades
        'age_days': 'mean'  # average age for reference
    }
    
    try:
        grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
        grouped = grouped.rename(columns={
            'is_win_count': 'Trades_OOS',
            'is_win_sum': 'Wins_OOS',
            'is_win_<lambda_0>': 'Weighted_Wins',
            'R_mean': 'OOS_ExpectancyR',
            'R_count': 'Total_Trades',  # Should match Trades_OOS
            'age_days_mean': 'Avg_Age_Days'
        })
        
        # Compute weighted win rate
        grouped['OOS_WinRate_raw'] = grouped['Weighted_Wins'] / grouped['Trades_OOS']
        
        # Fill NaN
        grouped = grouped.fillna({
            'OOS_WinRate_raw': 0.5,
            'OOS_ExpectancyR': 0.0,
            'Weighted_Wins': 0.0,
            'Wins_OOS': 0.0
        })
        
    except Exception as e:
        print(f"Warning: Could not aggregate OOS data: {e}")
        return pd.DataFrame()
    
    return grouped


def empirical_bayes_shrink(wins: float, trades: float, prior_p: float, lambda_: float) -> float:
    """
    Empirical Bayes shrinkage toward prior.
    """
    if trades == 0:
        return prior_p
    return (wins + lambda_ * prior_p) / (trades + lambda_)


def posterior_beta_ci(wins: float, losses: float, a_prior: float = 1.0, b_prior: float = 1.0, q: Tuple[float, float] = (0.05, 0.95)) -> Tuple[float, float, float]:
    """
    Beta posterior: mean and confidence interval.
    """
    a_post = a_prior + wins
    b_post = b_prior + losses
    
    mean = a_post / (a_post + b_post)
    ci_low = beta.ppf(q[0], a_post, b_post)
    ci_high = beta.ppf(q[1], a_post, b_post)
    
    return mean, ci_low, ci_high


def calibrate_probabilities(preds: pd.Series, outcomes: pd.Series) -> pd.Series:
    """
    Isotonic regression calibration.
    """
    try:
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(preds.values, outcomes.values)
        calibrated = iso_reg.predict(preds.values)
        return pd.Series(np.clip(calibrated, 0.05, 0.95), index=preds.index)
    except Exception:
        # Fallback: return original predictions clipped
        return np.clip(preds, 0.05, 0.95)


def build_hierarchical_model(bt_root: str, today: pd.Timestamp, t_min: int = 30) -> pd.DataFrame:
    """
    Build hierarchical success model with empirical Bayes and Beta posteriors.
    """
    # Load and prepare data
    trades_df = load_oos_trades(bt_root)
    if trades_df.empty:
        return pd.DataFrame(columns=[
            'Strategy', 'Sector', 'Symbol', 'RSI14_Status', 'GoldenBull_Flag', 'GoldenBear_Flag',
            'Trend_OK', 'RVOL20_bucket', 'ATRpct_bucket', 'Trades_OOS', 'Wins_OOS', 'OOS_WinRate_raw',
            'CalibratedWinRate', 'CI_low', 'CI_high', 'OOS_ExpectancyR', 'Reliability', 'WFO_Efficiency', 'CoverageNote'
        ])
    
    bucketed_df = make_buckets(trades_df, today)
    aggregated_df = aggregate_oos(bucketed_df)
    
    if aggregated_df.empty:
        # Fall back to simple strategy-level model
        print("Complex bucketing failed, using simple strategy-level model")
        if not trades_df.empty:
            # Group by strategy and compute basic metrics
            strategy_model = trades_df.groupby('Strategy').agg({
                'is_win': ['count', 'mean'],
                'R': 'mean'
            }).reset_index()
            
            # Flatten columns
            strategy_model.columns = ['Strategy', 'Trades_OOS', 'OOS_WinRate_raw', 'OOS_ExpectancyR']
            
            # Add required columns for compatibility
            strategy_model['CalibratedWinRate'] = strategy_model['OOS_WinRate_raw']
            strategy_model['CI_low'] = strategy_model['OOS_WinRate_raw'] - 0.1
            strategy_model['CI_high'] = strategy_model['OOS_WinRate_raw'] + 0.1
            strategy_model['CoverageNote'] = 'StrategyLevel'
            strategy_model['Reliability'] = 1.0
            strategy_model['WFO_Efficiency'] = 1.0
            
            return strategy_model
    
    if aggregated_df.empty:
        return pd.DataFrame()
    
    # Build hierarchical priors
    priors = _build_priors(aggregated_df)
    
    # Apply empirical Bayes and Beta posteriors
    results = []
    for _, row in aggregated_df.iterrows():
        strategy = row['Strategy']
        sector = row['Sector']
        symbol = row['Symbol']
        
        # Get prior from hierarchy
        prior_p = _get_prior_from_hierarchy(priors, strategy, sector, symbol)
        
        # Empirical Bayes shrinkage
        wins = row['Weighted_Wins']
        trades = row['Trades_OOS']
        lambda_shrink = max(5, 50 - trades)  # Shrinkage strength decreases with more data
        shrunk_p = empirical_bayes_shrink(wins, trades, prior_p, lambda_shrink)
        
        # Beta posterior CI
        losses = trades - wins
        mean_p, ci_low, ci_high = posterior_beta_ci(wins, losses)
        
        # Reliability
        reliability = min(1.0, trades / t_min)
        
        # Final calibrated rate (combine shrunk and posterior)
        calibrated = shrunk_p * 0.7 + mean_p * 0.3  # Weighted combination
        calibrated = np.clip(calibrated, 0.05, 0.95)
        
        results.append({
            **row.to_dict(),
            'CalibratedWinRate': calibrated,
            'CI_low': ci_low,
            'CI_high': ci_high,
            'Reliability': reliability,
            'WFO_Efficiency': 1.0,  # Placeholder
            'CoverageNote': _determine_coverage_note(row, priors)
        })
    
    return pd.DataFrame(results)


def _build_priors(aggregated_df: pd.DataFrame) -> Dict:
    """
    Build hierarchical priors: global -> strategy -> sector -> symbol.
    """
    priors = {}
    
    # Global prior
    total_wins = aggregated_df['Weighted_Wins'].sum()
    total_trades = aggregated_df['Trades_OOS'].sum()
    priors['global'] = total_wins / total_trades if total_trades > 0 else 0.5
    
    # Strategy priors
    strategy_priors = aggregated_df.groupby('Strategy').agg({
        'Weighted_Wins': 'sum',
        'Trades_OOS': 'sum'
    }).apply(lambda x: x['Weighted_Wins'] / x['Trades_OOS'] if x['Trades_OOS'] > 0 else priors['global'], axis=1)
    priors['strategy'] = strategy_priors.to_dict()
    
    # Sector priors (within strategy)
    sector_priors = {}
    for strategy in aggregated_df['Strategy'].unique():
        strat_df = aggregated_df[aggregated_df['Strategy'] == strategy]
        sect_priors = strat_df.groupby('Sector').agg({
            'Weighted_Wins': 'sum',
            'Trades_OOS': 'sum'
        }).apply(lambda x: x['Weighted_Wins'] / x['Trades_OOS'] if x['Trades_OOS'] > 0 else priors['strategy'].get(strategy, priors['global']), axis=1)
        sector_priors[strategy] = sect_priors.to_dict()
    priors['sector'] = sector_priors
    
    # Symbol priors (within strategy-sector)
    symbol_priors = {}
    for (strategy, sector), group in aggregated_df.groupby(['Strategy', 'Sector']):
        sym_priors = group.groupby('Symbol').agg({
            'Weighted_Wins': 'sum',
            'Trades_OOS': 'sum'
        }).apply(lambda x: x['Weighted_Wins'] / x['Trades_OOS'] if x['Trades_OOS'] > 0 else priors['sector'].get(strategy, {}).get(sector, priors['strategy'].get(strategy, priors['global'])), axis=1)
        if strategy not in symbol_priors:
            symbol_priors[strategy] = {}
        symbol_priors[strategy][sector] = sym_priors.to_dict()
    priors['symbol'] = symbol_priors
    
    return priors


def _get_prior_from_hierarchy(priors: Dict, strategy: str, sector: str, symbol: str) -> float:
    """
    Get the most specific available prior from hierarchy.
    """
    # Try symbol-level
    if strategy in priors.get('symbol', {}) and sector in priors['symbol'][strategy] and symbol in priors['symbol'][strategy][sector]:
        return priors['symbol'][strategy][sector][symbol]
    
    # Try sector-level
    if strategy in priors.get('sector', {}) and sector in priors['sector'][strategy]:
        return priors['sector'][strategy][sector]
    
    # Try strategy-level
    if strategy in priors.get('strategy', {}):
        return priors['strategy'][strategy]
    
    # Global
    return priors.get('global', 0.5)


def _determine_coverage_note(row: pd.Series, priors: Dict) -> str:
    """
    Determine coverage note based on data availability.
    """
    strategy = row['Strategy']
    sector = row['Sector']
    symbol = row['Symbol']
    trades = row['Trades_OOS']
    
    if trades >= 50:
        return 'ContextExact'
    elif trades >= 20:
        return 'SymbolPool'
    elif strategy in priors.get('symbol', {}) and sector in priors['symbol'][strategy] and symbol in priors['symbol'][strategy][sector]:
        return 'SymbolPrior'
    elif strategy in priors.get('sector', {}) and sector in priors['sector'][strategy]:
        return 'SectorPool'
    elif strategy in priors.get('strategy', {}):
        return 'StrategyPrior'
    else:
        return 'GlobalPrior'


def lookup_confidence(model_df: pd.DataFrame, row_ctx: dict, strategy: str) -> dict:
    """
    Find best matching bucket with backoff logic.
    """
    if model_df.empty:
        return {
            'DecisionConfidence': 0.5,
            'CI_low': 0.4,
            'CI_high': 0.6,
            'OOS_WinRate': 0.5,
            'OOS_ExpectancyR': 0.0,
            'Trades_OOS': 0,
            'CoverageNote': 'NoModel'
        }
    
    # Start with strategy-only match (for simple models)
    match = model_df[model_df['Strategy'] == strategy]
    
    if not match.empty:
        # Found strategy match, return it
        bucket = match.iloc[0]
        return {
            'DecisionConfidence': bucket.get('CalibratedWinRate', 0.5),
            'CI_low': bucket.get('CI_low', 0.4),
            'CI_high': bucket.get('CI_high', 0.6),
            'OOS_WinRate': bucket.get('OOS_WinRate_raw', 0.5),
            'OOS_ExpectancyR': bucket.get('OOS_ExpectancyR', 0.0),
            'Trades_OOS': int(bucket.get('Trades_OOS', 0)),
            'CoverageNote': bucket.get('CoverageNote', 'StrategyLevel'),
            'Reliability': bucket.get('Reliability', 1.0),
            'WFO_Efficiency': bucket.get('WFO_Efficiency', 1.0)
        }
    
    # Handle CompositeScore ensemble
    if strategy == 'CompositeScore':
        # Get available strategies in model
        available_strategies = model_df['Strategy'].unique()
        if len(available_strategies) > 0:
            # Weight by OOS trades and reliability
            weights = {}
            total_weight = 0
            
            for strat in available_strategies:
                strat_data = model_df[model_df['Strategy'] == strat]
                if not strat_data.empty:
                    # Weight by trades * reliability
                    weight = strat_data['Trades_OOS'].sum() * strat_data['Reliability'].mean()
                    weights[strat] = weight
                    total_weight += weight
            
            if total_weight > 0:
                # Compute weighted average confidence
                weighted_confidence = 0
                weighted_ci_low = 0
                weighted_ci_high = 0
                weighted_winrate = 0
                weighted_expectancy = 0
                total_trades = 0
                
                for strat, weight in weights.items():
                    strat_data = model_df[model_df['Strategy'] == strat].iloc[0]
                    w = weight / total_weight
                    weighted_confidence += w * strat_data.get('CalibratedWinRate', 0.5)
                    weighted_ci_low += w * strat_data.get('CI_low', 0.4)
                    weighted_ci_high += w * strat_data.get('CI_high', 0.6)
                    weighted_winrate += w * strat_data.get('OOS_WinRate_raw', 0.5)
                    weighted_expectancy += w * strat_data.get('OOS_ExpectancyR', 0.0)
                    total_trades += strat_data.get('Trades_OOS', 0)
                
                return {
                    'DecisionConfidence': weighted_confidence,
                    'CI_low': weighted_ci_low,
                    'CI_high': weighted_ci_high,
                    'OOS_WinRate': weighted_winrate,
                    'OOS_ExpectancyR': weighted_expectancy,
                    'Trades_OOS': int(total_trades),
                    'CoverageNote': 'Ensemble',
                    'Reliability': 1.0,
                    'WFO_Efficiency': 1.0
                }
    # Fall back to original hierarchical matching if no strategy match
    match = model_df[
        (model_df['Strategy'] == strategy) &
        (model_df.get('Sector', pd.Series(['Unknown'] * len(model_df))) == row_ctx.get('Sector', 'Unknown')) &
        (model_df.get('Symbol', pd.Series([''] * len(model_df))) == row_ctx.get('Symbol', '')) &
        (model_df['RSI14_Status'] == row_ctx['RSI14_Status']) &
        (model_df['MACD_Regime'] == row_ctx['MACD_Regime']) &
        (model_df['MACD_Cross_Status'] == row_ctx['MACD_Cross_Status']) &
        (model_df['GoldenBull_Flag'] == row_ctx['GoldenBull_Flag']) &
        (model_df['GoldenBear_Flag'] == row_ctx['GoldenBear_Flag']) &
        (model_df['Trend_OK'] == row_ctx['Trend_OK']) &
        (model_df['RVOL20_bucket'] == row_ctx['RVOL20_bucket']) &
        (model_df['ATRpct_bucket'] == row_ctx['ATRpct_bucket'])
    ]
    
    # Backoff sequence
    backoffs = [
        # Drop ATR
        lambda: model_df[
            (model_df['Strategy'] == strategy) &
            (model_df.get('Sector', pd.Series(['Unknown'] * len(model_df))) == row_ctx.get('Sector', 'Unknown')) &
            (model_df.get('Symbol', pd.Series([''] * len(model_df))) == row_ctx.get('Symbol', '')) &
            (model_df['RSI14_Status'] == row_ctx['RSI14_Status']) &
            (model_df['MACD_Regime'] == row_ctx['MACD_Regime']) &
            (model_df['MACD_Cross_Status'] == row_ctx['MACD_Cross_Status']) &
            (model_df['GoldenBull_Flag'] == row_ctx['GoldenBull_Flag']) &
            (model_df['GoldenBear_Flag'] == row_ctx['GoldenBear_Flag']) &
            (model_df['Trend_OK'] == row_ctx['Trend_OK']) &
            (model_df['RVOL20_bucket'] == row_ctx['RVOL20_bucket'])
        ],
        # Drop RVOL
        lambda: model_df[
            (model_df['Strategy'] == strategy) &
            (model_df.get('Sector', pd.Series(['Unknown'] * len(model_df))) == row_ctx.get('Sector', 'Unknown')) &
            (model_df.get('Symbol', pd.Series([''] * len(model_df))) == row_ctx.get('Symbol', '')) &
            (model_df['RSI14_Status'] == row_ctx['RSI14_Status']) &
            (model_df['MACD_Regime'] == row_ctx['MACD_Regime']) &
            (model_df['MACD_Cross_Status'] == row_ctx['MACD_Cross_Status']) &
            (model_df['GoldenBull_Flag'] == row_ctx['GoldenBull_Flag']) &
            (model_df['GoldenBear_Flag'] == row_ctx['GoldenBear_Flag']) &
            (model_df['Trend_OK'] == row_ctx['Trend_OK'])
        ],
        # Drop Golden
        lambda: model_df[
            (model_df['Strategy'] == strategy) &
            (model_df.get('Sector', pd.Series(['Unknown'] * len(model_df))) == row_ctx.get('Sector', 'Unknown')) &
            (model_df.get('Symbol', pd.Series([''] * len(model_df))) == row_ctx.get('Symbol', '')) &
            (model_df['RSI14_Status'] == row_ctx['RSI14_Status']) &
            (model_df['MACD_Regime'] == row_ctx['MACD_Regime']) &
            (model_df['MACD_Cross_Status'] == row_ctx['MACD_Cross_Status'])
        ],
        # Strategy-level aggregate
        lambda: model_df[model_df['Strategy'] == strategy].groupby('Strategy').agg({
            'Trades_OOS': 'sum',
            'Weighted_Wins': 'sum',
            'OOS_ExpectancyR': 'mean',
            'CalibratedWinRate': 'mean',
            'CI_low': 'mean',
            'CI_high': 'mean',
            'Reliability': 'mean',
            'CoverageNote': lambda x: 'StrategyAggregate'
        }).reset_index()
    ]
    
    for backoff_func in backoffs:
        if not match.empty:
            break
        match = backoff_func()
    
    if match.empty:
        return {
            'DecisionConfidence': 0.5,
            'CI_low': 0.4,
            'CI_high': 0.6,
            'OOS_WinRate': 0.5,
            'OOS_ExpectancyR': 0.0,
            'Trades_OOS': 0,
            'CoverageNote': 'GlobalPrior',
            'Reliability': 1.0,
            'WFO_Efficiency': 1.0
        }
    
    # Take first match
    bucket = match.iloc[0]
    
    return {
        'DecisionConfidence': bucket.get('CalibratedWinRate', 0.5),
        'CI_low': bucket.get('CI_low', 0.4),
        'CI_high': bucket.get('CI_high', 0.6),
        'OOS_WinRate': bucket.get('OOS_WinRate_raw', 0.5),
        'OOS_ExpectancyR': bucket.get('OOS_ExpectancyR', 0.0),
        'Trades_OOS': int(bucket.get('Trades_OOS', 0)),
        'CoverageNote': bucket.get('CoverageNote', 'Unknown'),
        'Reliability': bucket.get('Reliability', 1.0),
        'WFO_Efficiency': bucket.get('WFO_Efficiency', 1.0)
    }
    """
    Build success model from OOS backtest results.

    Aggregates OOS performance by strategy and context buckets:
    - RSI14_Status: Oversold/Neutral/Overbought
    - GoldenBull_Flag: 0/1
    - GoldenBear_Flag: 0/1
    - Trend_OK: 0/1 (if available)
    - RVOL20_bucket: <1.0, 1.0–1.5, >1.5
    - ATRpct_bucket: <1%, 1–2%, >2%

    Returns DataFrame with OOS metrics per bucket.
    """
    bt_path = Path(bt_dir)
    if not bt_path.exists():
        # Return empty model if no backtests available
        return pd.DataFrame(columns=[
            'Strategy', 'RSI14_Status', 'GoldenBull_Flag', 'GoldenBear_Flag',
            'Trend_OK', 'RVOL20_bucket', 'ATRpct_bucket',
            'Trades_OOS', 'Wins_OOS', 'Losses_OOS', 'OOS_WinRate', 'OOS_ExpectancyR',
            'CalibratedWinRate', 'Reliability', 'WFO_Efficiency'
        ])

    all_trades = []
    strategy_kpis = {}

    # Collect all OOS trades from backtest directories
    for strategy_dir in bt_path.glob("*/"):
        if not strategy_dir.is_dir():
            continue

        strategy = strategy_dir.name
        trades_file = strategy_dir / "trades.csv"
        kpi_file = strategy_dir / "kpi.csv"

        if trades_file.exists():
            try:
                trades_df = pd.read_csv(trades_file)
                # Assume trades.csv has OOS column or filter by date
                if 'OOS' in trades_df.columns:
                    oos_trades = trades_df[trades_df['OOS'] == 1].copy()
                else:
                    # If no OOS flag, assume all are OOS for simplicity
                    oos_trades = trades_df.copy()

                oos_trades['Strategy'] = strategy
                all_trades.append(oos_trades)
            except Exception as e:
                print(f"Warning: Could not read trades for {strategy}: {e}")
                continue

        if kpi_file.exists():
            try:
                kpi_df = pd.read_csv(kpi_file)
                # Extract OOS KPIs
                oos_kpis = kpi_df[kpi_df.get('Sample', '').str.contains('OOS', na=False)]
                if not oos_kpis.empty:
                    strategy_kpis[strategy] = oos_kpis.iloc[0].to_dict()
            except Exception as e:
                print(f"Warning: Could not read KPIs for {strategy}: {e}")

    if not all_trades:
        # Return empty model with correct columns
        return pd.DataFrame(columns=[
            'Strategy', 'RSI14_Status', 'GoldenBull_Flag', 'GoldenBear_Flag',
            'Trend_OK', 'RVOL20_bucket', 'ATRpct_bucket',
            'Trades_OOS', 'Wins_OOS', 'Losses_OOS', 'OOS_WinRate', 'OOS_ExpectancyR',
            'CalibratedWinRate', 'Reliability', 'WFO_Efficiency'
        ])

    # Combine all trades
    combined_trades = pd.concat(all_trades, ignore_index=True)

    # Add context buckets
    try:
        combined_trades = _add_context_buckets(combined_trades)
    except Exception as e:
        print(f"Warning: Could not add context buckets: {e}")
        return pd.DataFrame(columns=[
            'Strategy', 'RSI14_Status', 'GoldenBull_Flag', 'GoldenBear_Flag',
            'Trend_OK', 'RVOL20_bucket', 'ATRpct_bucket',
            'Trades_OOS', 'Wins_OOS', 'Losses_OOS', 'OOS_WinRate', 'OOS_ExpectancyR',
            'CalibratedWinRate', 'Reliability', 'WFO_Efficiency'
        ])

    # Group by strategy and buckets
    group_cols = ['Strategy', 'RSI14_Status', 'GoldenBull_Flag', 'GoldenBear_Flag',
                  'Trend_OK', 'RVOL20_bucket', 'ATRpct_bucket']

    # Aggregate metrics
    # First ensure Win column exists
    if 'Win' not in combined_trades.columns:
        # Fallback: assume positive R = win
        combined_trades['Win'] = (combined_trades.get('R', 0) > 0).astype(int)

    try:
        grouped = combined_trades.groupby(group_cols).agg({
            'Win': ['count', lambda x: (x == 1).sum(), lambda x: (x == 0).sum()],
            'R': 'mean'
        }).reset_index()

        # Flatten column names
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
        grouped = grouped.rename(columns={
            'Win_count': 'Trades_OOS',
            'Win_<lambda_0>': 'Wins_OOS',
            'Win_<lambda_1>': 'Losses_OOS',
            'R_mean': 'OOS_ExpectancyR'
        })
    except Exception as e:
        print(f"Warning: Could not aggregate trades: {e}")
        return pd.DataFrame(columns=[
            'Strategy', 'RSI14_Status', 'GoldenBull_Flag', 'GoldenBear_Flag',
            'Trend_OK', 'RVOL20_bucket', 'ATRpct_bucket',
            'Trades_OOS', 'Wins_OOS', 'Losses_OOS', 'OOS_WinRate', 'OOS_ExpectancyR',
            'CalibratedWinRate', 'Reliability', 'WFO_Efficiency'
        ])

    # Compute OOS_WinRate
    grouped['OOS_WinRate'] = grouped['Wins_OOS'] / grouped['Trades_OOS']

    # Beta smoothing (Jeffreys prior)
    grouped['CalibratedWinRate'] = (grouped['Wins_OOS'] + 0.5) / (grouped['Trades_OOS'] + 1.0)

    # Reliability down-weighting
    T_min = 30
    grouped['Reliability'] = np.minimum(1.0, grouped['Trades_OOS'] / T_min)

    # WFO Efficiency (placeholder - would need IS/OOS comparison)
    grouped['WFO_Efficiency'] = 1.0  # Default; could be computed from strategy_kpis

    # Fill NaN values
    grouped = grouped.fillna({
        'OOS_WinRate': 0.5,
        'OOS_ExpectancyR': 0.0,
        'CalibratedWinRate': 0.5,
        'Reliability': 0.0,
        'WFO_Efficiency': 1.0
    })

    return grouped


def _add_context_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add context bucket columns to trades DataFrame."""
    # RSI14_Status
    rsi_series = df.get('RSI14', pd.Series([50] * len(df)))
    if isinstance(rsi_series, pd.Series):
        df['RSI14_Status'] = pd.cut(rsi_series.values,
                                    bins=[-np.inf, 30, 70, np.inf],
                                    labels=['Oversold', 'Neutral', 'Overbought'])
    else:
        df['RSI14_Status'] = pd.cut(rsi_series,
                                    bins=[-np.inf, 30, 70, np.inf],
                                    labels=['Oversold', 'Neutral', 'Overbought'])

    # Golden flags (assume already present or default to 0)
    df['GoldenBull_Flag'] = df.get('GoldenBull_Flag', 0).astype(int)
    df['GoldenBear_Flag'] = df.get('GoldenBear_Flag', 0).astype(int)

    # Trend_OK (assume present or default)
    if 'Trend_OK' in df.columns:
        df['Trend_OK'] = df['Trend_OK'].astype(int)
    else:
        df['Trend_OK'] = 0

    # RVOL20_bucket
    df['RVOL20_bucket'] = pd.cut(df.get('RVOL20', 1.0),
                                 bins=[-np.inf, 1.0, 1.5, np.inf],
                                 labels=['<1.0', '1.0–1.5', '>1.5'])

    # ATRpct_bucket (assume ATRpct = ATR14 / Close * 100)
    if 'ATRpct' not in df.columns and 'ATR14' in df.columns and 'Close' in df.columns:
        df['ATRpct'] = (df['ATR14'] / df['Close']) * 100

    df['ATRpct_bucket'] = pd.cut(df.get('ATRpct', 1.5),
                                 bins=[-np.inf, 1.0, 2.0, np.inf],
                                 labels=['<1%', '1–2%', '>2%'])

    return df