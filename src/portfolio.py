"""
Portfolio Construction and Risk Management
Ensemble weighting, deflated Sharpe selection, and risk constraints
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize


class PortfolioConstructor:
    """
    Portfolio construction with ensemble weighting and risk management
    """

    def __init__(self, max_weight: float = 0.05, max_sector_weight: float = 0.20,
                 min_sharpe: float = 0.5, max_correlation: float = 0.7):
        """
        Initialize portfolio constructor

        Args:
            max_weight: Maximum weight per position
            max_sector_weight: Maximum weight per sector
            min_sharpe: Minimum Sharpe ratio for inclusion
            max_correlation: Maximum correlation between positions
        """
        self.max_weight = max_weight
        self.max_sector_weight = max_sector_weight
        self.min_sharpe = min_sharpe
        self.max_correlation = max_correlation

    def compute_strategy_returns(self, signals_df: pd.DataFrame,
                                price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute returns for each strategy signal

        Args:
            signals_df: DataFrame with strategy signals
            price_df: DataFrame with price data

        Returns:
            DataFrame with strategy returns
        """
        returns_df = pd.DataFrame(index=signals_df.index)

        # Strategy columns (assuming they end with '_Flag' or '_Signal')
        strategy_cols = [col for col in signals_df.columns
                        if col.endswith('_Flag') or col.endswith('_Signal')]

        for strategy in strategy_cols:
            # Forward returns (next day)
            returns_df[f'{strategy}_return'] = (
                price_df['Close'].shift(-1) / price_df['Close'] - 1
            ).where(signals_df[strategy] == 1)

        return returns_df

    def compute_deflated_sharpe(self, returns_df: pd.DataFrame,
                               benchmark_returns: pd.Series,
                               min_periods: int = 60) -> pd.Series:
        """
        Compute deflated Sharpe ratios for strategies

        Args:
            returns_df: Strategy returns DataFrame
            benchmark_returns: Benchmark returns series
            min_periods: Minimum periods for calculation

        Returns:
            Series with deflated Sharpe ratios
        """
        deflated_sharpes = {}

        for col in returns_df.columns:
            if col.endswith('_return'):
                strategy_returns = returns_df[col].dropna()

                if len(strategy_returns) < min_periods:
                    deflated_sharpes[col.replace('_return', '_sharpe')] = 0
                    continue

                # Basic Sharpe
                mean_ret = strategy_returns.mean()
                std_ret = strategy_returns.std()
                sharpe = mean_ret / std_ret if std_ret > 0 else 0

                # Deflate by benchmark Sharpe
                bench_sharpe = benchmark_returns.mean() / benchmark_returns.std()
                deflated_sharpe = sharpe - bench_sharpe

                deflated_sharpes[col.replace('_return', '_sharpe')] = deflated_sharpe

        return pd.Series(deflated_sharpes)

    def compute_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation matrix of strategy returns

        Args:
            returns_df: Strategy returns DataFrame

        Returns:
            Correlation matrix
        """
        return returns_df.corr()

    def select_strategies_by_sharpe(self, sharpe_series: pd.Series,
                                   min_sharpe: Optional[float] = None) -> List[str]:
        """
        Select strategies based on deflated Sharpe ratio

        Args:
            sharpe_series: Series with Sharpe ratios
            min_sharpe: Minimum Sharpe threshold

        Returns:
            List of selected strategy names
        """
        threshold = min_sharpe or self.min_sharpe
        selected = sharpe_series[sharpe_series >= threshold].index.tolist()
        return [s.replace('_sharpe', '') for s in selected]

    def compute_ensemble_weights(self, signals_df: pd.DataFrame,
                                returns_df: pd.DataFrame,
                                selected_strategies: List[str]) -> pd.Series:
        """
        Compute ensemble weights using deflated Sharpe and correlation

        Args:
            signals_df: Strategy signals
            returns_df: Strategy returns
            selected_strategies: List of selected strategies

        Returns:
            Series with strategy weights
        """
        if not selected_strategies:
            return pd.Series()

        weights = {}

        # Base weight from Sharpe ratio
        sharpe_cols = [f'{s}_sharpe' for s in selected_strategies]
        sharpe_weights = returns_df[sharpe_cols].mean() / returns_df[sharpe_cols].mean().sum()

        # Adjust for correlation (diversification)
        corr_matrix = self.compute_correlation_matrix(
            returns_df[[f'{s}_return' for s in selected_strategies]]
        )

        # Reduce weight for highly correlated strategies
        for strategy in selected_strategies:
            strategy_corr = corr_matrix.loc[f'{strategy}_return'].mean()
            correlation_penalty = max(0, (strategy_corr - 0.3) / 0.4)  # Penalty above 0.3 correlation

            weights[strategy] = sharpe_weights[f'{strategy}_sharpe'] * (1 - correlation_penalty * 0.5)

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}

        return pd.Series(weights)

    def apply_risk_constraints(self, weights: pd.Series,
                              signals_df: pd.DataFrame,
                              sector_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Apply risk constraints to portfolio weights

        Args:
            weights: Initial strategy weights
            signals_df: Signals data with position info
            sector_df: Sector classification data

        Returns:
            Constrained weights
        """
        constrained_weights = weights.copy()

        # Maximum weight constraint
        constrained_weights = constrained_weights.clip(upper=self.max_weight)

        # Sector constraints (if sector data available)
        if sector_df is not None and not sector_df.empty:
            # Group by sector and apply sector limits
            sector_weights = {}
            for sector in sector_df['sector'].unique():
                sector_positions = sector_df[sector_df['sector'] == sector].index
                sector_weight = constrained_weights[sector_positions].sum()

                if sector_weight > self.max_sector_weight:
                    # Scale down sector positions
                    scale_factor = self.max_sector_weight / sector_weight
                    constrained_weights[sector_positions] *= scale_factor

        # Re-normalize
        total_weight = constrained_weights.sum()
        if total_weight > 0:
            constrained_weights = constrained_weights / total_weight

        return constrained_weights

    def construct_portfolio(self, signals_df: pd.DataFrame,
                           price_df: pd.DataFrame,
                           benchmark_returns: pd.Series,
                           sector_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Construct complete portfolio with all constraints

        Args:
            signals_df: Strategy signals
            price_df: Price data
            benchmark_returns: Benchmark returns
            sector_df: Sector data

        Returns:
            Dict with portfolio construction results
        """
        # Compute strategy returns
        returns_df = self.compute_strategy_returns(signals_df, price_df)

        # Compute deflated Sharpe ratios
        sharpe_series = self.compute_deflated_sharpe(returns_df, benchmark_returns)

        # Select strategies
        selected_strategies = self.select_strategies_by_sharpe(sharpe_series)

        if not selected_strategies:
            return {
                'weights': pd.Series(),
                'selected_strategies': [],
                'sharpe_ratios': sharpe_series,
                'correlation_matrix': pd.DataFrame(),
                'portfolio_return': 0,
                'portfolio_volatility': 0
            }

        # Compute ensemble weights
        weights = self.compute_ensemble_weights(signals_df, returns_df, selected_strategies)

        # Apply risk constraints
        constrained_weights = self.apply_risk_constraints(weights, signals_df, sector_df)

        # Portfolio statistics
        portfolio_return = (constrained_weights * returns_df[[f'{s}_return' for s in selected_strategies]].mean()).sum()
        portfolio_vol = np.sqrt(
            constrained_weights.T @ returns_df[[f'{s}_return' for s in selected_strategies]].cov() @ constrained_weights
        )

        return {
            'weights': constrained_weights,
            'selected_strategies': selected_strategies,
            'sharpe_ratios': sharpe_series,
            'correlation_matrix': self.compute_correlation_matrix(returns_df),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        }


def compute_portfolio_metrics(portfolio_result: Dict, risk_free_rate: float = 0.02) -> Dict:
    """
    Compute comprehensive portfolio metrics

    Args:
        portfolio_result: Portfolio construction result
        risk_free_rate: Risk-free rate for Sharpe calculation

    Returns:
        Dict with portfolio metrics
    """
    weights = portfolio_result.get('weights', pd.Series())
    portfolio_return = portfolio_result.get('portfolio_return', 0)
    portfolio_vol = portfolio_result.get('portfolio_volatility', 0)

    if weights.empty:
        return {'error': 'No portfolio positions'}

    metrics = {
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0,
        'max_weight': weights.max(),
        'min_weight': weights.min(),
        'num_positions': len(weights),
        'concentration_ratio': weights.max() / weights.sum() if weights.sum() > 0 else 0
    }

    return metrics


def risk_parity_allocation(cov: pd.DataFrame, target_risk: float = 0.01) -> pd.Series:
    """
    Compute risk parity weights: equal risk contribution.

    cov: N x N covariance matrix
    target_risk: target portfolio volatility
    """
    n = len(cov)
    init_weights = np.ones(n) / n

    def risk_contrib(weights):
        port_vol = np.sqrt(weights @ cov @ weights)
        marginal_risk = cov @ weights / port_vol
        risk_contrib = weights * marginal_risk
        return risk_contrib

    def obj_func(weights):
        rc = risk_contrib(weights)
        return np.var(rc)  # minimize variance of risk contributions

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum to 1
        {'type': 'ineq', 'fun': lambda w: target_risk - np.sqrt(w @ cov @ w)}  # vol <= target
    ]
    bounds = [(0, 1) for _ in range(n)]

    res = minimize(obj_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success:
        return pd.Series(res.x, index=cov.columns)
    else:
        return pd.Series(init_weights, index=cov.columns)


def black_litterman_tilt(base_weights: pd.Series, views: Dict[str, float], cov: pd.DataFrame,
                         tau: float = 0.05, confidence: float = 0.5) -> pd.Series:
    """
    Apply Black-Litterman tilt to base weights.

    views: dict of asset -> expected return view
    confidence: confidence in views (0-1)
    """
    n = len(base_weights)
    pi = np.zeros(n)  # market equilibrium returns (simplified)
    P = np.eye(n)  # view matrix (identity for absolute views)
    Q = np.array([views.get(asset, 0) for asset in base_weights.index])
    omega = np.diag([tau * cov.iloc[i, i] / confidence for i in range(n)])  # uncertainty

    # BL posterior
    inv_cov = np.linalg.inv(cov)
    inv_omega = np.linalg.inv(omega)
    bl_cov = np.linalg.inv(inv_cov + P.T @ inv_omega @ P)
    bl_mu = bl_cov @ (inv_cov @ pi + P.T @ inv_omega @ Q)

    # Optimize to get weights
    def obj(w):
        return - (w @ bl_mu - 0.5 * tau * w @ cov @ w)  # maximize expected utility

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]

    res = minimize(obj, base_weights.values, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success:
        return pd.Series(res.x, index=base_weights.index)
    else:
        return base_weights


def construct_portfolio(candidates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Construct portfolio from candidates using risk parity and optional BL.

    candidates: dataframe with Symbol, CompositeScore, etc.
    cfg: config dict with portfolio settings
    """
    # Filter top N respecting sector caps (placeholder: assume no sectors)
    top_n = cfg.get('portfolio', {}).get('top_n', 25)
    max_positions = cfg.get('risk', {}).get('max_positions', 8)
    selected = candidates.nlargest(min(top_n, max_positions), 'CompositeScore')

    # Compute returns (placeholder: use historical returns if available)
    # For simplicity, assume equal weight or risk parity on synthetic cov
    n = len(selected)
    synthetic_cov = pd.DataFrame(np.eye(n) * 0.04, index=selected['Symbol'], columns=selected['Symbol'])  # 20% vol

    weights = risk_parity_allocation(synthetic_cov)

    # Optional BL tilt
    if cfg.get('portfolio', {}).get('bl_enable', False):
        views = {sym: score / 100 for sym, score in zip(selected['Symbol'], selected['CompositeScore'])}
        weights = black_litterman_tilt(weights, views, synthetic_cov)

    selected = selected.assign(Weight=weights.values)
    return selected