import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize


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