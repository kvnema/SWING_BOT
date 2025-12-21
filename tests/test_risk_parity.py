import pytest
import pandas as pd
import numpy as np
from src.portfolio import risk_parity_allocation


def test_risk_parity_equal_contribution():
    # Mock returns
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(100, 3) * 0.02, columns=['A', 'B', 'C'])
    cov = returns.cov()

    weights = risk_parity_allocation(cov)
    assert abs(weights.sum() - 1) < 1e-6
    assert all(w >= 0 for w in weights)

    # Check risk contributions are roughly equal
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal_risk = cov @ weights / port_vol
    risk_contrib = weights * marginal_risk
    # Variance of contributions should be low
    assert np.var(risk_contrib) < 0.01