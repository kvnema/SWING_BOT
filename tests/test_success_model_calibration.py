import numpy as np
import pandas as pd
from src.success_model import calibrate_probabilities


def test_isotonic_calibration_bounds_and_monotonic():
    # Synthetic predictions and outcomes
    preds = pd.Series([0.1, 0.2, 0.4, 0.6, 0.8])
    outcomes = pd.Series([0, 0, 1, 1, 1])

    calibrated = calibrate_probabilities(preds, outcomes)
    assert isinstance(calibrated, (pd.Series, np.ndarray))
    cal = np.array(calibrated)
    # Bounds
    assert np.all(cal >= 0.05) and np.all(cal <= 0.95)
    # Monotonic non-decreasing
    assert np.all(np.diff(cal) >= -1e-8)
