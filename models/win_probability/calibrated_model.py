"""
IsotonicCalibratedXGB: thin wrapper combining a trained XGBoost model
with an isotonic regression layer for post-hoc probability calibration.

Kept in its own module so joblib always serializes the class with a stable
__module__ path (models.win_probability.calibrated_model), regardless of
whether calibrate.py is run as a script or imported.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibratedXGB:
    """XGBoost + isotonic regression post-hoc calibration.

    Wraps the raw XGBoost model so callers use predict_proba() identically
    to the uncalibrated version. The isotonic layer maps raw scores to
    calibrated probabilities using a monotone step function fit on the val set.
    """

    def __init__(self, base_model, isotonic: IsotonicRegression):
        self.base_model = base_model
        self.isotonic   = isotonic

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.isotonic.transform(raw)
        return np.column_stack([1.0 - cal, cal])
