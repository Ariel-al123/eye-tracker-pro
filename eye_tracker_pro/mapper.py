from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _poly_features(x: np.ndarray) -> np.ndarray:
    """Second-order polynomial features for [x1..x8]. 
    Keep it bounded to avoid overfitting: use linear + squares + key cross terms + bias.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    # 8 linear + 8 squares + 4 cross terms + bias = 21 features
    cross = [
        x[0] * x[1],  # avg_x * avg_y
        x[2] * x[3],  # left x*y
        x[4] * x[5],  # right x*y
        x[6] * x[7],  # nose x*y
    ]
    return np.concatenate([x, x * x, np.array(cross, dtype=np.float64), [1.0]])


@dataclass
class CalibrationSample:
    feature: np.ndarray
    screen_x: float
    screen_y: float


class GazeMapper:
    def __init__(self):
        self.samples: list[CalibrationSample] = []
        self._wx: Optional[np.ndarray] = None
        self._wy: Optional[np.ndarray] = None
        self.calibrated = False
        # Affine remap to expand predictions to screen edges.
        self._rx_min = 0.0
        self._rx_max = 1.0
        self._ry_min = 0.0
        self._ry_max = 1.0
        self._tx_min = 0.0
        self._tx_max = 1.0
        self._ty_min = 0.0
        self._ty_max = 1.0

    def clear(self) -> None:
        self.samples.clear()
        self._wx = None
        self._wy = None
        self.calibrated = False
        self._rx_min = self._ry_min = self._tx_min = self._ty_min = 0.0
        self._rx_max = self._ry_max = self._tx_max = self._ty_max = 1.0

    def add_sample(self, feature: np.ndarray, screen_x: float, screen_y: float) -> None:
        self.samples.append(CalibrationSample(feature=feature.astype(np.float32), screen_x=float(screen_x), screen_y=float(screen_y)))

    def fit(self, ridge: float = 1e-2) -> bool:
        if len(self.samples) < 5:
            self.calibrated = False
            return False
        X = np.stack([_poly_features(s.feature) for s in self.samples], axis=0)
        yx = np.array([s.screen_x for s in self.samples], dtype=np.float64)
        yy = np.array([s.screen_y for s in self.samples], dtype=np.float64)

        # Ridge regression: w = (X^T X + λI)^-1 X^T y
        XtX = X.T @ X
        reg = ridge * np.eye(XtX.shape[0], dtype=np.float64)
        reg[-1, -1] = 0.0  # don't regularize bias
        try:
            A = np.linalg.solve(XtX + reg, X.T)
        except np.linalg.LinAlgError:
            self.calibrated = False
            return False
        self._wx = A @ yx
        self._wy = A @ yy

        # Learn output remap so predictions cover the calibrated target range.
        pred_x = X @ self._wx
        pred_y = X @ self._wy
        self._rx_min, self._rx_max = float(np.min(pred_x)), float(np.max(pred_x))
        self._ry_min, self._ry_max = float(np.min(pred_y)), float(np.max(pred_y))
        self._tx_min, self._tx_max = float(np.min(yx)), float(np.max(yx))
        self._ty_min, self._ty_max = float(np.min(yy)), float(np.max(yy))
        self.calibrated = True
        return True

    @staticmethod
    def _remap(v: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
        span = src_max - src_min
        if abs(span) < 1e-6:
            return v
        t = (v - src_min) / span
        # small extrapolation allowance beyond calibrated range
        t = max(-0.15, min(1.15, t))
        return dst_min + t * (dst_max - dst_min)

    def predict(self, feature: np.ndarray) -> Optional[tuple[float, float]]:
        if not self.calibrated or self._wx is None or self._wy is None:
            return None
        phi = _poly_features(feature)
        x = float(phi @ self._wx)
        y = float(phi @ self._wy)
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        x = self._remap(x, self._rx_min, self._rx_max, self._tx_min, self._tx_max)
        y = self._remap(y, self._ry_min, self._ry_max, self._ty_min, self._ty_max)
        return x, y
