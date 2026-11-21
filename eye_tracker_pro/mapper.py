from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _gaze_features(x: np.ndarray) -> np.ndarray:
    """Compact, well-conditioned features for nine calibration targets."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    # Average gaze carries the main signal; binocular disagreement and nose
    # position help compensate for asymmetry and modest head movement.
    return np.array([x[0], x[1], x[2] - x[4], x[3] - x[5], x[6], x[7],
                     x[8], x[9], x[10], 1.0])


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
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_scale: Optional[np.ndarray] = None
        self._design_mean: Optional[np.ndarray] = None
        self._design_scale: Optional[np.ndarray] = None

    def clear(self) -> None:
        self.samples.clear()
        self._wx = None
        self._wy = None
        self.calibrated = False
        self._feature_mean = self._feature_scale = None
        self._design_mean = self._design_scale = None

    def add_sample(self, feature: np.ndarray, screen_x: float, screen_y: float) -> None:
        self.samples.append(CalibrationSample(feature=feature.astype(np.float32), screen_x=float(screen_x), screen_y=float(screen_y)))

    def save(self, path: Path, screen_w: int, screen_h: int) -> bool:
        """Persist calibration samples in normalized screen coordinates."""
        if not self.calibrated or not self.samples or screen_w <= 0 or screen_h <= 0:
            return False
        payload = {
            "format": 1,
            "screen_width": int(screen_w),
            "screen_height": int(screen_h),
            "samples": [
                {
                    "feature": sample.feature.astype(float).tolist(),
                    "x": sample.screen_x / screen_w,
                    "y": sample.screen_y / screen_h,
                }
                for sample in self.samples
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)
            return True
        except (OSError, ValueError, TypeError):
            return False

    def load(self, path: Path, screen_w: int, screen_h: int) -> bool:
        """Load and refit a saved calibration, adapting it to the current display."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") != 1:
                raise ValueError("Unsupported calibration format")
            samples = payload.get("samples", [])
            if len(samples) < 5:
                raise ValueError("Calibration has too few samples")
            loaded: list[CalibrationSample] = []
            for sample in samples:
                feature = np.asarray(sample["feature"], dtype=np.float32).reshape(-1)
                x, y = float(sample["x"]), float(sample["y"])
                if feature.size < 11 or not np.all(np.isfinite(feature)) or not np.isfinite(x + y):
                    raise ValueError("Invalid calibration sample")
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError("Calibration coordinates are out of bounds")
                loaded.append(CalibrationSample(feature, x * (screen_w - 1), y * (screen_h - 1)))
            self.samples = loaded
            if self.fit():
                return True
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            pass
        self.clear()
        return False

    def fit(self, ridge: float = 1e-2) -> bool:
        if len(self.samples) < 5:
            self.calibrated = False
            return False
        raw = np.stack([s.feature for s in self.samples], axis=0).astype(np.float64)
        self._feature_mean = np.median(raw, axis=0)
        self._feature_scale = np.maximum(np.std(raw, axis=0), 0.025)
        X0 = np.stack([_gaze_features((s.feature - self._feature_mean) / self._feature_scale) for s in self.samples])
        self._design_mean = X0.mean(axis=0)
        # A meaningful floor prevents nearly constant head-pose signals from
        # being magnified when the user moves after calibration.
        self._design_scale = np.maximum(X0.std(axis=0), 0.15)
        self._design_mean[-1], self._design_scale[-1] = 0.0, 1.0
        X = (X0 - self._design_mean) / self._design_scale
        yx = np.array([s.screen_x for s in self.samples], dtype=np.float64)
        yy = np.array([s.screen_y for s in self.samples], dtype=np.float64)

        # Ridge regression: w = (X^T X + λI)^-1 X^T y
        XtX = X.T @ X
        reg = max(ridge, 0.5) * np.eye(XtX.shape[0], dtype=np.float64)
        reg[-1, -1] = 0.0  # the final column is the constant bias
        try:
            A = np.linalg.solve(XtX + reg, X.T)
        except np.linalg.LinAlgError:
            self.calibrated = False
            return False
        self._wx = A @ (yx - np.mean(yx))
        self._wy = A @ (yy - np.mean(yy))
        self._wx[-1] += np.mean(yx)
        self._wy[-1] += np.mean(yy)
        self.calibrated = True
        return True

    def predict(self, feature: np.ndarray) -> Optional[tuple[float, float]]:
        if not self.calibrated or self._wx is None or self._wy is None:
            return None
        if self._feature_mean is None or self._feature_scale is None or self._design_mean is None or self._design_scale is None:
            return None
        phi0 = _gaze_features((feature - self._feature_mean) / self._feature_scale)
        phi = (phi0 - self._design_mean) / self._design_scale
        x = float(phi @ self._wx)
        y = float(phi @ self._wy)
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return x, y
