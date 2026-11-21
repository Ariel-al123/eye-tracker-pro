from __future__ import annotations

import math
import time


class OneEuroFilter:
    """One Euro Filter for low-latency smoothing.

    Based on Casiez et al., 2012. Good balance between jitter reduction and
    responsiveness, which matters for gaze control.
    """

    def __init__(self, min_cutoff: float = 0.7, beta: float = 0.005, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, x: float, t: float | None = None) -> float:
        if t is None:
            t = time.perf_counter()
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = float(x)
            self._dx_prev = 0.0
            return float(x)

        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t

        dx = (float(x) - float(self._x_prev)) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * float(x) + (1.0 - a) * float(self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class PointFilter:
    def __init__(self, min_cutoff: float = 0.7, beta: float = 0.005, d_cutoff: float = 1.0):
        self.x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.y = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.x.reset()
        self.y.reset()

    def filter(self, x: float, y: float, t: float | None = None) -> tuple[float, float]:
        if t is None:
            t = time.perf_counter()
        return self.x.filter(x, t), self.y.filter(y, t)
