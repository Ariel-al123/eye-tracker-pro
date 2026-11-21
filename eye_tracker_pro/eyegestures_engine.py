from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EyeGesturesFrame:
    face_detected: bool
    point: Optional[tuple[int, int]]
    target: Optional[tuple[int, int]]
    acceptance_radius: int
    blink: bool


class EyeGesturesEngine:
    """Adapter for EyeGestures 3.x, which supports the project's Python 3.11."""

    def __init__(self, screen_width: int, screen_height: int):
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._engine = self._new_tracker()
        self.calibrating = False
        self.ready = False
        self.last_error: Optional[str] = None
        self._reported_error: Optional[str] = None
        self.last_frame = EyeGesturesFrame(False, None, None, 0, False)

    @staticmethod
    def _new_tracker():
        try:
            from eyeGestures import EyeGestures_v3, FaceFinder
        except ImportError as exc:
            raise RuntimeError(
                "EyeGestures no está instalado. Ejecuta: "
                "uv pip install --cache-dir .uv-cache --python .venv\\Scripts\\python.exe "
                "-r requirements-eyegestures.txt"
            ) from exc
        tracker = EyeGestures_v3()
        # EyeGestures 3.2.4 defaults FaceMesh to static_image_mode=True. Keep
        # MediaPipe's temporal tracker alive between frames for steadier face reacquisition.
        finder = FaceFinder(static_image_mode=False)
        # EyeGestures_v3.getLandmarks already converts BGR to RGB before the
        # finder is called. Avoid FaceFinder's second channel swap.
        finder.find = lambda frame_rgb: finder.mp_face_mesh.process(frame_rgb)
        tracker.finder = finder
        return tracker

    def start_calibration(self) -> None:
        old_finder = getattr(self._engine, "finder", None)
        old_mesh = getattr(old_finder, "mp_face_mesh", None)
        if old_mesh is not None:
            old_mesh.close()
        self._engine = self._new_tracker()
        self.calibrating = True
        self.ready = False

    def finish_calibration(self) -> None:
        self.calibrating = False
        self.ready = True

    def cancel_calibration(self) -> None:
        self.calibrating = False
        self.ready = False

    def close(self) -> None:
        finder = getattr(self._engine, "finder", None)
        mesh = getattr(finder, "mp_face_mesh", None)
        if mesh is not None:
            mesh.close()

    def detect(self, frame_bgr: np.ndarray) -> EyeGesturesFrame:
        try:
            event, calibration_event = self._engine.step(
                frame_bgr,
                self.calibrating,
                self._screen_width,
                self._screen_height,
                context="eye-tracker-pro",
            )
            face = event.sub_frame is not None and event.sub_frame.size > 0
            point = None
            if face and np.all(np.isfinite(event.point)):
                point = (
                    int(np.clip(event.point[0], 0, self._screen_width - 1)),
                    int(np.clip(event.point[1], 0, self._screen_height - 1)),
                )
            target_raw = calibration_event.point
            target = (
                int(np.clip(target_raw[0], 0, self._screen_width - 1)),
                int(np.clip(target_raw[1], 0, self._screen_height - 1)),
            )
            radius = int(getattr(calibration_event, "acceptance_radius", 0))
            self.last_frame = EyeGesturesFrame(face, point, target, radius, bool(event.blink))
            self.last_error = None
            self._reported_error = None
        except Exception as exc:
            # FaceFinder/landmark failures should pause tracking, not terminate the UI loop.
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.last_error != self._reported_error:
                print(f"EyeGestures tracking error: {self.last_error}", flush=True)
                self._reported_error = self.last_error
            self.last_frame = EyeGesturesFrame(False, None, None, 0, False)
        return self.last_frame
