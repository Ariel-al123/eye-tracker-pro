from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import cv2
import mediapipe as mp
import numpy as np

EYE_GESTURES_CALIBRATION_POINTS = 25
EYE_GESTURES_POINT_DELAY_SECONDS = 1.0


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
        self.completed_points = 0
        self.total_points = EYE_GESTURES_CALIBRATION_POINTS
        self._collect_after = 0.0
        self.last_error: Optional[str] = None
        self._reported_error: Optional[str] = None
        self.last_frame = EyeGesturesFrame(False, None, None, 0, False)

    @staticmethod
    def _new_tracker():
        try:
            from eyeGestures import EyeGestures_v3
        except ImportError as exc:
            raise RuntimeError(
                "EyeGestures no está instalado. Ejecuta: "
                "uv pip install --cache-dir .uv-cache --python .venv\\Scripts\\python.exe "
                "-r requirements-eyegestures.txt"
            ) from exc
        tracker = EyeGestures_v3()
        # EyeGestures 3.2.4 defaults FaceMesh to static_image_mode=True. Keep
        # MediaPipe's temporal tracker alive between frames for steadier face reacquisition.
        finder = tracker.finder
        # EyeGestures_v3.getLandmarks already converts BGR to RGB before the
        # finder is called. Avoid FaceFinder's second channel swap.
        finder.mp_face_mesh.close()
        finder.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.4,
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def find_face(frame_rgb):
            result = finder.mp_face_mesh.process(frame_rgb)
            if result.multi_face_landmarks:
                return result
            if float(frame_rgb.mean()) < 5.0:
                return result
            ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            enhanced_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            return finder.mp_face_mesh.process(enhanced_rgb)

        finder.find = find_face
        tracker.finder = finder
        # 5 x 5 grid spans the entire screen. Values 0 and 1 intentionally
        # place the outer targets at the first/last pixel, including corners.
        calibration_map = np.array(
            [(x / 4, y / 4) for y in range(5) for x in range(5)],
            dtype=np.float32,
        )
        tracker.uploadCalibrationMap(calibration_map, context="eye-tracker-pro")
        return tracker

    def start_calibration(self) -> None:
        old_finder = getattr(self._engine, "finder", None)
        old_mesh = getattr(old_finder, "mp_face_mesh", None)
        if old_mesh is not None:
            old_mesh.close()
        self._engine = self._new_tracker()
        self.calibrating = True
        self.ready = False
        self.completed_points = 0
        self._collect_after = time.monotonic() + EYE_GESTURES_POINT_DELAY_SECONDS

    def finish_calibration(self) -> None:
        self.calibrating = False
        self.ready = True

    def cancel_calibration(self) -> None:
        self.calibrating = False
        self.ready = False
        self.completed_points = 0
        self._collect_after = 0.0

    def close(self) -> None:
        finder = getattr(self._engine, "finder", None)
        mesh = getattr(finder, "mp_face_mesh", None)
        if mesh is not None:
            mesh.close()

    def detect(self, frame_bgr: np.ndarray) -> EyeGesturesFrame:
        if float(frame_bgr.mean()) < 2.0 and float(frame_bgr.std()) < 12.0:
            self.last_error = "la camara entrega una imagen negra"
            if self.last_error != self._reported_error:
                print(f"EyeGestures tracking error: {self.last_error}", flush=True)
                self._reported_error = self.last_error
            self.last_frame = EyeGesturesFrame(False, None, None, 0, False)
            return self.last_frame
        try:
            context = "eye-tracker-pro"
            now = time.monotonic()
            collect_calibration = self.calibrating and now >= self._collect_after
            matrix = self._engine.clb.get(context)
            previous_index = matrix.matrix.iterator if matrix is not None else 0
            event, calibration_event = self._engine.step(
                frame_bgr,
                collect_calibration,
                self._screen_width,
                self._screen_height,
                context=context,
            )
            matrix = self._engine.clb[context].matrix
            current_index = matrix.iterator
            if self.calibrating and current_index != previous_index:
                if previous_index == self.total_points - 1 and current_index == 0:
                    self.completed_points = self.total_points
                    self.calibrating = False
                else:
                    self.completed_points = min(self.total_points, current_index)
                self._collect_after = now + EYE_GESTURES_POINT_DELAY_SECONDS
                try:
                    import winsound
                    winsound.Beep(880, 75)
                except Exception:
                    pass
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
