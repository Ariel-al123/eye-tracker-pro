from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


# MediaPipe FaceLandmarker indices (478 points with refined face model)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
NOSE_TIP = 1


@dataclass
class GazeFeatures:
    feature: np.ndarray  # shape (8,)
    face_detected: bool
    left_eye_open: float
    right_eye_open: float
    landmarks: Optional[list] = None


class FaceLandmarkerDetector:
    def __init__(self, model_path: Path, min_detection: float = 0.5, min_tracking: float = 0.5):
        if not model_path.exists():
            raise FileNotFoundError(f"Face landmarker model not found: {model_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    @staticmethod
    def _pt(lm, idx: int, w: int, h: int) -> np.ndarray:
        p = lm[idx]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    @staticmethod
    def _mean_pt(lm, indices: list[int], w: int, h: int) -> np.ndarray:
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in indices], dtype=np.float32)
        return pts.mean(axis=0)

    @staticmethod
    def _eye_openness(top: np.ndarray, bottom: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> float:
        vert = np.linalg.norm(top - bottom)
        horiz = np.linalg.norm(outer - inner)
        if horiz < 1e-6:
            return 0.0
        return float(vert / horiz)

    @staticmethod
    def _norm_in_eye(iris: np.ndarray, outer: np.ndarray, inner: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> tuple[float, float]:
        # Normalize iris position inside eye bounding box. ~0..1 center ~0.5.
        # Allow slight overshoot so gaze can reach screen edges/corners.
        min_x = min(outer[0], inner[0])
        max_x = max(outer[0], inner[0])
        min_y = min(top[1], bottom[1])
        max_y = max(top[1], bottom[1])
        dx = max(max_x - min_x, 1e-6)
        dy = max(max_y - min_y, 1e-6)
        nx = (iris[0] - min_x) / dx
        ny = (iris[1] - min_y) / dy
        return float(np.clip(nx, -0.5, 1.5)), float(np.clip(ny, -0.5, 1.5))

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> GazeFeatures:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.face_landmarks or len(results.face_landmarks[0]) < 478:
            return GazeFeatures(feature=np.zeros(8, dtype=np.float32), face_detected=False,
                                left_eye_open=0.0, right_eye_open=0.0)

        lm = results.face_landmarks[0]

        l_iris = self._mean_pt(lm, LEFT_IRIS, w, h)
        r_iris = self._mean_pt(lm, RIGHT_IRIS, w, h)
        l_outer = self._pt(lm, LEFT_EYE_OUTER, w, h)
        l_inner = self._pt(lm, LEFT_EYE_INNER, w, h)
        l_top = self._pt(lm, LEFT_EYE_TOP, w, h)
        l_bottom = self._pt(lm, LEFT_EYE_BOTTOM, w, h)
        r_outer = self._pt(lm, RIGHT_EYE_OUTER, w, h)
        r_inner = self._pt(lm, RIGHT_EYE_INNER, w, h)
        r_top = self._pt(lm, RIGHT_EYE_TOP, w, h)
        r_bottom = self._pt(lm, RIGHT_EYE_BOTTOM, w, h)
        nose = self._pt(lm, NOSE_TIP, w, h)

        lnx, lny = self._norm_in_eye(l_iris, l_outer, l_inner, l_top, l_bottom)
        rnx, rny = self._norm_in_eye(r_iris, r_outer, r_inner, r_top, r_bottom)
        l_open = self._eye_openness(l_top, l_bottom, l_outer, l_inner)
        r_open = self._eye_openness(r_top, r_bottom, r_outer, r_inner)

        # Head compensation features: nose position relative to frame center and face width.
        face_left = self._pt(lm, 234, w, h)
        face_right = self._pt(lm, 454, w, h)
        face_w = max(np.linalg.norm(face_right - face_left), 1e-6)
        face_cx = (face_left[0] + face_right[0]) / 2.0
        face_cy = (l_top[1] + r_top[1] + l_bottom[1] + r_bottom[1]) / 4.0
        nose_nx = float(np.clip((nose[0] - face_cx) / face_w + 0.5, -0.5, 1.5))
        nose_ny = float(np.clip((nose[1] - face_cy) / face_w + 0.5, -0.5, 1.5))

        # Average iris gives stability; individual eyes give resolution.
        avg_x = (lnx + rnx) / 2.0
        avg_y = (lny + rny) / 2.0

        feat = np.array([avg_x, avg_y, lnx, lny, rnx, rny, nose_nx, nose_ny], dtype=np.float32)
        return GazeFeatures(feature=feat, face_detected=True,
                            left_eye_open=l_open, right_eye_open=r_open, landmarks=lm)
