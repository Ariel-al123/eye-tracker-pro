from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyautogui

from .config import AppConfig, load_config, save_config
from .controller import CursorController
from .detector import FaceLandmarkerDetector, GazeFeatures
from .filter import PointFilter
from .mapper import GazeMapper
from .overlay import OverlayWindow


CALIB_5 = [(0.05, 0.05), (0.95, 0.05), (0.5, 0.5), (0.05, 0.95), (0.95, 0.95)]
CALIB_9 = [
    (0.04, 0.04), (0.5, 0.04), (0.96, 0.04),
    (0.04, 0.5),  (0.5, 0.5),  (0.96, 0.5),
    (0.04, 0.96), (0.5, 0.96), (0.96, 0.96),
]


class EyeTrackerApp:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cfg: AppConfig = load_config(base_dir)
        self.screen_w, self.screen_h = pyautogui.size()
        self.controller = CursorController(self.screen_w, self.screen_h, self.cfg.move_deadzone_px)
        self.mapper = GazeMapper()
        self.filter = PointFilter(self.cfg.filter_min_cutoff, self.cfg.filter_beta, self.cfg.filter_d_cutoff)

        model_path = base_dir / "models" / "face_landmarker.task"
        self.detector = FaceLandmarkerDetector(model_path)
        self.overlay = OverlayWindow(self.cfg.buttons) if self.cfg.overlay_enabled else None

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.calibrating = False
        self.calib_points: list[tuple[float, float]] = []
        self.calib_index = 0
        self.calib_samples: list[np.ndarray] = []
        self.calib_point_started_at = 0.0
        self.last_face_seen_at = 0.0
        self.status = "Presiona C para calibrar"

    def _open_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        if not self.cap.isOpened():
            # fallback without DSHOW
            self.cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara. Verifica permisos/conexión.")

    def _calibration_targets(self) -> list[tuple[int, int]]:
        pts = CALIB_9 if self.cfg.calibration_points >= 9 else CALIB_5
        self.calib_points = pts
        return [(int(x * self.screen_w), int(y * self.screen_h)) for x, y in pts]

    def start_calibration(self) -> None:
        self._calibration_targets()
        self.calibrating = True
        self.calib_index = 0
        self.calib_samples = []
        self.calib_point_started_at = time.perf_counter()
        self.mapper.clear()
        self.filter.reset()
        self.status = "Calibrando: mira el punto rojo"
        self.controller.set_paused(True)

    def _finish_calibration(self) -> None:
        ok = self.mapper.fit()
        self.calibrating = False
        self.calib_samples = []
        self.filter.reset()
        if ok:
            self.status = "Calibración completa. Mirada activa."
            self.controller.set_paused(False)
        else:
            self.status = "Calibración insuficiente. Presiona C para reintentar."

    def _draw_calibration(self, frame: np.ndarray, feats: GazeFeatures) -> np.ndarray:
        canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)
        targets = [(int(x * self.screen_w), int(y * self.screen_h)) for x, y in self.calib_points]
        tx, ty = targets[self.calib_index]

        # draw all targets
        for i, (x, y) in enumerate(targets):
            color = (90, 90, 90) if i < self.calib_index else (140, 140, 140)
            cv2.circle(canvas, (x, y), 8, color, -1, cv2.LINE_AA)
        # active target
        pulse = int(6 * (1 + np.sin(time.perf_counter() * 4)))
        cv2.circle(canvas, (tx, ty), self.cfg.calibration_point_radius + pulse, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 7, (0, 0, 255), -1, cv2.LINE_AA)

        msg = f"Calibrando {self.calib_index + 1}/{len(targets)} - mira el punto rojo"
        cv2.putText(canvas, msg, (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Mantén la cabeza estable. ESC cancela.", (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)

        # settle period
        settle = 0.7
        elapsed = time.perf_counter() - self.calib_point_started_at
        if elapsed < settle:
            cv2.putText(canvas, "Preparando...", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
            return canvas

        if feats.face_detected:
            self.calib_samples.append(feats.feature.copy())
            progress = len(self.calib_samples) / max(1, self.cfg.calibration_samples_per_point)
            bar_w = 500
            cv2.rectangle(canvas, (60, 170), (60 + bar_w, 195), (80, 80, 80), -1)
            cv2.rectangle(canvas, (60, 170), (60 + int(bar_w * progress), 195), (0, 200, 0), -1)
            cv2.putText(canvas, f"{int(progress*100)}%", (570, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Rostro no detectado", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2, cv2.LINE_AA)

        if len(self.calib_samples) >= self.cfg.calibration_samples_per_point:
            # Use median sample to reduce outliers
            med = np.median(np.stack(self.calib_samples, axis=0), axis=0)
            self.mapper.add_sample(med, tx, ty)
            self.calib_index += 1
            self.calib_samples = []
            self.calib_point_started_at = time.perf_counter()
            if self.calib_index >= len(targets):
                self._finish_calibration()
        return canvas

    def _handle_action(self, action: str) -> None:
        if action == "calibrate":
            self.start_calibration()
            return
        result = self.controller.perform_action(action)
        if action == "pause":
            self.status = "Pausado" if self.controller.paused else "Mirada activa"
        elif result not in {"ignored_paused"}:
            self.status = f"Acción: {action}"

    def _process_gaze(self, feats: GazeFeatures) -> tuple[Optional[tuple[int, int]], float]:
        now = time.perf_counter()
        if not feats.face_detected:
            if self.cfg.pause_on_face_lost and (now - self.last_face_seen_at) * 1000 > self.cfg.face_lost_pause_ms:
                self.status = "Rostro no detectado - cursor en pausa"
            return None, 0.0

        self.last_face_seen_at = now
        pred = self.mapper.predict(feats.feature)
        if pred is None:
            self.status = "Sin calibración. Presiona C."
            return None, 0.0

        sx, sy = pred
        sx = max(0, min(self.screen_w - 1, sx))
        sy = max(0, min(self.screen_h - 1, sy))
        fx, fy = self.filter.filter(sx, sy, now)
        ix, iy = self.controller.clamp(fx, fy)

        if self.cfg.move_cursor_enabled and not self.controller.paused:
            self.controller.move(ix, iy)

        dwell_progress = 0.0
        action_target = None
        if self.overlay:
            btn = self.overlay.button_at(ix, iy)
            if btn is not None:
                dwell_progress = self.controller.dwell_progress(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                if self.controller.dwell_ready(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                    action_target = btn.action
            else:
                # regular dwell click on screen (optional): only if enabled and no button
                if self.cfg.dwell_enabled and not self.controller.paused:
                    dwell_progress = self.controller.dwell_progress(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                    if self.controller.dwell_ready(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                        action_target = "left_click"
                else:
                    self.controller.dwell.active = False
        else:
            if self.cfg.dwell_enabled and not self.controller.paused:
                dwell_progress = self.controller.dwell_progress(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                if self.controller.dwell_ready(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                    action_target = "left_click"

        if action_target:
            self._handle_action(action_target)

        if not self.controller.paused and self.mapper.calibrated:
            self.status = "Mirada activa"
        return (ix, iy), dwell_progress

    def _draw_preview(self, frame: np.ndarray, feats: GazeFeatures, gaze: Optional[tuple[int, int]], dwell: float) -> np.ndarray:
        view = frame.copy()
        if self.cfg.mirror_preview:
            view = cv2.flip(view, 1)
        h, w = view.shape[:2]
        color = (0, 200, 0) if feats.face_detected else (0, 0, 255)
        cv2.putText(view, f"Face: {'OK' if feats.face_detected else 'NO'}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(view, f"Calibrado: {'SI' if self.mapper.calibrated else 'NO'}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f"Mouse: {'OFF' if self.controller.paused else 'ON'}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if not self.controller.paused else (0, 165, 255), 2, cv2.LINE_AA)
        if gaze:
            cv2.putText(view, f"Gaze: {gaze[0]},{gaze[1]} dwell {int(dwell*100)}%", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(view, "C=calibrar  P=pausa  M=mouse  ESC=salir", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
        return view

    def run(self) -> None:
        self._open_camera()
        self.running = True
        self.last_face_seen_at = time.perf_counter()
        cv2.namedWindow("EyeTrackerPro - Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("EyeTrackerPro - Preview", 960, 540)

        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.status = "Error leyendo cámara"
                    break

                ts = int(time.perf_counter() * 1000)
                feats = self.detector.detect(frame, ts)

                if self.calibrating:
                    calib_view = self._draw_calibration(frame, feats)
                    cv2.namedWindow("EyeTrackerPro - Calibracion", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("EyeTrackerPro - Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("EyeTrackerPro - Calibracion", calib_view)
                else:
                    try:
                        cv2.destroyWindow("EyeTrackerPro - Calibracion")
                    except Exception:
                        pass

                gaze, dwell = (None, 0.0)
                if not self.calibrating:
                    gaze, dwell = self._process_gaze(feats)

                if self.overlay:
                    self.overlay.state.status = self.status
                    self.overlay.state.paused = self.controller.paused
                    if gaze:
                        self.overlay.state.gaze_x, self.overlay.state.gaze_y = gaze
                    self.overlay.state.dwell_progress = dwell
                    self.overlay.render()

                preview = self._draw_preview(frame, feats, gaze, dwell)
                cv2.imshow("EyeTrackerPro - Preview", preview)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    if self.calibrating:
                        self.calibrating = False
                        self.status = "Calibración cancelada"
                        self.controller.set_paused(True)
                    else:
                        self.running = False
                elif key in (ord('c'), ord('C')):
                    self.start_calibration()
                elif key in (ord('p'), ord('P')):
                    self.controller.toggle_pause()
                    self.status = "Pausado" if self.controller.paused else "Mirada activa"
                elif key in (ord('m'), ord('M')):
                    self.cfg.move_cursor_enabled = not self.cfg.move_cursor_enabled
                    self.status = f"Movimiento cursor: {'ON' if self.cfg.move_cursor_enabled else 'OFF'}"
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            self.detector.close()
        except Exception:
            pass
        if self.overlay:
            self.overlay.destroy()
        cv2.destroyAllWindows()
        save_config(self.base_dir, self.cfg)
