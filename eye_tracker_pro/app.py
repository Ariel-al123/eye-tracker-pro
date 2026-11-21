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
from .eyetheia_engine import EyeTheiaEngine


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
        if self.cfg.gaze_engine.lower() == "eyetheia":
            self.eye_theia = EyeTheiaEngine(base_dir, self.screen_w, self.screen_h)
            self.detector = None
        else:
            self.eye_theia = None
            self.detector = FaceLandmarkerDetector(model_path)
        self.eye_observation = None
        self.overlay = OverlayWindow(self.cfg.buttons) if self.cfg.overlay_enabled else None

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.calibrating = False
        self.calib_points: list[tuple[float, float]] = []
        self.calib_index = 0
        self.calib_samples: list[np.ndarray] = []
        self.calib_point_started_at = 0.0
        self.last_face_seen_at = 0.0
        self.fine_tune_mode = False
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
        if self.eye_theia and self.eye_theia.training:
            self.status = "Espera a que termine la personalización de EyeTheia."
            return
        self._calibration_targets()
        self.calibrating = True
        self.calib_index = 0
        self.calib_samples = []
        self.calib_point_started_at = time.perf_counter()
        self.mapper.clear()
        self.filter.reset()
        if self.eye_theia:
            self.eye_theia.begin_calibration()
        self.status = "Calibrando: mira el punto rojo"
        self.controller.set_paused(True)

    def _finish_calibration(self) -> None:
        if self.eye_theia:
            self.calibrating = False
            self.calib_samples = []
            self.filter.reset()
            try:
                self.eye_theia.train_async()
                self.status = "Personalizando EyeTheia con tu calibración..."
            except Exception as exc:
                self.status = f"Falló la personalización: {exc}"
                self.controller.set_paused(True)
            return
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
        settle = 1.0
        elapsed = time.perf_counter() - self.calib_point_started_at
        if elapsed < settle:
            cv2.putText(canvas, "Preparando...", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
            return canvas

        eyes_open = feats.left_eye_open > 0.08 and feats.right_eye_open > 0.08
        required_samples = (self.cfg.eyetheia_samples_per_target if self.eye_theia
                            else self.cfg.calibration_samples_per_point)
        if feats.face_detected and eyes_open:
            self.calib_samples.append(feats.feature.copy())
            if self.eye_theia:
                self.eye_theia.add_calibration_sample(self.eye_observation, tx, ty)
            progress = len(self.calib_samples) / max(1, required_samples)
            bar_w = 500
            cv2.rectangle(canvas, (60, 170), (60 + bar_w, 195), (80, 80, 80), -1)
            cv2.rectangle(canvas, (60, 170), (60 + int(bar_w * progress), 195), (0, 200, 0), -1)
            cv2.putText(canvas, f"{int(progress*100)}%", (570, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Rostro no detectado", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2, cv2.LINE_AA)

        if len(self.calib_samples) >= required_samples:
            # Use median sample to reduce outliers
            med = np.median(np.stack(self.calib_samples, axis=0), axis=0)
            if not self.eye_theia:
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
        if action == "fine_tune":
            self.fine_tune_mode = not self.fine_tune_mode
            self.status = "Ajuste fino: ON" if self.fine_tune_mode else "Ajuste fino: OFF"
            return
        result = self.controller.perform_action(action)
        if action == "pause":
            self.status = "Pausado" if self.controller.paused else "Mirada activa"
        elif result not in {"ignored_paused"}:
            self.status = f"Acción: {action}"

    def _process_gaze(self, feats: GazeFeatures) -> tuple[Optional[tuple[int, int]], float]:
        now = time.perf_counter()
        if not feats.face_detected:
            self.filter.reset()
            self.controller.dwell.active = False
            if self.cfg.pause_on_face_lost and (now - self.last_face_seen_at) * 1000 > self.cfg.face_lost_pause_ms:
                self.status = "Rostro no detectado - cursor en pausa"
            return None, 0.0

        self.last_face_seen_at = now
        # Do not let a blink or a nearly closed eyelid pull the cursor.
        if feats.left_eye_open <= 0.08 or feats.right_eye_open <= 0.08:
            self.controller.dwell.active = False
            return None, 0.0
        if self.eye_theia:
            if self.eye_theia.training:
                self.status = "Personalizando EyeTheia..."
                return None, 0.0
            pred = self.eye_theia.predict(self.eye_observation) if self.eye_observation else None
        else:
            pred = self.mapper.predict(feats.feature)
        if pred is None:
            self.status = "Sin calibración. Presiona C."
            return None, 0.0

        sx, sy = pred
        sx = max(0, min(self.screen_w - 1, sx))
        sy = max(0, min(self.screen_h - 1, sy))
        fx, fy = self.filter.filter(sx, sy, now)

        # Fine adjustment: gain around center + offset
        cx, cy = self.screen_w / 2.0, self.screen_h / 2.0
        fx = cx + (fx - cx) * float(self.cfg.fine_gain_x) + float(self.cfg.fine_offset_x)
        fy = cy + (fy - cy) * float(self.cfg.fine_gain_y) + float(self.cfg.fine_offset_y)

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

        if not self.controller.paused and self._gaze_ready():
            self.status = "Mirada activa"
        return (ix, iy), dwell_progress

    def _gaze_ready(self) -> bool:
        if self.eye_theia:
            return self.eye_theia.ready
        return self.mapper.calibrated

    def _finish_eye_theia_training_if_ready(self) -> None:
        if not self.eye_theia or not self.eye_theia.training_done:
            return
        self.eye_theia.training_done = False
        if self.eye_theia.training_error is not None:
            self.eye_theia.ready = False
            self.status = f"Falló EyeTheia: {self.eye_theia.training_error}"
            self.controller.set_paused(True)
            return
        self.filter.reset()
        self.eye_theia.ready = True
        self.controller.set_paused(False)
        self.status = "EyeTheia listo. Mirada activa."

    def _draw_preview(self, frame: np.ndarray, feats: GazeFeatures, gaze: Optional[tuple[int, int]], dwell: float) -> np.ndarray:
        view = frame.copy()
        if self.cfg.mirror_preview:
            view = cv2.flip(view, 1)
        h, w = view.shape[:2]
        color = (0, 200, 0) if feats.face_detected else (0, 0, 255)
        cv2.putText(view, f"Face: {'OK' if feats.face_detected else 'NO'}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(view, f"Calibrado: {'SI' if self._gaze_ready() else 'NO'}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f"Mouse: {'OFF' if self.controller.paused else 'ON'}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if not self.controller.paused else (0, 165, 255), 2, cv2.LINE_AA)
        if gaze:
            cv2.putText(view, f"Gaze: {gaze[0]},{gaze[1]} dwell {int(dwell*100)}%", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        ft = f"FINE gx={self.cfg.fine_gain_x:.2f} gy={self.cfg.fine_gain_y:.2f} ox={int(self.cfg.fine_offset_x)} oy={int(self.cfg.fine_offset_y)}"
        ft_color = (0, 255, 255) if self.fine_tune_mode else (160, 160, 160)
        cv2.putText(view, ft, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ft_color, 2, cv2.LINE_AA)
        if self.fine_tune_mode:
            cv2.putText(view, "A/D ox  W/S oy  J/L gx  I/K gy  R reset  F/ESC salir", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, "C=calibrar  P=pausa  M=mouse  F=ajuste fino  ESC=salir", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
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
                if self.eye_theia:
                    self.eye_observation = None if self.eye_theia.training else self.eye_theia.observe(frame)
                    observation = self.eye_observation
                    feats = GazeFeatures(
                        feature=np.zeros(1, dtype=np.float32),
                        face_detected=observation is not None,
                        left_eye_open=observation.left_eye_open if observation else 0.0,
                        right_eye_open=observation.right_eye_open if observation else 0.0,
                    )
                    self._finish_eye_theia_training_if_ready()
                else:
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

                key_raw = cv2.waitKeyEx(1)
                key = key_raw & 0xFF if key_raw != -1 else -1

                if key == 27:  # ESC
                    if self.calibrating:
                        self.calibrating = False
                        self.status = "Calibración cancelada"
                        self.controller.set_paused(True)
                    elif self.fine_tune_mode:
                        self.fine_tune_mode = False
                        self.status = "Ajuste fino desactivado"
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
                elif key in (ord('f'), ord('F')):
                    self.fine_tune_mode = not self.fine_tune_mode
                    self.status = "Ajuste fino: ON (WASD offset, IJKL gain, R reset)" if self.fine_tune_mode else "Ajuste fino: OFF"
                elif self.fine_tune_mode:
                    step = int(self.cfg.fine_step_px)
                    gstep = float(self.cfg.fine_gain_step)
                    if key in (ord('a'), ord('A')):  # offset X -
                        self.cfg.fine_offset_x -= step
                    elif key in (ord('d'), ord('D')):  # offset X +
                        self.cfg.fine_offset_x += step
                    elif key in (ord('w'), ord('W')):  # offset Y -
                        self.cfg.fine_offset_y -= step
                    elif key in (ord('s'), ord('S')):  # offset Y +
                        self.cfg.fine_offset_y += step
                    elif key in (ord('j'), ord('J')):  # gain X -
                        self.cfg.fine_gain_x = max(0.5, round(self.cfg.fine_gain_x - gstep, 3))
                    elif key in (ord('l'), ord('L')):  # gain X +
                        self.cfg.fine_gain_x = min(2.0, round(self.cfg.fine_gain_x + gstep, 3))
                    elif key in (ord('i'), ord('I')):  # gain Y +
                        self.cfg.fine_gain_y = min(2.0, round(self.cfg.fine_gain_y + gstep, 3))
                    elif key in (ord('k'), ord('K')):  # gain Y -
                        self.cfg.fine_gain_y = max(0.5, round(self.cfg.fine_gain_y - gstep, 3))
                    elif key in (ord('r'), ord('R')):
                        self.cfg.fine_gain_x = 1.0
                        self.cfg.fine_gain_y = 1.0
                        self.cfg.fine_offset_x = 0.0
                        self.cfg.fine_offset_y = 0.0
                    self.status = (
                        f"Ajuste fino gx={self.cfg.fine_gain_x:.2f} gy={self.cfg.fine_gain_y:.2f} "
                        f"ox={self.cfg.fine_offset_x:.0f} oy={self.cfg.fine_offset_y:.0f}"
                    )
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
        if self.eye_theia:
            try:
                self.eye_theia.close()
            except Exception:
                pass
        if self.overlay:
            self.overlay.destroy()
        cv2.destroyAllWindows()
        save_config(self.base_dir, self.cfg)
