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
from .eyegestures_engine import EyeGesturesEngine


CALIB_5 = [(0.05, 0.05), (0.95, 0.05), (0.5, 0.5), (0.05, 0.95), (0.95, 0.95)]
CALIB_9 = [
    (0.04, 0.04), (0.5, 0.04), (0.96, 0.04),
    (0.04, 0.5),  (0.5, 0.5),  (0.96, 0.5),
    (0.04, 0.96), (0.5, 0.96), (0.96, 0.96),
]
CALIB_25 = [(x / 4, y / 4) for y in range(5) for x in range(5)]


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
            self.eye_gestures = None
        elif self.cfg.gaze_engine.lower() == "eyegestures":
            self.eye_theia = None
            self.detector = None
            self.eye_gestures = EyeGesturesEngine(self.screen_w, self.screen_h)
        else:
            self.eye_theia = None
            self.eye_gestures = None
            self.detector = FaceLandmarkerDetector(model_path)
        self.eye_observation = None
        self.eye_gestures_point = None
        self.overlay = OverlayWindow(self.cfg.buttons) if self.cfg.overlay_enabled else None

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.calibrating = False
        self.calib_points: list[tuple[float, float]] = []
        self.calib_index = 0
        self.calib_samples: list[np.ndarray] = []
        self.calib_point_started_at = 0.0
        self.last_calibration_sample_at = 0.0
        self.calib_gaze_history: list[tuple[float, float]] = []
        self.last_face_seen_at = 0.0
        self.paused_by_face_loss = False
        self.fine_tune_mode = False
        self.status = "Presiona C para calibrar"
        self.calibration_window_created = False

    def _open_camera(self) -> None:
        configured = max(0, int(self.cfg.camera_index))
        camera_indices = [configured] + [i for i in range(6) if i != configured]
        black_indices: list[int] = []

        for index in camera_indices:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            frame = None
            for _ in range(8):
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    frame = candidate
                time.sleep(0.04)

            if frame is None:
                cap.release()
                continue

            # Some virtual/blocked cameras report isOpened=True but return a black image.
            if float(frame.mean()) < 2.0 and float(frame.std()) < 12.0:
                black_indices.append(index)
                cap.release()
                continue

            self.cap = cap
            self.cfg.camera_index = index
            return

        if black_indices:
            indexes = ", ".join(str(i) for i in black_indices)
            raise RuntimeError(
                f"Camara(s) {indexes} abre(n), pero envia(n) imagen negra. "
                "Activa la camara fisica, quita la tapa de privacidad y permite el acceso "
                "en Configuracion de Windows > Privacidad y seguridad > Camara."
            )
        raise RuntimeError(
            "Windows no ofrece una camara que OpenCV pueda abrir "
            f"(indices revisados: {', '.join(map(str, camera_indices))}). "
            "Conecta/activa la webcam y revisa Configuracion de Windows > "
            "Privacidad y seguridad > Camara."
        )

    def _calibration_targets(self) -> list[tuple[int, int]]:
        if self.cfg.calibration_points >= 25:
            pts = CALIB_25
        else:
            pts = CALIB_9 if self.cfg.calibration_points >= 9 else CALIB_5
        self.calib_points = pts
        return [
            (int(round(x * (self.screen_w - 1))), int(round(y * (self.screen_h - 1))))
            for x, y in pts
        ]

    def _show_calibration(self, frame: np.ndarray) -> None:
        name = "EyeTrackerPro - Calibracion"
        if not self.calibration_window_created:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.calibration_window_created = True
        cv2.imshow(name, frame)

    def start_calibration(self) -> None:
        if self.eye_gestures:
            if self.calibrating:
                self.eye_gestures.finish_calibration()
                self.calibrating = False
                self.filter.reset()
                self.controller.set_paused(False)
                self.status = "Calibracion terminada. Mirada activa."
            else:
                self.eye_gestures.start_calibration()
                self.calibrating = True
                self.filter.reset()
                self.controller.set_paused(True)
                self.status = f"EyeGestures: punto 1/{self.eye_gestures.total_points}"
            return
        if self.eye_theia and self.eye_theia.training:
            self.status = "Espera a que termine la personalización de EyeTheia."
            return
        self._calibration_targets()
        self.calibrating = True
        self.calib_index = 0
        self.calib_samples = []
        self.calib_gaze_history = []
        self.calib_point_started_at = time.perf_counter()
        self.last_calibration_sample_at = 0.0
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
        targets = [
            (int(round(x * (self.screen_w - 1))), int(round(y * (self.screen_h - 1))))
            for x, y in self.calib_points
        ]
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
        stable = True
        if feats.face_detected and eyes_open:
            now = time.perf_counter()
            interval = max(0, self.cfg.calibration_sample_delay_ms) / 1000.0
            sample_due = now - self.last_calibration_sample_at >= interval
            if self.eye_theia and self.eye_observation:
                raw_gaze = self.eye_theia.predict(self.eye_observation)
                if raw_gaze:
                    self.calib_gaze_history.append((raw_gaze[0] / self.screen_w, raw_gaze[1] / self.screen_h))
                    self.calib_gaze_history = self.calib_gaze_history[-8:]
                    stable = len(self.calib_gaze_history) >= 5 and bool(
                        np.max(np.std(np.asarray(self.calib_gaze_history), axis=0)) <= 0.055
                    )
                else:
                    stable = False
            if sample_due and stable:
                self.last_calibration_sample_at = now
                self.calib_samples.append(feats.feature.copy())
                if self.eye_theia:
                    self.eye_theia.add_calibration_sample(self.eye_observation, tx, ty)
            progress = len(self.calib_samples) / max(1, required_samples)
            bar_w = 500
            cv2.rectangle(canvas, (60, 170), (60 + bar_w, 195), (80, 80, 80), -1)
            cv2.rectangle(canvas, (60, 170), (60 + int(bar_w * progress), 195), (0, 200, 0), -1)
            cv2.putText(canvas, f"{int(progress*100)}%", (570, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            self.calib_gaze_history = []
            if not feats.face_detected:
                dim = bool(self.eye_theia and self.eye_theia.last_brightness < 82)
                hint = "Aumenta la luz frente a tu cara" if dim else "Centra tu cara y alejate un poco de la camara"
                cv2.putText(canvas, "Rostro no detectado - " + hint, (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2, cv2.LINE_AA)
            elif not eyes_open:
                cv2.putText(canvas, "Abre los dos ojos y mira al centro del punto", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 150, 255), 2, cv2.LINE_AA)
            elif self.eye_theia and not stable:
                cv2.putText(canvas, "Fija la mirada y mantente quieto para estabilizar la muestra", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        if len(self.calib_samples) >= required_samples:
            # Use median sample to reduce outliers
            med = np.median(np.stack(self.calib_samples, axis=0), axis=0)
            if not self.eye_theia:
                self.mapper.add_sample(med, tx, ty)
            self.calib_index += 1
            self.calib_samples = []
            self.calib_gaze_history = []
            self.calib_point_started_at = time.perf_counter()
            self.last_calibration_sample_at = 0.0
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
                if not self.controller.paused:
                    self.controller.set_paused(True)
                    self.paused_by_face_loss = True
                self.status = "Rostro no detectado - cursor en pausa"
            return None, 0.0

        self.last_face_seen_at = now
        if self.paused_by_face_loss:
            self.controller.set_paused(False)
            self.paused_by_face_loss = False
            self.filter.reset()
            self.status = "Rostro recuperado - seguimiento activo"
        # Do not let a blink or a nearly closed eyelid pull the cursor.
        if feats.left_eye_open <= 0.08 or feats.right_eye_open <= 0.08:
            self.controller.dwell.active = False
            return None, 0.0
        if self.eye_theia:
            if self.eye_theia.training:
                self.status = "Personalizando EyeTheia..."
                return None, 0.0
            pred = self.eye_theia.predict(self.eye_observation) if self.eye_observation else None
        elif self.eye_gestures:
            pred = self.eye_gestures_point if self.eye_gestures.ready else None
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
        if self.eye_gestures:
            return self.eye_gestures.ready
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
        if self.eye_theia and self.eye_theia.last_face_box:
            x1, y1, x2, y2 = self.eye_theia.last_face_box
            sx, sy = w / max(frame.shape[1], 1), h / max(frame.shape[0], 1)
            cv2.rectangle(view, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(view, f"Face: {'OK' if feats.face_detected else 'NO'}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        if self.eye_theia and not feats.face_detected:
            hint = "ADD FRONT LIGHT" if self.eye_theia.last_brightness < 105 else "CENTER FACE / REDUCE GLASSES GLARE"
            cv2.putText(view, hint, (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
        elif self.eye_theia and self.eye_theia.last_detection_retry:
            cv2.putText(view, self.eye_theia.last_detection_source.upper(), (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2, cv2.LINE_AA)
        elif self.eye_gestures and self.eye_gestures.last_error:
            cv2.putText(view, "EyeGestures error - revisa la terminal", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f"Calibrado: {'SI' if self._gaze_ready() else 'NO'}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        mouse_on = not self.controller.paused and self.cfg.move_cursor_enabled
        cv2.putText(view, f"Mouse: {'ON' if mouse_on else 'OFF'}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if mouse_on else (0, 165, 255), 2, cv2.LINE_AA)
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
        try:
            self._open_camera()
        except Exception:
            self.cleanup()
            raise
        self.running = True
        self.last_face_seen_at = time.perf_counter()
        cv2.namedWindow("EyeTrackerPro - Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("EyeTrackerPro - Preview", 960, 540)
        cv2.moveWindow("EyeTrackerPro - Preview", 10, 10)
        if self.overlay:
            self.overlay.win_x = max(10, self.screen_w - self.overlay.width - 10)
            self.overlay.win_y = max(10, self.screen_h - self.overlay.height - 10)

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
                elif self.eye_gestures:
                    gesture_frame = self.eye_gestures.detect(frame)
                    self.eye_gestures_point = gesture_frame.point
                    feats = GazeFeatures(
                        feature=np.zeros(1, dtype=np.float32),
                        face_detected=gesture_frame.face_detected,
                        left_eye_open=0.0 if gesture_frame.blink else 1.0,
                        right_eye_open=0.0 if gesture_frame.blink else 1.0,
                    )
                    if self.calibrating:
                        if self.eye_gestures.completed_points >= self.eye_gestures.total_points:
                            self.eye_gestures.finish_calibration()
                            self.calibrating = False
                            self.filter.reset()
                            self.controller.set_paused(False)
                            self.status = "Calibracion completa. Cursor activo."
                        else:
                            self.status = (
                                f"EyeGestures: {self.eye_gestures.completed_points + 1}/"
                                f"{self.eye_gestures.total_points} - fija la mirada en el punto"
                            )
                else:
                    feats = self.detector.detect(frame, ts)

                if self.calibrating and self.eye_gestures:
                    calib_view = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                    calib_view[:] = (18, 18, 18)
                    target = self.eye_gestures.last_frame.target
                    if target and self.eye_gestures.completed_points < self.eye_gestures.total_points:
                        radius = max(12, self.eye_gestures.last_frame.acceptance_radius)
                        cv2.circle(calib_view, target, radius, (60, 60, 60), 2, cv2.LINE_AA)
                        cv2.circle(calib_view, target, 12, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.circle(calib_view, target, 4, (0, 0, 255), -1, cv2.LINE_AA)
                    if self.eye_gestures_point and feats.face_detected:
                        cv2.circle(calib_view, self.eye_gestures_point, 18, (255, 120, 0), -1, cv2.LINE_AA)
                        cv2.circle(calib_view, self.eye_gestures_point, 24, (255, 255, 255), 2, cv2.LINE_AA)
                    progress_text = (
                        f"Calibracion: {self.eye_gestures.completed_points + 1}/"
                        f"{self.eye_gestures.total_points} - mira el punto; espera 1 s entre puntos"
                    )
                    cv2.putText(calib_view, progress_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
                    cv2.putText(calib_view, "Punto azul: posicion estimada de la mirada", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 160, 40), 2, cv2.LINE_AA)
                    if not feats.face_detected:
                        cv2.putText(calib_view, "SIN ROSTRO: revisa la imagen y el indice de la camara", (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)
                    self._show_calibration(calib_view)
                elif self.calibrating:
                    calib_view = self._draw_calibration(frame, feats)
                    self._show_calibration(calib_view)
                else:
                    try:
                        cv2.destroyWindow("EyeTrackerPro - Calibracion")
                        self.calibration_window_created = False
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
                        if self.eye_gestures:
                            self.eye_gestures.cancel_calibration()
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
                    self.paused_by_face_loss = False
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
        if self.eye_gestures:
            try:
                self.eye_gestures.close()
            except Exception:
                pass
        if self.overlay:
            self.overlay.destroy()
        cv2.destroyAllWindows()
        save_config(self.base_dir, self.cfg)
