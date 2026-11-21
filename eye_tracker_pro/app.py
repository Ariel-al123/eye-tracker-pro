from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyautogui

from .config import AppConfig, load_config, save_config, shortcut_kind
from .controller import CursorController
from .detector import FaceLandmarkerDetector, GazeFeatures
from .filter import PointFilter
from .eyegestures_engine import EyeGesturesEngine
from .hotkeys import LowLevelHotkeyListener
from .mapper import GazeMapper
from .overlay import OverlayWindow
from .suggestions import SuggestionEngine


CALIB_5 = [(0.05, 0.05), (0.95, 0.05), (0.5, 0.5), (0.05, 0.95), (0.95, 0.95)]
CALIB_9 = [
    (0.04, 0.04), (0.5, 0.04), (0.96, 0.04),
    (0.04, 0.5),  (0.5, 0.5),  (0.96, 0.5),
    (0.04, 0.96), (0.5, 0.96), (0.96, 0.96),
]
CALIB_25 = [
    (x, y)
    for y in (0.04, 0.27, 0.5, 0.73, 0.96)
    for x in (0.04, 0.27, 0.5, 0.73, 0.96)
]


class _WinRect(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class _GuiThreadInfo(ctypes.Structure):
    _fields_ = [
        ("cb_size", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("active", wintypes.HWND),
        ("focus", wintypes.HWND),
        ("capture", wintypes.HWND),
        ("menu_owner", wintypes.HWND),
        ("move_size", wintypes.HWND),
        ("caret", wintypes.HWND),
        ("caret_rect", _WinRect),
    ]


class EyeTrackerApp:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cfg: AppConfig = load_config(base_dir)
        self.suggestion_engine = SuggestionEngine(self.cfg.suggestion_history)
        self.screen_w, self.screen_h = pyautogui.size()
        self.controller = CursorController(self.screen_w, self.screen_h, self.cfg.move_deadzone_px)
        self.mapper = GazeMapper()
        self.calibration_path = base_dir / "calibration.json"
        self.eye_gestures: Optional[EyeGesturesEngine] = None
        self.eye_gestures_point: Optional[tuple[int, int]] = None
        self.engine_warning: Optional[str] = None
        requested_engine = str(self.cfg.gaze_engine).lower()
        if requested_engine == "eyegestures":
            try:
                self.eye_gestures = EyeGesturesEngine(
                    self.screen_w,
                    self.screen_h,
                    base_dir / "eye_gestures_calibration.json",
                )
            except Exception as exc:
                self.engine_warning = f"EyeGestures unavailable; using MediaPipe mapper ({exc})"
        elif requested_engine not in {"mediapipe", ""}:
            self.engine_warning = f"Unknown gaze engine '{requested_engine}'; using MediaPipe mapper"
        self.detector: Optional[FaceLandmarkerDetector] = None
        if self.eye_gestures is None:
            model_path = base_dir / "models" / "face_landmarker.task"
            self.detector = FaceLandmarkerDetector(model_path)
            self.active_gaze_engine = "mediapipe"
            self.calibration_loaded = self.mapper.load(self.calibration_path, self.screen_w, self.screen_h)
        else:
            self.active_gaze_engine = "eyegestures"
            self.calibration_loaded = self.eye_gestures.ready
        self.filter = PointFilter(self.cfg.filter_min_cutoff, self.cfg.filter_beta, self.cfg.filter_d_cutoff)

        self.overlay = OverlayWindow(self.screen_w, self.screen_h, self.cfg.keybind_pages) if self.cfg.overlay_enabled else None
        if self.overlay:
            self.overlay.mode = self.cfg.overlay_mode
            self.overlay.language = self.cfg.language
            self.overlay.on_activate = self._activate_debug_control
            self.overlay.set_mouse_debug(self.cfg.debug_mouse_interaction)

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.calibrating = False
        self.calib_points: list[tuple[float, float]] = []
        self.calib_index = 0
        self.calib_samples: list[np.ndarray] = []
        self.calib_feature_history: list[np.ndarray] = []
        self.calib_point_started_at = 0.0
        self.last_calibration_sample_at = 0.0
        self.last_face_seen_at = 0.0
        self.paused_by_face_loss = False
        self.fine_tune_mode = False
        if self.engine_warning:
            self.status = self.engine_warning
        else:
            self.status = "Calibración guardada cargada" if self.calibration_loaded else "Presiona C para calibrar"
        self._latched_control_id: Optional[str] = None
        self._global_keys_down: set[int] = set()
        self._last_text_focus_poll = 0.0
        self._text_focus_id: Optional[tuple[int, int, int]] = None
        self._text_focus_since: Optional[float] = None
        self._text_focus_missing_since: Optional[float] = None
        self._auto_keyboard_handled_focus: Optional[tuple[int, int, int]] = None
        self._wink_candidate_since: Optional[float] = None
        self._wink_candidate_side: Optional[str] = None
        self._wink_fired = False
        self._last_wink_click_at = 0.0
        self.hotkeys = LowLevelHotkeyListener(
            fine_tune_active=lambda: self.fine_tune_mode,
            escape_allowed=lambda: self.calibrating or self.fine_tune_mode or self._is_eyetracker_foreground(),
        )

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
            if float(frame.mean()) < 2.0 and float(frame.std()) < 12.0:
                black_indices.append(index)
                cap.release()
                continue
            self.cap = cap
            self.cfg.camera_index = index
            return
        if black_indices:
            indexes = ", ".join(map(str, black_indices))
            raise RuntimeError(
                f"Camara(s) {indexes} abre(n), pero envia(n) imagen negra. "
                "Activa la webcam, quita la tapa de privacidad y revisa los permisos de cámara de Windows."
            )
        raise RuntimeError(
            "No se encontró una cámara utilizable en los índices: "
            + ", ".join(map(str, camera_indices))
        )

    def _calibration_targets(self) -> list[tuple[int, int]]:
        if self.cfg.calibration_points >= 25:
            pts = CALIB_25
        else:
            pts = CALIB_9 if self.cfg.calibration_points >= 9 else CALIB_5
        self.calib_points = pts
        return [(int(round(x * (self.screen_w - 1))), int(round(y * (self.screen_h - 1)))) for x, y in pts]

    def start_calibration(self) -> None:
        if self.eye_gestures:
            self.eye_gestures.start_calibration()
            self.calibrating = True
            self.calib_index = 0
            self.filter.reset()
            self.status = f"EyeGestures: punto 1/{self.eye_gestures.total_points}"
            self.controller.set_paused(True)
            self._begin_calibration_display()
            return
        self._calibration_targets()
        self.calibrating = True
        self.calib_index = 0
        self.calib_samples = []
        self.calib_feature_history = []
        self.calib_point_started_at = time.perf_counter()
        self.last_calibration_sample_at = 0.0
        self.mapper.clear()
        self.filter.reset()
        self.status = "Calibrando: mira el punto rojo"
        self.controller.set_paused(True)
        self._begin_calibration_display()

    def _begin_calibration_display(self) -> None:
        # Remove other application windows from view while targets are shown.
        try:
            cv2.setWindowProperty("EyeTrackerPro - Preview", cv2.WND_PROP_VISIBLE, 0)
        except Exception:
            pass
        if self.overlay:
            self.overlay.destroy()

    def _show_calibration(self, frame: np.ndarray) -> None:
        name = "EyeTrackerPro - Calibracion"
        if not getattr(self, "calibration_window_created", False):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.calibration_window_created = True
        cv2.imshow(name, frame)

    def _activate_debug_control(self, control) -> None:
        if not self.cfg.debug_mouse_interaction:
            return
        if control.action.startswith("game_move:"):
            self.controller.move_game_camera(control.action.split(":", 1)[1])
        else:
            self._handle_action(control.action)

    def _toggle_mouse_debug(self) -> None:
        self.cfg.debug_mouse_interaction = not self.cfg.debug_mouse_interaction
        if self.overlay:
            self.overlay.set_mouse_debug(self.cfg.debug_mouse_interaction)
        self.status = f"Mouse debug: {'ON' if self.cfg.debug_mouse_interaction else 'OFF'}"

    def _is_eyetracker_foreground(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            user32 = ctypes.windll.user32
            user32.GetForegroundWindow.restype = ctypes.c_void_p
            user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
            user32.GetWindowTextW.restype = ctypes.c_int
            user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
            user32.GetAsyncKeyState.restype = ctypes.c_short
            hwnd = user32.GetForegroundWindow()
            title = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title, len(title))
            return "EyeTrackerPro" in title.value
        except Exception:
            return False

    @staticmethod
    def _get_focused_text_control() -> Optional[tuple[int, int, int]]:
        """Use the foreground Windows caret as a no-dependency text-focus signal."""
        if sys.platform != "win32":
            return None
        try:
            user32 = ctypes.windll.user32
            user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(_GuiThreadInfo)]
            user32.GetGUIThreadInfo.restype = wintypes.BOOL
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
            user32.GetWindowThreadProcessId.restype = wintypes.DWORD
            foreground = user32.GetForegroundWindow()
            if not foreground:
                return None
            thread_id = user32.GetWindowThreadProcessId(foreground, None)
            info = _GuiThreadInfo()
            info.cb_size = ctypes.sizeof(_GuiThreadInfo)
            if not user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)) or not info.caret:
                return None
            focus_hwnd = int(info.focus or info.caret)
            caret_hwnd = int(info.caret)
            # Quantize only the vertical caret location so separate browser fields
            # can be distinguished without restarting the timer as text is typed.
            caret_row = int(info.caret_rect.top) // 8
            return focus_hwnd, caret_hwnd, caret_row
        except Exception:
            return None

    def _check_auto_keyboard(self, now: float) -> None:
        if (
            not self.cfg.auto_open_keyboard_on_text_focus
            or not self.overlay
            or self.calibrating
            or self.overlay.mode not in {"navigation", "keyboard"}
            or sys.platform != "win32"
            or now - self._last_text_focus_poll < 0.15
        ):
            return
        self._last_text_focus_poll = now
        focus_id = self._get_focused_text_control()
        if focus_id is None:
            self._text_focus_id = None
            self._text_focus_since = None
            if self._text_focus_missing_since is None:
                self._text_focus_missing_since = now
            elif now - self._text_focus_missing_since >= 0.7:
                self._auto_keyboard_handled_focus = None
            return

        self._text_focus_missing_since = None
        if focus_id != self._text_focus_id:
            self._text_focus_id = focus_id
            self._text_focus_since = now
        if (
            self._text_focus_since is not None
            and focus_id != self._auto_keyboard_handled_focus
            and now - self._text_focus_since >= max(0, self.cfg.auto_keyboard_focus_delay_ms) / 1000.0
        ):
            self._handle_action("mode:keyboard")
            self._auto_keyboard_handled_focus = focus_id

    @staticmethod
    def _eye_openness_from_landmarks(landmarks: Optional[list], frame_shape: tuple[int, ...]) -> Optional[tuple[float, float]]:
        """Compute the same vertical-to-horizontal eye ratios for EyeGestures frames."""
        if not landmarks or len(landmarks) < 478:
            return None
        h, w = frame_shape[:2]

        def point(index: int) -> tuple[float, float]:
            landmark = landmarks[index]
            return landmark.x * w, landmark.y * h

        def ratio(top: int, bottom: int, outer: int, inner: int) -> float:
            tx, ty = point(top)
            bx, by = point(bottom)
            ox, oy = point(outer)
            ix, iy = point(inner)
            width = ((ox - ix) ** 2 + (oy - iy) ** 2) ** 0.5
            height = ((tx - bx) ** 2 + (ty - by) ** 2) ** 0.5
            return height / width if width > 1e-6 else 0.0

        return ratio(159, 145, 33, 133), ratio(386, 374, 362, 263)

    def _check_wink(self, feats: GazeFeatures, now: float, eye_openness: Optional[tuple[float, float]] = None) -> None:
        """Fire one right-click per deliberate one-eye closure."""
        if (self.calibrating or self.controller.paused or not feats.face_detected or not self._gaze_ready()
                or (self.overlay and self.overlay.mode in {"settings", "information"})):
            self._wink_candidate_since = None
            self._wink_candidate_side = None
            return

        left_open, right_open = eye_openness or (feats.left_eye_open, feats.right_eye_open)
        closed_left = left_open <= 0.08 and right_open > 0.12
        closed_right = right_open <= 0.08 and left_open > 0.12
        side = "left" if closed_left else "right" if closed_right else None
        if side is None:
            if left_open > 0.12 and right_open > 0.12:
                self._wink_fired = False
            self._wink_candidate_since = None
            self._wink_candidate_side = None
            return

        if side != self._wink_candidate_side:
            self._wink_candidate_side = side
            self._wink_candidate_since = now
            return
        if (self._wink_candidate_since is not None and not self._wink_fired
                and (now - self._wink_candidate_since) >= 0.18
                and (now - self._last_wink_click_at) >= 0.8):
            result = self.controller.perform_action("right_click")
            self._last_wink_click_at = now
            self._wink_fired = True
            self.status = "Guiño: clic derecho" if result == "right_click" else f"Guiño: {result}"

    def _sync_shortcut_states(self) -> None:
        if not self.overlay:
            return
        active: set[str] = set()
        for page_index, page in enumerate(self.cfg.keybind_pages):
            for binding_index, entry in enumerate(page):
                if shortcut_kind(entry) != "toggle":
                    continue
                if entry.get("mouse"):
                    is_active = entry["mouse"] in self.controller.held_mouse_buttons
                else:
                    keys = frozenset(str(key).lower() for key in entry.get("keys", []) if key)
                    is_active = bool(keys and keys in self.controller.held_shortcuts)
                if is_active:
                    active.add(f"{page_index}:{binding_index}")
        self.overlay.active_shortcuts = active

    def _end_calibration_display(self) -> None:
        try:
            cv2.destroyWindow("EyeTrackerPro - Calibracion")
        except Exception:
            pass
        self.calibration_window_created = False
        try:
            cv2.setWindowProperty("EyeTrackerPro - Preview", cv2.WND_PROP_VISIBLE, 1)
        except Exception:
            pass

    def _finish_calibration(self) -> None:
        if self.eye_gestures:
            saved = self.eye_gestures.finish_calibration()
            self.calibrating = False
            self.calibration_loaded = self.eye_gestures.ready
            self.filter.reset()
            self.controller.set_paused(not self.calibration_loaded)
            self.status = (
                "EyeGestures calibration complete and saved. Gaze active."
                if saved else "EyeGestures calibration complete. Gaze active."
            )
            self._end_calibration_display()
            return
        ok = self.mapper.fit()
        self.calibrating = False
        self.calib_samples = []
        self.filter.reset()
        if ok:
            self.calibration_loaded = self.mapper.save(self.calibration_path, self.screen_w, self.screen_h)
            self.status = (
                "Calibración completa y guardada. Mirada activa."
                if self.calibration_loaded else "Calibración completa. Mirada activa."
            )
            self.controller.set_paused(False)
        else:
            self.calibration_loaded = self.mapper.load(self.calibration_path, self.screen_w, self.screen_h)
            if self.calibration_loaded:
                self.status = "Calibración nueva insuficiente; se restauró la anterior."
                self.controller.set_paused(False)
            else:
                self.status = "Calibración insuficiente. Presiona C para reintentar."
        self._end_calibration_display()

    def _draw_eyegestures_calibration(self, frame: np.ndarray) -> np.ndarray:
        canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)
        state = self.eye_gestures.last_frame
        target = state.target
        if target is not None and self.eye_gestures.completed_points < self.eye_gestures.total_points:
            radius = max(12, state.acceptance_radius)
            cv2.circle(canvas, target, radius, (80, 80, 80), 2, cv2.LINE_AA)
            cv2.circle(canvas, target, 12, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(canvas, target, 4, (0, 0, 255), -1, cv2.LINE_AA)
        if self.eye_gestures_point is not None and state.face_detected:
            cv2.circle(canvas, self.eye_gestures_point, 18, (255, 120, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, self.eye_gestures_point, 24, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"EyeGestures calibration: {min(self.eye_gestures.completed_points + 1, self.eye_gestures.total_points)}/{self.eye_gestures.total_points} - look at the target",
            (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA,
        )
        cv2.putText(canvas, "Blue dot: estimated gaze position", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 160, 40), 2, cv2.LINE_AA)
        if not state.face_detected:
            cv2.putText(canvas, "Face not detected: check camera framing and lighting", (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)
        elif self.eye_gestures.waiting_for_target:
            cv2.putText(canvas, "Hold your gaze; preparing next target...", (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        return canvas

    def _draw_calibration(self, frame: np.ndarray, feats: GazeFeatures) -> np.ndarray:
        canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)
        targets = [(int(round(x * (self.screen_w - 1))), int(round(y * (self.screen_h - 1)))) for x, y in self.calib_points]
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
        stable = False
        if feats.face_detected and eyes_open:
            self.calib_feature_history.append(feats.feature[:6].copy())
            self.calib_feature_history = self.calib_feature_history[-max(2, self.cfg.calibration_stability_frames):]
            stable = len(self.calib_feature_history) >= max(2, self.cfg.calibration_stability_frames) and bool(
                np.max(np.std(np.asarray(self.calib_feature_history), axis=0)) <= self.cfg.calibration_stability_threshold
            )
            now = time.perf_counter()
            sample_interval = max(0, self.cfg.calibration_sample_delay_ms) / 1000.0
            if stable and now - self.last_calibration_sample_at >= sample_interval:
                self.calib_samples.append(feats.feature.copy())
                self.last_calibration_sample_at = now
            progress = len(self.calib_samples) / max(1, self.cfg.calibration_samples_per_point)
            bar_w = 500
            cv2.rectangle(canvas, (60, 170), (60 + bar_w, 195), (80, 80, 80), -1)
            cv2.rectangle(canvas, (60, 170), (60 + int(bar_w * progress), 195), (0, 200, 0), -1)
            cv2.putText(canvas, f"{int(progress*100)}%", (570, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            self.calib_feature_history = []
            if not feats.face_detected:
                cv2.putText(canvas, "Rostro no detectado", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "Abre ambos ojos y mira al centro del punto", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2, cv2.LINE_AA)
        if feats.face_detected and eyes_open and not stable:
            cv2.putText(canvas, "Fija la mirada y mantén la cabeza estable", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

        if len(self.calib_samples) >= self.cfg.calibration_samples_per_point:
            # Use median sample to reduce outliers
            med = np.median(np.stack(self.calib_samples, axis=0), axis=0)
            self.mapper.add_sample(med, tx, ty)
            self.calib_index += 1
            self.calib_samples = []
            self.calib_feature_history = []
            self.calib_point_started_at = time.perf_counter()
            self.last_calibration_sample_at = 0.0
            if self.calib_index >= len(targets):
                self._finish_calibration()
        return canvas

    def _handle_action(self, action: str) -> None:
        if action == "noop":
            return
        if action == "calibrate":
            self.start_calibration()
            return
        if action.startswith("mode:"):
            mode = action.split(":", 1)[1]
            if self.overlay:
                previous_mode = self.overlay.mode
                if previous_mode == "keyboard" and mode != "keyboard":
                    self._auto_keyboard_handled_focus = self._text_focus_id
                self.overlay.mode = mode
                self.cfg.overlay_mode = mode if mode in {"navigation", "game"} else self.cfg.overlay_mode
                if mode in {"settings", "information"}:
                    self.overlay.menu_open = True
                elif mode == "navigation" and previous_mode in {"settings", "information"}:
                    self.overlay.menu_open = False
                self.overlay.settings_selected = None
                self.overlay.locked_direction = None
                self.controller.set_game_hold(set())
                self.overlay.refresh_controls()
                if mode == "keyboard":
                    self._update_suggestions()
            return
        if action == "menu_toggle":
            if self.overlay:
                if self.overlay.mode in {"settings", "information"}:
                    self.overlay.mode = "navigation"
                    self.overlay.menu_open = False
                    self.overlay.settings_selected = None
                    self.overlay.refresh_controls()
                else:
                    self.overlay.menu_open = not self.overlay.menu_open
                    self.overlay.refresh_controls()
            return
        if action == "debug_mouse_toggle":
            self._toggle_mouse_debug()
            return
        if action.startswith("page:"):
            if self.overlay:
                self.overlay.page_index = (self.overlay.page_index + int(action.split(":", 1)[1])) % max(1, len(self.cfg.keybind_pages))
                if self.overlay.mode == "settings":
                    self.overlay.settings_selected = None
            return
        if action == "add_page":
            new_page = []
            for index in range(4):
                new_page.append({"label": f"Keybind {len(self.cfg.keybind_pages) * 4 + index + 1}", "keys": [f"f{index + 1}"], "kind": "click"})
            self.cfg.keybind_pages.append(new_page)
            if self.overlay:
                self.overlay.keybind_pages = self.cfg.keybind_pages
                self.overlay.page_index = len(self.cfg.keybind_pages) - 1
                self.overlay.settings_selected = (self.overlay.page_index, 0)
            return
        if action.startswith("shortcut_add:"):
            index = int(action.split(":", 1)[1])
            if self.overlay:
                page = self.cfg.keybind_pages[self.overlay.page_index]
                while len(page) <= index:
                    page.append({"label": f"Keybind {len(page) + 1}", "keys": ["f1"], "kind": "click"})
                self.overlay.settings_selected = (self.overlay.page_index, index)
                self.overlay.mode = "settings"
                self.overlay.menu_open = True
                self.overlay.refresh_controls()
            return
        if action.startswith("edit_binding:"):
            if self.overlay:
                self.overlay.settings_selected = (self.overlay.page_index, int(action.split(":", 1)[1]))
            return
        if action in {"binding_cycle", "binding_add_key", "binding_remove_key", "binding_modifier", "binding_type", "binding_mouse"}:
            if self.overlay and self.overlay.settings_selected:
                page_idx, binding_idx = self.overlay.settings_selected
                entry = self.cfg.keybind_pages[page_idx][binding_idx]
                modifiers = {"ctrl", "shift", "alt", "win"}
                keys = list(entry.get("keys", []))
                base = next((key for key in reversed(keys) if key.lower() not in modifiers), "a")
                mods = [key for key in keys if key.lower() in modifiers]
                choices = list("abcdefghijklmnopqrstuvwxyz") + [f"f{i}" for i in range(1, 13)] + ["space", "enter", "backspace", "left", "right", "up", "down", "tab"]
                if action == "binding_cycle":
                    next_key = choices[(choices.index(base) + 1) % len(choices)] if base in choices else choices[0]
                    entry["keys"] = mods + [next_key]
                    entry.pop("mouse", None)
                elif action == "binding_add_key":
                    entry["keys"] = keys + [next((key for key in choices if key not in keys), "a")]
                    entry.pop("mouse", None)
                elif action == "binding_remove_key":
                    if keys:
                        entry["keys"] = keys[:-1]
                    entry.pop("mouse", None)
                elif action == "binding_modifier":
                    choices = [[], ["ctrl"], ["shift"], ["alt"], ["ctrl", "shift"], ["ctrl", "alt"], ["shift", "alt"], ["ctrl", "shift", "alt"]]
                    entry["keys"] = choices[(choices.index(mods) + 1) % len(choices)] + [base]
                    entry.pop("mouse", None)
                elif action == "binding_type":
                    entry["kind"] = "click" if shortcut_kind(entry) == "toggle" else "toggle"
                    entry.pop("hold", None)
                else:
                    choices = ["", "left", "right"]
                    current = entry.get("mouse", "")
                    entry["mouse"] = choices[(choices.index(current) + 1) % len(choices)] if current in choices else "left"
                self.overlay.keybind_pages = self.cfg.keybind_pages
            return
        if action == "language_next":
            if self.overlay:
                langs = ["en", "es", "zh", "fr"]
                self.cfg.language = langs[(langs.index(self.cfg.language) + 1) % len(langs)] if self.cfg.language in langs else "en"
                self.overlay.language = self.cfg.language
            return
        if action.startswith("shortcut:"):
            _, page, index = action.split(":")
            entry = self.cfg.keybind_pages[int(page)][int(index)]
            try:
                kind = shortcut_kind(entry)
                if entry.get("mouse"):
                    result = self.controller.hold_mouse(entry["mouse"]) if kind == "toggle" else self.controller.click_mouse(entry["mouse"])
                else:
                    result = self.controller.shortcut(entry.get("keys", []), kind == "toggle")
                self.status = f"{entry.get('label', 'Atajo')}: {result}"
            except Exception as exc:
                self.status = f"Error de atajo: {exc}"
            return
        if action.startswith("key:"):
            key = action.split(":", 1)[1]
            if key in {"ctrl", "alt", "shift"}:
                try:
                    self.controller.shortcut([key], hold=True)
                except Exception as exc:
                    self.status = f"Error de tecla: {exc}"
                return
            if key == "space" and self.overlay:
                self.overlay.keyboard_text += " "
                self.suggestion_engine.learn_completed_text(self.overlay.keyboard_text, self.cfg.language)
                self.cfg.suggestion_history = self.suggestion_engine.to_dict()
                self._update_suggestions()
                self.controller.type_key("space")
                return
            if self.overlay and len(key) == 1 and (key.isalnum() or key in "-=[];',./"):
                self.overlay.keyboard_text += key
                self._update_suggestions()
                self.controller.type_key(key)
                return
            if key == "backspace" and self.overlay:
                self.overlay.keyboard_text = self.overlay.keyboard_text[:-1]
                self._update_suggestions()
            self.controller.type_key(key)
            return
        if action.startswith("suggest:") and self.overlay:
            word = action.split(":", 1)[1]
            previous_text = self.overlay.keyboard_text
            self.suggestion_engine.learn_selection(previous_text, word, self.cfg.language)
            self.cfg.suggestion_history = self.suggestion_engine.to_dict()
            has_partial_word = bool(self.overlay.keyboard_text) and not self.overlay.keyboard_text[-1].isspace()
            current_word = self.overlay.keyboard_text.split()[-1] if has_partial_word and self.overlay.keyboard_text.split() else ""
            if current_word:
                self.overlay.keyboard_text = self.overlay.keyboard_text[:-len(current_word)]
            self.overlay.keyboard_text += word + " "
            self._update_suggestions()
            try:
                if current_word:
                    pyautogui.press("backspace", presses=len(current_word))
                self.controller.type_text(word + " ")
            except Exception:
                pass
            return
        if action.startswith("nav:"):
            direction = action.split(":", 1)[1]
            if self.overlay and self.overlay.locked_direction and self.overlay.locked_direction != direction:
                return
            if self.overlay:
                self.overlay.last_direction = direction
            movement = {"up": "scroll_up", "down": "scroll_down", "left": "altleft", "right": "altright"}
            if direction in {"left", "right"}:
                try:
                    self.controller.shortcut(["alt", direction])
                except Exception:
                    pass
            else:
                self.controller.perform_action(movement[direction])
            return
        if action == "lock":
            if self.overlay:
                if self.overlay.locked_direction:
                    self.overlay.locked_direction = None
                else:
                    self.overlay.locked_direction = self.overlay.last_direction
            return
        if action == "hold_mouse":
            self.controller.hold_mouse()
            return
        if action.startswith("game_hold:"):
            self.controller.set_game_hold(set(action.split(":", 1)[1].split("+")))
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

    def _update_suggestions(self) -> None:
        if not self.overlay:
            return
        self.overlay.suggestions = self.suggestion_engine.suggest(self.overlay.keyboard_text, self.cfg.language, limit=5)

    def _process_gaze(self, feats: GazeFeatures) -> tuple[Optional[tuple[int, int]], float]:
        now = time.perf_counter()
        if not feats.face_detected:
            self.filter.reset()
            self.controller.dwell.active = False
            self.controller.release_all()
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
            self.controller.set_game_hold(set())
            return None, 0.0
        pred = (
            self.eye_gestures_point
            if self.eye_gestures and self.eye_gestures.ready
            else self.mapper.predict(feats.feature)
        )
        if pred is None:
            self.status = "Sin calibración. Presiona C."
            return None, 0.0

        sx, sy = pred
        sx = max(0, min(self.screen_w - 1, sx))
        sy = max(0, min(self.screen_h - 1, sy))
        if self.eye_gestures:
            # EyeGestures already stabilizes its direct screen estimate; avoid adding cursor lag.
            fx, fy = float(sx), float(sy)
        else:
            fx, fy = self.filter.filter(sx, sy, now)

        # Fine adjustment: gain around center + offset
        cx, cy = self.screen_w / 2.0, self.screen_h / 2.0
        if not self.eye_gestures or self.fine_tune_mode:
            fx = cx + (fx - cx) * float(self.cfg.fine_gain_x) + float(self.cfg.fine_offset_x)
            fy = cy + (fy - cy) * float(self.cfg.fine_gain_y) + float(self.cfg.fine_offset_y)

        ix, iy = self.controller.clamp(fx, fy)

        game_mode = bool(self.overlay and self.overlay.mode == "game")
        if self.cfg.move_cursor_enabled and not self.controller.paused and not game_mode:
            self.controller.move(ix, iy)

        dwell_progress = 0.0
        action_target = None
        if self.overlay:
            btn = self.overlay.button_at(ix, iy)
            if btn is None or btn.id != self._latched_control_id:
                self._latched_control_id = None
            if btn and btn.action.startswith("game_move:"):
                self.controller.set_game_hold(set())
                self.controller.move_game_camera(btn.action.split(":", 1)[1])
                self.controller.dwell.active = False
            elif btn and btn.action.startswith("game_hold:"):
                keys = set(btn.action.split(":", 1)[1].split("+"))
                self.controller.set_game_hold(keys)
                self.controller.dwell.active = False
            elif game_mode:
                self.controller.set_game_hold(set())
                self.controller.dwell.active = False
                if btn is not None and btn.id != self._latched_control_id:
                    dwell_progress = self.controller.dwell_progress(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                    if self.controller.dwell_ready(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                        action_target = btn.action
            else:
                if btn is not None:
                    if btn.id != self._latched_control_id:
                        dwell_progress = self.controller.dwell_progress(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                        if self.controller.dwell_ready(ix, iy, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                            action_target = btn.action
                    else:
                        self.controller.dwell.active = False
                else:
                    # Ordinary dwell clicks remain available away from controls outside game mode.
                    modal_mode = self.overlay.mode in {"settings", "information"}
                    if not modal_mode and self.cfg.dwell_enabled and not self.controller.paused:
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
            if self.overlay and btn is not None:
                self._latched_control_id = btn.id
            self._handle_action(action_target)

        if not self.controller.paused and self._gaze_ready():
            self.status = "Mirada activa"
        return (ix, iy), dwell_progress

    def _gaze_ready(self) -> bool:
        return self.eye_gestures.ready if self.eye_gestures else self.mapper.calibrated

    def _draw_preview(self, frame: np.ndarray, feats: GazeFeatures, gaze: Optional[tuple[int, int]], dwell: float) -> np.ndarray:
        view = frame.copy()
        if self.cfg.mirror_preview:
            view = cv2.flip(view, 1)
        h, w = view.shape[:2]
        if feats.face_detected and feats.landmarks:
            self._draw_gaze_guides(view, feats.landmarks, gaze)
        color = (0, 200, 0) if feats.face_detected else (0, 0, 255)
        cv2.putText(view, f"Face: {'OK' if feats.face_detected else 'NO'}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(view, f"Calibrado: {'SI' if self._gaze_ready() else 'NO'}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f"Engine: {self.active_gaze_engine}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2, cv2.LINE_AA)
        if self.engine_warning:
            cv2.putText(view, "EyeGestures unavailable; using built-in tracker", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2, cv2.LINE_AA)
        elif self.eye_gestures and self.eye_gestures.last_error:
            cv2.putText(view, "EyeGestures tracking error - see terminal", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f"Mouse: {'OFF' if self.controller.paused else 'ON'}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if not self.controller.paused else (0, 165, 255), 2, cv2.LINE_AA)
        if gaze:
            cv2.putText(view, f"Gaze: {gaze[0]},{gaze[1]} dwell {int(dwell*100)}%", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        ft = f"FINE gx={self.cfg.fine_gain_x:.2f} gy={self.cfg.fine_gain_y:.2f} ox={int(self.cfg.fine_offset_x)} oy={int(self.cfg.fine_offset_y)}"
        ft_color = (0, 255, 255) if self.fine_tune_mode else (160, 160, 160)
        cv2.putText(view, ft, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ft_color, 2, cv2.LINE_AA)
        if self.fine_tune_mode:
            cv2.putText(view, "A/D ox  W/S oy  J/L gx  I/K gy  R reset  F/ESC salir", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, "C=calibrar  P=pausa  M=mouse  B=mouse debug  F=ajuste fino  ESC=salir", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        return view

    def _draw_gaze_guides(self, view: np.ndarray, landmarks: list, gaze: Optional[tuple[int, int]]) -> None:
        """Draw user-only landmarks and screen gaze direction on the camera preview."""
        h, w = view.shape[:2]
        points = []
        for landmark in landmarks:
            x = int(landmark.x * w)
            if self.cfg.mirror_preview:
                x = w - 1 - x
            points.append((x, int(landmark.y * h)))
        if not points:
            return

        # Sparse facial mesh dots make eye/head movement easier to read without
        # obscuring the camera image. These are display-only; tracker inputs stay raw.
        for index in range(0, len(points), 6):
            cv2.circle(view, points[index], 1, (100, 220, 255), -1, cv2.LINE_AA)
        eyes = (([474, 475, 476, 477], [33, 133], (0, 255, 255)),
                ([469, 470, 471, 472], [362, 263], (255, 220, 0)))
        for iris_indices, corners, color in eyes:
            if max(*iris_indices, *corners) >= len(points):
                continue
            iris = [points[i] for i in iris_indices]
            iris_center = tuple(int(sum(p[axis] for p in iris) / len(iris)) for axis in (0, 1))
            eye_center = tuple((points[corners[0]][axis] + points[corners[1]][axis]) // 2 for axis in (0, 1))
            # Short eye-direction vector gives a second, local guide alongside the screen map.
            vector_end = (eye_center[0] + 3 * (iris_center[0] - eye_center[0]),
                          eye_center[1] + 3 * (iris_center[1] - eye_center[1]))
            cv2.arrowedLine(view, eye_center, vector_end, color, 2, cv2.LINE_AA, tipLength=0.3)
            cv2.circle(view, iris_center, 5, color, 2, cv2.LINE_AA)
            cv2.circle(view, iris_center, 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Small screen map translates the calibrated gaze point into a visible guide.
        if gaze:
            box_w, box_h, margin = min(180, w // 3), min(120, h // 3), 16
            if box_w >= 70 and box_h >= 55:
                x0, y0 = w - box_w - margin, margin
                layer = view.copy()
                cv2.rectangle(layer, (x0, y0), (x0 + box_w, y0 + box_h), (12, 12, 12), -1)
                cv2.addWeighted(layer, 0.55, view, 0.45, 0, view)
                cv2.rectangle(view, (x0, y0), (x0 + box_w, y0 + box_h), (230, 230, 230), 1)
                for frac in (0.25, 0.5, 0.75):
                    gx, gy = x0 + int(box_w * frac), y0 + int(box_h * frac)
                    cv2.line(view, (gx, y0), (gx, y0 + box_h), (95, 95, 95), 1)
                    cv2.line(view, (x0, gy), (x0 + box_w, gy), (95, 95, 95), 1)
                sx = x0 + int(np.clip(gaze[0] / max(1, self.screen_w - 1), 0, 1) * box_w)
                sy = y0 + int(np.clip(gaze[1] / max(1, self.screen_h - 1), 0, 1) * box_h)
                cv2.circle(view, (sx, sy), 8, (0, 80, 255), -1, cv2.LINE_AA)
                cv2.circle(view, (sx, sy), 11, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view, "SCREEN GAZE", (x0 + 5, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    def run(self) -> None:
        self._open_camera()
        self.running = True
        self.last_face_seen_at = time.perf_counter()
        cv2.namedWindow("EyeTrackerPro - Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("EyeTrackerPro - Preview", 960, 540)
        hotkeys_available = self.hotkeys.start()
        if sys.platform == "win32" and not hotkeys_available:
            self.status = "Global hotkeys unavailable; keep the preview focused for keyboard controls."
        # Reuse a valid saved calibration; otherwise calibrate before gaze interaction.
        if not self._gaze_ready():
            self.start_calibration()

        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.status = "Error leyendo cámara"
                    break

                ts = int(time.perf_counter() * 1000)
                wink_eye_openness = None
                if self.eye_gestures:
                    gesture_frame = self.eye_gestures.detect(frame)
                    self.eye_gestures_point = gesture_frame.point
                    wink_eye_openness = self._eye_openness_from_landmarks(gesture_frame.landmarks, frame.shape)
                    feats = GazeFeatures(
                        feature=np.zeros(11, dtype=np.float32),
                        face_detected=gesture_frame.face_detected,
                        left_eye_open=0.0 if gesture_frame.blink else 1.0,
                        right_eye_open=0.0 if gesture_frame.blink else 1.0,
                        landmarks=gesture_frame.landmarks,
                    )
                    if self.calibrating:
                        if self.eye_gestures.completed_points >= self.eye_gestures.total_points:
                            self._finish_calibration()
                        else:
                            self.status = (
                                f"EyeGestures: {self.eye_gestures.completed_points + 1}/"
                                f"{self.eye_gestures.total_points} - look at the target"
                            )
                else:
                    feats = self.detector.detect(frame, ts)

                if self.calibrating:
                    calib_view = (
                        self._draw_eyegestures_calibration(frame)
                        if self.eye_gestures else self._draw_calibration(frame, feats)
                    )
                    self._show_calibration(calib_view)
                    try:
                        cv2.setWindowProperty("EyeTrackerPro - Calibracion", cv2.WND_PROP_TOPMOST, 1)
                    except Exception:
                        pass
                else:
                    try:
                        cv2.destroyWindow("EyeTrackerPro - Calibracion")
                    except Exception:
                        pass
                    self.calibration_window_created = False

                gaze, dwell = (None, 0.0)
                if not self.calibrating:
                    self._check_wink(feats, time.perf_counter(), wink_eye_openness)
                    gaze, dwell = self._process_gaze(feats)
                    self._check_auto_keyboard(time.perf_counter())

                if self.overlay and not self.calibrating:
                    self._sync_shortcut_states()
                    self.overlay.status = self.status
                    self.overlay.paused = self.controller.paused
                    self.overlay.gaze = gaze if gaze else self.overlay.gaze
                    self.overlay.dwell_progress = dwell
                    self.overlay.render()

                preview = self._draw_preview(frame, feats, gaze, dwell)
                cv2.imshow("EyeTrackerPro - Preview", preview)

                key_raw = cv2.waitKeyEx(1)
                key = self.hotkeys.poll() if hotkeys_available else None
                if key is None:
                    key = -1 if hotkeys_available else (key_raw & 0xFF if key_raw != -1 else -1)

                if key == 27:  # ESC
                    if self.calibrating:
                        self.calibrating = False
                        if self.eye_gestures:
                            self.eye_gestures.cancel_calibration()
                            self.calibration_loaded = self.eye_gestures.restore_saved_calibration()
                        else:
                            self.calibration_loaded = self.mapper.load(self.calibration_path, self.screen_w, self.screen_h)
                        if self.calibration_loaded:
                            self.status = "Calibración cancelada; se restauró la anterior."
                            self.controller.set_paused(False)
                        else:
                            self.status = "Calibración cancelada"
                            self.controller.set_paused(True)
                        self._end_calibration_display()
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
                elif key in (ord('b'), ord('B')):
                    self._toggle_mouse_debug()
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
        self.hotkeys.stop()
        self.controller.release_all()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.detector:
                self.detector.close()
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
