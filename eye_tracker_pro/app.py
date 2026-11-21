from __future__ import annotations

import ctypes
import os
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
from .cursor_overlay import CursorOverlay
from .detector import FaceLandmarkerDetector, GazeFeatures
from .filter import PointFilter
from .eyegestures_engine import EyeGesturesEngine
from .eyetheia_engine import EyeTheiaEngine
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


class EyeTrackerApp:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cfg: AppConfig = load_config(base_dir)
        self.suggestion_engine = SuggestionEngine(self.cfg.suggestion_history)
        self.screen_w, self.screen_h = pyautogui.size()
        self.controller = CursorController(
            self.screen_w,
            self.screen_h,
            self.cfg.move_deadzone_px,
            self.cfg.cursor_max_speed_px_s,
        )
        self.controller.scroll_step = max(1, int(self.cfg.edge_scroll_amount) * 2)
        self.mapper = GazeMapper()
        self.calibration_path = base_dir / "calibration.json"
        self.eye_gestures: Optional[EyeGesturesEngine] = None
        self.eye_gestures_point: Optional[tuple[int, int]] = None
        self.eye_theia: Optional[EyeTheiaEngine] = None
        self.eye_observation = None
        self.engine_warning: Optional[str] = None
        requested_engine = str(self.cfg.gaze_engine).lower()
        if requested_engine == "eyetheia":
            try:
                self.eye_theia = EyeTheiaEngine(base_dir, self.screen_w, self.screen_h)
            except Exception as exc:
                self.engine_warning = f"EyeTheia unavailable; using MediaPipe mapper ({exc})"
        elif requested_engine == "eyegestures":
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
        if self.eye_theia is not None:
            self.active_gaze_engine = "eyetheia"
            # EyeTheia is personalized per user, so a previous session is never reused.
            self.calibration_loaded = False
        elif self.eye_gestures is None:
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
            self.overlay.precision_mode = self.controller.precision_mode
            self.overlay.edge_scroll_enabled = self.cfg.edge_scroll_enabled
        # Click-through dot from the debug branch; the Tk control layer never drew one.
        self.cursor_overlay = CursorOverlay(self.screen_w, self.screen_h) if self.cfg.show_gaze_dot else None

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
        self.calibration_window_created = False
        self._last_external_window: Optional[int] = None
        self._last_edge_scroll_at = 0.0
        if self.engine_warning:
            self.status = self.engine_warning
        elif self.eye_theia:
            self.status = "Presiona C para personalizar EyeTheia"
        else:
            self.status = "Calibración guardada cargada" if self.calibration_loaded else "Presiona C para calibrar"
        self._latched_control_id: Optional[str] = None
        self._global_keys_down: set[int] = set()
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
        if self.eye_theia:
            if self.eye_theia.training:
                self.status = "Espera a que termine la personalización de EyeTheia."
                return
            self._calibration_targets()
            self.eye_theia.begin_calibration()
            self.calibrating = True
            self.calib_index = 0
            self.calib_samples = []
            self.calib_feature_history = []
            self.calib_point_started_at = time.perf_counter()
            self.last_calibration_sample_at = 0.0
            self.filter.reset()
            self.status = "EyeTheia: mira el punto rojo"
            self.controller.set_paused(True)
            self._begin_calibration_display()
            return
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
        if self.eye_theia:
            self.calibrating = False
            self.calib_samples = []
            self.calib_feature_history = []
            self.filter.reset()
            try:
                self.eye_theia.train_async()
                self.status = "Personalizando EyeTheia con tu calibración..."
            except Exception as exc:
                self.status = f"Falló la personalización: {exc}"
                self.controller.set_paused(True)
            self._end_calibration_display()
            return
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
            needed = self.cfg.eyetheia_samples_per_target if self.eye_theia else self.cfg.calibration_samples_per_point
            progress = len(self.calib_samples) / max(1, needed)
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

        needed = self.cfg.eyetheia_samples_per_target if self.eye_theia else self.cfg.calibration_samples_per_point
        if len(self.calib_samples) >= needed:
            if self.eye_theia:
                self.eye_theia.add_calibration_sample(self.eye_observation, tx, ty)
            else:
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
        if action == "precision_toggle":
            self.controller.set_direct_mode(False)
            self.controller.set_precision_mode(not self.controller.precision_mode)
            if self.overlay:
                self.overlay.precision_mode = self.controller.precision_mode
            self.status = f"Precisión: {'ON' if self.controller.precision_mode else 'OFF'}"
            return
        if action == "edge_scroll_toggle":
            self.cfg.edge_scroll_enabled = not self.cfg.edge_scroll_enabled
            if self.overlay:
                self.overlay.edge_scroll_enabled = self.cfg.edge_scroll_enabled
            self.status = f"Auto-scroll: {'ON' if self.cfg.edge_scroll_enabled else 'OFF'}"
            return
        if action in {
            "browser_back", "browser_forward", "browser_reload", "browser_address",
            "browser_prev_tab", "browser_next_tab", "browser_zoom_in", "browser_zoom_out",
            "browser_zoom_reset", "browser_undo", "scroll_up", "scroll_down",
        }:
            self._focus_browser_window()
        result = self.controller.perform_action(action)
        if action == "pause":
            self.status = "Pausado" if self.controller.paused else "Mirada activa"
        elif result not in {"ignored_paused"}:
            self.status = f"Acción: {action}"

    def _update_suggestions(self) -> None:
        if not self.overlay:
            return
        self.overlay.suggestions = self.suggestion_engine.suggest(self.overlay.keyboard_text, self.cfg.language, limit=5)

    @staticmethod
    def _handle_value(handle: object) -> int:
        return int(getattr(handle, "value", handle) or 0)

    def _focus_browser_window(self) -> None:
        """Send browser shortcuts to the last external window, not the overlay."""
        if os.name != "nt" or not self._last_external_window:
            return
        try:
            user32 = ctypes.windll.user32
            hwnd = wintypes.HWND(self._last_external_window)
            user32.GetForegroundWindow.argtypes = []
            user32.GetForegroundWindow.restype = wintypes.HWND
            user32.IsWindow.argtypes = [wintypes.HWND]
            user32.IsWindow.restype = wintypes.BOOL
            if not user32.IsWindow(hwnd):
                self._last_external_window = None
                return
            current = user32.GetForegroundWindow()
            if self._handle_value(current) != self._handle_value(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.03)
        except (AttributeError, OSError):
            pass

    def _update_edge_scroll(self, x: int, y: int) -> None:
        if not self.cfg.edge_scroll_enabled or self.controller.paused:
            return
        if self.overlay and self.overlay.button_at(x, y):
            return
        now = time.monotonic()
        if (now - self._last_edge_scroll_at) * 1000 < self.cfg.edge_scroll_interval_ms:
            return
        action = None
        if y <= self.cfg.edge_scroll_margin_px:
            action = "scroll_up"
        elif y >= self.screen_h - 1 - self.cfg.edge_scroll_margin_px:
            action = "scroll_down"
        if action:
            self._focus_browser_window()
            amount = max(1, int(self.cfg.edge_scroll_amount))
            self.controller.scroll_by(amount if action == "scroll_up" else -amount)
            self._last_edge_scroll_at = now

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
        if self.eye_theia:
            if self.eye_theia.training:
                self.status = "Personalizando EyeTheia..."
                return None, 0.0
            pred = self.eye_theia.predict(self.eye_observation) if self.eye_observation else None
        elif self.eye_gestures and self.eye_gestures.ready:
            pred = self.eye_gestures_point
        else:
            pred = self.mapper.predict(feats.feature)
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
        cursor_x, cursor_y = ix, iy
        if self.cfg.move_cursor_enabled and not self.controller.paused and not game_mode:
            cursor_x, cursor_y = self.controller.move(ix, iy)
            self._update_edge_scroll(cursor_x, cursor_y)

        dwell_progress = 0.0
        action_target = None
        if self.overlay:
            btn = self.overlay.button_at(cursor_x, cursor_y)
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
                    dwell_progress = self.controller.dwell_progress(cursor_x, cursor_y, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                    if self.controller.dwell_ready(cursor_x, cursor_y, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                        action_target = btn.action
            else:
                if btn is not None:
                    if btn.id != self._latched_control_id:
                        dwell_progress = self.controller.dwell_progress(cursor_x, cursor_y, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                        if self.controller.dwell_ready(cursor_x, cursor_y, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                            action_target = btn.action
                    else:
                        self.controller.dwell.active = False
                else:
                    # Ordinary dwell clicks remain available away from controls outside game mode.
                    modal_mode = self.overlay.mode in {"settings", "information"}
                    if not modal_mode and self.cfg.dwell_enabled and not self.controller.paused:
                        dwell_progress = self.controller.dwell_progress(cursor_x, cursor_y, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                        if self.controller.dwell_ready(cursor_x, cursor_y, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                            action_target = "left_click"
                    else:
                        self.controller.dwell.active = False
        else:
            if self.cfg.dwell_enabled and not self.controller.paused:
                dwell_progress = self.controller.dwell_progress(cursor_x, cursor_y, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms)
                if self.controller.dwell_ready(cursor_x, cursor_y, self.cfg.dwell_radius_px, self.cfg.dwell_time_ms, self.cfg.dwell_cooldown_ms):
                    action_target = "left_click"

        if action_target:
            if self.overlay and btn is not None:
                self._latched_control_id = btn.id
            self._handle_action(action_target)

        if not self.controller.paused and self._gaze_ready():
            self.status = "Mirada activa"
        return (cursor_x, cursor_y), dwell_progress

    def _gaze_ready(self) -> bool:
        if self.eye_theia:
            return self.eye_theia.ready
        if self.eye_gestures:
            return self.eye_gestures.ready
        return self.mapper.calibrated

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
        cv2.putText(view, f"Calibrado: {'SI' if self._gaze_ready() else 'NO'}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f"Engine: {self.active_gaze_engine}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2, cv2.LINE_AA)
        if self.engine_warning:
            cv2.putText(view, "Engine unavailable; using built-in tracker", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2, cv2.LINE_AA)
        elif self.eye_theia and not feats.face_detected:
            hint = "ADD FRONT LIGHT" if self.eye_theia.last_brightness < 105 else "CENTER FACE / REDUCE GLARE"
            cv2.putText(view, hint, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
        elif self.eye_theia and self.eye_theia.last_detection_retry:
            cv2.putText(view, self.eye_theia.last_detection_source.upper(), (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2, cv2.LINE_AA)
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
        cv2.putText(view, "C=calibrar  P=pausa  M=mouse  V=precision  G=auto-scroll  F=ajuste  ESC=salir", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)
        return view

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
                        feature=np.zeros(11, dtype=np.float32),
                        face_detected=gesture_frame.face_detected,
                        left_eye_open=0.0 if gesture_frame.blink else 1.0,
                        right_eye_open=0.0 if gesture_frame.blink else 1.0,
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
                    gaze, dwell = self._process_gaze(feats)

                if self.overlay and not self.calibrating:
                    self._sync_shortcut_states()
                    self.overlay.status = self.status
                    self.overlay.paused = self.controller.paused
                    self.overlay.gaze = gaze if gaze else self.overlay.gaze
                    self.overlay.dwell_progress = dwell
                    self.overlay.precision_mode = self.controller.precision_mode
                    self.overlay.edge_scroll_enabled = self.cfg.edge_scroll_enabled
                    self.overlay.render()

                if self.cursor_overlay:
                    self.cursor_overlay.update(
                        gaze,
                        visible=bool(
                            gaze
                            and self.cfg.move_cursor_enabled
                            and not self.controller.paused
                            and not self.calibrating
                        ),
                        dwell_progress=dwell,
                    )

                # Remember the last non-tracker window so browser actions do not
                # steal focus back to the transparent overlay.
                if sys.platform == "win32" and not self.calibrating:
                    try:
                        foreground = ctypes.windll.user32.GetForegroundWindow()
                        if foreground and not self._is_eyetracker_foreground():
                            self._last_external_window = self._handle_value(foreground)
                    except Exception:
                        pass

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
                        elif self.eye_theia:
                            self.eye_theia.ready = False
                            self.eye_observation = None
                            self.calibration_loaded = False
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
                elif key in (ord('v'), ord('V')):
                    self._handle_action("precision_toggle")
                elif key in (ord('g'), ord('G')):
                    self._handle_action("edge_scroll_toggle")
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
        if self.cursor_overlay:
            try:
                self.cursor_overlay.destroy()
            except Exception:
                pass
        cv2.destroyAllWindows()
        save_config(self.base_dir, self.cfg)
