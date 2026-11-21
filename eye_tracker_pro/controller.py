from __future__ import annotations

import time
import ctypes
import sys
from dataclasses import dataclass

import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


@dataclass
class DwellState:
    candidate_x: float = 0.0
    candidate_y: float = 0.0
    started_at: float = 0.0
    last_fired_at: float = 0.0
    active: bool = False


class CursorController:
    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        move_deadzone_px: int = 8,
        max_speed_px_s: int = 950,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.paused = False
        self.dragging = False
        self.dwell = DwellState()
        self.last_move_at = 0.0
        self.base_deadzone_px = max(0, int(move_deadzone_px))
        self.base_max_speed_px_s = max(1, int(max_speed_px_s))
        self.move_deadzone_px = self.base_deadzone_px
        self.max_speed_px_s = self.base_max_speed_px_s
        self.precision_mode = False
        self.direct_mode = False
        self.scroll_step = 4
        self._last_sent: tuple[int, int] | None = None
        self.held_keys: set[str] = set()
        self.held_shortcuts: set[frozenset[str]] = set()
        self.held_mouse_buttons: set[str] = set()
        self.game_held_keys: set[str] = set()

    def set_direct_mode(self, enabled: bool) -> None:
        self.direct_mode = bool(enabled)

    def set_precision_mode(self, enabled: bool) -> None:
        self.precision_mode = bool(enabled)
        if self.precision_mode:
            self.move_deadzone_px = max(3, self.base_deadzone_px // 2)
            self.max_speed_px_s = max(350, int(self.base_max_speed_px_s * 0.5))
        else:
            self.move_deadzone_px = self.base_deadzone_px
            self.max_speed_px_s = self.base_max_speed_px_s

    def set_paused(self, paused: bool) -> None:
        if self.paused != paused:
            self.last_move_at = time.perf_counter()
        self.paused = paused
        if paused:
            self.release_all()

    def release_all(self) -> None:
        for key in tuple(self.held_keys | self.game_held_keys):
            try:
                pyautogui.keyUp(key)
            except Exception:
                pass
        for button in tuple(self.held_mouse_buttons):
            try:
                pyautogui.mouseUp(button=button)
            except Exception:
                pass
        if self.dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        self.held_keys.clear()
        self.held_shortcuts.clear()
        self.game_held_keys.clear()
        self.held_mouse_buttons.clear()
        self.dragging = False

    def set_game_hold(self, keys: set[str]) -> None:
        """Keep game movement keys down only while gaze is in their zones."""
        wanted = set() if self.paused else keys
        for key in self.game_held_keys - wanted:
            try:
                pyautogui.keyUp(key)
            except Exception:
                pass
        for key in wanted - self.game_held_keys:
            try:
                pyautogui.keyDown(key)
            except Exception:
                pass
        self.game_held_keys = wanted

    def move_game_camera(self, direction: str, speed: int = 4) -> None:
        if self.paused:
            return
        movement = {
            "up_left": (-speed, -speed), "up_right": (speed, -speed),
            "down_left": (-speed, speed), "down_right": (speed, speed),
        }
        delta = movement.get(direction)
        if delta:
            try:
                pyautogui.moveRel(*delta, duration=0)
            except Exception:
                pass

    def shortcut(self, keys: list[str], hold: bool = False) -> str:
        if self.paused:
            return "ignored_paused"
        normalized = [str(key).lower() for key in keys if key]
        if not normalized:
            return "empty_shortcut"
        if hold:
            requested = frozenset(normalized)
            if requested in self.held_shortcuts:
                self.held_shortcuts.remove(requested)
                still_needed = set().union(*self.held_shortcuts) if self.held_shortcuts else set()
                for key in set(requested) - still_needed:
                    try:
                        pyautogui.keyUp(key)
                    except Exception:
                        pass
                    self.held_keys.discard(key)
                return "released"
            pressed_now: list[str] = []
            try:
                for key in normalized:
                    if key not in self.held_keys:
                        pyautogui.keyDown(key)
                        self.held_keys.add(key)
                        pressed_now.append(key)
            except Exception:
                for key in reversed(pressed_now):
                    try:
                        pyautogui.keyUp(key)
                    except Exception:
                        pass
                    self.held_keys.discard(key)
                raise
            self.held_shortcuts.add(requested)
            return "held"
        pyautogui.hotkey(*normalized)
        return "shortcut"

    def type_key(self, key: str) -> str:
        if self.paused:
            return "ignored_paused"
        try:
            pyautogui.press(key)
            return "key"
        except Exception as exc:
            return f"error:{exc}"

    def type_text(self, text: str) -> str:
        if self.paused:
            return "ignored_paused"
        if sys.platform != "win32":
            pyautogui.write(text, interval=0.01)
            return "text"
        class KeyboardInput(ctypes.Structure):
            _fields_ = [
                ("virtual_key", ctypes.c_ushort),
                ("scan_code", ctypes.c_ushort),
                ("flags", ctypes.c_uint),
                ("time", ctypes.c_uint),
                ("extra_info", ctypes.c_size_t),
            ]

        class InputUnion(ctypes.Union):
            _fields_ = [("keyboard", KeyboardInput)]

        class Input(ctypes.Structure):
            _anonymous_ = ("value",)
            _fields_ = [("type", ctypes.c_uint), ("value", InputUnion)]

        events = []
        for unit in (int.from_bytes(text.encode("utf-16-le")[i:i + 2], "little") for i in range(0, len(text.encode("utf-16-le")), 2)):
            events.append(Input(type=1, value=InputUnion(keyboard=KeyboardInput(0, unit, 0x0004, 0, 0))))
            events.append(Input(type=1, value=InputUnion(keyboard=KeyboardInput(0, unit, 0x0004 | 0x0002, 0, 0))))
        if not events:
            return "text"
        batch = (Input * len(events))(*events)
        user32 = ctypes.windll.user32
        user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(Input), ctypes.c_int]
        user32.SendInput.restype = ctypes.c_uint
        sent = user32.SendInput(len(events), batch, ctypes.sizeof(Input))
        if sent != len(events):
            return f"error: Unicode input sent {sent}/{len(events)} events"
        return "text"

    def hold_mouse(self, button: str = "left") -> str:
        if self.paused:
            return "ignored_paused"
        if button in self.held_mouse_buttons:
            pyautogui.mouseUp(button=button)
            self.held_mouse_buttons.remove(button)
            return "released"
        pyautogui.mouseDown(button=button)
        self.held_mouse_buttons.add(button)
        return "held"

    def click_mouse(self, button: str = "left") -> str:
        if self.paused:
            return "ignored_paused"
        if button not in {"left", "right", "middle"}:
            return "invalid_mouse_button"
        pyautogui.click(button=button)
        return "click"

    def toggle_pause(self) -> bool:
        self.set_paused(not self.paused)
        return self.paused

    def clamp(self, x: float, y: float) -> tuple[int, int]:
        xi = int(max(0, min(self.screen_w - 1, round(x))))
        yi = int(max(0, min(self.screen_h - 1, round(y))))
        return xi, yi

    def move(self, x: float, y: float) -> tuple[int, int]:
        xi, yi = self.clamp(x, y)
        if not self.paused:
            now = time.perf_counter()
            if self._last_sent is None:
                position = pyautogui.position()
                self._last_sent = self.clamp(position.x, position.y)
                self.last_move_at = now

            if self._last_sent is not None and not self.direct_mode:
                dx = xi - self._last_sent[0]
                dy = yi - self._last_sent[1]
                if (dx * dx + dy * dy) < (self.move_deadzone_px * self.move_deadzone_px):
                    return self._last_sent

                distance = (dx * dx + dy * dy) ** 0.5
                elapsed = max(0.001, min(0.05, now - self.last_move_at))
                max_step = self.max_speed_px_s * elapsed
                if distance > max_step:
                    scale = max_step / distance
                    xi = int(round(self._last_sent[0] + dx * scale))
                    yi = int(round(self._last_sent[1] + dy * scale))

            pyautogui.moveTo(xi, yi)
            self._last_sent = (xi, yi)
            self.last_move_at = now
        return xi, yi

    def perform_action(self, action: str) -> str:
        if action == "pause":
            self.toggle_pause()
            return "paused" if self.paused else "resumed"
        if self.paused and action not in {"calibrate"}:
            return "ignored_paused"
        try:
            if action == "left_click":
                pyautogui.click()
            elif action == "right_click":
                pyautogui.rightClick()
            elif action == "double_click":
                pyautogui.doubleClick()
            elif action == "scroll_up":
                self.scroll_by(self.scroll_step)
            elif action == "scroll_down":
                self.scroll_by(-self.scroll_step)
            elif action == "browser_back":
                pyautogui.hotkey("alt", "left")
            elif action == "browser_forward":
                pyautogui.hotkey("alt", "right")
            elif action == "browser_reload":
                pyautogui.hotkey("ctrl", "r")
            elif action == "browser_address":
                pyautogui.hotkey("ctrl", "l")
            elif action == "browser_prev_tab":
                pyautogui.hotkey("ctrl", "shift", "tab")
            elif action == "browser_next_tab":
                pyautogui.hotkey("ctrl", "tab")
            elif action == "browser_zoom_in":
                pyautogui.hotkey("ctrl", "=")
            elif action == "browser_zoom_out":
                pyautogui.hotkey("ctrl", "-")
            elif action == "browser_zoom_reset":
                pyautogui.hotkey("ctrl", "0")
            elif action == "browser_undo":
                pyautogui.hotkey("ctrl", "z")
            elif action == "cancel_dwell":
                self.dwell.active = False
            elif action == "drag_toggle":
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                else:
                    pyautogui.mouseDown()
                    self.dragging = True
            elif action == "hold_mouse":
                return self.hold_mouse()
            elif action == "calibrate":
                return "calibrate"
            else:
                return f"unknown:{action}"
        except Exception as e:
            return f"error:{e}"
        return action

    @staticmethod
    def scroll_by(amount: int) -> None:
        pyautogui.scroll(int(amount))

    def dwell_progress(self, x: float, y: float, radius_px: int, dwell_ms: int, cooldown_ms: int) -> float:
        """Returns 0..1 progress; resets when cursor leaves radius or during cooldown."""
        now = time.perf_counter()
        if (now - self.dwell.last_fired_at) * 1000 < cooldown_ms:
            self.dwell.active = False
            return 0.0

        dist = ((x - self.dwell.candidate_x) ** 2 + (y - self.dwell.candidate_y) ** 2) ** 0.5
        if not self.dwell.active or dist > radius_px:
            self.dwell.active = True
            self.dwell.candidate_x = x
            self.dwell.candidate_y = y
            self.dwell.started_at = now
            return 0.0

        elapsed_ms = (now - self.dwell.started_at) * 1000
        return max(0.0, min(1.0, elapsed_ms / max(dwell_ms, 1)))

    def dwell_ready(self, x: float, y: float, radius_px: int, dwell_ms: int, cooldown_ms: int) -> bool:
        progress = self.dwell_progress(x, y, radius_px, dwell_ms, cooldown_ms)
        if progress >= 1.0:
            self.dwell.last_fired_at = time.perf_counter()
            self.dwell.active = False
            return True
        return False
