from __future__ import annotations

import time
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
        if paused and self.dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self.dragging = False

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
