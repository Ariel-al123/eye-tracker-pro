from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import ButtonConfig


@dataclass
class OverlayState:
    gaze_x: int = 0
    gaze_y: int = 0
    dwell_progress: float = 0.0
    status: str = "Iniciando..."
    paused: bool = False


class OverlayWindow:
    """Always-on-top OpenCV control bar with customizable action buttons."""

    def __init__(self, buttons: list[ButtonConfig], window_name: str = "EyeTrackerPro - Controles"):
        self.buttons = [b for b in buttons if b.enabled]
        self.window_name = window_name
        self.state = OverlayState()
        self.win_x = 10
        self.win_y = 10
        self.width = max((b.x + b.w for b in self.buttons), default=800) + 20
        self.height = max((b.y + b.h for b in self.buttons), default=100) + 55
        self.visible = True
        self._created = False
        self._pending_action: Optional[str] = None

    def create(self) -> None:
        if self._created:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.moveWindow(self.window_name, self.win_x, self.win_y)
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._created = True

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.x <= x <= button.x + button.w and button.y <= y <= button.y + button.h:
                self._pending_action = button.action
                return

    def consume_action(self) -> Optional[str]:
        action = self._pending_action
        self._pending_action = None
        return action

    def button_at(self, screen_x: int, screen_y: int) -> Optional[ButtonConfig]:
        lx, ly = screen_x - self.win_x, screen_y - self.win_y
        for b in self.buttons:
            if b.x <= lx <= b.x + b.w and b.y <= ly <= b.y + b.h:
                return b
        return None

    def render(self) -> None:
        if not self.visible:
            return
        self.create()
        try:
            rect = cv2.getWindowImageRect(self.window_name)
            if rect and len(rect) == 4:
                self.win_x, self.win_y, _, _ = [int(v) for v in rect]
        except Exception:
            pass
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)

        hovered = self.button_at(self.state.gaze_x, self.state.gaze_y)
        for b in self.buttons:
            x1, y1, x2, y2 = b.x, b.y, b.x + b.w, b.y + b.h
            bg = (55, 55, 55) if hovered and hovered.id == b.id else (40, 40, 40)
            if self.state.paused and b.action not in {"pause", "calibrate"}:
                bg = (30, 30, 30)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (250, 170, 96), 2)
            label = b.label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tx = x1 + (b.w - tw) // 2
            ty = y1 + (b.h + th) // 2
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)

        status = self.state.status + ("  [PAUSADO]" if self.state.paused else "")
        cv2.putText(canvas, status, (20, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

        # Gaze dot in overlay-local coords
        gx, gy = self.state.gaze_x - self.win_x, self.state.gaze_y - self.win_y
        cv2.circle(canvas, (int(gx), int(gy)), 7, (80, 200, 80), -1, cv2.LINE_AA)
        progress = max(0.0, min(1.0, self.state.dwell_progress))
        if progress > 0:
            cv2.ellipse(canvas, (int(gx), int(gy)), (14, 14), -90, 0, 360 * progress, (0, 180, 255), 3, cv2.LINE_AA)

        cv2.imshow(self.window_name, canvas)

    def destroy(self) -> None:
        if self._created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._created = False
