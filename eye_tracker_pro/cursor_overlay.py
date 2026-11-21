from __future__ import annotations

import ctypes
import os
import tkinter as tk
from typing import Optional
from ctypes import wintypes


class CursorOverlay:
    """Small click-through gaze marker shown over other Windows applications."""

    _TRANSPARENT_COLOR = "#ff00ff"
    _GWL_EXSTYLE = -20
    _WS_EX_TRANSPARENT = 0x00000020
    _WS_EX_LAYERED = 0x00080000
    _WS_EX_TOOLWINDOW = 0x00000080
    _WS_EX_NOACTIVATE = 0x08000000
    _SIZE = 64
    _CENTER = _SIZE // 2

    def __init__(self, screen_width: int, screen_height: int):
        self._enabled = os.name == "nt"
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._screen_width = screen_width
        self._screen_height = screen_height
        if not self._enabled:
            return

        try:
            root = tk.Tk()
            self._root = root
            root.overrideredirect(True)
            root.geometry(f"{self._SIZE}x{self._SIZE}+0+0")
            root.withdraw()
            root.configure(bg=self._TRANSPARENT_COLOR)
            root.wm_attributes("-topmost", True)
            root.wm_attributes("-transparentcolor", self._TRANSPARENT_COLOR)
            canvas = tk.Canvas(
                root,
                width=self._SIZE,
                height=self._SIZE,
                bg=self._TRANSPARENT_COLOR,
                bd=0,
                highlightthickness=0,
            )
            self._canvas = canvas
            canvas.pack(fill="both", expand=True)
            root.update_idletasks()

            # Keep the marker above the desktop while passing all mouse input
            # through to the real Windows cursor underneath it.
            hwnd = root.winfo_id()
            user32 = ctypes.windll.user32
            get_parent = user32.GetParent
            get_parent.argtypes = [wintypes.HWND]
            get_parent.restype = wintypes.HWND
            get_window_long = user32.GetWindowLongPtrW
            get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
            get_window_long.restype = ctypes.c_ssize_t
            set_window_long = user32.SetWindowLongPtrW
            set_window_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
            set_window_long.restype = ctypes.c_ssize_t
            set_window_pos = user32.SetWindowPos
            set_window_pos.argtypes = [
                wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, wintypes.UINT,
            ]
            set_window_pos.restype = wintypes.BOOL

            hwnd = wintypes.HWND(hwnd)
            # Tk's widget handle can be the client child window; apply the
            # click-through styles to its top-level window instead.
            top_hwnd = get_parent(hwnd) or hwnd.value
            top_hwnd = wintypes.HWND(top_hwnd)
            ex_style = get_window_long(top_hwnd, self._GWL_EXSTYLE)
            ex_style |= (
                self._WS_EX_LAYERED | self._WS_EX_TRANSPARENT
                | self._WS_EX_TOOLWINDOW | self._WS_EX_NOACTIVATE
            )
            set_window_long(top_hwnd, self._GWL_EXSTYLE, ex_style)
            set_window_pos(top_hwnd, wintypes.HWND(-1), 0, 0, self._SIZE, self._SIZE, 0x0020)

        except Exception as exc:
            self._enabled = False
            print(f"No se pudo crear el cursor circular: {exc}", flush=True)
            try:
                if self._root:
                    self._root.destroy()
            except Exception:
                pass
            self._root = None
            self._canvas = None

    def update(
        self,
        point: Optional[tuple[int, int]],
        visible: bool,
        dwell_progress: float = 0.0,
    ) -> None:
        if not self._enabled or self._root is None or self._canvas is None:
            return
        try:
            canvas = self._canvas
            canvas.delete("cursor")
            if visible and point is not None:
                x = max(0, min(self._screen_width - 1, int(point[0])))
                y = max(0, min(self._screen_height - 1, int(point[1])))
                left = max(0, min(self._screen_width - self._SIZE, x - self._CENTER))
                top = max(0, min(self._screen_height - self._SIZE, y - self._CENTER))
                cursor_x, cursor_y = x - left, y - top
                if not self._root.winfo_viewable():
                    self._root.deiconify()
                self._root.geometry(f"{self._SIZE}x{self._SIZE}+{left}+{top}")
                # Match the calibration indicator: a large colored center,
                # white outline, and an optional dwell progress ring.
                canvas.create_oval(
                    cursor_x - 18, cursor_y - 18,
                    cursor_x + 18, cursor_y + 18,
                    fill="#0078ff", outline="", tags="cursor",
                )
                canvas.create_oval(
                    cursor_x - 24, cursor_y - 24,
                    cursor_x + 24, cursor_y + 24,
                    outline="#ffffff", width=3, tags="cursor",
                )
                progress = max(0.0, min(1.0, float(dwell_progress)))
                if progress > 0:
                    canvas.create_arc(
                        cursor_x - 29, cursor_y - 29,
                        cursor_x + 29, cursor_y + 29,
                        start=90, extent=-360 * progress,
                        style="arc", outline="#00b4ff", width=4, tags="cursor",
                    )
            else:
                if self._root.winfo_viewable():
                    self._root.withdraw()
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._enabled = False

    def destroy(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except tk.TclError:
                pass
            self._root = None
            self._canvas = None
