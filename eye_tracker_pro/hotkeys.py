"""Non-blocking Windows low-level keyboard listener for app controls."""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
from collections.abc import Callable


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _Message(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("w_param", ctypes.c_size_t),
        ("l_param", ctypes.c_ssize_t),
        ("time", ctypes.c_uint),
        ("point", _Point),
        ("l_private", ctypes.c_uint),
    ]


class _KeyboardHookData(ctypes.Structure):
    _fields_ = [
        ("vk_code", ctypes.c_uint),
        ("scan_code", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("time", ctypes.c_uint),
        ("extra_info", ctypes.c_size_t),
    ]


class LowLevelHotkeyListener:
    """Listen for app hotkeys globally without consuming the user's key presses."""

    WH_KEYBOARD_LL = 13
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105
    WM_QUIT = 0x0012
    PM_NOREMOVE = 0

    MODIFIER_KEYS = {0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5}
    APP_KEYS = {ord(char): ord(char.lower()) for char in "CPMBFVGXZ"}
    FINE_TUNE_KEYS = {ord(char): ord(char.lower()) for char in "ADWSJLIK R" if char != " "}
    APP_KEYS[0x1B] = 0x1B

    def __init__(self, fine_tune_active: Callable[[], bool], escape_allowed: Callable[[], bool]):
        self.fine_tune_active = fine_tune_active
        self.escape_allowed = escape_allowed
        self.events: queue.Queue[int] = queue.Queue()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._hook = None
        self._callback = None
        self.available = False
        self.error: str | None = None

    def start(self) -> bool:
        if sys.platform != "win32":
            return False
        if self._thread and self._thread.is_alive():
            return self.available
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_guarded, name="EyeTrackerHotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(2.0)
        return self.available

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            self._ready.set()

    def _run(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentThreadId.restype = ctypes.c_uint
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        self._thread_id = int(kernel32.GetCurrentThreadId())
        user32.PeekMessageW.argtypes = [ctypes.POINTER(_Message), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
        user32.PeekMessageW.restype = ctypes.c_int
        user32.GetMessageW.argtypes = [ctypes.POINTER(_Message), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
        user32.GetMessageW.restype = ctypes.c_int
        user32.TranslateMessage.argtypes = [ctypes.POINTER(_Message)]
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(_Message)]
        user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
        user32.UnhookWindowsHookEx.restype = ctypes.c_int
        user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t]
        user32.CallNextHookEx.restype = ctypes.c_ssize_t

        callback_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t)
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, callback_type, ctypes.c_void_p, ctypes.c_uint]
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        down_keys: set[int] = set()
        modifiers_down: set[int] = set()

        def on_keyboard(code: int, message: int, data_ptr: int) -> int:
            if code >= 0 and data_ptr:
                event = ctypes.cast(data_ptr, ctypes.POINTER(_KeyboardHookData)).contents
                vk = int(event.vk_code)
                if event.flags & 0x10:  # Ignore key events synthesized by the on-screen keyboard.
                    return int(user32.CallNextHookEx(self._hook, code, message, data_ptr))
                is_down = message in (self.WM_KEYDOWN, self.WM_SYSKEYDOWN)
                is_up = message in (self.WM_KEYUP, self.WM_SYSKEYUP)
                if is_down and vk in self.MODIFIER_KEYS:
                    modifiers_down.add(vk)
                elif is_up and vk in self.MODIFIER_KEYS:
                    modifiers_down.discard(vk)
                elif is_up:
                    down_keys.discard(vk)
                elif is_down and vk not in down_keys:
                    down_keys.add(vk)
                    if not modifiers_down:
                        key = self.APP_KEYS.get(vk)
                        if vk == 0x1B:
                            if self.escape_allowed():
                                self.events.put(0x1B)
                        elif key is not None:
                            self.events.put(key)
                        elif self.fine_tune_active() and vk in self.FINE_TUNE_KEYS:
                            self.events.put(self.FINE_TUNE_KEYS[vk])
            return int(user32.CallNextHookEx(self._hook, code, message, data_ptr))

        self._callback = callback_type(on_keyboard)
        user32.PeekMessageW(ctypes.byref(_Message()), None, 0, 0, self.PM_NOREMOVE)
        self._hook = user32.SetWindowsHookExW(self.WH_KEYBOARD_LL, self._callback, kernel32.GetModuleHandleW(None), 0)
        if not self._hook:
            self.error = f"SetWindowsHookExW failed ({ctypes.get_last_error()})"
            self._ready.set()
            return
        self.available = True
        self._ready.set()
        msg = _Message()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
            self.available = False

    def poll(self) -> int | None:
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._thread_id is not None and sys.platform == "win32":
            try:
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                user32.PostThreadMessageW.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
                user32.PostThreadMessageW.restype = ctypes.c_int
                user32.PostThreadMessageW(self._thread_id, self.WM_QUIT, 0, 0)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread_id = None
        self.available = False
