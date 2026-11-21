from __future__ import annotations

import ctypes
import tkinter as tk
from dataclasses import dataclass
from typing import Callable, Optional

from .config import shortcut_kind


TRANSPARENT = "#010203"
I18N = {
    "en": {"settings": "Settings", "calibrate": "Calibrate", "info": "Information", "game": "Game mode", "close": "Close", "back": "Back", "enter": "Enter", "keyboard": "Keyboard", "lock": "Lock", "unlock": "Unlock", "add_page": "Add page", "edit": "Edit shortcut", "change_key": "Change key", "modifier": "Modifier", "mouse_binding": "Mouse: none / left / right", "mouse_debug": "Mouse debug", "hold": "Hold", "release": "Release", "tap": "Tap", "toggle": "Toggle", "add_key": "Add key", "remove_key": "Remove key", "binding_type": "Action type", "language": "Language", "navigation": "Navigation", "done": "Done", "space": "SPACE", "suggestions": "Suggestions"},
    "es": {"settings": "Ajustes", "calibrate": "Calibrar", "info": "Información", "game": "Modo juego", "close": "Cerrar", "back": "Atrás", "enter": "Enter", "keyboard": "Teclado", "lock": "Bloquear", "unlock": "Desbloquear", "add_page": "Añadir página", "edit": "Editar atajo", "change_key": "Cambiar tecla", "modifier": "Modificador", "mouse_binding": "Ratón: ninguno / izq. / der.", "mouse_debug": "Depuración ratón", "hold": "Mantener", "release": "Soltar", "tap": "Toque", "toggle": "Alternar", "add_key": "Añadir tecla", "remove_key": "Quitar tecla", "binding_type": "Tipo de acción", "language": "Idioma", "navigation": "Navegación", "done": "Listo", "space": "ESPACIO", "suggestions": "Sugerencias"},
    "fr": {"settings": "Réglages", "calibrate": "Calibrer", "info": "Informations", "game": "Mode jeu", "close": "Fermer", "back": "Retour", "enter": "Entrée", "keyboard": "Clavier", "lock": "Verrouiller", "unlock": "Déverrouiller", "add_page": "Ajouter page", "edit": "Modifier raccourci", "change_key": "Changer touche", "modifier": "Modificateur", "mouse_binding": "Souris : aucune / gauche / droite", "mouse_debug": "Débogage souris", "hold": "Maintenir", "release": "Relâcher", "tap": "Clic", "toggle": "Basculer", "add_key": "Ajouter touche", "remove_key": "Retirer touche", "binding_type": "Type d’action", "language": "Langue", "navigation": "Navigation", "done": "Terminé", "space": "ESPACE", "suggestions": "Suggestions"},
    "zh": {"settings": "设置", "calibrate": "校准", "info": "信息", "game": "游戏模式", "close": "关闭", "back": "返回", "enter": "确认", "keyboard": "键盘", "lock": "锁定", "unlock": "解锁", "add_page": "添加页面", "edit": "编辑快捷键", "change_key": "更改按键", "modifier": "修饰键", "mouse_binding": "鼠标：无 / 左键 / 右键", "mouse_debug": "鼠标调试", "hold": "按住", "release": "释放", "tap": "单击", "toggle": "切换", "add_key": "添加按键", "remove_key": "移除按键", "binding_type": "操作类型", "language": "语言", "navigation": "导航", "done": "完成", "space": "空格", "suggestions": "建议"},
}
INFO_TEXT = {
    "en": ["Look at a control until its dwell ring completes.", "Direction pad: navigation and Space.", "Back / Enter: common page and form controls.", "Shortcut buttons can press or hold key combinations.", "Game mode holds camera keys while you look at a zone.", "Settings edits shortcut pages and language."],
    "es": ["Mira un control hasta que se complete el indicador.", "La cruceta permite navegar y usar Espacio.", "Atrás / Enter: navegación y formularios.", "Los atajos pueden pulsar o mantener combinaciones.", "En modo juego, la mirada mantiene teclas de cámara.", "Ajustes permite editar atajos e idioma."],
    "fr": ["Fixez un contrôle jusqu'à la fin de l'indicateur.", "La croix directionnelle sert à naviguer et utiliser Espace.", "Retour / Entrée : navigation et formulaires.", "Les raccourcis peuvent maintenir des combinaisons.", "En jeu, le regard maintient les touches caméra.", "Les réglages modifient raccourcis et langue."],
    "zh": ["注视控件直到凝视进度完成。", "方向键用于导航和空格操作。", "返回 / 确认：页面和表单操作。", "快捷键支持按下或持续按住组合键。", "游戏模式会在注视区域时持续按住视角键。", "设置可编辑快捷键页面和语言。"],
}


@dataclass
class Control:
    id: str
    label: str
    action: str
    x: int
    y: int
    w: int
    h: int


class OverlayWindow:
    """Transparent, full-screen gaze control layer rendered with Tk Canvas."""

    def __init__(self, screen_w: int, screen_h: int, keybind_pages: list[list[dict]]):
        self.screen_w, self.screen_h = screen_w, screen_h
        self.keybind_pages = keybind_pages
        self.page_index = 0
        self.mode = "navigation"
        self.menu_open = True
        self.language = "en"
        self.status = ""
        self.paused = False
        self.mouse_debug = False
        self.modal = False
        self.on_activate: Optional[Callable[[Control], None]] = None
        self.gaze = (0, 0)
        self.dwell_progress = 0.0
        self.keyboard_text = ""
        self.suggestions: list[str] = []
        self.active_shortcuts: set[str] = set()
        self.settings_selected: Optional[tuple[int, int]] = None
        self.locked_direction: Optional[str] = None
        self.last_direction: Optional[str] = None
        self.precision_mode = False
        self.edge_scroll_enabled = False
        self.controls: list[Control] = []
        self._created = False
        self._tk: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._scrim_hwnd: Optional[int] = None

    def _tr(self, key: str) -> str:
        return I18N.get(self.language, I18N["en"]).get(key, I18N["en"].get(key, key))

    def _control(self, ident: str, label: str, action: str, x: int, y: int, w: int, h: int) -> None:
        self.controls.append(Control(ident, label, action, x, y, w, h))

    def _build_controls(self) -> None:
        self.controls = []
        w, h = self.screen_w, self.screen_h
        s = min(w, h) / 700
        size = max(46, int(58 * s))
        pad = max(8, int(10 * s))
        menu_size = max(56, min(78, int(54 * s)))
        menu_y = 10
        if self.mode in {"settings", "information"}:
            # Modal screens expose only their own controls and a close affordance.
            self._control("menu_toggle", "×", "menu_toggle", 10, menu_y, menu_size, menu_size)
        else:
            self._control("menu_toggle", "◀" if self.menu_open else "▶", "menu_toggle", 10, menu_y, menu_size, menu_size)
            menu_x = 10 + menu_size + pad
            if self.menu_open:
                menu_w = max(92, int(menu_size * 1.42))
                options = [("⚙", self._tr("settings"), "mode:settings"), ("◎", self._tr("calibrate"), "calibrate"), ("ⓘ", self._tr("info"), "mode:information")]
                for i, (icon, label, action) in enumerate(options):
                    self._control(f"menu_{i}", f"{icon}\n{label}", action, menu_x + i * (menu_w + pad), menu_y, menu_w, menu_size)
                menu_x += len(options) * (menu_w + pad)
            target_mode = "navigation" if self.mode == "game" else "game"
            mode_label = self._tr("navigation") if target_mode == "navigation" else self._tr("game")
            mode_w = max(142, int(menu_size * 2.15))
            self._control("mode_switch", f"⇄\n{mode_label}", f"mode:{target_mode}", menu_x, menu_y, mode_w, menu_size)

        if self.mode in {"navigation", "keyboard"}:
            # Direction pad and movement lock, bottom left.
            bx, by = pad * 2, h - pad * 2 - size * 3
            for name, label, dx, dy in (("up", "▲", 1, 0), ("left", "◀", 0, 1), ("right", "▶", 2, 1), ("down", "▼", 1, 2)):
                self._control(f"nav_{name}", label, f"nav:{name}", bx + dx * size, by + dy * size, size, size)
            self._control("nav_space", self._tr("space"), "key:space", bx + size, by + size, size, size)
            self._control("nav_lock", "🔒" if self.locked_direction else "🔓", "lock", bx + size * 2 + pad, by, size, size)

        elif self.mode == "game":
            # A familiar cross-shaped movement pad, separate from the camera zones.
            game_size = max(48, int(58 * s))
            gx = pad * 2
            camera_zone_h = int(h * .22)
            movement_top = menu_y + menu_size + pad + camera_zone_h + pad
            movement_bottom = h - camera_zone_h - 12 - pad
            gy = max(movement_top, (movement_top + movement_bottom - game_size * 3) // 2)
            self._control("game_move_up", "▲", "game_hold:w", gx + game_size, gy, game_size, game_size)
            self._control("game_move_left", "◀", "game_hold:a", gx, gy + game_size, game_size, game_size)
            self._control("game_jump", self._tr("space"), "game_hold:space", gx + game_size, gy + game_size, game_size, game_size)
            self._control("game_move_right", "▶", "game_hold:d", gx + game_size * 2, gy + game_size, game_size, game_size)
            self._control("game_move_down", "▼", "game_hold:s", gx + game_size, gy + game_size * 2, game_size, game_size)

        if self.mode == "navigation":
            self._add_shortcuts()
            # Tracking helpers from the debug branch, always reachable by gaze.
            self._control(
                "precision_toggle",
                f"PREC\n{'ON' if self.precision_mode else 'OFF'}",
                "precision_toggle",
                w - size - pad, menu_y, size, menu_size,
            )
            self._control(
                "edge_scroll_toggle",
                f"SCROLL\n{'ON' if self.edge_scroll_enabled else 'OFF'}",
                "edge_scroll_toggle",
                w - size * 2 - pad * 2, menu_y, size, menu_size,
            )
            cy = h - pad * 2 - size
            keyboard_w = size * 2
            group_w = size * 2 + keyboard_w + pad * 2
            cx = (w - group_w) // 2
            self._control("back", "↶\n" + self._tr("back"), "key:backspace", cx, cy, size, size)
            self._control("open_keyboard", "▦  " + self._tr("keyboard"), "mode:keyboard", cx + size + pad, cy, keyboard_w, size)
            self._control("enter", "↵\n" + self._tr("enter"), "key:enter", cx + size * 3 + pad * 2, cy, size, size)

        elif self.mode == "keyboard":
            self._add_shortcuts()
            self._add_keyboard(size)

        elif self.mode == "game":
            # Gaze-held camera directions at the four screen corners.
            zone_w, zone_h = int(w * .13), int(h * .22)
            camera_top = menu_y + menu_size + pad
            corners = (("top_left", "↖", 12, camera_top, "up_left"), ("top_right", "↗", w - zone_w - 12, camera_top, "up_right"), ("bottom_left", "↙", 12, h - zone_h - 12, "down_left"), ("bottom_right", "↘", w - zone_w - 12, h - zone_h - 12, "down_right"))
            for name, label, x, y, direction in corners:
                self._control(f"game_{name}", label, f"game_move:{direction}", x, y, zone_w, zone_h)
            self._add_shortcuts(game_mode=True)

        elif self.mode == "settings":
            self._add_settings(size)
        elif self.mode == "information":
            self._add_information(size)

    def refresh_controls(self) -> None:
        self._build_controls()
        self.set_modal(self.mode in {"settings", "information"})

    def _add_shortcuts(self, game_mode: bool = False) -> None:
        w, h = self.screen_w, self.screen_h
        size = max(58, int(82 * min(w, h) / 700))
        pad = max(8, int(10 * min(w, h) / 700))
        pages = self.keybind_pages or [[]]
        self.page_index %= max(1, len(pages))
        if game_mode:
            grid_w, grid_h = size * 2 + pad, size * 2 + pad
            zone_w = int(w * .13)
            x = w - zone_w - grid_w - pad * 3
            y = (h - grid_h) // 2
        else:
            x, y = w - size * 3 - pad * 3, h - (size * 2 + pad) - pad * 2
        self._control("page_prev", "◀", "page:-1", x - size - pad, y + (size + pad) // 2, size, size)
        self._control("page_next", "▶", "page:1", x + 2 * (size + pad), y + (size + pad) // 2, size, size)
        for idx in range(4):
            row, col = divmod(idx, 2)
            if idx < len(pages[self.page_index]):
                entry = pages[self.page_index][idx]
                label = str(entry.get("label", f"Keybind {idx + 1}"))
                if label.lower().startswith("atajo"):
                    label = "Keybind" + label[5:]
                combo = " + ".join(str(key).upper() for key in entry.get("keys", [])) or "(none)"
                if entry.get("mouse"):
                    combo = f"Mouse {str(entry['mouse']).title()}"
                shortcut_id = f"{self.page_index}:{idx}"
                kind = shortcut_kind(entry)
                state = "ON" if shortcut_id in self.active_shortcuts else "OFF"
                type_label = f"{self._tr('toggle')} {state}" if kind == "toggle" else self._tr("tap")
                label = f"{label} · {type_label}\n{combo}"
            else:
                label = "+"
            action = f"shortcut:{self.page_index}:{idx}" if idx < len(pages[self.page_index]) else f"shortcut_add:{idx}"
            self._control(f"shortcut_{idx}", label, action, x + col * (size + pad), y + row * (size + pad), size, size)

    def _add_keyboard(self, size: int) -> None:
        w, h = self.screen_w, self.screen_h
        panel_w, key_h = int(w * .68), max(42, int(size * .7))
        key_w = int((panel_w - 24) / 10)
        x0, y0 = (w - panel_w) // 2, int(h * .17)
        self._control("keyboard_close", "×", "mode:navigation", x0 + panel_w - key_w, y0, key_w, key_h)
        sug_y = y0 + key_h + 8
        input_w = max(key_w * 2, int(panel_w * .24))
        gap = 6
        self._control("keyboard_input", self.keyboard_text[-24:] or "…", "noop", x0, sug_y, input_w, key_h)
        suggestion_x = x0 + input_w + gap
        suggestion_w = max(24, (panel_w - input_w - gap - 4 * gap) // 5)
        for i, suggestion in enumerate(self.suggestions[:5]):
            self._control(f"suggest_{i}", suggestion, f"suggest:{suggestion}", suggestion_x + i * (suggestion_w + gap), sug_y, suggestion_w, key_h)
        rows = [list("1234567890"), list("qwertyuiop"), list("asdfghjkl"), list("zxcvbnm"), list("-=[];',./")]
        y = sug_y + key_h + 8
        for ri, row in enumerate(rows):
            offset = (10 - len(row)) * key_w // 2
            kw = int(key_w * .86)
            for ci, char in enumerate(row):
                self._control(f"key_{char}", char.upper(), f"key:{char}", x0 + 12 + offset + ci * key_w, y + ri * (key_h + 8), kw, key_h)
        y += len(rows) * (key_h + 8)
        controls = [("F1", "F1"), ("F2", "F2"), ("F3", "F3"), ("F4", "F4"), ("F5", "F5"), ("F6", "F6"), ("F7", "F7"), ("F8", "F8"), ("F9", "F9"), ("F10", "F10"), ("F11", "F11"), ("F12", "F12")]
        fw = (panel_w - 24) // len(controls)
        for i, (label, value) in enumerate(controls):
            self._control(f"key_f{i+1}", label, f"key:{value.lower()}", x0 + i * fw, y, fw - 3, key_h)
        y += key_h + 8
        bottom = [("shift", "⇧"), ("capslock", "Caps"), ("tab", "Tab"), ("ctrl", "Ctrl"), ("alt", "Alt"), ("space", self._tr("space")), ("backspace", "⌫"), ("enter", "↵"), ("esc", "Esc")]
        bw = panel_w // len(bottom)
        for i, (key, label) in enumerate(bottom):
            self._control(f"key_bottom_{key}", label, f"key:{key}", x0 + i * bw, y, bw - 4, key_h)

    def _add_settings(self, size: int) -> None:
        w, h = self.screen_w, self.screen_h
        pages = self.keybind_pages or [[]]
        title = f"{self._tr('edit')} — {self.page_index + 1}/{len(pages)}"
        top = max(size + 20, int(h * .1))
        self._control("settings_title", title, "noop", w * .25, top + 10, w * .5, size)
        grid_w, cell_w, cell_h = int(w * .62), int(w * .29), int(size * 1.3)
        x0, y0 = (w - grid_w) // 2, top + size + 22
        for i, entry in enumerate(pages[self.page_index]):
            row, col = divmod(i, 2)
            keys = f"Mouse {entry['mouse']}" if entry.get("mouse") else "+".join(entry.get("keys", [])) or "(none)"
            kind = shortcut_kind(entry)
            type_text = self._tr("toggle") if kind == "toggle" else self._tr("tap")
            binding_label = str(entry.get("label", f"Keybind {i + 1}"))
            if binding_label.lower().startswith("atajo"):
                binding_label = "Keybind" + binding_label[5:]
            label = f"{binding_label}  [{keys}]  {type_text}"
            self._control(f"setting_bind_{i}", label, f"edit_binding:{i}", x0 + col * cell_w, y0 + row * (cell_h + 12), cell_w - 8, cell_h)
        y = int(h * .68)
        self._control("settings_prev", "◀", "page:-1", w * .16, y, size, size)
        self._control("settings_next", "▶", "page:1", w * .22, y, size, size)
        self._control("settings_add", "+  " + self._tr("add_page"), "add_page", w * .31, y, w * .22, size)
        self._control("settings_language", self._tr("language") + f": {self.language.upper()}", "language_next", w * .55, y, w * .2, size)
        debug_state = "ON" if self.mouse_debug else "OFF"
        self._control("settings_debug_mouse", f"{self._tr('mouse_debug')}: {debug_state}", "debug_mouse_toggle", w * .77, y, w * .21, size)
        if self.settings_selected is not None:
            selected_page, selected_index = self.settings_selected
            selected_kind = shortcut_kind(pages[selected_page][selected_index]) if selected_page < len(pages) and selected_index < len(pages[selected_page]) else "click"
            y1, y2 = h * .76, h * .84
            self._control("binding_cycle", self._tr("change_key"), "binding_cycle", w * .10, y1, w * .18, size)
            self._control("binding_add_key", "+  " + self._tr("add_key"), "binding_add_key", w * .30, y1, w * .18, size)
            self._control("binding_remove_key", "−  " + self._tr("remove_key"), "binding_remove_key", w * .50, y1, w * .18, size)
            self._control("binding_type", self._tr("binding_type") + ": " + self._tr("toggle" if selected_kind == "toggle" else "tap"), "binding_type", w * .70, y1, w * .24, size)
            self._control("binding_modifier", self._tr("modifier"), "binding_modifier", w * .25, y2, w * .20, size)
            self._control("binding_mouse", self._tr("mouse_binding"), "binding_mouse", w * .48, y2, w * .30, size)
        self._control("settings_done", self._tr("done"), "mode:navigation", w - size * 2, h - size * 2, size * 2, size)

    def _add_information(self, size: int) -> None:
        w, h = self.screen_w, self.screen_h
        lines = INFO_TEXT.get(self.language, INFO_TEXT["en"])
        top = max(size + 20, int(h * .1))
        self._control("info_title", self._tr("info"), "noop", w * .3, top + 10, w * .4, size)
        for i, line in enumerate(lines):
            self._control(f"info_line_{i}", line, "noop", w * .2, top + size + 18 + i * max(30, int(h * .07)), w * .6, size)
        self._control("info_language", self._tr("language") + f": {self.language.upper()}", "language_next", w * .36, h * .76, w * .28, size)
        self._control("info_done", self._tr("done"), "mode:navigation", w - size * 2, h - size * 2, size * 2, size)

    def button_at(self, screen_x: int, screen_y: int) -> Optional[Control]:
        for control in reversed(self.controls):
            if control.action == "display" or not (control.x <= screen_x <= control.x + control.w and control.y <= screen_y <= control.y + control.h):
                continue
            if control.action.startswith("shortcut:"):
                _, page, index = control.action.split(":")
                entry = self.keybind_pages[int(page)][int(index)]
                if shortcut_kind(entry) == "click":
                    notch = min(12, control.w // 5, control.h // 5)
                    if screen_x > control.x + control.w - notch and screen_y < control.y + notch:
                        continue
            return control
        return None

    def set_mouse_debug(self, enabled: bool) -> None:
        self.mouse_debug = bool(enabled)
        self._apply_input_style()

    def set_modal(self, enabled: bool) -> None:
        enabled = bool(enabled)
        changed = self.modal != enabled
        self.modal = enabled
        if self.modal:
            self._create_scrim()
        else:
            self._destroy_scrim()
        if changed:
            self._apply_input_style()

    def _create_scrim(self) -> None:
        if self._scrim_hwnd is not None or __import__("sys").platform != "win32":
            return
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            user32.CreateWindowExW.argtypes = [ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            user32.CreateWindowExW.restype = ctypes.c_void_p
            user32.SetLayeredWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_uint]
            user32.SetLayeredWindowAttributes.restype = ctypes.c_bool
            user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            user32.SetWindowPos.restype = ctypes.c_bool
            user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
            user32.ShowWindow.restype = ctypes.c_bool
            kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            kernel32.GetModuleHandleW.restype = ctypes.c_void_p
            exstyle = 0x00080000 | 0x08000000 | 0x00000080  # layered, no-activate, tool window
            style = 0x80000000 | 0x00000004  # popup, black static control
            hwnd = user32.CreateWindowExW(exstyle, "STATIC", "EyeTrackerPro dim layer", style, 0, 0, self.screen_w, self.screen_h, None, None, kernel32.GetModuleHandleW(None), None)
            if not hwnd:
                return
            user32.SetLayeredWindowAttributes(hwnd, 0, 150, 0x2)  # LWA_ALPHA dims the desktop below
            user32.SetWindowPos(hwnd, ctypes.c_void_p(-1), 0, 0, self.screen_w, self.screen_h, 0x0010 | 0x0040)
            user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
            self._scrim_hwnd = int(hwnd)
            self._raise_control_window()
        except Exception:
            self._scrim_hwnd = None

    def _destroy_scrim(self) -> None:
        if self._scrim_hwnd is not None and __import__("sys").platform == "win32":
            try:
                ctypes.windll.user32.DestroyWindow(ctypes.c_void_p(self._scrim_hwnd))
            except Exception:
                pass
        self._scrim_hwnd = None

    def _raise_control_window(self) -> None:
        if self._created and self._tk is not None and __import__("sys").platform == "win32":
            try:
                user32 = ctypes.windll.user32
                user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
                user32.SetWindowPos.restype = ctypes.c_bool
                user32.SetWindowPos(ctypes.c_void_p(self._tk.winfo_id()), ctypes.c_void_p(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)
            except Exception:
                pass

    def _apply_input_style(self) -> None:
        if not self._created or self._tk is None or __import__("sys").platform != "win32":
            return
        try:
            hwnd = self._tk.winfo_id()
            user32 = ctypes.windll.user32
            user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
            user32.GetWindowLongW.restype = ctypes.c_long
            user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
            user32.SetWindowLongW.restype = ctypes.c_long
            user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            user32.SetWindowPos.restype = ctypes.c_bool
            style = user32.GetWindowLongW(hwnd, -20)
            click_through = 0x20
            style = style & ~click_through if self.mouse_debug or self.modal else style | click_through
            user32.SetWindowLongW(hwnd, -20, style)
            user32.SetWindowPos(ctypes.c_void_p(hwnd), None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)
        except Exception:
            pass

    def _mouse_click(self, event: tk.Event) -> None:
        if not self.mouse_debug or self.on_activate is None:
            return
        control = self.button_at(int(event.x), int(event.y))
        if control is not None:
            self.on_activate(control)

    @staticmethod
    def _font_size(label: str, width: int, height: int) -> int:
        if label.strip().upper() in {"SPACE", "ESPACIO", "空格"}:
            return min(10, max(8, (height - 8) // 2))
        if label.lower().startswith("keybind"):
            return min(10, max(8, (height - 8) // 2))
        lines = label.splitlines() or [label]
        longest = max((len(line) for line in lines), default=1)
        width_factor = 1.0 if any(ord(char) > 127 for char in label) else 0.62
        by_width = (width - 12) / max(1.0, width_factor * longest)
        by_height = (height - 8) / max(1.0, 1.25 * len(lines))
        fitted = min(24, int(min(by_width, by_height)))
        if label.lstrip().startswith("↵"):
            return max(8, fitted)
        return max(8, int(fitted * 0.78))

    def _create(self) -> None:
        if self._created:
            return
        root = tk.Tk()
        root.overrideredirect(True)
        root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        root.attributes("-topmost", True)
        root.configure(bg=TRANSPARENT)
        try:
            root.attributes("-transparentcolor", TRANSPARENT)
        except tk.TclError:
            if __import__("sys").platform != "win32":
                root.attributes("-alpha", 0.86)
        canvas = tk.Canvas(root, width=self.screen_w, height=self.screen_h, bg=TRANSPARENT, bd=0, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.bind("<Button-1>", self._mouse_click)
        root.update_idletasks()
        # Keep the game/desktop as the active window while the control layer stays topmost.
        if __import__("sys").platform == "win32":
            try:
                hwnd = root.winfo_id()
                user32 = ctypes.windll.user32
                GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_NOACTIVATE = -20, 0x80000, 0x20, 0x08000000
                user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
                user32.GetWindowLongW.restype = ctypes.c_long
                user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
                user32.SetWindowLongW.restype = ctypes.c_long
                user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
                user32.SetWindowPos.restype = ctypes.c_bool
                user32.SetLayeredWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_uint]
                user32.SetLayeredWindowAttributes.restype = ctypes.c_bool
                style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE)
                # COLORREF for TRANSPARENT (#010203), LWA_COLORKEY.
                user32.SetLayeredWindowAttributes(ctypes.c_void_p(hwnd), 0x030201, 0, 0x1)
                user32.SetWindowPos(ctypes.c_void_p(hwnd), ctypes.c_void_p(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)
            except Exception:
                pass
        self._tk, self._canvas, self._created = root, canvas, True
        self.set_mouse_debug(self.mouse_debug)
        self.set_modal(self.modal)

    def render(self) -> None:
        self.set_modal(self.mode in {"settings", "information"})
        self._create()
        assert self._canvas is not None and self._tk is not None
        self._build_controls()
        canvas = self._canvas
        canvas.delete("all")
        if self.modal and __import__("sys").platform != "win32":
            # Non-Windows fallback; Windows uses a translucent native input-blocking scrim.
            canvas.create_rectangle(0, 0, self.screen_w, self.screen_h, fill="#080808", outline="")
            size = max(46, int(58 * min(self.screen_w, self.screen_h) / 700))
            top = max(size + 20, int(self.screen_h * .1))
            canvas.create_rectangle(int(self.screen_w * .1), top, int(self.screen_w * .9), int(self.screen_h * .94), fill="#171717", outline="#777777", width=2)
        hovered = self.button_at(*self.gaze)
        for control in self.controls:
            if control.id == "keyboard_input":
                canvas.create_rectangle(control.x, control.y, control.x + control.w, control.y + control.h, fill="#080808", outline="#FFFFFF", width=2, stipple="gray50")
                canvas.create_text(control.x + control.w // 2, control.y + control.h // 2, text=control.label, fill="white", font=("Segoe UI", self._font_size(control.label, control.w, control.h)), width=control.w - 8)
                continue
            if control.action == "noop":
                x, y = control.x + control.w // 2, control.y + control.h // 2
                font_size = self._font_size(control.label, control.w, control.h)
                if control.id.startswith("info_"):
                    # Tk Canvas has no stroke option. Use the exact same glyph size
                    # and weight for a 2 px dark edge, then draw white text on top.
                    outline_font = ("Segoe UI", font_size, "bold")
                    for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1)):
                        canvas.create_text(x + ox, y + oy, text=control.label, fill="#080808", font=outline_font, width=control.w - 12)
                    canvas.create_text(x, y, text=control.label, fill="#FFFFFF", font=("Segoe UI", font_size, "bold"), width=control.w - 12)
                else:
                    canvas.create_text(x, y, text=control.label, fill="#f5f5f5", font=("Segoe UI", font_size), width=control.w - 12)
                continue
            is_hovered = hovered is not None and hovered.id == control.id
            fill = "#303030" if is_hovered else "#111111"
            if control.action.startswith("shortcut:"):
                _, page, index = control.action.split(":")
                entry = self.keybind_pages[int(page)][int(index)]
                shortcut_id = f"{page}:{index}"
                if shortcut_kind(entry) == "toggle":
                    if shortcut_id in self.active_shortcuts:
                        fill = "#15544C"
                    canvas.create_rectangle(control.x, control.y, control.x + control.w, control.y + control.h, fill=fill, outline="#64E6CF" if shortcut_id in self.active_shortcuts else "#FFFFFF", width=2, stipple="gray50")
                else:
                    x1, y1, x2, y2 = control.x, control.y, control.x + control.w, control.y + control.h
                    notch = min(12, control.w // 5, control.h // 5)
                    canvas.create_polygon(x1, y1, x2 - notch, y1, x2, y1 + notch, x2, y2, x1, y2, fill=fill, outline="#FFFFFF", width=2, stipple="gray50")
            else:
                canvas.create_rectangle(control.x, control.y, control.x + control.w, control.y + control.h, fill=fill, outline="#FFFFFF", width=2, stipple="gray50")
            canvas.create_text(control.x + control.w // 2, control.y + control.h // 2, text=control.label, fill="white", font=("Segoe UI", self._font_size(control.label, control.w, control.h), "bold"), width=control.w - 8)
        if hovered and self.dwell_progress > 0:
            cx, cy = hovered.x + hovered.w // 2, hovered.y + hovered.h // 2
            radius = max(9, min(hovered.w, hovered.h) // 2 - 3)
            canvas.create_arc(cx - radius, cy - radius, cx + radius, cy + radius, start=90, extent=-360 * max(0, min(1, self.dwell_progress)), style="arc", outline="#38d46a", width=5)
        if self.paused:
            canvas.create_text(self.screen_w // 2, 24, text="PAUSED", fill="#ffcf4a", font=("Segoe UI", 14, "bold"))
        if self.mouse_debug:
            canvas.create_text(self.screen_w // 2, 44 if self.paused else 24, text="MOUSE DEBUG", fill="#ffcf4a", font=("Segoe UI", 12, "bold"))
        try:
            self._tk.update_idletasks()
            self._tk.update()
            self._tk.attributes("-topmost", True)
            self._raise_control_window()
        except tk.TclError:
            self._created = False

    def destroy(self) -> None:
        self._destroy_scrim()
        if self._tk is not None:
            try:
                self._tk.destroy()
            except tk.TclError:
                pass
        self._tk = self._canvas = None
        self._created = False
