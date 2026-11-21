from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class ButtonConfig:
    id: str
    label: str
    action: str  # left_click,right_click,double_click,scroll_up,scroll_down,drag_toggle,pause,calibrate
    x: int
    y: int
    w: int = 140
    h: int = 64
    enabled: bool = True


@dataclass
class AppConfig:
    gaze_engine: str = "eyegestures"
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    show_preview: bool = True
    mirror_preview: bool = True

    # Calibration
    calibration_points: int = 25  # 5, 9, or a full-screen 5x5 grid
    calibration_samples_per_point: int = 40
    eyetheia_samples_per_target: int = 12
    calibration_sample_delay_ms: int = 40
    calibration_point_radius: int = 22

    # Smoothing / filter
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.025
    filter_d_cutoff: float = 1.0
    move_deadzone_px: int = 20
    cursor_max_speed_px_s: int = 950

    # Dwell click
    dwell_enabled: bool = True
    dwell_time_ms: int = 450
    dwell_radius_px: int = 35
    dwell_cooldown_ms: int = 700
    edge_scroll_enabled: bool = False
    edge_scroll_margin_px: int = 45
    edge_scroll_interval_ms: int = 170
    edge_scroll_amount: int = 2

    # Safety
    pause_on_face_lost: bool = True
    face_lost_pause_ms: int = 700
    move_cursor_enabled: bool = True

    # Fine adjustment (applied after gaze prediction)
    fine_gain_x: float = 1.0
    fine_gain_y: float = 1.0
    fine_offset_x: float = 0.0
    fine_offset_y: float = 0.0
    fine_step_px: int = 10
    fine_gain_step: float = 0.02

    # UI
    overlay_enabled: bool = True
    overlay_alpha: float = 0.92
    show_gaze_dot: bool = True

    buttons: List[ButtonConfig] = field(default_factory=lambda: [
        ButtonConfig("btn_left", "Clic izq", "left_click", 20, 20, 110, 54),
        ButtonConfig("btn_right", "Clic der", "right_click", 140, 20, 110, 54),
        ButtonConfig("btn_double", "Doble", "double_click", 260, 20, 110, 54),
        ButtonConfig("btn_up", "Scroll ↑", "scroll_up", 380, 20, 110, 54),
        ButtonConfig("btn_down", "Scroll ↓", "scroll_down", 500, 20, 110, 54),
        ButtonConfig("btn_pause", "Pausa", "pause", 620, 20, 110, 54),
        ButtonConfig("btn_cal", "Calibrar", "calibrate", 740, 20, 110, 54),
        ButtonConfig("btn_back", "Atrás", "browser_back", 20, 85, 110, 54),
        ButtonConfig("btn_forward", "Adelante", "browser_forward", 140, 85, 110, 54),
        ButtonConfig("btn_reload", "Recargar", "browser_reload", 260, 85, 110, 54),
        ButtonConfig("btn_address", "Dirección", "browser_address", 380, 85, 110, 54),
        ButtonConfig("btn_prev_tab", "Pestaña ←", "browser_prev_tab", 500, 85, 110, 54),
        ButtonConfig("btn_next_tab", "Pestaña →", "browser_next_tab", 620, 85, 110, 54),
        ButtonConfig("btn_zoom_in", "Zoom +", "browser_zoom_in", 740, 85, 110, 54),
        ButtonConfig("btn_zoom_out", "Zoom −", "browser_zoom_out", 20, 150, 110, 54),
        ButtonConfig("btn_zoom_reset", "Zoom 100%", "browser_zoom_reset", 140, 150, 110, 54),
        ButtonConfig("btn_precision", "Precisión", "precision_toggle", 260, 150, 110, 54),
        ButtonConfig("btn_edge_scroll", "Auto-scroll", "edge_scroll_toggle", 380, 150, 110, 54),
        ButtonConfig("btn_cancel", "Cancelar", "cancel_dwell", 500, 150, 110, 54),
        ButtonConfig("btn_undo", "Deshacer", "browser_undo", 620, 150, 110, 54),
    ])


DEFAULT_CONFIG_NAME = "config.json"


def get_config_path(base_dir: Path) -> Path:
    return base_dir / DEFAULT_CONFIG_NAME


def load_config(base_dir: Path) -> AppConfig:
    path = get_config_path(base_dir)
    if not path.exists():
        cfg = AppConfig()
        save_config(base_dir, cfg)
        return cfg
    data = json.loads(path.read_text(encoding="utf-8"))
    buttons = [ButtonConfig(**b) for b in data.get("buttons", [])]
    known = {f.name for f in AppConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known and k != "buttons"}
    cfg = AppConfig(**filtered)
    if buttons:
        cfg.buttons = buttons
    return cfg


def save_config(base_dir: Path, cfg: AppConfig) -> None:
    path = get_config_path(base_dir)
    payload = asdict(cfg)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
