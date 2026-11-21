# Eye Tracker Pro — versión definitiva (`fast`)

Sistema de control por mirada con webcam para navegar la computadora sin usar las manos. Esta rama conserva la **interfaz y el sistema de escritura** de `fast` (overlay Tk, teclado en pantalla, sugerencias, ajustes, información y modo juego) e integra el **mejor seguimiento y calibración** de `debug/screenpipe-eye-tracking` (EyeGestures de 25 puntos, zonas muertas/velocidad, marcador de mirada y motor EyeTheia opcional).

> **Nota honesta**: no reemplaza un producto comercial como Beam Eye Tracker. Es una base local, configurable y en desarrollo activo.

## Qué incluye

- **EyeGestures 3.2.4** por defecto, con calibración de **25 puntos** guardada en `eye_gestures_calibration.json` y reutilizada al iniciar.
- **Motor fallback `mediapipe`**: mapper propio con regresión ridge + One Euro Filter, calibración guardada en `calibration.json`.
- **Motor opcional `eyetheia`**: modelo iTracker profundo con personalización por calibración (requiere dependencias y modelo extra; ver [EYETHEIA.md](EYETHEIA.md)).
- **Zonas muertas y límite de velocidad** del cursor (`move_deadzone_px`, `cursor_max_speed_px_s`) para evitar saltos y temblores. `V` alterna modo precisión.
- **Marcador de mirada** flotante y click-through (`show_gaze_dot`), con anillo de progreso de dwell.
- **Dwell click** configurable y **pausa de seguridad** al perder el rostro.
- **UI completa de `fast`**: navegación, menú, teclado, sugerencias locales, información, modo juego, páginas de atajos editables y hotkeys globales en Windows.
- **Auto-scroll en bordes** (opcional, `G`) y acciones de navegador en el controlador.
- **Procesamiento local**: no envía video ni datos a internet.

EyeGestures se distribuye bajo GPL-3.0; revisa esa licencia antes de redistribuir el programa.

## Estructura

```text
eye-tracker-pro/
  main.py
  config.json
  requirements.txt / requirements-eyegestures.txt / requirements-eyetheia.txt
  run.bat / dependencias.bat
  models/face_landmarker.task
  eye_tracker_pro/
    app.py               # loop principal, motores y calibración
    config.py            # configuración (UI de fast + tracking de debug)
    controller.py        # mouse, dwell, zonas muertas, acciones
    detector.py          # MediaPipe Tasks FaceLandmarker
    eyegestures_engine.py
    eyetheia_engine.py   # opcional
    cursor_overlay.py    # marcador de mirada
    hotkeys.py           # hotkeys globales (fast)
    suggestions.py       # sugerencias del teclado (fast)
    mapper.py / filter.py / overlay.py
  vendor/EyeTheia/       # modelo y fuentes del motor opcional
```

## Requisitos

- Windows (objetivo principal; usa `cv2.CAP_DSHOW`).
- Python 3.10+ para la versión por defecto (3.11 recomendado si vas a usar EyeTheia).
- Webcam y buena iluminación frontal.

Instalación rápida:

```bash
pip install -r requirements.txt
```

O en Windows: doble clic en `run.bat` (usa `.venv` si existe y luego instala lo que falte).

Para EyeTheia, además:

```bash
pip install -r requirements-eyetheia.txt
```

## Ejecutar

```bash
python main.py
```

Si no hay calibración guardada válida, el programa inicia la calibración automáticamente.

## Controles

- `C`: iniciar/reiniciar calibración.
- `P`: pausar/reanudar control.
- `M`: activar/desactivar movimiento del cursor.
- `B`: activar/desactivar clics del ratón en los controles (depuración).
- `F`: activar/desactivar ajuste fino.
- `V`: alternar modo precisión (zona muerta menor y velocidad reducida).
- `G`: alternar auto-scroll en bordes.
- `X`: clic izquierdo inmediato en el punto donde está la mirada.
- `Z`: clic derecho inmediato en el punto donde está la mirada.
- `ESC`: cancelar calibración, salir del ajuste fino, o salir.
- Capa de controles: mira un botón hasta completar el dwell. El triángulo superior abre/cierra el menú.
- Los botones de Navegación son grandes y los grupos están pegados a los bordes: menú arriba-izquierda, precisión/scroll arriba-derecha, cruceta abajo-izquierda, atajos abajo-derecha y teclado/enter abajo-centro.
- El programa **siempre inicia en Navegación** para que el clic por mirada funcione desde el primer momento. El modo juego se activa con el botón `⇄` del menú.
- Teclado y sugerencias: accesibles desde la capa; las flechas cambian de página de atajos.
- Modo juego: mira las esquinas para mover la cámara; el menú permite volver a Navegación.

### Modo ajuste fino (`F` activo)

`A/D` offset X, `W/S` offset Y, `J/L` ganancia X, `I/K` ganancia Y, `R` reset, `F`/`ESC` salir.

## Calibración recomendada

1. Siéntate a 50–70 cm, con luz frontal y sin contraluz.
2. Presiona `C` y mira cada punto rojo (25 por defecto) hasta completar el círculo/barra.
3. Mantén la cabeza estable; espera el aviso entre puntos.
4. Si el cursor deriva, repite `C`. La calibración válida se guarda y se reutiliza.

## Configuración (`config.json`)

- `camera_use_max_resolution`: si está en `true` (por defecto), el programa consulta a la cámara y usa el mayor tamaño de frame que el driver acepta. Ponlo en `false` para usar `camera_width` / `camera_height` fijos.
- `gaze_engine`: `eyegestures` (por defecto), `mediapipe` o `eyetheia`.
- `calibration_points`: `5`, `9` o `25`.
- `move_deadzone_px` / `cursor_max_speed_px_s`: zona muerta y velocidad máxima.
- `dwell_time_ms` / `dwell_radius_px`: tiempo y tolerancia del dwell (por defecto `300` ms y `45` px para hacer clic fácil).
- `show_gaze_dot`: marcador flotante de mirada.
- `overlay_mode`: se fuerza a `navigation` al iniciar; el cambio a `game` es solo temporal durante la sesión.
- `keybind_pages`, `language`, `suggestion_history`: UI y escritura.
- `edge_scroll_*`: auto-scroll opcional en bordes.

## Limitaciones

- La precisión depende de cámara, luz, distancia y estabilidad de la cabeza.
- El mapper `mediapipe` es 2D; EyeTheia es un modelo profundo opcional y más pesado.
- Las sugerencias son locales y pequeñas, no predicen frases completas.
- Algunos juegos a pantalla completa pueden bloquear entradas sintéticas.

## RISE / accesibilidad

- **Inclusive**: acceso alternativo sin manos.
- **Empowering**: acciones y atajos personalizables.
- **Safe**: pausa al perder rostro, procesamiento local.
- **Responsible**: límites explícitos y calibración transparente.
