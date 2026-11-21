# Eye Tracker Pro

> **Branch `feature-eyetheia-gaze`:** esta version usa el modelo profundo iTracker de EyeTheia y ajuste personalizado con los nueve puntos. Antes de ejecutar, instala `requirements-eyetheia.txt` y sigue la guia [EYETHEIA.md](EYETHEIA.md). El motor anterior sigue disponible con `gaze_engine: "mediapipe"` en `config.json`.

El motor seleccionado por defecto en esta rama es EyeGestures 3.2.4. Instálalo en el entorno Python 3.11 del proyecto con:

```powershell
uv pip install --cache-dir .uv-cache --python .venv\Scripts\python.exe -r requirements-eyegestures.txt
```

En `config.json` puedes cambiar `gaze_engine` entre `eyegestures`, `eyetheia` y `mediapipe`. EyeGestures usa su calibración interna: pulsa `C`, sigue los puntos que presenta y pulsa `C` otra vez para terminar. Para EyeTheia instala además `requirements-eyetheia.txt`.

EyeGestures 3.2.4 se usa porque admite Python 3.11; la versión 4.x requiere Python 3.13 según sus metadatos de PyPI y cambió la API. EyeGestures está bajo GPLv3. Revisa las obligaciones de esa licencia antes de distribuir una aplicación que lo incluya.

Prototipo robusto de control por mirada para navegación básica de computadora, pensado para personas con movilidad limitada en manos/brazos. Usa webcam, MediaPipe Face Landmarker, calibración por puntos, suavizado One Euro Filter, dwell click y una barra de botones configurables.

> **Nota honesta**: esto no reemplaza un producto comercial pulido como Beam Eye Tracker. Es una base mucho más limpia y robusta que los prototipos anteriores, diseñada para iterar rápido.

## Qué incluye

- **Seguimiento facial/iris** con MediaPipe Tasks (`face_landmarker.task`).
- **Calibración 5 o 9 puntos** con muestreo y mediana para reducir outliers.
- **Mapeo mirada -> pantalla** por regresión polinomial regularizada (ridge).
- **Suavizado** con One Euro Filter para menos jitter y mejor respuesta.
- **Dwell click** configurable (tiempo y radio).
- **Barra de botones personalizables** siempre visible (`config.json`): clic, doble clic, scroll, pausa, calibrar.
- **Pausa de seguridad** cuando se pierde el rostro.
- **Procesamiento local**: no envía video ni datos a internet.

## Estructura

```text
eye-tracker-pro/
  main.py
  config.json
  requirements.txt
  run.bat
  models/
    face_landmarker.task
  eye_tracker_pro/
    app.py          # loop principal y calibración
    camera.py       # (reservado/extendible)
    config.py       # carga/guarda configuración
    controller.py   # mouse, dwell, acciones
    detector.py     # MediaPipe + features de mirada
    filter.py       # One Euro Filter
    mapper.py       # calibración y regresión a pantalla
    overlay.py      # barra de botones configurable
```

## Requisitos

- Windows (probado como objetivo principal; usa `cv2.CAP_DSHOW`).
- Python 3.10+ recomendado.
- Webcam.

Instalar dependencias en el entorno del proyecto (Python 3.11 recomendado):

```powershell
uv venv --python 3.11 .venv
uv pip install --cache-dir .uv-cache --python .venv\Scripts\python.exe -r requirements.txt
```

En la rama `feature-eyetheia-gaze`, instala además las dependencias del motor profundo en ese mismo entorno con `uv pip install --cache-dir .uv-cache --python .venv\Scripts\python.exe -r requirements-eyetheia.txt`. No uses el `pip` global de Python 3.15 para estos pins. Consulta [EYETHEIA.md](EYETHEIA.md) para configuración, calibración y licencia.

## Ejecutar

```bash
.venv\Scripts\python.exe main.py
```

O en Windows: doble clic en `run.bat`.

## Controles

- `C`: iniciar/reiniciar calibración.
- `P`: pausar/reanudar control.
- `M`: activar/desactivar movimiento del cursor.
- `F`: activar/desactivar modo de ajuste fino.
- `ESC`: cancelar calibración, salir del ajuste fino, o salir.
- Barra de botones: mira un botón hasta completar el dwell para activar la acción.

### Modo ajuste fino (cuando `F` está activo)

- `A/D`: offset horizontal (`fine_offset_x`) -/+
- `W/S`: offset vertical (`fine_offset_y`) -/+
- `J/L`: ganancia horizontal (`fine_gain_x`) -/+
- `I/K`: ganancia vertical (`fine_gain_y`) +/-
- `R`: resetear ajuste fino a valores por defecto
- `F` o `ESC`: salir del modo ajuste fino

## Calibración (recomendado)

1. Siéntate a una distancia cómoda y estable (50–70 cm).
2. Buena iluminación frontal, sin luz fuerte detrás.
3. Presiona `C`.
4. Mira cada punto rojo hasta completar la barra. Mantén la cabeza lo más estable posible.
5. Si el cursor deriva, recalibra con `C`.

## Configuración

Edita `config.json`:

- `calibration_points`: `5` o `9`.
- `dwell_time_ms`: tiempo para activar dwell click.
- `dwell_radius_px`: tolerancia de movimiento durante dwell.
- `filter_min_cutoff`, `filter_beta`: suavizado (menos jitter = min_cutoff más bajo; más latencia).
- `buttons`: lista de botones personalizables.

- `fine_gain_x`, `fine_gain_y`, `fine_offset_x`, `fine_offset_y`: ajuste fino manual.
- `fine_step_px`, `fine_gain_step`: tamaño de paso del ajuste fino.

Acciones de botón soportadas:
`left_click`, `right_click`, `double_click`, `scroll_up`, `scroll_down`, `drag_toggle`, `pause`, `calibrate`, `fine_tune`.

Ejemplo de botón:

```json
{ "id": "btn_custom", "label": "Mi acción", "action": "left_click", "x": 20, "y": 100, "w": 140, "h": 64, "enabled": true }
```

## Limitaciones actuales

- La precisión depende mucho de cámara, luz, distancia y estabilidad de cabeza.
- El mapeo actual es 2D con compensación simple de cabeza; no es un modelo 3D completo como el de `MonitorTracking.py`.
- La detección de dwell en botones usa la posición conocida de la ventana overlay; si mueves la ventana, puede requerir ajuste.
- No incluye teclado en pantalla completo todavía (se puede integrar).
- PyAutoGUI controla el mouse del sistema; usa `P` para pausar en cualquier momento.

## Próximos pasos recomendados

- Teclado en pantalla accesible.
- Perfiles por usuario (calibración guardada).
- Corrección de deriva por recalibración rápida de 1 punto.
- Modelo 3D/híbrido para mayor tolerancia a movimiento de cabeza.
- Pruebas con usuarios reales y métricas de precisión (error en px, tiempo de selección).
- Modo “zonas seguras” y confirmación para acciones críticas.

## RISE / accesibilidad

Este proyecto se puede enmarcar en RISE como:
- **Inclusive**: acceso alternativo a la computadora sin manos.
- **Empowering**: personalización de acciones y mayor agencia del usuario.
- **Safe**: pausa ante pérdida de rostro, procesamiento local, acciones configurables.
- **Responsible**: transparencia de límites, calibración explícita, sin captura/envío de datos sensibles por defecto.
