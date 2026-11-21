# Eye Tracker Pro

Prototipo robusto de control por mirada para navegación básica de computadora, pensado para personas con movilidad limitada en manos/brazos. Usa webcam, MediaPipe Face Landmarker, calibración por puntos, suavizado One Euro Filter, dwell click y una capa de controles transparente.

> **Nota honesta**: esto no reemplaza un producto comercial pulido como Beam Eye Tracker. Es una base mucho más limpia y robusta que los prototipos anteriores, diseñada para iterar rápido.

## Qué incluye

- **Seguimiento de mirada con EyeGestures 3.2.4**, como en la configuración de tu amigo, con el mapper MediaPipe original disponible como fallback.
- **Calibración de 5, 9 o 25 puntos** con muestreo temporizado, filtro de fijación estable y mediana para reducir errores.
- La calibración válida se guarda en `calibration.json` y se reutiliza al iniciar. `calibration_points` en `config.json` acepta `5`, `9` o `25`.
- **Mapeo mirada -> pantalla** por regresión polinomial regularizada (ridge).
- **Suavizado** con One Euro Filter para menos jitter y mejor respuesta.
- **Dwell click** configurable (tiempo y radio).
- **Capa de controles transparente y siempre encima**. Los controles negros se muestran sobre el escritorio o juego sin taparlo por completo.
- Diseños de navegación, menú, teclado, ajustes, información y modo juego activados por mirada.
- Atajos organizados en páginas, con acciones de toque o alternancia, combinaciones editables y botones del ratón.
- Teclas de control detectadas globalmente en Windows para que funcionen mientras una aplicación mantiene el foco.
- Teclado en pantalla con sugerencias locales, teclas F1–F12 y compatibilidad con inglés, español, chino y francés en la interfaz.
- Controles de cámara para juegos: mirar una esquina mueve el ratón relativo en esa dirección.
- **Pausa de seguridad** cuando se pierde el rostro.
- **Procesamiento local**: no envía video ni datos a internet.

EyeGestures se instala como dependencia ([`eyeGestures==3.2.4`](https://pypi.org/project/eyeGestures/3.2.4/)) y se distribuye bajo GPL-3.0; revisa esa licencia antes de redistribuir el programa.

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
    overlay.py      # capa transparente con los controles por mirada
```

## Requisitos

- Windows (probado como objetivo principal; usa `cv2.CAP_DSHOW`).
- Python 3.10+ recomendado.
- Webcam.

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutar

```bash
python main.py
```

O en Windows: doble clic en `run.bat`.

## Controles

- `C`: iniciar/reiniciar calibración.
- `P`: pausar/reanudar control.
- `M`: activar/desactivar movimiento del cursor.
- `B`: activar/desactivar clics del ratón en los controles (modo de depuración, desactivado por defecto; también se cambia desde Ajustes).
- `F`: activar/desactivar modo de ajuste fino.
- `ESC`: cancelar calibración, salir del ajuste fino, o salir.
- Capa de controles: mira un botón hasta completar el dwell para activar la acción. Para volver a activar el mismo botón, aparta la mirada y vuelve a mirarlo.
- Triángulo superior izquierdo: abre/cierra el menú.
- Teclado y accesos rápidos: están en la parte inferior; las flechas junto a los accesos cambian de página.
- Ajustes: cambia las teclas, añade o quita teclas de combinaciones, elige toque o alternancia, cambia el idioma y añade páginas.
- Ajustes e Información son pantallas modales: oscurecen el fondo y desactivan los controles de navegación hasta volver a Navegación. Cerrar el menú también cierra la pantalla modal.
- Modo juego: mira las esquinas para mover la cámara. La opción del menú cambia a Navegación para volver.
- Si no hay una calibración guardada válida, se inicia automáticamente al arrancar. Pulsa `C` para recalibrar cuando quieras.
- EyeGestures guarda su calibración de 25 puntos en `eye_gestures_calibration.json`. Cambia `gaze_engine` a `mediapipe` en `config.json` para usar el mapper original.

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
- `keybind_pages`: páginas de accesos directos, creadas y editadas desde Ajustes.

- `fine_gain_x`, `fine_gain_y`, `fine_offset_x`, `fine_offset_y`: ajuste fino manual.
- `fine_step_px`, `fine_gain_step`: tamaño de paso del ajuste fino.

## Limitaciones actuales

- La precisión depende mucho de cámara, luz, distancia y estabilidad de cabeza.
- El mapeo actual es 2D con compensación simple de cabeza; no es un modelo 3D completo como el de `MonitorTracking.py`.
- Las sugerencias usan listas locales pequeñas y no predicen frases por contexto.
- Algunos juegos en pantalla completa pueden bloquear entradas de teclado o ratón sintéticas según su modo de entrada y permisos.
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
