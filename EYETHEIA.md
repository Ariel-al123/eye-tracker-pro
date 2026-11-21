# EyeTheia branch

This branch integrates the official EyeTheia iTracker model with the existing cursor, dwell actions, and calibration screen.

## Setup

Use Python 3.10 or 3.11. Do not install these pinned dependencies into Python 3.15: the pinned MediaPipe, OpenCV, and PyTorch versions do not provide a compatible set of wheels there. This repository already uses a Python 3.11 virtual environment at `.venv`.

```powershell
uv pip install --cache-dir .uv-cache --python .venv\Scripts\python.exe -r requirements-eyetheia.txt
```

Run the app with `.venv\Scripts\python.exe main.py` or double-click `run.bat`. EyeTheia is selected by default in `config.json`. Set `gaze_engine` to `mediapipe` to switch back to the original landmark mapper.

Press `C` and look at each of the nine targets. The tracker waits for a stable fixation and collects twelve valid frames per target. It fine-tunes only the final regression layers, then fits a small quadratic correction to the calibrated screen points. The cursor remains paused until both steps finish. If the face is not detected, the preview shows a framing or lighting hint; detection retries with local contrast enhancement and a second MediaPipe detector before giving up.

For the most reliable tracking, place the webcam close to the top center of the display, keep both eyes visible, sit far enough back to show the full face, and add light in front of you. The camera preview shows the detected face box so you can adjust the position before calibrating.

The pretrained MPIIFaceGaze checkpoint and image means are in `vendor/EyeTheia/src/`. No training dataset download is required.

## License and attribution

EyeTheia source and model assets are included from the official [EyeTheia repository](https://github.com/patherstevenson/EyeTheia), pinned to commit `13c3d444da4f7ef9e26a856bc9f46756ae818d87`. Its included license is GPL-3.0; review `vendor/EyeTheia/LICENSE` before redistributing this application.
