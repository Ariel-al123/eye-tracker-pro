# EyeTheia branch

This branch integrates the official EyeTheia iTracker model with the existing cursor, dwell actions, and calibration screen.

## Setup

Use Python 3.10 or 3.11, then install:

```powershell
pip install -r requirements-eyetheia.txt
```

Run the app with `python main.py`. EyeTheia is selected by default in `config.json`. Set `gaze_engine` to `mediapipe` to switch back to the original landmark mapper.

Press `C` and look at each of the nine targets. EyeTheia collects six valid camera frames per target and fine-tunes only the final regression layers so the pretrained visual feature extractors remain stable with a small personal calibration set. On CPU, this adaptation may take a little while; the cursor remains paused until it finishes.

The pretrained MPIIFaceGaze checkpoint and image means are in `vendor/EyeTheia/src/`. No training dataset download is required.

## License and attribution

EyeTheia source and model assets are included from the official [EyeTheia repository](https://github.com/patherstevenson/EyeTheia), pinned to commit `13c3d444da4f7ef9e26a856bc9f46756ae818d87`. Its included license is GPL-3.0; review `vendor/EyeTheia/LICENSE` before redistributing this application.
