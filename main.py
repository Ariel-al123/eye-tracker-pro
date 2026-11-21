from pathlib import Path

from eye_tracker_pro.app import EyeTrackerApp


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    app = EyeTrackerApp(base_dir)
    app.run()
