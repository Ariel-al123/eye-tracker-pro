"""Install only project dependencies whose Python modules are unavailable."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


# Distribution names in requirements.txt and their importable module names.
REQUIREMENT_IMPORTS = {
    "opencv-python": "cv2",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "pyautogui": "pyautogui",
    "eyegestures": "eyeGestures",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
}


def main() -> int:
    requirements = Path(__file__).with_name("requirements.txt")
    missing: list[str] = []

    for line in requirements.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        distribution = requirement.split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].strip()
        module = REQUIREMENT_IMPORTS.get(distribution.lower().replace("_", "-"), distribution)
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            installed = False
        if not installed:
            missing.append(requirement)

    if not missing:
        print("All dependencies are already installed.")
        return 0

    print("Installing missing dependencies: " + ", ".join(missing))
    return subprocess.call([sys.executable, "-m", "pip", "install", *missing])


if __name__ == "__main__":
    raise SystemExit(main())
