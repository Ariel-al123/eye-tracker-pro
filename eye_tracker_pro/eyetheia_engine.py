from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class EyeTheiaObservation:
    inputs: tuple
    left_eye_open: float
    right_eye_open: float


class EyeTheiaEngine:
    """EyeTheia iTracker model with this app's calibration and cursor UI."""

    def __init__(self, project_dir: Path, screen_w: int, screen_h: int):
        source_dir = project_dir / "vendor" / "EyeTheia" / "src"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"No se encuentra EyeTheia: {source_dir}")
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

        try:
            import torch
            if not hasattr(torch.nn, "CrossMapLRN2d"):
                # EyeTheia checkpoints use the older name for this standard layer.
                torch.nn.CrossMapLRN2d = torch.nn.LocalResponseNorm
            from tracker.GazeModel import GazeModel
            from utils.utils import (
                FACE_OVAL, LEFT_EYE, RIGHT_EYE, generate_face_grid,
                get_bounding_box, loadMetadata, preprocess_roi,
            )
            from tracker.CalibrationDataset import CalibrationDataset
        except ImportError as exc:
            raise RuntimeError(
                "Faltan dependencias de EyeTheia. Instala con: "
                "pip install -r requirements-eyetheia.txt"
            ) from exc

        self.torch = torch
        if not hasattr(torch.nn, "CrossMapLRN2d"):
            # EyeTheia checkpoints use the older name for this standard layer.
            torch.nn.CrossMapLRN2d = torch.nn.LocalResponseNorm
        self.screen_w, self.screen_h = screen_w, screen_h
        self._GazeModel = GazeModel
        self._face_oval = FACE_OVAL
        self._left_eye = LEFT_EYE
        self._right_eye = RIGHT_EYE
        self._get_bbox = get_bounding_box
        self._make_grid = generate_face_grid
        self._preprocess_roi = preprocess_roi
        self._CalibrationDataset = CalibrationDataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = project_dir / "vendor" / "EyeTheia" / "src" / "models"
        mean_dir = project_dir / "vendor" / "EyeTheia" / "src" / "mat"
        checkpoint_path = model_dir / "itracker_mpiiface.tar"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No se encuentra el modelo EyeTheia: {checkpoint_path}")

        self.face_mean = torch.as_tensor(
            loadMetadata(str(mean_dir / "mean_face_224_MPIIFace.mat"))["mean_face"] / 255.0,
            dtype=torch.float32, device=self.device,
        )
        self.left_mean = torch.as_tensor(
            loadMetadata(str(mean_dir / "mean_left_224_MPIIFace.mat"))["mean_eye_left"] / 255.0,
            dtype=torch.float32, device=self.device,
        )
        self.right_mean = torch.as_tensor(
            loadMetadata(str(mean_dir / "mean_right_224_MPIIFace.mat"))["mean_eye_right"] / 255.0,
            dtype=torch.float32, device=self.device,
        )

        self.model = GazeModel().to(self.device)
        loaded = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(loaded["state_dict"], strict=False)
        self.model.eval()
        self.base_weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.55, min_tracking_confidence=0.55,
        )
        self.samples: list[tuple[tuple, tuple[float, float]]] = []
        self.training = False
        self.training_done = False
        self.ready = False
        self.training_error: Exception | None = None
        self._training_thread: threading.Thread | None = None

    @staticmethod
    def _openness(landmarks, top: int, bottom: int, outer: int, inner: int) -> float:
        def pt(index: int) -> np.ndarray:
            p = landmarks[index]
            return np.array([p.x, p.y], dtype=np.float32)
        return float(np.linalg.norm(pt(top) - pt(bottom)) / max(np.linalg.norm(pt(outer) - pt(inner)), 1e-6))

    def observe(self, frame: np.ndarray) -> EyeTheiaObservation | None:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        face = result.multi_face_landmarks[0]
        lm = face.landmark
        left_open = self._openness(lm, 159, 145, 33, 133)
        right_open = self._openness(lm, 386, 374, 362, 263)
        points = [(int(p.x * w), int(p.y * h)) for p in lm]
        left_box = self._get_bbox(self._left_eye, points, w, h, x_margin=20, y_margin=20)
        right_box = self._get_bbox(self._right_eye, points, w, h, x_margin=20, y_margin=20)
        face_box = self._get_bbox(self._face_oval, points, w, h, x_margin=20, y_margin=20)
        boxes = (left_box, right_box, face_box)
        if any(x2 <= x1 or y2 <= y1 for x1, y1, x2, y2 in boxes):
            return None

        left_roi = self._preprocess_roi(frame[left_box[1]:left_box[3], left_box[0]:left_box[2]])[0]
        right_roi = self._preprocess_roi(frame[right_box[1]:right_box[3], right_box[0]:right_box[2]])[0]
        face_roi = self._preprocess_roi(frame[face_box[1]:face_box[3], face_box[0]:face_box[2]])[0]
        torch = self.torch

        def image_tensor(roi: np.ndarray, mean) -> object:
            return torch.as_tensor(roi, dtype=torch.float32, device=self.device).permute(2, 0, 1).sub(mean).unsqueeze(0)

        inputs = (
            image_tensor(face_roi, self.face_mean),
            image_tensor(left_roi, self.left_mean),
            image_tensor(right_roi, self.right_mean),
            torch.as_tensor(self._make_grid(face_box, frame.shape), dtype=torch.float32,
                            device=self.device).view(1, -1),
        )
        return EyeTheiaObservation(inputs, left_open, right_open)

    def predict(self, observation: EyeTheiaObservation) -> tuple[float, float] | None:
        if self.training or observation is None:
            return None
        with self.torch.inference_mode():
            pred = self.model(*observation.inputs).detach().cpu().numpy().reshape(-1)
        if pred.size != 2 or not np.isfinite(pred).all():
            return None
        return float(pred[0] * self.screen_w), float(pred[1] * self.screen_h)

    def begin_calibration(self) -> None:
        self.samples.clear()
        self.training_done = False
        self.training_error = None
        self.ready = False
        self.model.load_state_dict(self.base_weights)
        self.model.eval()

    def add_calibration_sample(self, observation: EyeTheiaObservation, x: int, y: int) -> None:
        if observation is None:
            return
        # Keep calibration tensors on CPU; move each batch to the model device
        # during adaptation so webcam samples do not consume VRAM.
        stored = tuple(t.detach().cpu() for t in observation.inputs)
        target = (x / max(self.screen_w, 1), y / max(self.screen_h, 1))
        self.samples.append((stored, target))

    def train_async(self, epochs: int = 3) -> None:
        if len(self.samples) < 9:
            raise RuntimeError("EyeTheia necesita muestras de todos los puntos para personalizarse.")
        self.training = True
        self.training_done = False
        self.training_error = None

        def worker() -> None:
            try:
                torch = self.torch
                data = self._CalibrationDataset(self.samples)
                loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
                for parameter in self.model.parameters():
                    parameter.requires_grad_(False)
                for module in (self.model.eyesFC, self.model.fc):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
                self.model.train()
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(trainable, lr=1e-4)
                loss_fn = torch.nn.MSELoss()
                for _ in range(epochs):
                    for face, left, right, grid, target in loader:
                        face = face.to(self.device)
                        left = left.to(self.device)
                        right = right.to(self.device)
                        grid = grid.to(self.device)
                        target = target.to(self.device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_fn(self.model(face, left, right, grid), target)
                        loss.backward()
                        optimizer.step()
                self.model.eval()
            except Exception as exc:
                self.training_error = exc
                self.model.load_state_dict(self.base_weights)
                self.model.eval()
            finally:
                self.training = False
                self.training_done = True

        self._training_thread = threading.Thread(target=worker, name="EyeTheia-finetune", daemon=True)
        self._training_thread.start()

    def close(self) -> None:
        self.face_mesh.close()
