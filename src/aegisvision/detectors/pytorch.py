"""PyTorch-backed detector via Ultralytics (YOLO / RT-DETR)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from ultralytics import RTDETR, YOLO

from .base import Detection

# Friendly aliases mapping to (weights filename, Ultralytics class).
_REGISTRY: dict[str, tuple[str, type]] = {
    "yolo": ("yolo26n.pt", YOLO),
    "yolo26n": ("yolo26n.pt", YOLO),
    "yolo11n": ("yolo11n.pt", YOLO),
    "rtdetr": ("rtdetr-l.pt", RTDETR),
    "rtdetr-l": ("rtdetr-l.pt", RTDETR),
    "rtdetr-x": ("rtdetr-x.pt", RTDETR),
}


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_model(name: str) -> tuple[str, type]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    # Allow passing a raw weights filename, infer class from prefix.
    cls = RTDETR if name.startswith("rtdetr") else YOLO
    return name if name.endswith(".pt") else f"{name}.pt", cls


def _load(weights_path: Path, model_cls: type):
    weights_path = weights_path.resolve()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        return model_cls(str(weights_path))
    # Ultralytics downloads relative to cwd; cd into the weights dir so the
    # file lands in models/ rather than the project root.
    cwd = Path.cwd()
    try:
        os.chdir(weights_path.parent)
        return model_cls(weights_path.name)
    finally:
        os.chdir(cwd)


class PyTorchDetector:
    """Ultralytics detector running via PyTorch (MPS on Apple Silicon)."""

    def __init__(
        self,
        model: str = "yolo",
        weights_dir: Path | str = "models",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int | None = None,
        device: str | None = None,
        classes: list[int] | None = None,
    ) -> None:
        weights_name, model_cls = _resolve_model(model)
        weights_path = Path(weights_dir) / weights_name
        self._model = _load(weights_path, model_cls)

        self.name = weights_path.stem
        self.device = device or _pick_device()
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._classes = classes
        self._class_names: dict[int, str] = self._model.names

        # Warm-up run to amortize lazy init cost out of the first frame's FPS.
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._predict(dummy)

    def _predict(self, frame: np.ndarray):
        kwargs: dict = {
            "source": frame,
            "device": self.device,
            "conf": self._conf,
            "iou": self._iou,
            "classes": self._classes,
            "verbose": False,
        }
        if self._imgsz is not None:
            kwargs["imgsz"] = self._imgsz
        return self._model.predict(**kwargs)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._predict(frame)
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(
                x1=float(b[0]),
                y1=float(b[1]),
                x2=float(b[2]),
                y2=float(b[3]),
                confidence=float(c),
                class_id=int(k),
                class_name=self._class_names.get(int(k), str(k)),
            )
            for b, c, k in zip(xyxy, confs, cls_ids)
        ]
