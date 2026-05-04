"""ONNX Runtime detector with configurable execution provider.

Supports the same YOLO and RT-DETR weights as the PyTorch detector, but via
ONNX Runtime. The same .onnx file works on Mac (CoreML EP) and inside Linux
containers (CPU EP); only the provider config differs.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort

from .base import Detection

ProviderName = Literal["coreml", "cpu"]
ComputeUnits = Literal["CPUOnly", "CPUAndGPU", "CPUAndNeuralEngine", "ALL"]
_VALID_COMPUTE_UNITS: tuple[str, ...] = ("CPUOnly", "CPUAndGPU", "CPUAndNeuralEngine", "ALL")

_ALIASES: dict[str, str] = {
    "yolo": "yolo26n.onnx",
    "yolo26n": "yolo26n.onnx",
    "rtdetr": "rtdetr-l.onnx",
    "rtdetr-l": "rtdetr-l.onnx",
}


def _build_session(
    weights_path: Path, provider: ProviderName, compute_units: ComputeUnits
) -> ort.InferenceSession:
    if provider == "coreml":
        providers = [
            ("CoreMLExecutionProvider", {"MLComputeUnits": compute_units}),
            "CPUExecutionProvider",
        ]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"unknown provider: {provider!r}")
    return ort.InferenceSession(str(weights_path), providers=providers)


def _letterbox(
    img: np.ndarray, size: int = 640, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def _preprocess(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    img, r, pad = _letterbox(frame, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]  # NCHW
    return img, r, pad


def _undo_letterbox(
    boxes: np.ndarray, r: float, pad: tuple[int, int], orig_shape: tuple[int, ...]
) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= r
    h, w = orig_shape[:2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    return boxes


def _decode_yolo(
    out: np.ndarray,
    orig_shape: tuple[int, ...],
    r: float,
    pad: tuple[int, int],
    conf: float,
    classes: set[int] | None,
    names: dict[int, str],
) -> list[Detection]:
    """YOLO26 output: (1, 300, 6) = [x1,y1,x2,y2,conf,cls], pixel coords on letterboxed 640."""
    rows = out[0]
    mask = rows[:, 4] >= conf
    if classes is not None:
        mask &= np.isin(rows[:, 5].astype(int), list(classes))
    rows = rows[mask]
    if rows.size == 0:
        return []
    boxes = _undo_letterbox(rows[:, :4], r, pad, orig_shape)
    return [
        Detection(
            x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
            confidence=float(c), class_id=int(k),
            class_name=names.get(int(k), str(int(k))),
        )
        for b, c, k in zip(boxes, rows[:, 4], rows[:, 5])
    ]


def _decode_rtdetr(
    out: np.ndarray,
    orig_shape: tuple[int, ...],
    r: float,
    pad: tuple[int, int],
    conf: float,
    classes: set[int] | None,
    names: dict[int, str],
    imgsz: int,
) -> list[Detection]:
    """RT-DETR output: (1, 300, 84) = [cx,cy,w,h] (normalized 0-1) + 80 class scores."""
    rows = out[0]
    cls_scores = rows[:, 4:]
    cls_id = cls_scores.argmax(axis=1)
    cls_conf = cls_scores.max(axis=1)
    mask = cls_conf >= conf
    if classes is not None:
        mask &= np.isin(cls_id, list(classes))
    if not mask.any():
        return []
    rows = rows[mask]
    cls_id = cls_id[mask]
    cls_conf = cls_conf[mask]
    cx = rows[:, 0] * imgsz
    cy = rows[:, 1] * imgsz
    w = rows[:, 2] * imgsz
    h = rows[:, 3] * imgsz
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes = _undo_letterbox(boxes, r, pad, orig_shape)
    return [
        Detection(
            x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
            confidence=float(c), class_id=int(k),
            class_name=names.get(int(k), str(int(k))),
        )
        for b, c, k in zip(boxes, cls_conf, cls_id)
    ]


def _load_class_names(session: ort.InferenceSession) -> dict[int, str]:
    meta = session.get_modelmeta().custom_metadata_map
    if "names" in meta:
        d = ast.literal_eval(meta["names"])
        return {int(k): v for k, v in d.items()}
    return {}


class OnnxDetector:
    """ONNX Runtime detector with selectable execution provider."""

    def __init__(
        self,
        model: str = "yolo",
        weights_dir: Path | str = "models",
        provider: ProviderName = "coreml",
        compute_units: ComputeUnits = "ALL",
        conf: float = 0.25,
        classes: list[int] | None = None,
        imgsz: int = 640,
    ) -> None:
        if compute_units not in _VALID_COMPUTE_UNITS:
            raise ValueError(
                f"compute_units must be one of {_VALID_COMPUTE_UNITS}, got {compute_units!r}"
            )

        weights_name = _ALIASES.get(model, model if model.endswith(".onnx") else f"{model}.onnx")
        weights_path = Path(weights_dir) / weights_name
        if not weights_path.exists():
            raise FileNotFoundError(
                f"ONNX weights not found at {weights_path}. "
                f"Run `uv run python scripts/export_onnx.py` first."
            )

        self._session = _build_session(weights_path, provider, compute_units)
        self._input_name = self._session.get_inputs()[0].name
        self._class_names = _load_class_names(self._session)

        self.name = weights_path.stem
        self.provider = provider
        self.compute_units = compute_units if provider == "coreml" else None
        suffix = f"({compute_units})" if provider == "coreml" else ""
        self.device = f"onnx:{provider}{suffix}"

        self._conf = conf
        self._classes = set(classes) if classes else None
        self._imgsz = imgsz
        self._is_rtdetr = "rtdetr" in self.name.lower()

        # Warm-up. Surfaces compile/lazy-init cost out of the first frame's FPS.
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self._session.run(None, {self._input_name: dummy})

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob, r, pad = _preprocess(frame, self._imgsz)
        out = self._session.run(None, {self._input_name: blob})[0]
        if self._is_rtdetr:
            return _decode_rtdetr(
                out, frame.shape, r, pad, self._conf, self._classes,
                self._class_names, self._imgsz,
            )
        return _decode_yolo(
            out, frame.shape, r, pad, self._conf, self._classes, self._class_names,
        )
