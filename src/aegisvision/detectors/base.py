"""Detector interface and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# COCO class IDs we treat as vehicles for traffic monitoring.
# 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck (skipping 6=train).
VEHICLE_CLASS_IDS: frozenset[int] = frozenset({1, 2, 3, 5, 7})


@dataclass(frozen=True, slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Detector(Protocol):
    name: str
    device: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...
