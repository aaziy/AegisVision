"""ByteTrack tracker wrapper.

Wraps Ultralytics' BYTETracker so it operates on our `Detection` list
(rather than Ultralytics' internal Boxes objects). Stateful: instantiate
once per video stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from .detectors.base import Detection


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class _DetectionAdapter:
    """Boxes-like view over a Detection list, matching what BYTETracker reads."""

    __slots__ = ("xyxy", "xywh", "conf", "cls")

    def __init__(self, detections: list[Detection] | None = None) -> None:
        if not detections:
            self.xyxy = np.zeros((0, 4), dtype=np.float32)
            self.xywh = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros((0,), dtype=np.float32)
            self.cls = np.zeros((0,), dtype=np.float32)
            return
        xyxy = np.array(
            [[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32
        )
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        self.xyxy = xyxy
        self.xywh = np.stack([cx, cy, w, h], axis=1)
        self.conf = np.array([d.confidence for d in detections], dtype=np.float32)
        self.cls = np.array([d.class_id for d in detections], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_DetectionAdapter":
        sub = _DetectionAdapter.__new__(_DetectionAdapter)
        sub.xyxy = self.xyxy[idx]
        sub.xywh = self.xywh[idx]
        sub.conf = self.conf[idx]
        sub.cls = self.cls[idx]
        return sub


class ByteTracker:
    """Stateful ByteTrack tracker over our Detection objects."""

    def __init__(
        self,
        fps: float = 30.0,
        track_high_thresh: float = 0.45,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.50,
        match_thresh: float = 0.80,
        track_buffer: int = 30,
    ) -> None:
        args = IterableSimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            fuse_score=False,
        )
        self._tracker = BYTETracker(args, frame_rate=int(round(fps)))
        self._class_names: dict[int, str] = {}

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        for d in detections:
            self._class_names.setdefault(d.class_id, d.class_name)

        tracks = self._tracker.update(_DetectionAdapter(detections))
        if len(tracks) == 0:
            return []

        # Ultralytics' STrack.result columns: [x1, y1, x2, y2, track_id, score, cls, idx]
        return [
            TrackedDetection(
                track_id=int(row[4]),
                x1=float(row[0]), y1=float(row[1]),
                x2=float(row[2]), y2=float(row[3]),
                confidence=float(row[5]),
                class_id=int(row[6]),
                class_name=self._class_names.get(int(row[6]), str(int(row[6]))),
            )
            for row in tracks
        ]
