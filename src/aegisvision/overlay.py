"""OpenCV HUD: bbox labels, counting line, FPS + counts overlay."""

from __future__ import annotations

from typing import Union

import cv2
import numpy as np

from .counter import LineCrossingCounter, LineSegment
from .detectors.base import Detection
from .tracker import TrackedDetection

#: Any object with bbox fields + class_id; accepts Detection and TrackedDetection.
AnyDetection = Union[Detection, TrackedDetection]

# BGR colors keyed by COCO class id.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 255, 255),    # bicycle    - yellow
    2: (0, 255, 0),      # car        - green
    3: (0, 128, 255),    # motorcycle - orange
    5: (255, 0, 255),    # bus        - magenta
    7: (0, 165, 255),    # truck      - amber
}
DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)
LINE_COLOR: tuple[int, int, int] = (0, 200, 255)


def draw_box(frame: np.ndarray, d: AnyDetection) -> None:
    """Draw a bbox + label."""
    color = CLASS_COLORS.get(d.class_id, DEFAULT_COLOR)
    x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = (
        f"#{d.track_id} {d.class_name} {d.confidence:.2f}"
        if hasattr(d, "track_id")
        else f"{d.class_name} {d.confidence:.2f}"
    )
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - bl - 2), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)


def draw_line(frame: np.ndarray, line: LineSegment) -> None:
    p1 = (int(line.p1[0]), int(line.p1[1]))
    p2 = (int(line.p2[0]), int(line.p2[1]))
    cv2.line(frame, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)


def outlined_text(
    frame: np.ndarray, text: str, org: tuple[int, int],
    scale: float = 0.7, color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray, fps: float, label: str, counter: LineCrossingCounter | None,
) -> None:
    outlined_text(frame, f"{label}  FPS: {fps:5.1f}", (12, 32), 0.9)
    if counter is None:
        return
    totals = (
        f"{counter.in_label}: {counter.total_in}  "
        f"{counter.out_label}: {counter.total_out}  "
        f"total: {counter.total}"
    )
    outlined_text(frame, totals, (12, 64), 0.7)
    if counter.counts:
        parts = []
        for cls in sorted(counter.counts.keys()):
            d = counter.counts[cls]
            parts.append(f"{cls}:{d[counter.in_label]}/{d[counter.out_label]}")
        outlined_text(frame, "  ".join(parts), (12, 92), 0.55, color=(220, 220, 0))
