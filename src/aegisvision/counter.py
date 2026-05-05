"""Line-crossing counter for tracked detections.

A tracked object is counted exactly once when its centroid crosses a line
segment between two consecutive frames. Direction is inferred from which
side of the line the centroid was on before the crossing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .tracker import TrackedDetection


@dataclass(frozen=True, slots=True)
class LineSegment:
    p1: tuple[float, float]
    p2: tuple[float, float]

    def side(self, point: tuple[float, float]) -> float:
        """Signed cross product. +ve on one side, -ve on the other, 0 on the line."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        x, y = point
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    track_id: int
    class_id: int
    class_name: str
    direction: str
    point: tuple[float, float]
    frame: int
    confidence: float


class LineCrossingCounter:
    """Counts tracks crossing a line segment, with direction and per-class tallies."""

    def __init__(
        self,
        line: LineSegment,
        in_label: str = "in",
        out_label: str = "out",
    ) -> None:
        self.line = line
        self.in_label = in_label
        self.out_label = out_label
        self._prev_centroid: dict[int, tuple[float, float]] = {}
        self._counted: set[int] = set()
        self._frame_idx = 0
        # counts[class_name][direction] -> int
        self.counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {self.in_label: 0, self.out_label: 0}
        )

    def update(self, tracks: Iterable[TrackedDetection]) -> list[CrossingEvent]:
        self._frame_idx += 1
        events: list[CrossingEvent] = []
        active_ids: set[int] = set()

        for t in tracks:
            active_ids.add(t.track_id)
            curr = t.centroid
            prev = self._prev_centroid.get(t.track_id)
            self._prev_centroid[t.track_id] = curr

            if prev is None or t.track_id in self._counted:
                continue

            curr_side = self.line.side(curr)
            prev_side = self.line.side(prev)
            if curr_side * prev_side >= 0:  # same side or grazing the line
                continue

            direction = self.in_label if prev_side > 0 else self.out_label
            events.append(CrossingEvent(
                track_id=t.track_id,
                class_id=t.class_id,
                class_name=t.class_name,
                direction=direction,
                point=curr,
                frame=self._frame_idx,
                confidence=t.confidence,
            ))
            self.counts[t.class_name][direction] += 1
            self._counted.add(t.track_id)

        # Periodically prune centroids of tracks we haven't seen in a while.
        if len(self._prev_centroid) > len(active_ids) + 200:
            for tid in list(self._prev_centroid):
                if tid not in active_ids:
                    self._prev_centroid.pop(tid, None)

        return events

    @property
    def total_in(self) -> int:
        return sum(d[self.in_label] for d in self.counts.values())

    @property
    def total_out(self) -> int:
        return sum(d[self.out_label] for d in self.counts.values())

    @property
    def total(self) -> int:
        return self.total_in + self.total_out

    def summary(self) -> str:
        if not self.counts:
            return f"{self.in_label}: 0  {self.out_label}: 0  total: 0"
        parts = []
        for cls, d in sorted(self.counts.items()):
            parts.append(f"{cls}({d[self.in_label]}/{d[self.out_label]})")
        return f"{self.in_label}: {self.total_in}  {self.out_label}: {self.total_out}  | " + " ".join(parts)


def load_config(path: Path | str) -> tuple[LineSegment, str, str]:
    """Load a counting-line YAML config; returns (line, in_label, out_label)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cl = cfg["counting_line"]
    line = LineSegment(p1=tuple(cl["p1"]), p2=tuple(cl["p2"]))
    return line, cl.get("in_label", "in"), cl.get("out_label", "out")
