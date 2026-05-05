"""Line-delimited JSON event logger for the pipeline.

Writes one JSON object per line to a `.jsonl` file. Events:

  - start      : pipeline start, source / model / backend / device.
  - line_cross : a track crossed the counting line.
  - summary    : periodic snapshot of rolling FPS + cumulative counts.
  - end        : final summary at shutdown (always written).

Use as a context manager so close() flushes and writes the end event.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLogger:
    """Append-only JSONL writer with periodic flush."""

    def __init__(self, path: Path | str, flush_every: int = 20) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._since_flush = 0
        self._flush_every = flush_every

    def emit(self, **fields: Any) -> None:
        event = {"ts": _iso_now(), **fields}
        self._fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self._fh.flush()
            self._since_flush = 0

    def start(
        self, source: str, model: str, backend: str, device: str,
        resolution: tuple[int, int], fps: float,
    ) -> None:
        self.emit(
            event="start", source=source, model=model, backend=backend,
            device=device, width=resolution[0], height=resolution[1], src_fps=round(fps, 2),
        )
        self._fh.flush()

    def line_cross(
        self, frame: int, track_id: int, class_name: str, direction: str,
        confidence: float, point: tuple[float, float],
    ) -> None:
        self.emit(
            event="line_cross", frame=frame, track_id=track_id,
            **{"class": class_name},
            direction=direction, confidence=round(confidence, 4),
            x=round(point[0], 1), y=round(point[1], 1),
        )

    def summary(
        self, frame: int, fps: float, counts: dict[str, dict[str, int]],
        total_in: int, total_out: int,
    ) -> None:
        self.emit(
            event="summary", frame=frame, fps=round(fps, 2),
            total_in=total_in, total_out=total_out,
            counts={k: dict(v) for k, v in counts.items()},
        )
        self._fh.flush()

    def end(
        self, frame: int, elapsed_s: float, avg_fps: float,
        counts: dict[str, dict[str, int]], total_in: int, total_out: int,
    ) -> None:
        self.emit(
            event="end", frame=frame, elapsed_s=round(elapsed_s, 2),
            avg_fps=round(avg_fps, 2),
            total_in=total_in, total_out=total_out,
            counts={k: dict(v) for k, v in counts.items()},
        )
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
