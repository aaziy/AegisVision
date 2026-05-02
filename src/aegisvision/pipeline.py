"""Live detection pipeline: capture → detect → render → log."""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from aegisvision.detectors.base import Detection, VEHICLE_CLASS_IDS
from aegisvision.detectors.pytorch import PyTorchDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "samples" / "traffic_sample.mp4"

# BGR colors keyed by COCO class id.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 255, 255),    # bicycle    - yellow
    2: (0, 255, 0),      # car        - green
    3: (0, 128, 255),    # motorcycle - orange
    5: (255, 0, 255),    # bus        - magenta
    7: (0, 165, 255),    # truck      - amber
}
DEFAULT_COLOR = (200, 200, 200)


def _draw_box(frame: np.ndarray, d: Detection) -> None:
    color = CLASS_COLORS.get(d.class_id, DEFAULT_COLOR)
    x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{d.class_name} {d.confidence:.2f}"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - bl - 2), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)


def _draw_fps(frame: np.ndarray, fps: float, label: str) -> None:
    text = f"{label}  FPS: {fps:5.1f}"
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2, cv2.LINE_AA)


def run(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        print(f"source not found: {source}", file=sys.stderr)
        return 2

    classes = None if args.all_classes else sorted(VEHICLE_CLASS_IDS)
    detector = PyTorchDetector(
        model=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=classes,
    )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"failed to open video: {source}", file=sys.stderr)
        return 2

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_w, target_h = src_w, src_h
    if args.resize:
        target_h = args.resize
        target_w = int(round(src_w * target_h / src_h))

    print(f"[source]   {source.name} {src_w}x{src_h} @ {src_fps:.2f} fps, {src_frames} frames")
    print(f"[detector] {detector.name} on {detector.device} (classes={'all' if classes is None else classes})")
    if args.resize:
        print(f"[resize]   {src_w}x{src_h} -> {target_w}x{target_h}")
    print(f"[mode]     {'measure (no display)' if args.measure else 'live (press q to quit)'}")
    print()

    window: str | None = None
    if not args.measure:
        window = "AegisVision"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(target_w, 1280), min(target_h, 720))

    fps_window: collections.deque[float] = collections.deque(maxlen=30)
    frame_idx = 0
    t_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.resize:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            dt = time.perf_counter() - t0
            fps_window.append(1.0 / dt if dt > 0 else 0.0)
            rolling_fps = sum(fps_window) / len(fps_window)

            if window is not None:
                for d in detections:
                    _draw_box(frame, d)
                _draw_fps(frame, rolling_fps, detector.name)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frame_idx % 60 == 0:
                print(f"[frame {frame_idx:5d}] dets={len(detections):3d} rolling_fps={rolling_fps:5.1f}")

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if window is not None:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print()
    print(f"[summary] model={detector.name} backend=pytorch device={detector.device} "
          f"resolution={target_w}x{target_h} frames={frame_idx} elapsed={elapsed:.2f}s "
          f"avg_fps={avg_fps:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aegisvision.pipeline")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="video path")
    p.add_argument("--model", default="yolo",
                   help="model alias (yolo, rtdetr, yolo11n, ...) or weights filename")
    p.add_argument("--backend", default="pytorch", choices=["pytorch"], help="inference backend")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=None, help="inference image size")
    p.add_argument("--resize", type=int, default=None,
                   help="resize source to this height (e.g. 720 for 720p)")
    p.add_argument("--all-classes", action="store_true",
                   help="don't filter to vehicle classes (default: vehicles only)")
    p.add_argument("--measure", action="store_true", help="benchmark mode: no display, full pass")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
