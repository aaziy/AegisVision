"""Live detection pipeline: capture → detect → track → count → render → log."""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2

from aegisvision.counter import LineCrossingCounter, load_config
from aegisvision.detectors.base import VEHICLE_CLASS_IDS
from aegisvision.detectors.onnx import OnnxDetector
from aegisvision.detectors.pytorch import PyTorchDetector
from aegisvision.overlay import draw_box, draw_hud, draw_line
from aegisvision.telemetry import EventLogger
from aegisvision.tracker import ByteTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "samples" / "traffic_sample.mp4"
DEFAULT_COUNTING_LINE = REPO_ROOT / "configs" / "counting_line.yaml"
DEFAULT_LOG = REPO_ROOT / "logs" / "events.jsonl"


def run(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        print(f"source not found: {source}", file=sys.stderr)
        return 2

    classes = None if args.all_classes else sorted(VEHICLE_CLASS_IDS)
    if args.backend == "pytorch":
        detector = PyTorchDetector(
            model=args.model, conf=args.conf, imgsz=args.imgsz, classes=classes,
        )
    elif args.backend == "onnx":
        detector = OnnxDetector(
            model=args.model, provider=args.provider,
            compute_units=args.compute_units, conf=args.conf,
            classes=classes, imgsz=args.imgsz or 640,
        )
    else:
        print(f"unknown backend: {args.backend}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"failed to open video: {source}", file=sys.stderr)
        return 2

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_w, target_h = src_w, src_h
    if args.resize:
        target_h = args.resize
        target_w = int(round(src_w * target_h / src_h))

    tracker = ByteTracker(fps=src_fps) if args.track else None
    counter: LineCrossingCounter | None = None
    if args.count and tracker is not None:
        line, in_label, out_label = load_config(args.counting_line)
        counter = LineCrossingCounter(line, in_label, out_label)

    logger: EventLogger | None = None
    if args.log:
        logger = EventLogger(args.log_file)

    print(f"[source]    {source.name} {src_w}x{src_h} @ {src_fps:.2f} fps, {src_frames} frames")
    print(f"[detector]  {detector.name} on {detector.device} "
          f"(classes={'all' if classes is None else classes})")
    print(f"[tracker]   {'ByteTrack' if tracker else 'disabled'}")
    if counter is not None:
        print(f"[counter]   line {counter.line.p1} -> {counter.line.p2}  "
              f"({counter.in_label} / {counter.out_label})")
    else:
        print(f"[counter]   disabled")
    print(f"[telemetry] {logger.path if logger else 'disabled'}")
    if args.resize:
        print(f"[resize]    {src_w}x{src_h} -> {target_w}x{target_h}")
    print(f"[mode]      {'measure (no display)' if args.measure else 'live (press q to quit)'}")
    print()

    if logger:
        logger.start(
            source=str(source), model=detector.name, backend=args.backend,
            device=detector.device, resolution=(target_w, target_h), fps=src_fps,
        )

    window: str | None = None
    if not args.measure:
        window = "AegisVision"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, min(target_w, 1280), min(target_h, 720))

    writer: cv2.VideoWriter | None = None
    if args.output_video:
        Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, src_fps, (target_w, target_h))
        if not writer.isOpened():
            print(f"failed to open output video: {args.output_video}", file=sys.stderr)
            return 2
        print(f"[output]    annotated mp4 -> {args.output_video}")

    fps_window: collections.deque[float] = collections.deque(maxlen=30)
    frame_idx = 0
    t_start = time.perf_counter()
    last_summary = t_start
    summary_interval = max(0.5, float(args.summary_interval))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.resize:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            tracks = tracker.update(detections) if tracker else []
            crossings = counter.update(tracks) if counter is not None else []
            dt = time.perf_counter() - t0
            fps_window.append(1.0 / dt if dt > 0 else 0.0)
            rolling_fps = sum(fps_window) / len(fps_window)

            if logger and crossings:
                for ev in crossings:
                    logger.line_cross(
                        frame=ev.frame, track_id=ev.track_id,
                        class_name=ev.class_name, direction=ev.direction,
                        confidence=ev.confidence, point=ev.point,
                    )

            now = time.perf_counter()
            if logger and counter is not None and now - last_summary >= summary_interval:
                logger.summary(
                    frame=frame_idx, fps=rolling_fps, counts=counter.counts,
                    total_in=counter.total_in, total_out=counter.total_out,
                )
                last_summary = now

            if window is not None or writer is not None:
                items = tracks if tracker else detections
                for d in items:
                    draw_box(frame, d)
                if counter is not None:
                    draw_line(frame, counter.line)
                draw_hud(frame, rolling_fps, detector.name, counter)

            if window is not None:
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer is not None:
                writer.write(frame)

            if window is None and frame_idx % 60 == 0:
                ext = (
                    f" tracks={len(tracks):2d} counts={counter.total_in}/{counter.total_out}"
                    if counter else (f" tracks={len(tracks):2d}" if tracker else "")
                )
                print(f"[frame {frame_idx:5d}] dets={len(detections):3d}{ext}  "
                      f"rolling_fps={rolling_fps:5.1f}")

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if window is not None:
            cv2.destroyAllWindows()

        elapsed = time.perf_counter() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
        if logger:
            if counter is not None:
                logger.end(
                    frame=frame_idx, elapsed_s=elapsed, avg_fps=avg_fps,
                    counts=counter.counts,
                    total_in=counter.total_in, total_out=counter.total_out,
                )
            else:
                logger.emit(event="end", frame=frame_idx,
                            elapsed_s=round(elapsed, 2), avg_fps=round(avg_fps, 2))
            logger.close()

    print()
    print(f"[summary] model={detector.name} backend={args.backend} device={detector.device} "
          f"resolution={target_w}x{target_h} frames={frame_idx} elapsed={elapsed:.2f}s "
          f"avg_fps={avg_fps:.2f}")
    if counter is not None:
        print(f"[counts]  total: {counter.total} "
              f"({counter.total_in} {counter.in_label}, "
              f"{counter.total_out} {counter.out_label})")
        for cls in sorted(counter.counts.keys()):
            d = counter.counts[cls]
            print(f"          {cls:11s} {d[counter.in_label]:4d} {counter.in_label}  "
                  f"{d[counter.out_label]:4d} {counter.out_label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aegisvision.pipeline")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="video path")
    p.add_argument("--model", default="yolo",
                   help="model alias (yolo, rtdetr, yolo11n, ...) or weights filename")
    p.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx"],
                   help="inference backend")
    p.add_argument("--provider", default="coreml", choices=["coreml", "cpu"],
                   help="ONNX execution provider (only used when --backend=onnx)")
    p.add_argument("--compute-units", default="ALL",
                   choices=["CPUOnly", "CPUAndGPU", "CPUAndNeuralEngine", "ALL"],
                   help="CoreML EP compute units (only used when --provider=coreml)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=None, help="inference image size")
    p.add_argument("--resize", type=int, default=None,
                   help="resize source to this height (e.g. 720 for 720p)")
    p.add_argument("--all-classes", action="store_true",
                   help="don't filter to vehicle classes (default: vehicles only)")
    p.add_argument("--track", action=argparse.BooleanOptionalAction, default=True,
                   help="run ByteTrack on detections (default: on)")
    p.add_argument("--count", action=argparse.BooleanOptionalAction, default=True,
                   help="run line-crossing counter on tracks (default: on)")
    p.add_argument("--counting-line", default=str(DEFAULT_COUNTING_LINE),
                   help="path to counting-line YAML config")
    p.add_argument("--log", action=argparse.BooleanOptionalAction, default=True,
                   help="write JSONL event log (default: on)")
    p.add_argument("--log-file", default=str(DEFAULT_LOG),
                   help="path to JSONL event log")
    p.add_argument("--summary-interval", type=float, default=5.0,
                   help="seconds between summary events in the log (default: 5)")
    p.add_argument("--output-video", default=None,
                   help="path to write annotated MP4 (works headless; for Docker)")
    p.add_argument("--measure", action="store_true", help="benchmark mode: no display, full pass")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
