# AegisVision — Real-Time Edge Computer Vision for Traffic Monitoring

## Overview

AegisVision is a real-time traffic monitoring system that detects, tracks, and counts vehicles in video streams. It is engineered as a portfolio-grade demonstration of four things:

- **Object Detection** — YOLO26 (CNN) vs. RT-DETRv2 (Transformer), benchmarked head-to-head.
- **Edge Optimization** — CoreML for native Apple Silicon performance, ONNX for portable container deployment.
- **Real-Time Stream Processing** — Sustained 30+ FPS with multi-class counting via ByteTrack.
- **Production Hygiene** — Structured JSON event logging, reproducible benchmarks, Docker-packaged ARM64 deployment.

## Target Performance

- **Throughput:** ≥30 FPS sustained on M1 Max (32-core GPU, 64 GB unified memory) at 1080p input.
- **Accuracy:** Counting accuracy within ±2% of ground truth on a manually-labeled validation segment.
- **Footprint:** Docker image < 2 GB; cold-start to first frame < 5 s.

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12+ |
| Package manager | `uv` |
| Detection | Ultralytics (YOLO26, RT-DETRv2) |
| Tracking | ByteTrack (via Ultralytics) |
| Vision I/O | OpenCV |
| Native acceleration | CoreML (`coremltools`) |
| Portable acceleration | ONNX Runtime |
| Containerization | Docker (`linux/arm64`) |
| Logging | Line-delimited JSON to file |

## Deployment Paths

Two parallel paths share the same pipeline, tracker, counter, overlay, and logger — only the detector backend swaps:

1. **Native (Mac):** PyTorch baseline → CoreML for production. Targets the M1 Max GPU/ANE directly. This is the performance flex.
2. **Portable (Docker):** ONNX Runtime in a `linux/arm64` container. This is the deployability flex. Justifies its existence by being explicitly compared against the CoreML path in the benchmark suite.

## Repository Layout

```
AegisVision/
├── pyproject.toml          # uv-managed
├── uv.lock
├── README.md               # ships with demo GIF + benchmark table
├── AegisVision.md          # this plan
├── Dockerfile              # ONNX path, linux/arm64
├── Makefile                # setup / fetch / demo / benchmark / docker-*
├── data/
│   ├── samples/            # gitignored; fetched via script
│   └── README.md           # source URL + sha256 + license note
├── models/                 # exported weights (.pt, .mlpackage, .onnx)
├── configs/                # counting-line coordinates, class filters, etc.
├── src/aegisvision/
│   ├── __init__.py
│   ├── pipeline.py         # frame loop: capture → detect → track → count → render → log
│   ├── detectors/
│   │   ├── base.py         # Detector interface
│   │   ├── pytorch.py
│   │   ├── coreml.py
│   │   └── onnx.py
│   ├── tracker.py          # ByteTrack wrapper
│   ├── counter.py          # line-crossing + ROI variants
│   ├── overlay.py          # OpenCV HUD
│   └── telemetry.py        # JSON event logger
├── scripts/
│   ├── fetch_sample.py     # yt-dlp download, checksum verify
│   ├── export_coreml.py
│   ├── export_onnx.py
│   └── benchmark.py        # writes results/benchmark.md
├── results/
│   └── benchmark.md        # generated, committed
└── tests/
```

## Phases

### Phase 0 — Project Setup

**Goal:** Reproducible scaffolding before any model code touches the repo.

- `uv init`, pin Python 3.12+, add core deps: `ultralytics`, `opencv-python`, `coremltools`, `onnxruntime`, `numpy`, `yt-dlp`.
- Create the directory layout above.
- `scripts/fetch_sample.py` downloads a pinned YouTube traffic clip via `yt-dlp`, verifies SHA-256, and writes to `data/samples/`. Pin URL + hash in `data/README.md`. The video is gitignored — only the URL is committed, so reproducibility depends on the link staying live.
- `Makefile` skeleton: `setup`, `fetch`, `demo`, `benchmark`, `docker-build`, `docker-run`.

**Exit:** `make setup && make fetch` produces a working environment with the pinned video on disk.

### Phase 1 — Baseline Detection (PyTorch / MPS)

**Goal:** Both detectors running end-to-end on the sample clip with baseline numbers captured.

- Implement the `Detector` interface in `detectors/base.py`: `detect(frame) -> list[Detection]`.
- Implement `detectors/pytorch.py` for YOLO26 and RT-DETRv2 (MPS device).
- MVP `pipeline.py`: read frames, run detection, draw raw boxes, display window, print rolling FPS.
- Capture per-model baseline FPS at 720p and 1080p.

> Note: if `YOLO26` isn't yet the Ultralytics package alias by the time we run this, fall back to the current flagship (`YOLO12` / `YOLO11`). The plan stays the same.

**Exit:** `python -m aegisvision.pipeline --model yolo26 --backend pytorch` runs at measurable FPS with plausible boxes on screen.

### Phase 2 — CoreML Export & Validation

**Goal:** Native Apple Silicon acceleration without silently breaking accuracy.

- `scripts/export_coreml.py` exports both models to `.mlpackage`.
- Implement `detectors/coreml.py`.
- **Validation gate:** compare CoreML detections to PyTorch baseline on the same N frames. Require mean IoU ≥ 0.95 on matched boxes and per-frame detection-count delta ≤ 2%. Fail loudly if not met.
- Sweep CoreML compute units: `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`, `ALL`. On M1 Max the GPU path often beats ANE because some ops fall back silently — the data here is itself a finding worth writing up.

**Exit:** CoreML matches PyTorch within tolerance; best compute-unit setting documented.

### Phase 3 — Tracking & Counting

**Goal:** Convert per-frame detections into accurate, class-aware counts.

- Wire ByteTrack via Ultralytics' built-in tracker.
- `counter.py`:
  - User-defined polyline as the counting line (configurable in `configs/`).
  - Direction inference (in vs. out) from track centroid trajectory across the line.
  - Per-class tallies: car, truck, bus, motorcycle.
  - Deduplication by `track_id` to prevent double-counting on flicker.
- ROI / zone-occupancy variant behind a flag.
- Manually label a 30-second segment of the sample clip to validate counter accuracy (≥98% match).

**Exit:** Counts on the labeled segment are within tolerance.

### Phase 4 — Output & Telemetry

**Goal:** Production-ready observability.

- `overlay.py`: bounding boxes with class + track ID, FPS (instant + rolling avg), per-class counts, current direction tally.
- `telemetry.py`: line-delimited JSON to `logs/events.jsonl`. Schema:
  ```json
  {"ts": "<iso8601>", "frame": 1234, "event": "line_cross",
   "track_id": 42, "class": "car", "direction": "in", "confidence": 0.91}
  ```
- Periodic `summary` events every N seconds with rolling totals.
- Clean shutdown: flush buffer, write a final `summary` event.

**Exit:** A demo run produces a complete JSON log replayable into a counts time-series.

### Phase 5 — Benchmark Suite

**Goal:** Generate the artifact that sells the project on a résumé.

- `scripts/benchmark.py` runs the full matrix:
  - Models: YOLO26, RT-DETRv2.
  - Backends: PyTorch/MPS, CoreML (best compute unit), ONNX (CoreML EP), ONNX (CPU).
  - Resolutions: 720p, 1080p.
- Per-cell metrics: FPS mean / p50 / p95, mean per-frame latency, peak RSS, mAP delta vs. PyTorch baseline.
- Output: `results/benchmark.md` with formatted table + commentary on CNN vs. Transformer trade-offs.

**Exit:** Benchmark table committed; ≥30 FPS achieved by at least one model × backend at 1080p.

### Phase 6 — ONNX + Docker (Linux/ARM64)

**Goal:** Portable edge deployment, parallel to the CoreML native path.

- `scripts/export_onnx.py` exports both models to ONNX.
- Implement `detectors/onnx.py` with ONNX Runtime.
- `Dockerfile`:
  - `--platform=linux/arm64`, base `python:3.12-slim`.
  - `uv sync` without `coremltools`.
  - Copy ONNX models + source.
  - Default entrypoint: `python -m aegisvision.pipeline --backend onnx`.
- `make docker-build` / `make docker-run` with the sample video mounted as a volume; container writes annotated MP4 + JSON log to a host-mounted output dir (no GUI inside the container).
- Re-run Phase 5 benchmark inside the container; record container overhead vs. host.

**Exit:** `docker run` reproduces the demo end-to-end on M1 Max via `--platform linux/arm64`.

### Phase 7 — Polish & Reproducibility

**Goal:** Make it presentable.

- README with:
  - One-paragraph pitch.
  - Architecture diagram (Mermaid).
  - Demo GIF (~5 s of overlay output).
  - Benchmark table embedded from `results/benchmark.md`.
  - "How to reproduce" section with `make demo` and `make docker-run`.
- Verify `make demo` works from a fresh clone.
- Final pass on type hints, docstrings (only where the *why* is non-obvious), and error messages at I/O boundaries.

**Exit:** A new reader can clone the repo and run the demo with three commands.

## Non-Goals

- Multi-camera ingest, RTSP ingestion, or live-stream production deployment.
- Training or fine-tuning either detector — pretrained COCO weights only.
- Web dashboard or REST API beyond the JSON log.
- CUDA/TensorRT support. The "edge" story here is intentionally CoreML-native + portable ONNX-ARM64.

## Open Questions / Future Extensions

- Replace line-crossing with a more sophisticated zone-occupancy estimator.
- Stream the JSON log over WebSocket from a small FastAPI endpoint.
- Fine-tune a vehicle-specialized head on UA-DETRAC for higher accuracy than COCO weights.
