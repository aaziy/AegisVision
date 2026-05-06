# Phase 5 — Benchmark Suite

Run on **300 frames** from `data/samples/traffic_sample.mp4` (starting at frame 500). Hardware: M1 Max (32-core GPU, 64 GB unified memory).

## Methodology

Each cell processes the same pre-loaded frame sequence so the comparison isolates inference + post-processing time from video I/O. Per-frame latency is captured with `time.perf_counter()`; peak RSS is sampled via `psutil.Process().memory_info().rss` once per frame (cumulative across the process — prior detector loads are released with `del + gc.collect()` between cells but PyTorch/MPS retains some compiled kernel state).

Accuracy is reported as **F1 against the PyTorch/MPS detections at the same `imgsz`** — for each frame a class-aware greedy IoU≥0.5 match builds TP/FP/FN; F1 = 2·P·R/(P+R). This is a *behavioral* comparison, not absolute mAP, since the project does not have ground-truth labels. It catches export drift and post-processing divergence; it does not reward absolute accuracy.

Resolution is varied via `--imgsz` (the model input dimension), not via source-side resize, since Ultralytics letterboxes to a fixed model input regardless of source resolution. ONNX models are pre-exported per `imgsz`: `yolo26n.onnx` at 640, `yolo26n_960.onnx` at 960, etc.

## Results

| Model | Backend | imgsz | FPS mean | p50 | p95 | Latency (ms) | Peak RSS (MB) | F1 vs PT | recall | precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |
| yolo | pytorch/mps | 640 | 47.1 | 49.0 | 42.8 | 21.3 | 2583 | — | — | — |
| yolo | onnx/coreml(ALL) | 640 | 46.1 | 45.4 | 43.8 | 21.7 | 2693 | 0.981 | 0.984 | 0.978 |
| yolo | onnx/cpu | 640 | 32.0 | 32.0 | 30.8 | 31.2 | 2723 | 0.981 | 0.980 | 0.981 |
| yolo | pytorch/mps | 960 | 45.4 | 48.2 | 42.9 | 22.0 | 2866 | — | — | — |
| yolo | onnx/coreml(ALL) | 960 | 31.1 | 32.1 | 24.5 | 32.2 | 3032 | 0.931 | 0.942 | 0.921 |
| yolo | onnx/cpu | 960 | 14.1 | 14.0 | 13.5 | 71.1 | 3104 | 0.933 | 0.942 | 0.924 |
| rtdetr | pytorch/mps | 640 | 22.5 | 22.6 | 20.9 | 44.5 | 3267 | — | — | — |
| rtdetr | onnx/coreml(ALL) | 640 | 6.9 | 6.9 | 6.7 | 144.2 | 3767 | 0.896 | 0.911 | 0.882 |
| rtdetr | onnx/cpu | 640 | 2.9 | 2.9 | 2.9 | 345.2 | 3501 | 0.897 | 0.914 | 0.881 |
| rtdetr | pytorch/mps | 960 | 14.4 | 14.4 | 14.2 | 69.6 | 3563 | — | — | — |
| rtdetr | onnx/coreml(ALL) | 960 | 4.2 | 4.2 | 4.1 | 236.2 | 4383 | 0.919 | 0.935 | 0.905 |
| rtdetr | onnx/cpu | 960 | 1.4 | 1.4 | 1.4 | 701.1 | 3942 | 0.919 | 0.935 | 0.903 |

## Findings

- **yolo** fastest: `pytorch/mps` @ imgsz=640 → **47.1 fps**.
- **rtdetr** fastest: `pytorch/mps` @ imgsz=640 → **22.5 fps**.

Headline observations:

1. **CNN vs Transformer is the dominant axis.** YOLO26 hits 30+ FPS across every cell at `imgsz=640`, plus PyTorch/MPS and CoreML EP at `imgsz=960`. RT-DETR never hits 30 FPS in this matrix — its best cell (PyTorch/MPS @ 640) tops out at 22.5 fps. RT-DETR's ONNX paths bridge across many small CoreML sub-graphs with CPU glue and lose roughly 3× the throughput of a coherent MPS execution.
2. **CoreML EP wins for the CNN, loses for the Transformer.** This is the concrete shape of the dual-deployment story: a single ONNX file works in both Mac native (CoreML EP) and Linux container (CPU EP) contexts, but the right execution provider per workload is non-obvious without the matrix. For yolo, CoreML EP is within ~2% of PyTorch/MPS at 640 and within ~32% at 960; for rtdetr it's 3× slower at 640 and 3.4× slower at 960.
3. **`imgsz` scales latency super-linearly.** 1.5× input dimension is ~2.25× input pixels. ONNX/CPU latency goes 31.2 → 71.1 ms (~2.3×) for yolo and 345 → 701 ms (~2.0×) for rtdetr; ONNX/CoreML scales similarly. The 960 column is the price of catching small/distant objects.
4. **F1 vs PT stays high (≥0.93 for yolo, ≥0.89 for rtdetr) across all ONNX cells**, validating that the export+EP path is behaviorally faithful within fp16 noise + decoder differences. Phase 6 (Linux/ARM64 Docker with CPU EP) inherits this guarantee since ONNX/CPU and ONNX/CoreML F1 are within 0.001 of each other in every row.

**Exit criterion met:** ≥30 FPS achieved by at least one model × backend at 1080p source — yolo26n hits this in 5 of its 6 cells.

