# Phase 2 — ONNX + CoreML EP Validation

Run on 300 frames from `data/samples/traffic_sample.mp4` (starting at frame 500).

Validation gate: every CoreML EP config must hit **mean matched IoU ≥ 0.8** and **mean per-frame count delta ≤ 10.0%** vs the PyTorch/MPS baseline.

| Model | Backend | Config | FPS | Mean IoU | Count Δ | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- |
| yolo | pytorch | mps | 47.4 | — | — | baseline |
| yolo | onnx | cpu | 32.1 | 0.993 | 3.19% | PASS |
| yolo | onnx | coreml(CPUOnly) | 27.8 | 0.993 | 3.19% | PASS |
| yolo | onnx | coreml(CPUAndGPU) | 37.5 | 0.993 | 3.23% | PASS |
| yolo | onnx | coreml(CPUAndNeuralEngine) | 47.4 | 0.986 | 3.09% | PASS |
| yolo | onnx | coreml(ALL) | 46.8 | 0.986 | 3.09% | PASS |
| rtdetr | pytorch | mps | 22.6 | — | — | baseline |
| rtdetr | onnx | cpu | 2.9 | 0.877 | 9.40% | PASS |
| rtdetr | onnx | coreml(CPUOnly) | 4.4 | 0.877 | 9.40% | PASS |
| rtdetr | onnx | coreml(CPUAndGPU) | 6.6 | 0.877 | 9.47% | PASS |
| rtdetr | onnx | coreml(CPUAndNeuralEngine) | 7.1 | 0.834 | 9.41% | PASS |
| rtdetr | onnx | coreml(ALL) | 7.2 | 0.834 | 9.41% | PASS |

## Findings

Best ONNX configuration per model (highest FPS that passed the validation gate):

- **yolo**: ONNX/coreml(CPUAndNeuralEngine) (47.4 fps) is **0.1% slower** than the PyTorch/MPS baseline (47.4 fps).
- **rtdetr**: ONNX/coreml(ALL) (7.2 fps) is **68.2% slower** than the PyTorch/MPS baseline (22.6 fps).

The split between models is the headline trade-off this phase surfaces: ONNX Runtime's CoreML EP excels for CNN-style graphs (high op coverage by the partitioner), but transformer-heavy graphs like RT-DETR bridge across many small CoreML sub-graphs with CPU glue and lose to a coherent PyTorch/MPS execution. Phase 5's full benchmark suite is where this gets quantified across input sizes; Phase 6's Docker container intentionally drops to CPU EP only.
