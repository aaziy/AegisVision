---
title: AegisVision
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Real-time traffic monitoring — YOLO26n + ByteTrack via ONNX CPU EP
---

# AegisVision — Real-Time Traffic Monitoring

Upload a traffic video (or use the bundled sample) and the pipeline will:

1. **Detect** vehicles (car, truck, bus, motorcycle, bicycle) with **YOLO26n**
2. **Track** them across frames with **ByteTrack**
3. **Count** crossings of a configurable line with direction (in / out)
4. Return an **annotated MP4** and a structured **JSONL event log**

## Stack

| | |
|---|---|
| Detection | YOLO26n (COCO weights, Ultralytics) |
| Tracking | ByteTrack |
| Inference | ONNX Runtime — CPU Execution Provider |
| Frontend | Gradio |

## Performance

On the host (M1 Max, CoreML EP): **46 FPS** at 1080p.  
On HF free CPU tier (CPU EP, 720p resize): ~8–12 FPS → ~60 s for 150 frames.

Full benchmark table: [results/benchmark.md](https://github.com/aaziy/AegisVision/blob/main/results/benchmark.md)

## Source

[github.com/aaziy/AegisVision](https://github.com/aaziy/AegisVision)
