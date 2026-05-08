"""Gradio frontend for AegisVision — Hugging Face Spaces entry point.

Wraps the existing pipeline.run() so the same detector / tracker / counter
logic runs headlessly and returns an annotated MP4 + JSONL event log.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from aegisvision import pipeline

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_VIDEO = REPO_ROOT / "data" / "samples" / "traffic_sample.mp4"
COUNTING_LINE = REPO_ROOT / "configs" / "counting_line.yaml"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Maximum frames to process per request; keeps CPU-EP latency tolerable on HF.
MAX_FRAMES = 300


def _parse_log(jsonl_text: str) -> str:
    """Format the raw JSONL into a readable summary for display."""
    lines = [ln for ln in jsonl_text.strip().splitlines() if ln]
    if not lines:
        return "No events."
    end_event = None
    for ln in reversed(lines):
        try:
            ev = json.loads(ln)
            if ev.get("event") == "end":
                end_event = ev
                break
        except json.JSONDecodeError:
            continue

    parts: list[str] = []
    if end_event:
        parts.append(
            f"Frames processed : {end_event.get('frame', '?')}\n"
            f"Avg FPS          : {end_event.get('avg_fps', '?')}\n"
            f"Elapsed          : {end_event.get('elapsed_s', '?')} s\n"
            f"Total in         : {end_event.get('total_in', 0)}\n"
            f"Total out        : {end_event.get('total_out', 0)}"
        )
        counts = end_event.get("counts", {})
        if counts:
            parts.append("\nPer-class counts:")
            for cls, d in sorted(counts.items()):
                parts.append(f"  {cls:<12} in={list(d.values())[0]}  out={list(d.values())[1]}")
    parts.append(f"\n--- raw log ({len(lines)} events) ---")
    parts.append("\n".join(lines[-30:]))  # last 30 events
    return "\n".join(parts)


def process(
    video_path: str | None,
    use_sample: bool,
    imgsz: int,
    conf: float,
    max_frames: int,
) -> tuple[str | None, str]:
    src = SAMPLE_VIDEO if use_sample else (Path(video_path) if video_path else None)
    if src is None or not src.exists():
        return None, "No input video. Upload a clip or tick 'Use bundled sample'."

    with tempfile.TemporaryDirectory() as tmp:
        out_video = Path(tmp) / "annotated.mp4"
        out_log = Path(tmp) / "events.jsonl"

        args = argparse.Namespace(
            source=str(src),
            model="yolo",
            backend="onnx",
            provider="cpu",
            compute_units="ALL",
            conf=conf,
            imgsz=imgsz,
            resize=720,          # cap source to 720p so CPU EP is tolerable
            all_classes=False,
            track=True,
            count=True,
            counting_line=str(COUNTING_LINE),
            log=True,
            log_file=str(out_log),
            summary_interval=5.0,
            output_video=str(out_video),
            measure=True,        # headless — no cv2.imshow
            max_frames=max_frames,
        )

        exit_code = pipeline.run(args)
        if exit_code != 0:
            return None, f"Pipeline exited with code {exit_code}."

        final_path = OUTPUT_DIR / "annotated.mp4"
        shutil.copy(out_video, final_path)
        log_text = out_log.read_text(encoding="utf-8") if out_log.exists() else ""

    return str(final_path), _parse_log(log_text)


# ── UI ──────────────────────────────────────────────────────────────────────

_DESCRIPTION = """
## AegisVision — Real-Time Traffic Monitoring

Detects, tracks, and counts vehicles using **YOLO26n + ByteTrack** via ONNX Runtime (CPU EP).

**How it works:** upload a traffic video (or use the bundled sample) → the pipeline runs \
detection → tracking → line-crossing counting → returns an annotated MP4 and a structured event summary.

> Running on CPU — first run may take ~60 s for 300 frames. Subsequent runs are faster (warm model cache).
"""

with gr.Blocks(title="AegisVision", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            video_in = gr.Video(label="Upload a traffic video", sources=["upload"])
            use_sample = gr.Checkbox(
                label="Use bundled sample video (1080p, ~10 s segment)",
                value=True,
            )

            with gr.Accordion("Options", open=False):
                imgsz_sl = gr.Slider(
                    320, 640, value=640, step=32,
                    label="Model input size (imgsz)",
                    info="Lower = faster, less detail",
                )
                conf_sl = gr.Slider(
                    0.1, 0.9, value=0.25, step=0.05,
                    label="Confidence threshold",
                )
                frames_sl = gr.Slider(
                    30, MAX_FRAMES, value=150, step=30,
                    label="Max frames to process",
                    info=f"Cap at {MAX_FRAMES} to keep latency reasonable",
                )

            run_btn = gr.Button("Run pipeline", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            video_out = gr.Video(label="Annotated video")
            log_box = gr.Textbox(
                label="Count summary + event log",
                lines=18,
                max_lines=30,
                interactive=False,
            )

    run_btn.click(
        process,
        inputs=[video_in, use_sample, imgsz_sl, conf_sl, frames_sl],
        outputs=[video_out, log_box],
    )

    gr.Markdown(
        "Model: **YOLO26n** (COCO weights) · Tracker: **ByteTrack** · "
        "Backend: **ONNX Runtime CPU EP** · "
        "[GitHub](https://github.com/aaziy/AegisVision) · "
        "[Benchmark results](https://github.com/aaziy/AegisVision/blob/main/results/benchmark.md)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
