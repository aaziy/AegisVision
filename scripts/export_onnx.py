"""Export YOLO and RT-DETR weights to ONNX (.onnx).

Run via `make export-onnx` or directly: `uv run python scripts/export_onnx.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ultralytics import RTDETR, YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

EXPORTS: list[tuple[str, type]] = [
    ("yolo26n.pt", YOLO),
    ("rtdetr-l.pt", RTDETR),
]


def export_one(weights: str, model_cls: type, imgsz: int, simplify: bool, opset: int) -> Path:
    src = MODELS_DIR / weights
    if not src.exists():
        raise FileNotFoundError(f"weights not found: {src}")

    cwd = Path.cwd()
    try:
        os.chdir(MODELS_DIR)
        model = model_cls(weights)
        out_rel = model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            nms=False,
            dynamic=False,
            opset=opset,
        )
        # Resolve while still chdir'd into MODELS_DIR; Ultralytics returns a
        # path relative to its working directory.
        out = (MODELS_DIR / Path(out_rel).name).resolve()
    finally:
        os.chdir(cwd)

    # Auto-suffix non-default sizes: yolo26n.onnx for 640, yolo26n_960.onnx for 960.
    if imgsz != 640:
        suffixed = out.parent / f"{out.stem}_{imgsz}.onnx"
        out.replace(suffixed)
        out = suffixed
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export Ultralytics weights to ONNX.")
    p.add_argument("--imgsz", type=int, default=640, help="Model input size (square)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--no-simplify", action="store_true", help="Disable onnx-simplifier pass")
    args = p.parse_args(argv)

    rc = 0
    for weights, cls in EXPORTS:
        print(f"=== exporting {weights} (imgsz={args.imgsz}, opset={args.opset}) ===")
        try:
            out = export_one(weights, cls, args.imgsz, not args.no_simplify, args.opset)
            size_mb = out.stat().st_size / 1e6
            print(f"  -> {out.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
