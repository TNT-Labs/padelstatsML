#!/usr/bin/env python3
"""
Export YOLOv8 model to ONNX for faster CPU inference on Raspberry Pi 5.

Run once after first setup:
    cd backend/
    python scripts/export_onnx.py
    # produces weights/yolov8n.onnx

Then set in .env:
    YOLO_WEIGHTS=weights/yolov8n.onnx

Ultralytics automatically uses onnxruntime when given an .onnx model,
giving 3-5x faster inference on ARM64 vs PyTorch.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX")
    parser.add_argument("--weights", default="yolov8n.pt", help="Source .pt weights")
    parser.add_argument("--out",     default="weights/yolov8n.onnx", help="Output .onnx path")
    parser.add_argument("--imgsz",   type=int, default=640, help="Input image size")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.weights} …")
    model = YOLO(args.weights)

    print(f"Exporting to ONNX (imgsz={args.imgsz}) …")
    # Export returns path to .onnx file
    # simplify=False: skips onnxslim which rewrites the model with a newer ONNX
    # IR version that onnxruntime 1.19.x cannot load (max IR version 10).
    exported = model.export(format="onnx", imgsz=args.imgsz, simplify=False, dynamic=False)

    # Move to desired output location (shutil.move handles cross-device links,
    # e.g. when weights/ is a bind-mounted volume on a different filesystem)
    exported_path = Path(exported)
    if exported_path.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_path), out_path)

    print(f"\nExported: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("\nNext steps:")
    print(f"  1. Add to .env: YOLO_WEIGHTS={out_path}")
    print("  2. Restart the worker")
    print("  3. Verify onnxruntime is installed: pip install onnxruntime==1.19.2")


if __name__ == "__main__":
    main()
