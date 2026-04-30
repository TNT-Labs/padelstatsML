#!/usr/bin/env python3
"""
Export YOLO and/or TrackNetV2 models to ONNX for faster CPU inference on Raspberry Pi 5.

Export YOLO (default):
    cd backend/
    python scripts/export_onnx.py
    # produces weights/yolov8n.onnx

Export TrackNetV2:
    python scripts/export_onnx.py --tracknet weights/tracknet_padel.pth
    # produces weights/tracknet_padel.onnx

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


def export_yolo(weights: str, out: str, imgsz: int) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {weights} …")
    model = YOLO(weights)

    print(f"Exporting to ONNX (imgsz={imgsz}) …")
    # simplify=False: skips onnxslim which rewrites the model with a newer ONNX
    # IR version that onnxruntime 1.19.x cannot load (max IR version 10).
    exported = model.export(format="onnx", imgsz=imgsz, simplify=False, dynamic=False)

    # shutil.move handles cross-device links (weights/ bind-mounted volume)
    exported_path = Path(exported)
    if exported_path.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported_path), out_path)

    print(f"\nExported: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("\nNext steps:")
    print(f"  1. Add to .env: YOLO_WEIGHTS={out_path}")
    print("  2. Restart the worker")
    print("  3. Verify onnxruntime is installed: pip install onnxruntime==1.19.2")


def export_tracknet(weights: str, out: str) -> None:
    try:
        import torch
    except ImportError:
        print("torch not installed.")
        sys.exit(1)

    try:
        from app.ml.ball import TrackNetV2
    except ImportError as e:
        print(f"Cannot import TrackNetV2: {e}")
        sys.exit(1)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading TrackNetV2 weights from {weights} …")
    model = TrackNetV2()
    state = torch.load(weights, map_location="cpu")
    # Handle checkpoints that wrap weights in a key
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # TrackNetV2 input: 3 consecutive RGB frames stacked → 9 channels, 360×640
    dummy = torch.zeros(1, 9, 360, 640)
    print("Exporting TrackNetV2 to ONNX (opset 12) …")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=12,
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes=None,  # fixed shape — avoids IR version bump from dynamic axes
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nExported: {out_path} ({size_mb:.1f} MB)")
    print("TrackNetV2 ONNX will be auto-detected by BallTracker at runtime.")
    print("(Place the .onnx file alongside the .pth file — no config change needed.)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO or TrackNetV2 weights to ONNX"
    )
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="Source YOLO .pt weights (ignored when --tracknet is set)")
    parser.add_argument("--out",     default="weights/yolov8n.onnx",
                        help="Output .onnx path for YOLO")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="YOLO input image size")
    parser.add_argument("--tracknet", metavar="PTH_PATH", default=None,
                        help="Export TrackNetV2 instead of YOLO. "
                             "Provide path to .pth file; .onnx is written alongside it.")
    args = parser.parse_args()

    if args.tracknet:
        pth = Path(args.tracknet)
        onnx_out = str(pth.with_suffix(".onnx"))
        export_tracknet(str(pth), onnx_out)
    else:
        export_yolo(args.weights, args.out, args.imgsz)


if __name__ == "__main__":
    main()
