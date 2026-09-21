#!/usr/bin/env python3
"""Export the YOLOv8 person detector to ONNX for the Pi runtime.

Run this once, anywhere with torch available — a laptop is fine, the file is
portable. The Pi itself never needs torch or ultralytics.

    pip install -r requirements.export.txt
    python scripts/export_yolo_onnx.py --imgsz 480 --out weights/yolov8n.onnx

Then copy weights/yolov8n.onnx onto the Pi and point DETECTOR_MODEL at it.

Why a fixed input size: a statically-shaped graph lets onnxruntime pre-plan
every allocation, which is worth roughly 10-15% on a Cortex-A76 compared to a
dynamic-axis export.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt",
                        help="ultralytics checkpoint (downloaded on first use)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="inference input size; must match DETECTOR_IMGSZ")
    parser.add_argument("--out", default="weights/yolov8n.onnx")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--int8", action="store_true",
                        help="apply dynamic int8 quantisation (faster, lower recall — verify before trusting)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics non installato. Esegui: pip install -r requirements.export.txt",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Esporto {args.model} a ONNX (imgsz={args.imgsz}, opset={args.opset})…")
    model = YOLO(args.model)
    produced = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=False,
            simplify=True,
            half=False,
        )
    )

    if produced.resolve() != out.resolve():
        shutil.move(str(produced), str(out))
    print(f"Scritto {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if args.int8:
        out = _quantise(out)

    _verify(out, args.imgsz)
    print("\nImposta nel .env del Pi:")
    print(f"  DETECTOR_MODEL={out}")
    print(f"  DETECTOR_IMGSZ={args.imgsz}")
    return 0


def _quantise(path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target = path.with_name(path.stem + "_int8.onnx")
    print(f"Quantizzazione dinamica int8 → {target}")
    quantize_dynamic(str(path), str(target), weight_type=QuantType.QUInt8)
    print(f"Scritto {target} ({target.stat().st_size / 1e6:.1f} MB)")
    print("ATTENZIONE: verifica il recall su un video reale prima di usarlo in produzione.")
    return target


def _verify(path: Path, imgsz: int) -> None:
    """Load the exported graph exactly as the Pi will, and check the output
    layout the detector expects."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    output = session.run(None, {name: dummy})[0]
    print(f"Verifica onnxruntime: input '{name}' → output {output.shape}")
    if output.ndim != 3 or min(output.shape[1:]) < 5:
        print("ATTENZIONE: layout di output inatteso per app/ml/detect.py", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
