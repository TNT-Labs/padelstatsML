#!/usr/bin/env python3
"""Check the AI HAT+ (Hailo) end to end and print why it fails, if it does.

    docker exec padel-worker python scripts/hailo_check.py [video]

What the worker's own check hides behind one line: the network chosen, its
input and output as HailoRT describes them, each way of allocating the
output tried by the trial inference, and — with a video — how many people a
frame of it holds on the board and on the CPU. Run it while no analysis is
in progress: the board serves one process at a time.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.ml.accel import hailo_status  # noqa: E402


def main() -> int:
    settings = get_settings()
    status = hailo_status(settings.accelerator, settings.hailo_model or None)
    print(f"Stato:   {status.reason}")
    print(f"Reti:    {', '.join(Path(m).name for m in status.models) or '-'}")
    if not status.usable:
        return 1

    try:
        from hailo_platform import VDevice

        from app.ml.accel import pick_model
        from app.ml.detect_hailo import _architecture, output_buffers

        with VDevice(VDevice.create_params()) as device:
            chip = _architecture(device)
            model_path = pick_model(status.models, chip)
            print(f"Chip:    {chip}")
            print(f"Rete:    {model_path}")
            model = device.create_infer_model(model_path)
            output = model.output()
            print(f"Ingresso: {model.input().name} shape={tuple(model.input().shape)}")
            print(f"Uscita:   {output.name} shape={tuple(output.shape)} "
                  f"nms={getattr(output, 'is_nms', '?')} nms_shape={_describe(getattr(output, 'nms_shape', None))}")
            print("Allocazioni provate dal rilevatore:", ", ".join(label for label, _ in output_buffers(output)))
    except Exception:                         # noqa: BLE001
        print("\nERRORE nell'apertura della scheda:")
        traceback.print_exc()
        return 1

    from app.ml.detect import DetectorError
    from app.ml.detect_hailo import HailoPersonDetector

    try:
        detector = HailoPersonDetector(status.model, alternatives=status.models,
                                       conf_threshold=settings.detector_conf_weak)
    except DetectorError:
        print("\nERRORE nell'inferenza di prova:")
        traceback.print_exc()
        return 1
    print(f"\nInferenza di prova riuscita: {detector.name}")

    if len(sys.argv) > 1:
        _compare(sys.argv[1], detector, settings)
    detector.close()
    print("\nTutto a posto: la scheda rileva.")
    return 0


def _compare(video: str, detector, settings) -> None:
    """People found in one frame a minute in, on the board and on the CPU."""
    import cv2

    from app.ml.detect import PersonDetector

    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_MSEC, 60_000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        print(f"Video non leggibile: {video}")
        return
    on_board = detector.detect(frame)
    print(f"Scheda:     {len(on_board)} persone "
          f"(confidenze {', '.join(f'{d.confidence:.2f}' for d in on_board[:8])})")
    try:
        cpu = PersonDetector(settings.detector_model, imgsz=settings.detector_imgsz,
                             conf_threshold=settings.detector_conf_weak).detect(frame)
        print(f"Processore: {len(cpu)} persone "
              f"(confidenze {', '.join(f'{d.confidence:.2f}' for d in cpu[:8])})")
    except Exception as exc:                  # noqa: BLE001
        print(f"Processore non confrontato: {exc}")


def _describe(value) -> str:
    if value is None:
        return "-"
    fields = {name: getattr(value, name) for name in dir(value)
              if not name.startswith("_") and not callable(getattr(value, name, None))}
    return str(fields or value)


if __name__ == "__main__":
    raise SystemExit(main())
