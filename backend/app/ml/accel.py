"""Is an AI accelerator fitted, and can the analysis use it?

The Raspberry Pi AI HAT+ (Hailo-8, 26 TOPS; Hailo-8L, 13 TOPS) sits on the
Pi 5's PCIe slot. With it the person detector runs on the accelerator: a
larger network (YOLOv8s instead of the nano), at a higher resolution, in a
fraction of the CPU time. Without it, or when anything about it is missing,
the analysis runs on the CPU exactly as before.

Three things must hold, checked in this order so the reason given is the
first one missing:

  1. the device node, /dev/hailo0: the board is fitted, the host driver is
     loaded and Docker passes the device to the container;
  2. the HailoRT Python runtime (hailo_platform) in the image;
  3. a compiled network (.hef) for the board.

`hailo_status` checks them without opening the device: the API asks while
the worker may be using it, and a Hailo device serves one process. The
worker then opens it for real (app.ml.detect_hailo); if that fails the
analysis falls back to the CPU and says why.

ACCELERATOR=cpu turns the accelerator off; ACCELERATOR=hailo makes its
absence an error instead of a fallback.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("padel.accel")

# Where the device appears once the driver is loaded.
DEVICE_GLOB = "/dev/hailo*"
# Networks looked for, in order, when HAILO_MODEL is not set: one placed in
# weights/ by hand first, then the ones the hailo-models package installs
# (raspberrypi/rpicam-apps uses the same files). A network is compiled for
# one chip: *_h8.hef for the Hailo-8 (26 TOPS), *_h8l.hef for the Hailo-8L
# (13 TOPS). The one for the chip found is taken (pick_model); an _h8l
# network also runs on a Hailo-8, slower, the reverse does not.
DEFAULT_MODELS = (
    "/srv/weights/yolov8m_h8.hef",
    "/srv/weights/yolov8s_h8.hef",
    "/usr/share/hailo-models/yolov8s_h8.hef",
)
FALLBACK_MODELS = (
    "/srv/weights/yolov8s_h8l.hef",
    "/usr/share/hailo-models/yolov8s_h8l.hef",
)

MODES = ("auto", "cpu", "hailo")


class AcceleratorError(RuntimeError):
    """ACCELERATOR=hailo and the accelerator cannot be used."""


@dataclass(frozen=True)
class HailoStatus:
    usable: bool
    # Why not, in Italian, for the reliability panel and /api/health; or what
    # is in use.
    reason: str
    device: str | None = None
    model: str | None = None
    # Every network found, in order of preference: the chip, known only once
    # the device is open, chooses among them (pick_model).
    models: tuple[str, ...] = ()


def accelerator_mode(value: str | None) -> str:
    mode = (value or "auto").strip().lower()
    return mode if mode in MODES else "auto"


def hailo_status(mode: str = "auto", model: str | None = None) -> HailoStatus:
    """Whether the analysis can run its detector on a Hailo accelerator."""
    mode = accelerator_mode(mode)
    if mode == "cpu":
        return HailoStatus(False, "disattivato (ACCELERATOR=cpu)")
    devices = sorted(glob.glob(DEVICE_GLOB))
    if not devices:
        return HailoStatus(False, "non rilevato")
    if importlib.util.find_spec("hailo_platform") is None:
        return HailoStatus(False, f"{devices[0]} presente ma runtime HailoRT assente nell'immagine",
                           device=devices[0])
    found = _models(model)
    if not found:
        wanted = model or "yolov8s_h8.hef"
        return HailoStatus(False, f"{devices[0]} presente ma modello {Path(wanted).name} non trovato",
                           device=devices[0])
    return HailoStatus(True, f"Hailo · {Path(found[0]).name}", device=devices[0], model=found[0],
                       models=found)


def _models(configured: str | None) -> tuple[str, ...]:
    if configured:
        return (configured,) if Path(configured).is_file() else ()
    return tuple(c for c in DEFAULT_MODELS + FALLBACK_MODELS if Path(c).is_file())


def pick_model(models: list[str] | tuple[str, ...], chip: str) -> str:
    """The network for `chip` ("Hailo-8", "Hailo-8L"): one compiled for it
    first. A Hailo-8L cannot run a network compiled for the Hailo-8."""
    def for_8l(path: str) -> bool:
        return Path(path).stem.lower().endswith("h8l")

    if chip == "Hailo-8L":
        preferred = [m for m in models if for_8l(m)]
    elif chip == "Hailo-8":
        preferred = [m for m in models if not for_8l(m)]
    else:
        preferred = []
    return (preferred or list(models))[0]


# What the worker found when it started, for /api/health: the API container
# is not given the device, the worker is.
STATUS_FILE = "accelerator.json"


def probe_and_record(mode: str, model: str | None, data_dir: Path) -> HailoStatus:
    """At worker start: check the accelerator, open it for real (the board
    answers, the network loads), and record the outcome in `data_dir`."""
    status = hailo_status(mode, model)
    if status.usable:
        from app.ml.detect import DetectorError
        from app.ml.detect_hailo import HailoPersonDetector

        try:
            detector = HailoPersonDetector(status.model, alternatives=status.models)
        except DetectorError as exc:
            status = HailoStatus(False, str(exc), device=status.device, model=status.model)
        else:
            status = HailoStatus(True, detector.name, device=status.device, model=status.model)
            detector.close()
    logger.info("Acceleratore: %s", status.reason)
    try:
        record = {**asdict(status), "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (Path(data_dir) / STATUS_FILE).write_text(json.dumps(record))
    except OSError as exc:
        logger.warning("Stato dell'acceleratore non salvato: %s", exc)
    return status


def recorded_status(data_dir: Path) -> dict | None:
    try:
        return json.loads((Path(data_dir) / STATUS_FILE).read_text())
    except (OSError, ValueError):
        return None
