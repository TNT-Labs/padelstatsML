#!/usr/bin/env python3
"""Export the person re-identification model (OSNet) to ONNX for the Pi runtime.

Run once, in the same virtualenv used for the detector export:

    python3 -m venv .export-venv && source .export-venv/bin/activate
    pip install -r backend/requirements.export.txt
    python backend/scripts/export_reid_onnx.py --out weights/osnet_x0_25_msmt17.onnx
    deactivate && rm -rf .export-venv

Why re-identification: identity used to rest on the colour of the kit, and
on a real match that could not tell the four players apart — team-mates in
the same kit, or four kits of similar dark and light tones. OSNet is trained
to tell people apart by appearance as a whole, and x0_25 is small enough for
a Cortex-A76: about 0.08 GFLOPs per crop, a few percent of the detector.

The checkpoint (trained on MSMT17) is downloaded from Google Drive on first
use. When Drive refuses — it rate-limits popular files — download it by hand
from the link printed below and pass it with --weights.

Only torch is needed to build the network: torchreid's `osnet.py` is loaded
on its own, without the package's `__init__`, which would otherwise pull in
scipy, torchvision, opencv and tensorboard for datasets that are never used.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

# Checkpoints trained on MSMT17, the largest and most varied public re-ID set.
# The same files torchreid and boxmot distribute.
CHECKPOINTS = {
    "osnet_x0_25": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    "osnet_x0_5": "https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv",
    "osnet_x1_0": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
}
# OSNet's training resolution: height x width.
INPUT_H, INPUT_W = 256, 128


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="osnet_x0_25", choices=sorted(CHECKPOINTS))
    parser.add_argument("--weights", default=None,
                        help="checkpoint .pt/.pth già scaricato (altrimenti lo scarica da Google Drive)")
    parser.add_argument("--out", default=None,
                        help="file .onnx da scrivere (predefinito: weights/<arch>_msmt17.onnx)")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("torch non installato. Esegui: pip install -r backend/requirements.export.txt", file=sys.stderr)
        return 1

    try:
        osnet = _load_osnet_module()
    except ImportError as exc:
        print(f"{exc}\nEsegui: pip install -r backend/requirements.export.txt", file=sys.stderr)
        return 1

    weights = Path(args.weights) if args.weights else _download(args.arch)
    if weights is None:
        return 1

    model = getattr(osnet, args.arch)(num_classes=1, pretrained=False, loss="softmax")
    try:
        _load_checkpoint(torch, model, weights)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    model.eval()

    out = Path(args.out or f"weights/{args.arch}_msmt17.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
    print(f"Esporto {args.arch} a ONNX ({INPUT_H}x{INPUT_W}, opset {args.opset})…")
    try:
        torch.onnx.export(
            model, dummy, str(out),
            input_names=["images"], output_names=["embeddings"],
            # The batch is the number of players detected in a frame: 1 to 6.
            dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=args.opset,
        )
    except PermissionError:
        print(
            f"Impossibile scrivere {out}: la cartella appartiene a un altro utente\n"
            f"(di solito root, se l'ha creata Docker). Correggi con:\n"
            f"  sudo chown -R $USER {out.parent}",
            file=sys.stderr,
        )
        return 1

    status = _verify(torch, model, out)
    if status == 0:
        if out.name == "osnet_x0_25_msmt17.onnx":
            print("È il modello predefinito: se il file è in weights/, non serve altro.")
        else:
            print(f"Per usarlo, nel .env: REID_MODEL=/srv/weights/{out.name}\n"
                  "poi docker compose up -d, e Rianalizza le partite già analizzate.")
    return status


def _load_osnet_module():
    """torchreid's osnet.py, loaded without executing the package."""
    spec = importlib.util.find_spec("torchreid")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("torchreid non installato.")
    path = Path(spec.submodule_search_locations[0]) / "reid" / "models" / "osnet.py"
    if not path.exists():
        raise ImportError(f"osnet.py non trovato in {path.parent}: versione di torchreid inattesa.")
    module_spec = importlib.util.spec_from_file_location("padel_osnet", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _download(arch: str) -> Path | None:
    url = CHECKPOINTS[arch]
    target = Path("weights") / f"{arch}_msmt17.pt"
    if target.exists():
        return target
    try:
        import gdown
    except ImportError:
        print("gdown non installato. Esegui: pip install -r backend/requirements.export.txt", file=sys.stderr)
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scarico i pesi {arch} (MSMT17)…")
    try:
        result = gdown.download(url, str(target), quiet=False)
    except Exception as exc:                     # noqa: BLE001 - gdown raises many types
        result, reason = None, exc
    else:
        reason = "download non riuscito"
    if not result or not target.exists():
        print(
            f"Google Drive non ha concesso il file ({reason}).\n"
            f"Scaricalo dal browser: {url}\n"
            f"poi riesegui con: --weights <file scaricato>",
            file=sys.stderr,
        )
        return None
    return target


def _load_checkpoint(torch, model, path: Path) -> None:
    """Load a torchreid checkpoint, whatever wrapper it was saved in."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"{path}: formato del checkpoint non riconosciuto.")

    own = model.state_dict()
    matched, skipped = {}, []
    for key, value in state.items():
        key = key[len("module."):] if key.startswith("module.") else key
        if key.startswith("classifier."):
            continue    # trained on MSMT17's 4,101 identities; not used for embeddings
        if key in own and own[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)

    missing = [k for k in own if k not in matched and not k.startswith("classifier.")]
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} parametri della rete mancano nel checkpoint "
            f"(es. {missing[0]}). Non è un checkpoint {type(model).__name__} compatibile."
        )
    model.load_state_dict(matched, strict=False)
    print(f"Caricati {len(matched)} tensori" + (f", ignorati {len(skipped)}" if skipped else ""))


def _verify(torch, model, out: Path) -> int:
    """Reload with onnxruntime, the way the Pi will, and compare with torch."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    batch = torch.randn(4, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        expected = model(batch).numpy()
    got = session.run(None, {session.get_inputs()[0].name: batch.numpy()})[0]
    error = float(np.abs(expected - got).max())
    if got.shape != expected.shape or error > 1e-3:
        print(f"Verifica fallita: forma {got.shape}, scarto massimo {error:.2e}", file=sys.stderr)
        return 1

    single = batch[:1].numpy()
    started = time.perf_counter()
    for _ in range(20):
        session.run(None, {session.get_inputs()[0].name: single})
    per_crop_ms = (time.perf_counter() - started) / 20 * 1000

    print(f"OK: {out} · embedding da {got.shape[1]} valori · scarto torch/onnx {error:.1e} · "
          f"{per_crop_ms:.1f} ms per giocatore su questa macchina")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
