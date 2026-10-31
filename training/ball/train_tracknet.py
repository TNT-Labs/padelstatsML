#!/usr/bin/env python3
"""Train a ball detector on the frames marked in the app (Annota palla).

    python train_tracknet.py palla_1.zip palla_2.zip --epochs 30 --out ball

TrackNet-style: three consecutive frames in (9 channels, 288 x 512), a heat
map of where the ball is in the last one out. The ball is a few pixels and a
smear; its movement over three frames is what tells it from a white sock or
a light in the roof. A small U-Net (TrackNetLite) — channels chosen to run at
30 frames a second on a Hailo-8L once compiled (see README.md).

Data: the zips the app exports (labels.csv, frames/<clip>/<n>.jpg at 640 px).
Each frame marked — ball at x, y or not in sight — is a sample with the two
frames before it; frames not marked only serve as those. Clips go whole to
training or to validation (VAL_SHARE), never split, or the score would be
measured on frames the model has nearly seen.

Out: <out>.pt (the weights) and <out>.onnx (for the Hailo compiler: frames
scaled 0-1, the chip scales them, see ball.alls), plus
the validation score: precision and recall of the ball within 4 px at 512.
Runs on Google Colab's free GPU in minutes for a few thousand frames.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH, HEIGHT = 512, 288
# The target: a disc of 1s this wide where the ball is (TrackNetV2), so the
# network learns to answer 1 there and the 0.5 threshold finds it.
RADIUS = 3.0           # pixels at 512
VAL_SHARE = 0.15
HIT_PX = 4.0           # a prediction this close to the mark is right
THRESHOLD = 0.5


# ── Data ─────────────────────────────────────────────────────────────────────

class Frames(torch.utils.data.Dataset):
    """(9 x 288 x 512 frames 0-255, 288 x 512 heat map, visible, x, y)."""

    def __init__(self, samples, images, augment: bool) -> None:
        self.samples, self.images, self.augment = samples, images, augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        files, visible, x, y = self.samples[i]
        stack = [self.images(f) for f in files]                      # 3 x H x W x 3, BGR
        if self.augment and random.random() < 0.5:
            stack = [im[:, ::-1] for im in stack]
            x = WIDTH - 1 - x if visible else x
        frames = np.concatenate(stack, axis=2).astype(np.float32)   # H x W x 9
        if self.augment:
            frames = np.clip(frames * random.uniform(0.8, 1.2) + random.uniform(-15, 15), 0, 255)
        heat = _heat(x, y) if visible else np.zeros((HEIGHT, WIDTH), np.float32)
        return (torch.from_numpy(np.ascontiguousarray(frames.transpose(2, 0, 1))),
                torch.from_numpy(heat), torch.tensor(float(visible)), torch.tensor([x, y], dtype=torch.float32))


def _heat(x: float, y: float) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    return (((xs - x) ** 2 + (ys - y) ** 2) <= RADIUS ** 2).astype(np.float32)


def _open(path: str) -> zipfile.ZipFile:
    """An exported zip, or why the file is not one."""
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"{path}: file non trovato.")
    if zipfile.is_zipfile(file):
        return zipfile.ZipFile(file)
    size = file.stat().st_size
    head = file.read_bytes()[:300]
    if head.startswith(b"PK"):
        why = ("lo zip è incompleto: il caricamento (barra in basso a sinistra in Colab) "
               "o il download non era finito. Ricaricalo e aspetta la fine.")
    elif head.lstrip().startswith((b"{", b"<")):
        why = ("non è lo zip ma una risposta del server: "
               + head.decode("utf-8", "replace").strip()[:200])
    else:
        why = "il file non è uno zip esportato da Annota palla."
    raise SystemExit(f"{path} ({size / 1e6:.1f} MB): {why}")


def load(zips: list[str]):
    """Samples per clip and an image reader over the zips."""
    archives = {z: _open(z) for z in zips}
    cache: dict[tuple[str, str], np.ndarray] = {}

    def images(key):
        if key not in cache:
            z, name = key
            data = np.frombuffer(archives[z].read(name), np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            cache[key] = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        return cache[key]

    clips = defaultdict(list)
    scales = {}
    for z, archive in archives.items():
        # The frames' size in the zip (640 x 360 for a 16:9 video): the marks
        # are in its pixels, the network sees 512 x 288.
        width, height = json.loads(archive.read("meta.json"))["size"]
        scales[z] = (WIDTH / width, HEIGHT / height)
        for row in csv.DictReader(io.StringIO(archive.read("labels.csv").decode())):
            clips[(z, row["clip"])].append((int(row["frame"]), (z, row["file"]), row["visible"],
                                            row["x"], row["y"]))
    samples = {}
    for key, rows in clips.items():
        rows.sort(key=lambda r: r[0])
        sx, sy = scales[key[0]]
        out = []
        for i in range(2, len(rows)):
            if rows[i][0] - rows[i - 2][0] != 2 or rows[i][2] == "":
                continue                                              # a gap, or not marked
            visible = rows[i][2] == "1"
            x = float(rows[i][3]) * sx if visible else 0.0
            y = float(rows[i][4]) * sy if visible else 0.0
            out.append(([rows[i - 2][1], rows[i - 1][1], rows[i][1]], visible, x, y))
        if out:
            samples[key] = out
    return samples, images


# ── Model ────────────────────────────────────────────────────────────────────

def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class TrackNetLite(nn.Module):
    """U-Net, 9 channels 0-255 in, ball heat map 0-1 out. Only layers the
    Hailo compiler takes: conv, batch norm, ReLU, max pool, nearest
    upsampling, concat, sigmoid."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        w = width
        self.e1, self.e2, self.e3, self.e4 = block(9, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w)
        self.d3, self.d2, self.d1 = block(12 * w, 4 * w), block(6 * w, 2 * w), block(3 * w, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        return self.net(x / 255.0)[:, 0]

    def net(self, x):
        """On 0-1 frames: the network the Hailo runs, its input scaled on
        the chip (normalization in ball.alls)."""
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(e4, scale_factor=2.0, mode="nearest"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2.0, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2.0, mode="nearest"), e1], 1))
        return torch.sigmoid(self.head(d1))          # N x 1 x H x W


def weighted_bce(pred, heat):
    """TrackNetV2's: the few ball pixels weigh as much as the whole court."""
    pred = pred.clamp(1e-6, 1 - 1e-6)
    return -((1 - pred) ** 2 * heat * torch.log(pred) + pred ** 2 * (1 - heat) * torch.log(1 - pred)).mean()


def locate(heat: np.ndarray) -> tuple[float, float] | None:
    """Where the ball is in a heat map: the centre of its brightest blob."""
    if heat.max() < THRESHOLD:
        return None
    mask = (heat >= THRESHOLD).astype(np.uint8)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(centres[best][0]), float(centres[best][1])


def score(model, loader, device) -> dict:
    """Precision and recall of the ball, TrackNet's way."""
    tp = fp = fn = tn = 0
    model.eval()
    with torch.no_grad():
        for frames, _, visible, xy in loader:
            heat = model(frames.to(device)).cpu().numpy()
            for h, v, p in zip(heat, visible.numpy(), xy.numpy()):
                found = locate(h)
                if v:
                    if found is not None and np.hypot(found[0] - p[0], found[1] - p[1]) <= HIT_PX:
                        tp += 1
                    elif found is not None:
                        fp += 1
                    else:
                        fn += 1
                else:
                    fp += found is not None
                    tn += found is None
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / max(1e-9, precision + recall), 3),
            "samples": tp + fp + fn + tn}


class _OnChip(nn.Module):
    """The exported graph: 0-1 frames in, without the /255 (the chip does it)."""

    def __init__(self, model: TrackNetLite) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.net(x)


def export_onnx(model: TrackNetLite, path: str) -> None:
    torch.onnx.export(_OnChip(model), torch.zeros(1, 9, HEIGHT, WIDTH), path, input_names=["frames"],
                      output_names=["heat"], opset_version=13, dynamo=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zips", nargs="+")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32, help="canali del primo livello")
    parser.add_argument("--out", default="ball")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples, images = load(args.zips)
    keys = sorted(samples)
    random.shuffle(keys)
    n_val = max(1, int(round(len(keys) * VAL_SHARE))) if len(keys) > 1 else 0
    val = [s for k in keys[:n_val] for s in samples[k]]
    train = [s for k in keys[n_val:] for s in samples[k]]
    print(f"Spezzoni {len(keys)} · campioni addestramento {len(train)} · validazione {len(val)}")
    if not train:
        raise SystemExit("Nessun fotogramma annotato con i due precedenti: annota spezzoni interi.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrackNetLite(args.width).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs)
    train_loader = torch.utils.data.DataLoader(Frames(train, images, True), batch_size=args.batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(Frames(val, images, False), batch_size=args.batch) if val else None
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for frames, heat, _, _ in train_loader:
            optimiser.zero_grad()
            loss = weighted_bce(model(frames.to(device)), heat.to(device))
            loss.backward()
            optimiser.step()
            total += float(loss) * len(frames)
        schedule.step()
        result = score(model, val_loader, device) if val_loader else {"f1": 0.0}
        print(f"epoca {epoch + 1}/{args.epochs} · perdita {total / len(train):.4f} · validazione {result}")
        if result["f1"] >= best:
            best = result["f1"]
            torch.save(model.state_dict(), f"{args.out}.pt")

    model.load_state_dict(torch.load(f"{args.out}.pt", map_location=device))
    export_onnx(model.eval().cpu(), f"{args.out}.onnx")
    print(f"Salvati {args.out}.pt e {args.out}.onnx · miglior F1 in validazione {best}")


if __name__ == "__main__":
    main()
