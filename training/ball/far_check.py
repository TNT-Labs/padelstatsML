#!/usr/bin/env python3
"""The second look at the far half, measured on a match the network never saw.

    python far_check.py ball.pt palla_*.zip lontana_*.zip --match 3628b11c

ball.pt trained with --holdout on that match (train_tracknet.py), the zips
of every match as exported ("Scarica i dati annotati" and "Scarica la metà
lontana"); --match names the match left out (its id, as in the zips' names).

On every marked frame of that match the network looks at the whole frame
and at the far crop, as the analysis would with BALL_FAR=on, and the ball
is measured within 15 video pixels of the mark (4 at the network's 512 on
the whole frame, as train_tracknet.py) — per frame and as flights
(backend app/ml/ball_track.py), split between the far crop and the rest of
the picture:

- whole frame only: the analysis as it is;
- with the far half, merged by each rule of ball_track.FarMerge.

Needs, beside it, train_tracknet.py and ball.py and ball_track.py from
backend/app/ml (found by themselves in the repository).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ML = HERE.parent.parent / "backend" / "app" / "ml"
if (REPO_ML / "ball_track.py").exists():
    sys.path.insert(0, str(REPO_ML))
try:
    import ball as ballmod  # noqa: E402
    import ball_track  # noqa: E402
except ImportError as exc:
    raise SystemExit(f"Manca {exc.name}.py: caricalo accanto a far_check.py (è in backend/app/ml).") from exc
import train_tracknet as tt  # noqa: E402

NEAR_PX = 15.0


def read_zip(path: str):
    archive = tt._open(path)
    meta = json.loads(archive.read("meta.json"))
    rows = {}
    for row in csv.DictReader(io.StringIO(archive.read("labels.csv").decode())):
        rows[(row["clip"], int(row["frame"]))] = row
    return archive, meta, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("zips", nargs="+")
    parser.add_argument("--match", required=True, help="l'id della partita esclusa dall'addestramento")
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()

    mine = [z for z in args.zips if args.match in Path(z).name]
    whole = [z for z in mine if json.loads(tt._open(z).read("meta.json")).get("kind") != "far"]
    far = [z for z in mine if json.loads(tt._open(z).read("meta.json")).get("kind") == "far"]
    if len(whole) != 1 or len(far) != 1:
        raise SystemExit(f"--match {args.match}: servono uno zip intero e uno della metà lontana di quella "
                         f"partita, trovati {len(whole)} e {len(far)}.")
    wa, wmeta, wrows = read_zip(whole[0])
    fa, fmeta, frows = read_zip(far[0])
    src_w, src_h = wmeta.get("source_size") or [1920, 1080]
    crop = (*fmeta["offset"], *fmeta["size"])
    sx_whole = src_w / tt.WIDTH
    sy_whole = src_h / tt.HEIGHT
    to_video = src_w / wmeta["size"][0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tt.TrackNetLite(args.width).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    cache: dict = {}

    def image(archive, name):
        key = (id(archive), name)
        if key not in cache:
            data = np.frombuffer(archive.read(name), np.uint8)
            cache[key] = cv2.resize(cv2.imdecode(data, cv2.IMREAD_COLOR), (tt.WIDTH, tt.HEIGHT),
                                    interpolation=cv2.INTER_AREA)
        return cache[key]

    def heat(frames):
        x9 = torch.from_numpy(np.ascontiguousarray(np.concatenate(frames, axis=2).transpose(2, 0, 1))).float()
        with torch.no_grad():
            return model(x9[None].to(device))[0].cpu().numpy()

    clips = defaultdict(list)
    for (clip, frame) in sorted(wrows):
        clips[clip].append(frame)
    runs = []          # per run of frames: (found whole, found far, marks)
    for clip, frames in clips.items():
        found, found_far, marks = [], [], {}
        for i, n in enumerate(frames):
            if i < 2 or frames[i - 2] != n - 2 or any((clip, n - k) not in frows for k in range(3)):
                continue
            row = wrows[(clip, n)]
            hw = heat([image(wa, wrows[(clip, n - k)]["file"]) for k in (2, 1, 0)])
            hf = heat([image(fa, frows[(clip, n - k)]["file"]) for k in (2, 1, 0)])
            a = ballmod.locate(hw)
            b = ballmod.locate(hf)
            t = n / 30.0
            if a:
                found.append((n, t, a[0] * sx_whole, a[1] * sy_whole, a[2]))
            if b:
                found_far.append((n, t, crop[0] + b[0] * crop[2] / tt.WIDTH, crop[1] + b[1] * crop[3] / tt.HEIGHT, b[2]))
            if row["visible"] != "":
                visible = row["visible"] == "1"
                marks[n] = (visible, float(row["x"] or 0) * to_video, float(row["y"] or 0) * to_video)
        runs.append((found, found_far, marks))

    def inside(x, y):
        return crop[0] <= x < crop[0] + crop[2] and crop[1] <= y < crop[1] + crop[3]

    def tally(points_of_run):
        c = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("lontana", "resto")}
        for points, (_, _, marks) in zip(points_of_run, runs):
            for n, (visible, mx, my) in marks.items():
                p = points.get(n)
                region = "lontana" if (inside(mx, my) if visible else (p is not None and inside(*p))) else "resto"
                if visible and p is not None and np.hypot(p[0] - mx, p[1] - my) <= NEAR_PX:
                    c[region]["tp"] += 1
                else:
                    if p is not None:
                        c["lontana" if inside(*p) else "resto"]["fp"] += 1
                    if visible:
                        c[region]["fn"] += 1
        return c

    def prf(c):
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        return f"precisione {p:.3f} · richiamo {r:.3f} · F1 {2 * p * r / max(1e-9, p + r):.3f}"

    def show(name, points_of_run):
        c = tally(points_of_run)
        print(f"  {name:38s} metà lontana: {prf(c['lontana'])}")
        print(f"  {'':38s} resto:        {prf(c['resto'])}")

    def per_frame(found):
        return {r[0]: (r[2], r[3]) for r in found}

    def flights(found):
        out = {}
        for f in ball_track.flights(found, src_h):
            out.update({fr: (x, y) for fr, x, y in zip(f.frames, f.xs, f.ys)})
        return out

    marked = sum(len(m) for _, _, m in runs)
    print(f"Partita {args.match}: {marked} fotogrammi segnati · ritaglio {crop} · palla entro {NEAR_PX:.0f} px")
    rules = [("unione, vince la metà lontana", ball_track.FarMerge("far")),
             ("unione, vince l'immagine intera", ball_track.FarMerge("full")),
             ("unione, solo se la metà lontana conferma", ball_track.FarMerge("far", True))]
    for kind, make in (("fotogramma per fotogramma", per_frame), ("traiettorie", flights)):
        print(f"\n{kind}:")
        show("solo immagine intera", [make(w) for w, _, _ in runs])
        for name, rule in rules:
            show(name, [make(ball_track.merge_far(w, f, crop, rule)) for w, f, _ in runs])


if __name__ == "__main__":
    main()
