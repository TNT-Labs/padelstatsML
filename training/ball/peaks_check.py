#!/usr/bin/env python3
"""How much of the ball the weaker peaks give back, on the marked frames.

    python peaks_check.py ball.pt palla_*.zip [--holdout palla_terzo.zip]

No training: the network already trained (ball.pt) is run over the
validation clips — the same as train_tracknet.py with the same zips, seed
and --holdout — and measured three ways against the marks:

1. per frame, as the analysis read the map until now: the largest blob over
   0.5 (precision, recall, F1 within 4 px at 512, as train_tracknet.py);
2. per frame, how often the ball is among the weaker peaks when the blob is
   not it, at each threshold — and how many weaker peaks a frame has (noise);
3. per clip, the flights (backend app/ml/ball_track.py) without and with the
   weaker peaks (ball_track.recover): the ball the statistics actually use.

Needs, beside it, train_tracknet.py, and ball.py and ball_track.py from
backend/app/ml (in the repository they are found by themselves; on Colab
upload them too).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# In the repository ball.py and ball_track.py are in backend/app/ml; on
# Colab, uploaded beside this script (/content has no parent to look in).
REPO_ML = HERE.parent.parent / "backend" / "app" / "ml"
if (REPO_ML / "ball_track.py").exists():
    sys.path.insert(0, str(REPO_ML))
try:
    import ball as ballmod  # noqa: E402
    import ball_track  # noqa: E402
except ImportError as exc:
    raise SystemExit(f"Manca {exc.name}.py: caricalo accanto a peaks_check.py (è in backend/app/ml).") from exc
import train_tracknet as tt  # noqa: E402

LOWS = (0.1, 0.15, 0.2, 0.3)


def clips(zips: list[str]):
    """Per clip (zip, clip): its runs of consecutive frames, each frame
    (file key, visible None/True/False, x, y at 512 x 288)."""
    archives = {z: tt._open(z) for z in zips}
    rows = defaultdict(list)
    for z, archive in archives.items():
        width, height = json.loads(archive.read("meta.json"))["size"]
        sx, sy = tt.WIDTH / width, tt.HEIGHT / height
        for row in csv.DictReader(io.StringIO(archive.read("labels.csv").decode())):
            visible = None if row["visible"] == "" else row["visible"] == "1"
            x = float(row["x"]) * sx if visible else 0.0
            y = float(row["y"]) * sy if visible else 0.0
            rows[(z, row["clip"])].append((int(row["frame"]), (z, row["file"]), visible, x, y))
    out = {}
    for key, rs in rows.items():
        rs.sort(key=lambda r: r[0])
        runs, run = [], [rs[0]]
        for r in rs[1:]:
            if r[0] == run[-1][0] + 1:
                run.append(r)
            else:
                runs.append(run)
                run = [r]
        runs.append(run)
        out[key] = runs
    return out, archives


def flights_score(runs_seen, how, near) -> dict:
    """The flights of every run against the marks: with `how` (a
    ball_track.Recovery) the weaker peaks followed, None without them; and
    the points the recovery added, right, misplaced or on a hidden ball."""
    c = {"tp": 0, "fp": 0, "fn": 0, "added_ok": 0, "added_wrong": 0, "added_hidden": 0}
    for found, weak, marks in runs_seen:
        def points(flights):
            out = {}
            for f in flights:
                out.update({fr: (x, y) for fr, x, y in zip(f.frames, f.xs, f.ys)})
            return out
        plain = points(ball_track.flights(found, tt.HEIGHT))
        mine = plain if how is None else points(ball_track.flights(found, tt.HEIGHT, weak, how))
        for n, (visible, bx, by) in marks.items():
            p = mine.get(n)
            if visible and near(p, bx, by):
                c["tp"] += 1
            elif p is not None:
                c["fp"] += 1
                c["fn"] += bool(visible)
            elif visible:
                c["fn"] += 1
            if p is not None and n not in plain:
                key = "added_hidden" if not visible else "added_ok" if near(p, bx, by) else "added_wrong"
                c[key] += 1
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="ball.pt")
    parser.add_argument("zips", nargs="+")
    parser.add_argument("--holdout", metavar="ZIP")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()

    all_clips, archives = clips(args.zips)
    random.seed(args.seed)
    _, val_keys = tt.split(list(all_clips), args.holdout, random)
    cache: dict = {}

    def image(key):
        if key not in cache:
            z, name = key
            data = np.frombuffer(archives[z].read(name), np.uint8)
            cache[key] = cv2.resize(cv2.imdecode(data, cv2.IMREAD_COLOR), (tt.WIDTH, tt.HEIGHT),
                                    interpolation=cv2.INTER_AREA)
        return cache[key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tt.TrackNetLite(args.width).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    frame = {"tp": 0, "fp": 0, "fn": 0}
    among = {low: 0 for low in LOWS}
    weak_count = {low: 0 for low in LOWS}
    missed = frames_seen = 0
    runs_seen = []                  # per run: (found, weak, marks), for the flights after

    def near(point, x, y):
        return point is not None and np.hypot(point[0] - x, point[1] - y) <= tt.HIT_PX

    for key in val_keys:
        for run in all_clips[key]:
            if len(run) < 3:
                continue
            found, weak, marks = [], {}, {}
            for i in range(2, len(run)):
                stack = np.concatenate([image(run[j][1]) for j in (i - 2, i - 1, i)], axis=2)
                x9 = torch.from_numpy(np.ascontiguousarray(stack.transpose(2, 0, 1))).float()[None].to(device)
                with torch.no_grad():
                    heat = model(x9)[0].cpu().numpy()
                n, (_, _, visible, bx, by) = run[i][0], run[i]
                top = ballmod.locate(heat)
                if top is not None:
                    found.append((n, n / 30, top[0], top[1], top[2]))
                peaks = ballmod.weak_peaks(heat, top[:2] if top else None, threshold=min(LOWS), count=8)
                kept = [p for p in peaks if p[2] >= ballmod.WEAK_THRESHOLD][:ballmod.WEAK_MAX]
                if kept:
                    weak[n] = (n / 30, kept)
                if visible is None:
                    continue
                marks[n] = (visible, bx, by)
                frames_seen += 1
                for low in LOWS:
                    weak_count[low] += sum(p[2] >= low for p in peaks[:ballmod.WEAK_MAX])
                if visible:
                    if near(top, bx, by):
                        frame["tp"] += 1
                    else:
                        frame["fp" if top is not None else "fn"] += 1
                        missed += 1
                        for low in LOWS:
                            among[low] += any(near(p, bx, by) for p in peaks[:ballmod.WEAK_MAX] if p[2] >= low)
                elif top is not None:
                    frame["fp"] += 1
            runs_seen.append((found, weak, marks))

    def prf(c):
        p = c["tp"] / max(1, c["tp"] + c["fp"])
        r = c["tp"] / max(1, c["tp"] + c["fn"])
        return f"precisione {p:.3f} · richiamo {r:.3f} · F1 {2 * p * r / max(1e-9, p + r):.3f}"

    print(f"Validazione: {len(val_keys)} spezzoni, {frames_seen} fotogrammi segnati"
          + (f" · partita {args.holdout}, mai vista se ball.pt è addestrato con --holdout" if args.holdout else
             " · spezzoni a caso: ottimista"))
    print(f"1. fotogramma per fotogramma, come finora: {prf(frame)}")
    print(f"2. palla visibile ma non presa ({missed}): è fra i picchi deboli (al più {ballmod.WEAK_MAX})")
    for low in LOWS:
        print(f"   sopra {low:.2f}: {among[low]} ({among[low] / max(1, missed):.0%}) · "
              f"picchi deboli per fotogramma {weak_count[low] / max(1, frames_seen):.2f}")
    base = flights_score(runs_seen, None, near)
    print(f"3. traiettorie, senza picchi deboli: {prf(base)}")
    print("   con i picchi deboli: dove · soglia · tolleranza → traiettorie · punti aggiunti giusti / "
          "nel posto sbagliato / dove la palla era coperta")
    for where, gaps, ends in (("buchi", True, False), ("capi", False, True), ("buchi+capi", True, True)):
        for low in (0.15, 0.2, 0.3):
            for tol in (1.0, 0.5):
                how = ball_track.Recovery(min_peak=low, gaps=gaps, ends=ends, tolerance=tol)
                c = flights_score(runs_seen, how, near)
                print(f"   {where:10s} · {low:.2f} · {tol:.1f} → {prf(c)} · "
                      f"{c['added_ok']} / {c['added_wrong']} / {c['added_hidden']}")


if __name__ == "__main__":
    main()
