#!/usr/bin/env python3
"""Where are the detections no player could keep, and what are they?

    python scripts/diagnose_extra.py latest

When identity warns that a share of the detections is contested ("conteso
fra due tracce dello stesso giocatore"), somebody besides the four players
is being detected on court: in those frames a player has two detections
and both are dropped. This finds where those detections stand, on the
court and in the picture, and writes a contact sheet of them cut from the
video — so one look says whether they are a person behind the glass, a
player of the next court, a banner, or a real player in a crowded frame.

The image goes in the data folder (diagnosi_<id>.jpg), visible from the
host where DATA_VOLUME points.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core.config import get_settings                     # noqa: E402
from app.core.storage import video_path                      # noqa: E402
from app.ml.artifacts import load_artifacts, resolve_artifacts_dir  # noqa: E402
from app.ml.identity import drop_static                      # noqa: E402
from app.ml.pipeline import PipelineConfig, analyse_tracklets  # noqa: E402
from app.ml.tracking import observation_key                  # noqa: E402

CELL_M = 1.0
TOP_SPOTS = 6
SAMPLES_PER_SPOT = 4
TILE = 220


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match_id", help="id della partita, un suo prefisso, oppure 'latest'")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--video", default=None, help="percorso video alternativo")
    args = parser.parse_args()
    settings = get_settings()

    try:
        directory = Path(args.artifacts) if args.artifacts else resolve_artifacts_dir(
            args.match_id, settings.artifacts_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    artifacts = load_artifacts(directory)
    config = PipelineConfig(
        detector_model=settings.detector_model,
        max_player_speed_ms=settings.max_player_speed_ms,
        rally_speed_threshold_ms=settings.rally_speed_threshold_ms,
        rally_min_duration_s=settings.rally_min_duration_s,
        rally_merge_gap_s=settings.rally_merge_gap_s,
    )
    _, identity, _ = analyse_tracklets(
        tracklets=artifacts.tracklets, calibration=artifacts.calibration, config=config,
        sample_hz=artifacts.sample_hz, frames_sampled=artifacts.frames_sampled,
        analysed_s=artifacts.analysed_s,
    )
    for warning in identity.warnings:
        print(f"! {warning}")

    usable = [t for t in artifacts.tracklets if len(t) >= 3]
    kept, _, _ = drop_static(usable)
    assigned = {observation_key(o) for p in identity.players for o in p.observations}
    candidates = [o for t in kept for o in t.observations]
    lost = [o for o in candidates if observation_key(o) not in assigned]
    total = len(candidates)
    print(f"\nRilevazioni (oggetti fermi esclusi): {total} · non attribuite: {len(lost)} "
          f"({100 * len(lost) / max(total, 1):.0f}%)")
    if not lost:
        return 0

    _bands(lost)
    spots = _spots(lost, [o for o in candidates if observation_key(o) in assigned])

    video = Path(args.video) if args.video else video_path(directory.name)
    if not video.exists():
        print(f"\nVideo non trovato ({video}): niente immagine.", file=sys.stderr)
        return 0
    out = settings.data_path / f"diagnosi_{directory.name[:8]}.jpg"
    if _contact_sheet(video, spots, candidates, assigned, out):
        print(f"\nImmagine: {out}")
    return 0


def _bands(lost) -> None:
    ys = np.array([o.foot_court[1] for o in lost])
    xs = np.array([o.foot_court[0] for o in lost])
    print("\nPer profondità (Y, 0 = fondo vicino, 10 = rete, 20 = fondo lontano):")
    for lo, hi, label in ((-99, 0, "dietro il fondo vicino"), (0, 10, "metà vicina"),
                          (10, 18, "metà lontana"), (18, 20, "ultimi 2 m lontani"),
                          (20, 99, "oltre il fondo lontano")):
        share = float(((ys >= lo) & (ys < hi)).mean())
        print(f"  {label:<28}{share:>6.0%}")
    outside = float(((xs < 0) | (xs > 10)).mean())
    print(f"  {'fuori dalle linee laterali':<28}{outside:>6.0%}")


def _spots(lost, assigned_obs):
    cells: dict[tuple[int, int], list] = defaultdict(list)
    for obs in lost:
        x, y = obs.foot_court
        cells[(int(np.floor(x / CELL_M)), int(np.floor(y / CELL_M)))].append(obs)
    heights_by_row: dict[int, list[float]] = defaultdict(list)
    for obs in assigned_obs:
        heights_by_row[int(obs.foot_px[1] // 20)].append(obs.bbox[3] - obs.bbox[1])

    ranked = sorted(cells.items(), key=lambda kv: -len(kv[1]))[:TOP_SPOTS]
    print(f"\nZone con più rilevazioni non attribuite (celle di {CELL_M:.0f} m):")
    print(f"  {'cella X,Y (m)':<15}{'n':>6}{'da':>8}{'a':>8}{'piede px':>14}{'alt. px':>9}{'alt. gioc.':>11}")
    for (cx, cy), group in ranked:
        times = [o.timestamp_s for o in group]
        foot = np.median([o.foot_px for o in group], axis=0)
        height = float(np.median([o.bbox[3] - o.bbox[1] for o in group]))
        typical = heights_by_row.get(int(foot[1] // 20))
        ref = f"{np.median(typical):.0f}" if typical else "-"
        print(f"  {f'{cx},{cy}':<15}{len(group):>6}{_clock(min(times)):>8}{_clock(max(times)):>8}"
              f"{f'{foot[0]:.0f},{foot[1]:.0f}':>14}{height:>9.0f}{ref:>11}")
    print("  (alt. gioc.: altezza mediana dei riquadri attribuiti ai giocatori alla stessa altezza "
          "nell'immagine; molto più alta di 'alt. px' = la rilevazione è più lontana, dietro il vetro)")
    return ranked


def _contact_sheet(video: Path, spots, candidates, assigned, out: Path) -> bool:
    """One row per spot: SAMPLES_PER_SPOT crops around the detection, then
    the whole frame of the first sample, every detection drawn (green
    given to a player, red not)."""
    by_frame: dict[int, list] = defaultdict(list)
    for obs in candidates:
        by_frame[obs.frame_index].append(obs)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print("Video illeggibile.", file=sys.stderr)
        return False
    rows = []
    for (cx, cy), group in spots:
        group = sorted(group, key=lambda o: o.timestamp_s)
        picks = [group[int(i)] for i in np.linspace(0, len(group) - 1, SAMPLES_PER_SPOT)]
        tiles = []
        overview = None
        for n, obs in enumerate(picks):
            capture.set(cv2.CAP_PROP_POS_FRAMES, obs.frame_index)
            ok, frame = capture.read()
            if not ok:
                tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
                continue
            tiles.append(_crop(frame, obs, f"{cx},{cy} m  {_clock(obs.timestamp_s)}"))
            if n == 0:
                overview = _overview(frame, by_frame[obs.frame_index], assigned)
        rows.append(np.hstack(tiles + [overview if overview is not None else np.zeros((TILE, TILE * 16 // 9, 3), np.uint8)]))
    capture.release()
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def _crop(frame, obs, label: str):
    x1, y1, x2, y2 = obs.bbox
    h, w = frame.shape[:2]
    side = max(3.0 * (y2 - y1), 120.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    view = frame.copy()
    cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    tile = cv2.resize(view[b:d, a:c], (TILE, TILE))
    cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def _overview(frame, detections, assigned):
    view = frame.copy()
    for obs in detections:
        colour = (0, 200, 0) if observation_key(obs) in assigned else (0, 0, 255)
        x1, y1, x2, y2 = (int(v) for v in obs.bbox)
        cv2.rectangle(view, (x1, y1), (x2, y2), colour, 3)
    return cv2.resize(view, (TILE * 16 // 9, TILE))


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


if __name__ == "__main__":
    raise SystemExit(main())
