#!/usr/bin/env python3
"""Try the ball network (weights/ball_h8l.hef) on a stretch of a match.

    docker exec padel-worker python scripts/ball_check.py [video or match id] [--start 60] [--seconds 20]

Prints whether the board loads the network, how fast it runs and in how
many frames it sees the ball, and writes /data/ball_check.mp4: the stretch
with the ball circled and its last half second of path. With no video, the
newest one uploaded. Run it while no analysis is in progress.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ml.ball import BALL_MODELS, HailoBallDetector, find_model, small  # noqa: E402

OUT_WIDTH = 960
TRAIL = 15


def _video(arg: str | None) -> Path | None:
    if arg and Path(arg).is_file():
        return Path(arg)
    videos = get_settings().videos_path
    if arg:
        path = videos / f"{arg}.mp4"
        return path if path.is_file() else None
    found = sorted(videos.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?")
    parser.add_argument("--start", type=float, default=60.0, help="secondo di partenza")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--model")
    parser.add_argument("--out", default=str(Path(get_settings().data_dir) / "ball_check.mp4"))
    args = parser.parse_args()

    model = find_model(args.model)
    if model is None:
        print(f"Rete della palla non trovata: copiala in weights/ ({', '.join(Path(m).name for m in BALL_MODELS)}).")
        return 1
    video = _video(args.video)
    if video is None:
        print("Video non trovato.")
        return 1

    try:
        detector = HailoBallDetector(model)
    except Exception as exc:              # noqa: BLE001
        print(f"La scheda non carica {Path(model).name}: {exc}")
        return 1
    print(f"Rete:    {detector.name} · ingresso {detector.input_shape}")

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    scale = OUT_WIDTH / width
    out_size = (OUT_WIDTH, int(round(height * scale)))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    print(f"Video:   {video.name} · {width}x{height} a {fps:.0f} fps · da {args.start:.0f}s per {args.seconds:.0f}s")

    recent: deque = deque(maxlen=3)
    trail: deque = deque(maxlen=TRAIL)
    frames = seen = 0
    infer_s = 0.0
    try:
        for _ in range(int(args.seconds * fps)):
            ok, frame = capture.read()
            if not ok:
                break
            recent.append(small(frame))
            ball = None
            if len(recent) == 3:
                t0 = time.perf_counter()
                ball = detector.find(list(recent), (width, height))
                infer_s += time.perf_counter() - t0
                frames += 1
                seen += ball is not None
            trail.append(ball)
            picture = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            points = [(int(b[0] * scale), int(b[1] * scale)) for b in trail if b is not None]
            for a, b in zip(points, points[1:]):
                cv2.line(picture, a, b, (0, 200, 255), 2)
            if ball is not None:
                cv2.circle(picture, (int(ball[0] * scale), int(ball[1] * scale)), 9, (0, 0, 255), 2)
                cv2.putText(picture, f"{ball[2]:.2f}", (int(ball[0] * scale) + 11, int(ball[1] * scale) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            writer.write(picture)
    finally:
        capture.release()
        writer.release()
        detector.close()

    if not frames:
        print("Nessun fotogramma letto.")
        return 1
    per = infer_s / frames
    print(f"Velocità: {1000 * per:.1f} ms per fotogramma · {1 / per:.0f} fotogrammi al secondo sulla scheda")
    print(f"Palla:   vista in {seen} di {frames} fotogrammi ({100 * seen / frames:.0f}%)")
    print(f"Video:   {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
