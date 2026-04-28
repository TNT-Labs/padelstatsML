#!/usr/bin/env python3
"""
Standalone ML pipeline smoke-test. No FastAPI, no Celery, no DB required.

Usage:
    cd backend/
    python scripts/test_pipeline.py path/to/match.mp4
    python scripts/test_pipeline.py path/to/match.mp4 --device cuda --tracknet weights/tracknet_padel.pth

The script runs the full AnalysisPipeline and prints a stats summary.
Useful for calibrating thresholds and validating model weights before
wiring the full async stack.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `app` importable when running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.pipeline import AnalysisPipeline, PipelineConfig


def _progress_bar(pct: int, msg: str) -> None:
    filled = pct // 5
    bar = "#" * filled + "." * (20 - filled)
    print(f"\r  [{bar}] {pct:3d}%  {msg:<40}", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test padel ML pipeline end-to-end")
    parser.add_argument("video", help="Local path to the match video")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu | cuda)")
    parser.add_argument("--yolo", default="yolov8n.pt", help="YOLO weights (ultralytics auto-downloads)")
    parser.add_argument("--tracknet", default=None, help="TrackNetV2 weights .pth (optional)")
    parser.add_argument("--stride", type=int, default=2, help="Player tracker frame stride (default: 2)")
    parser.add_argument("--dump", default=None, help="Dump full result JSON to this path")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if args.tracknet and not Path(args.tracknet).exists():
        print(f"Warning: tracknet weights not found at {args.tracknet}, running without ball tracking")
        args.tracknet = None

    config = PipelineConfig(
        yolo_weights=args.yolo,
        tracknet_weights=args.tracknet,
        device=args.device,
        player_stride=args.stride,
    )

    print(f"Video : {video_path}")
    print(f"Device: {args.device}")
    print(f"YOLO  : {args.yolo}")
    print(f"TrackNet: {args.tracknet or '(none — ball tracking disabled)'}")
    print()

    pipeline = AnalysisPipeline(config)

    t0 = time.perf_counter()
    try:
        result = pipeline.run(str(video_path), progress_callback=_progress_bar)
    except RuntimeError as e:
        print(f"\n\nPipeline error: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    print(f"\n\nCompleted in {elapsed:.1f}s\n")

    # ── Summary ───────────────────────────────────────────────────────
    rallies = result["rallies"]
    print(f"Rallies : {len(rallies)}")
    if rallies:
        lengths = [r["end_frame"] - r["start_frame"] for r in rallies]
        print(f"  avg length : {sum(lengths)/len(lengths):.0f} frames")

    print()
    print(f"{'Player':<10} {'Distance':>10} {'Shots':>7} {'Smash':>7} {'Volley':>8} {'Bandeja':>9} {'W':>4} {'E':>4}")
    print("-" * 70)
    for pid, ps in sorted(result["per_player"].items()):
        shots = ps["shots"]
        total = sum(shots.values())
        print(
            f"{pid:<10} {ps['distance_m']:>9.1f}m {total:>7d} "
            f"{shots.get('smash', 0):>7d} {shots.get('volley', 0):>8d} "
            f"{shots.get('bandeja', 0):>9d} {ps['winners']:>4d} {ps['errors']:>4d}"
        )

    cal = result["court_calibration"]
    print(f"\nCourt calibration: {'OK' if cal.get('homography') else 'FAILED'}")
    print(f"  corners_px: {cal.get('corners_px')}")

    if args.dump:
        out = Path(args.dump)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nFull result dumped to {out}")


if __name__ == "__main__":
    main()
