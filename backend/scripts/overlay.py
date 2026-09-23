#!/usr/bin/env python3
"""Render a debug overlay for an analysed match.

    python scripts/overlay.py <match-id> --out /tmp/check.mp4
    python scripts/overlay.py <match-id> --stills /tmp/frames --from 120 --to 180

Reads the artifacts stored by the analysis, so the detector never runs: a
three-minute clip renders in seconds.

What to look for, in order:
  1. Does the drawn court sit on the real court, and the yellow net line on
     the real net? If not, recalibrate — every metre-based number is wrong.
  2. Does each player keep the same colour throughout? A colour swapping
     between two people is an identity error; the mini-map makes it obvious.
  3. Do the green rally segments in the timeline line up with actual points?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings                   # noqa: E402
from app.core.storage import video_path                    # noqa: E402
from app.ml.artifacts import load_artifacts, resolve_artifacts_dir   # noqa: E402
from app.ml.overlay import build_context, render_stills, render_video  # noqa: E402
from app.ml.pipeline import PipelineConfig, analyse_tracklets          # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "match_id",
        help="id della partita, un suo prefisso, oppure 'latest' per l'ultima analizzata",
    )
    parser.add_argument("--out", default=None, help="file MP4 di destinazione")
    parser.add_argument("--stills", default=None, help="cartella per immagini JPEG invece del video")
    parser.add_argument("--count", type=int, default=12, help="numero di immagini con --stills")
    parser.add_argument("--from", dest="start_s", type=float, default=0.0, help="secondo iniziale")
    parser.add_argument("--to", dest="end_s", type=float, default=None, help="secondo finale")
    parser.add_argument("--fps", type=float, default=10.0, help="fps di riproduzione del video prodotto")
    parser.add_argument("--scale", type=float, default=1.0, help="ridimensiona l'uscita (es. 0.5)")
    parser.add_argument("--video", default=None, help="percorso video alternativo")
    parser.add_argument("--artifacts", default=None, help="cartella artefatti alternativa")
    args = parser.parse_args()

    settings = get_settings()
    if args.artifacts:
        directory = Path(args.artifacts)
    else:
        try:
            directory = resolve_artifacts_dir(args.match_id, settings.artifacts_path)
        except (FileNotFoundError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
    source = Path(args.video) if args.video else video_path(args.match_id)

    if not directory.exists():
        print(f"Artefatti non trovati in {directory}.", file=sys.stderr)
        print("La partita è stata analizzata con KEEP_ARTIFACTS attivo?", file=sys.stderr)
        return 1
    if not source.exists():
        print(f"Video non trovato: {source}", file=sys.stderr)
        return 1

    artifacts = load_artifacts(directory)
    if not artifacts.complete:
        print(
            "ATTENZIONE: questa analisi non è conclusa. Il file delle tracce è\n"
            "            parziale e i totali non sono ancora disponibili: i numeri\n"
            "            qui sotto riguardano solo la parte già elaborata.\n",
            file=sys.stderr,
        )

    print(
        f"Artefatti: {len(artifacts.tracklets)} tracce · "
        f"{artifacts.frames_sampled} frame · {artifacts.sample_hz:.1f} Hz"
    )

    # Identity and rallies are recomputed from the stored tracks using the
    # current settings, so the overlay always shows what the code does now —
    # not what it did when the match was analysed.
    config = _config_from(settings)
    _, identity, rallies = analyse_tracklets(
        tracklets=artifacts.tracklets,
        calibration=artifacts.calibration,
        config=config,
        sample_hz=artifacts.sample_hz,
        frames_sampled=artifacts.frames_sampled,
        analysed_s=artifacts.analysed_s,
    )
    print(f"Giocatori: {len(identity.players)} · Scambi: {len(rallies)}")
    for warning in identity.warnings:
        print(f"  ! {warning}")

    ctx = build_context(artifacts, identity.players, rallies)

    if args.stills:
        written = render_stills(
            source, ctx, args.stills, count=args.count, start_s=args.start_s, end_s=args.end_s
        )
        print(f"Scritte {len(written)} immagini in {args.stills}")
        return 0 if written else 1

    out = Path(args.out or f"overlay_{args.match_id[:8]}.mp4")
    written = render_video(
        source,
        ctx,
        out,
        start_s=args.start_s,
        end_s=args.end_s,
        playback_fps=args.fps,
        scale=args.scale,
        progress=lambda n, t: print(f"  {n} frame · {t / 60:.1f} min di video", flush=True),
    )
    if not written:
        print("Nessun frame nell'intervallo richiesto.", file=sys.stderr)
        return 1
    print(f"Scritti {written} frame in {out} ({written / args.fps:.0f}s di riproduzione)")
    return 0


def _config_from(settings) -> PipelineConfig:
    return PipelineConfig(
        detector_model=settings.detector_model,
        max_player_speed_ms=settings.max_player_speed_ms,
        track_max_age_s=settings.track_max_age_s,
        rally_speed_threshold_ms=settings.rally_speed_threshold_ms,
        rally_min_duration_s=settings.rally_min_duration_s,
        rally_merge_gap_s=settings.rally_merge_gap_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
