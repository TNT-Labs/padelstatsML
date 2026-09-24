#!/usr/bin/env python3
"""Re-run the tracker on a stored match and compare with the stored result.

    python scripts/retrack.py latest
    python scripts/retrack.py latest --max-age 1.6

The artifacts of a match hold every observation the tracker was fed, so
association can be repeated in seconds, without the detector. Run it on a
match analysed with the previous code and the comparison says whether a
tracker change helps on real footage — before re-analysing anything.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.core.config import get_settings                             # noqa: E402
from app.ml.artifacts import (  # noqa: E402
    completed_runs,
    load_artifacts,
    read_observations,
    resolve_artifacts_dir,
)
from app.ml.pipeline import PipelineConfig, analyse_tracklets        # noqa: E402
from app.ml.replay import replay_tracking                            # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "match_id",
        help="id della partita, un suo prefisso, oppure 'latest' per l'ultima analizzata",
    )
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--max-speed", type=float, default=None, help="m/s (default: quello della run)")
    parser.add_argument("--max-age", type=float, default=None, help="memoria di una traccia, s (default: quella della run)")
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
    if not directory.exists():
        print(f"Artefatti non trovati in {directory}.", file=sys.stderr)
        return 1

    artifacts = load_artifacts(directory)
    if not artifacts.complete:
        others = [p.name[:8] for p in completed_runs(directory.parent) if p != directory]
        print("Questa analisi non è conclusa: il confronto non avrebbe senso.", file=sys.stderr)
        if others:
            print(f"Analisi concluse: {', '.join(others)} — passa l'id (basta il prefisso).", file=sys.stderr)
        return 1

    run_config = artifacts.meta.get("config", {})
    max_speed = args.max_speed if args.max_speed is not None else run_config.get("max_player_speed_ms", 8.0)
    max_age = args.max_age if args.max_age is not None else run_config.get("track_max_age_s", 1.2)

    total = len(read_observations(directory))
    print(f"Partita {directory.name[:8]} · {artifacts.analysed_s / 60:.1f} min · "
          f"{artifacts.sample_hz:.1f} Hz · {total} osservazioni")
    print(f"velocità max {max_speed} m/s · memoria traccia {max_age} s\n")

    started = time.monotonic()
    replayed = replay_tracking(artifacts, max_speed_ms=max_speed, max_age_s=max_age)
    elapsed = time.monotonic() - started

    stored = artifacts.tracklets
    period = 1.0 / artifacts.sample_hz if artifacts.sample_hz else 0.2
    rows = [
        ("tracce", len(stored), len(replayed), "{}"),
        ("osservazioni per traccia (mediana)", _median_len(stored), _median_len(replayed), "{:.0f}"),
        ("osservazioni per traccia (p90)", _p90_len(stored), _p90_len(replayed), "{}"),
        ("osservazioni in tracce valide", _kept(stored) / total, _kept(replayed) / total, "{:.0%}"),
        ("interrotte con la traccia ancora viva",
         _live_splits(stored, max_speed, max_age, period),
         _live_splits(replayed, max_speed, max_age, period), "{}"),
    ]
    print(f"{'':<40}{'salvato':>10}{'attuale':>10}")
    for label, before, after, fmt in rows:
        print(f"{label:<40}{fmt.format(before):>10}{fmt.format(after):>10}")
    print(f"\n(tracker rieseguito in {elapsed:.1f}s)\n")

    config = PipelineConfig(
        detector_model=settings.detector_model,
        max_player_speed_ms=settings.max_player_speed_ms,
        rally_speed_threshold_ms=settings.rally_speed_threshold_ms,
        rally_min_duration_s=settings.rally_min_duration_s,
        rally_merge_gap_s=settings.rally_merge_gap_s,
    )
    print("Stessi stadi a valle (identità, scambi, metriche) sulle due serie di tracce:")
    for label, tracklets in (("salvato", stored), ("attuale", replayed)):
        try:
            result, identity, rallies = analyse_tracklets(
                tracklets=tracklets,
                calibration=artifacts.calibration,
                config=config,
                sample_hz=artifacts.sample_hz,
                frames_sampled=artifacts.frames_sampled,
                analysed_s=artifacts.analysed_s,
            )
        except RuntimeError as exc:
            print(f"  {label}: errore — {exc}")
            continue
        tracked = "  ".join(
            f"G{int(pid) + 1} {stats['tracked_ratio'] * 100:.0f}%"
            for pid, stats in sorted(result["per_player"].items())
        )
        print(f"  {label}: {len(identity.players)} giocatori da {identity.clusters_found} cluster · "
              f"{len(rallies)} scambi · tracciati {tracked}")

    print("\n'Interrotte con la traccia ancora viva' conta le tracce che finiscono")
    print("mentre un'altra nasce subito dopo, entro la memoria della traccia e a una")
    print("distanza percorribile: ri-associazioni che il tracker avrebbe potuto fare.")
    print("Le interruzioni più lunghe sono perdite del detector, e le ricuce il")
    print("collegamento delle identità.")
    return 0


def _median_len(tracklets) -> float:
    return statistics.median(len(t) for t in tracklets) if tracklets else 0


def _p90_len(tracklets) -> int:
    lengths = sorted(len(t) for t in tracklets)
    return lengths[int(len(lengths) * 0.9)] if lengths else 0


def _kept(tracklets) -> int:
    return sum(len(t) for t in tracklets)


def _live_splits(tracklets, max_speed: float, max_age: float, period: float) -> int:
    """Tracks followed, while still alive, by a reachable new track."""
    starts = sorted(tracklets, key=lambda t: t.start_s)
    begin = np.array([t.start_s for t in starts])
    count = 0
    for track in tracklets:
        end_pos = np.array(track.observations[-1].foot_court)
        lo = np.searchsorted(begin, track.end_s, side="right")
        hi = np.searchsorted(begin, track.end_s + max_age, side="right")
        for candidate in starts[lo:hi]:
            gap = candidate.start_s - track.end_s
            distance = float(np.linalg.norm(np.array(candidate.observations[0].foot_court) - end_pos))
            if distance <= max_speed * max(gap, period) + 0.6:
                count += 1
                break
    return count


if __name__ == "__main__":
    raise SystemExit(main())
