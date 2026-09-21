#!/usr/bin/env python3
"""Re-run the post-tracking stages with different parameters, in under a second.

    # what the current settings produce
    python scripts/retune.py <match-id>

    # try a different rally threshold
    python scripts/retune.py <match-id> --rally-speed 1.4

    # sweep a parameter and compare
    python scripts/retune.py <match-id> --sweep rally-speed 0.8 1.0 1.2 1.4 1.6

Inference is the only expensive stage. Everything after it — identity
linking, rally segmentation, metrics — runs on the stored observations in
milliseconds, so a threshold can be evaluated immediately instead of costing
a full re-analysis.

This is the tool that makes tuning measurable. Pick the value on real
matches, then change the default in .env or config.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings                # noqa: E402
from app.core.storage import artifacts_dir              # noqa: E402
from app.ml.artifacts import load_artifacts             # noqa: E402
from app.ml.pipeline import PipelineConfig, analyse_tracklets   # noqa: E402

SWEEPABLE = {
    "rally-speed": "rally_speed_threshold_ms",
    "rally-min": "rally_min_duration_s",
    "rally-gap": "rally_merge_gap_s",
    "max-speed": "max_player_speed_ms",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match_id")
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--rally-speed", type=float, default=None, help="m/s (default da .env)")
    parser.add_argument("--rally-min", type=float, default=None, help="durata minima scambio, s")
    parser.add_argument("--rally-gap", type=float, default=None, help="unione scambi ravvicinati, s")
    parser.add_argument("--max-speed", type=float, default=None, help="velocità massima giocatore, m/s")
    parser.add_argument(
        "--sweep", nargs="+", metavar=("PARAM", "VALORE"),
        help=f"confronta più valori di un parametro ({', '.join(SWEEPABLE)})",
    )
    args = parser.parse_args()

    directory = Path(args.artifacts) if args.artifacts else artifacts_dir(args.match_id)
    if not directory.exists():
        print(f"Artefatti non trovati in {directory}.", file=sys.stderr)
        return 1

    artifacts = load_artifacts(directory)
    print(f"Partita {args.match_id[:8]} · {len(artifacts.tracklets)} tracce · "
          f"{artifacts.analysed_s / 60:.1f} min · {artifacts.sample_hz:.1f} Hz\n")

    base = _base_config(get_settings(), args)

    if args.sweep:
        return _sweep(artifacts, base, args.sweep)

    result, identity, rallies = _run(artifacts, base)
    _print_run(base, result, identity, rallies, verbose=True)
    if artifacts.result:
        _print_delta(artifacts.result, result)
    return 0


def _run(artifacts, config: PipelineConfig):
    return analyse_tracklets(
        tracklets=artifacts.tracklets,
        calibration=artifacts.calibration,
        config=config,
        sample_hz=artifacts.sample_hz,
        frames_sampled=artifacts.frames_sampled,
        analysed_s=artifacts.analysed_s,
    )


def _base_config(settings, args) -> PipelineConfig:
    return PipelineConfig(
        detector_model=settings.detector_model,
        max_player_speed_ms=args.max_speed if args.max_speed is not None else settings.max_player_speed_ms,
        rally_speed_threshold_ms=(
            args.rally_speed if args.rally_speed is not None else settings.rally_speed_threshold_ms
        ),
        rally_min_duration_s=(
            args.rally_min if args.rally_min is not None else settings.rally_min_duration_s
        ),
        rally_merge_gap_s=(
            args.rally_gap if args.rally_gap is not None else settings.rally_merge_gap_s
        ),
    )


def _sweep(artifacts, base: PipelineConfig, sweep: list[str]) -> int:
    param = sweep[0]
    if param not in SWEEPABLE:
        print(f"Parametro '{param}' non supportato. Scegli fra: {', '.join(SWEEPABLE)}", file=sys.stderr)
        return 1
    try:
        values = [float(v) for v in sweep[1:]]
    except ValueError:
        print("I valori dello sweep devono essere numerici.", file=sys.stderr)
        return 1
    if not values:
        print("Servono almeno due valori da confrontare.", file=sys.stderr)
        return 1

    field = SWEEPABLE[param]
    print(f"{param:>14} │ {'scambi':>7} │ {'medio':>7} │ {'max':>7} │ {'gioco':>7} │ giocatori")
    print("─" * 66)
    for value in values:
        config = PipelineConfig(**{**base.__dict__, field: value})
        try:
            result, identity, rallies = _run(artifacts, config)
        except RuntimeError as exc:
            print(f"{value:>14.2f} │ errore: {exc}")
            continue
        summary = result["summary"]
        print(
            f"{value:>14.2f} │ {summary['rallies_count']:>7} │ "
            f"{summary['avg_rally_s']:>6.1f}s │ {summary['longest_rally_s']:>6.1f}s │ "
            f"{summary['active_ratio'] * 100:>6.0f}% │ {len(identity.players)}"
        )
    print("\nScegli il valore i cui scambi corrispondono ai punti reali nel video")
    print("(verificalo con scripts/overlay.py), poi aggiornalo nel .env.")
    return 0


def _print_run(config: PipelineConfig, result: dict, identity, rallies, verbose: bool) -> None:
    summary = result["summary"]
    print(f"rally_speed={config.rally_speed_threshold_ms}  "
          f"rally_min={config.rally_min_duration_s}  "
          f"rally_gap={config.rally_merge_gap_s}  "
          f"max_speed={config.max_player_speed_ms}")
    print(f"  Giocatori      {len(identity.players)} (da {identity.clusters_found} cluster)")
    print(f"  Scambi         {summary['rallies_count']} · medio {summary['avg_rally_s']:.1f}s · "
          f"max {summary['longest_rally_s']:.1f}s")
    print(f"  Gioco effettivo {summary['active_ratio'] * 100:.0f}%")

    if not verbose:
        return
    print("\n  Giocatore  squadra  distanza  tracciato  rete/metà/fondo")
    for pid, stats in sorted(result["per_player"].items()):
        zone = stats["zone_pct"]
        print(
            f"  G{int(pid) + 1:<9} {stats['team']:^7}  "
            f"{stats['distance_m']:>7.0f}m  {stats['tracked_ratio'] * 100:>8.0f}%  "
            f"{zone['net'] * 100:>3.0f}/{zone['mid'] * 100:>3.0f}/{zone['back'] * 100:>3.0f}%"
        )
    warnings = result["data_quality"]["warnings"]
    if warnings:
        print("\n  Avvisi:")
        for warning in warnings:
            print(f"   ! {warning}")


def _print_delta(stored: dict, fresh: dict) -> None:
    """Difference against the result saved at analysis time."""
    old, new = stored["summary"], fresh["summary"]
    if old == new:
        print("\nIdentico al risultato salvato.")
        return
    print("\nDifferenze rispetto al risultato salvato:")
    for key in ("rallies_count", "avg_rally_s", "longest_rally_s", "active_ratio"):
        if old.get(key) != new.get(key):
            print(f"  {key:<16} {old.get(key)} → {new.get(key)}")


if __name__ == "__main__":
    raise SystemExit(main())
