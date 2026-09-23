#!/usr/bin/env python3
"""Explain why observations end up outside the four players.

    python scripts/diagnose_identity.py latest

Identity linking joins tracklets into clusters and keeps the four largest as
the players; everything else is discarded. When the players are tracked for
a small share of the match, the question is what stops the leftover
clusters from joining them. For the largest leftovers this reports, against
each of the four players, the first rule that blocks the link:

  insieme      seen in the same frame while both on court: the share of
               samples in which that happens. Two players score 0.6-0.8;
               two pieces of one player sharing a few stray frames score far
               lower and are no longer kept apart (see COEXIST_DENSITY).
  colore       the shirts differ beyond the veto.
  salto        a junction no one could run, and the colour is not strong
               enough to call it a changeover.
  costo        linkable, but above the threshold.
  lontani      more than MAX_LINK_GAP_S apart: no evidence either way.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings                               # noqa: E402
from app.ml import identity                                            # noqa: E402
from app.ml.artifacts import load_artifacts, resolve_artifacts_dir     # noqa: E402
from app.ml.tracking import Tracklet                                   # noqa: E402

SHOWN = 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "match_id",
        help="id della partita, un suo prefisso, oppure 'latest' per l'ultima analizzata",
    )
    parser.add_argument("--artifacts", default=None)
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
    tracklets = [t for t in artifacts.tracklets if len(t) >= 3]
    if not tracklets:
        print("Nessuna traccia negli artefatti.", file=sys.stderr)
        return 1

    speed = settings.max_player_speed_ms
    period = identity._sample_period(tracklets)
    pieces = [
        Tracklet(id=t.id, observations=part)
        for t in tracklets
        for part in identity._split_identity_switches(t.observations)
    ]
    clusters = identity._link_tracklets(pieces, speed)
    clusters.sort(key=lambda c: -len(c.observations))
    players = identity._select_players(clusters, period)
    leftovers = [c for c in clusters if all(c is not p for p in players)]

    expected = max(int(round(artifacts.analysed_s * artifacts.sample_hz)), 1)
    total = sum(len(t) for t in tracklets)
    kept = sum(len(c.observations) for c in players)
    print(f"{len(tracklets)} tracce ({len(pieces) - len(tracklets)} divise per cambio di maglia) → "
          f"{len(clusters)} cluster · "
          f"{kept} di {total} osservazioni ai 4 giocatori ({100 * kept / total:.0f}%)")
    print("giocatori tracciati: " + "  ".join(
        f"G{i + 1} {100 * len(c.observations) / expected:.0f}%" for i, c in enumerate(players)))
    print()

    print(f"i {min(SHOWN, len(leftovers))} cluster esclusi più grandi, contro ciascun giocatore:")
    overlap_shares: list[float] = []
    for cluster in leftovers[:SHOWN]:
        covered = _covered_s(cluster)
        cells = []
        for player in players:
            reason, share = _reason(cluster, player, speed, period)
            cells.append(reason)
            if share is not None:
                overlap_shares.append(share)
        print(f"  {len(cluster.observations):>5} oss · {covered / 60:4.1f} min │ " + " │ ".join(cells))
    print()

    if overlap_shares:
        print(f"coppie bloccate perché viste insieme: {len(overlap_shares)} · "
              f"densità mediana {100 * statistics.median(overlap_shares):.0f}% "
              f"(soglia {100 * identity.COEXIST_DENSITY:.0f}%)")
        print("  Un cluster escluso visto insieme a tutti e quattro i giocatori è una")
        print("  quinta persona in campo (o appena fuori), non un giocatore spezzato.")
    return 0


def _covered_s(cluster) -> float:
    return sum(seg.end_s - seg.start_s for seg in cluster.segments)


def _density(cluster, player, period: float) -> float:
    import numpy as np

    overlap = identity._overlap_s(cluster.segments, player.segments)
    if overlap <= 0:
        return 0.0
    shared = len(np.intersect1d(cluster.frames(), player.frames(), assume_unique=True))
    return shared / (overlap / period + 1.0)


def _reason(cluster, player, speed: float, period: float) -> tuple[str, float | None]:
    if (cluster.start_s - player.end_s > identity.MAX_LINK_GAP_S
            or player.start_s - cluster.end_s > identity.MAX_LINK_GAP_S):
        return f"{'lontani':<17}", None
    density = _density(cluster, player, period)
    if identity._coexist(cluster, player, period):
        return f"insieme {100 * density:>3.0f}%".ljust(17), density
    color = identity._color_distance(cluster, player)
    if color > identity.MAX_COLOR_DISTANCE:
        return f"colore {color:.2f}".ljust(17), None
    cost = identity._disjoint_link_cost(cluster, player, speed, period)
    if cost is None:
        return "salto".ljust(17), None
    return f"costo {cost:.2f}".ljust(17), None


if __name__ == "__main__":
    raise SystemExit(main())
