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
  aspetto /    the two look different beyond the veto: re-ID embedding when the
  colore       match has a calibrated one, kit colour otherwise.
  salto        a junction no one could run, and the colour is not strong
               enough to call it a changeover.
  costo        linkable, but above the threshold.
  lontani      more than MAX_LINK_GAP_S apart: no evidence either way.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.core.config import get_settings                               # noqa: E402
from app.ml import identity                                            # noqa: E402
from app.ml.artifacts import load_artifacts, read_observations, resolve_artifacts_dir  # noqa: E402
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
    appearance, appearance_summary = identity.calibrate_appearance(tracklets)
    pieces = [
        Tracklet(id=t.id, observations=part)
        for t in tracklets
        for part in identity._split_identity_switches(t.observations, appearance)
    ]
    clusters = identity._link_tracklets(pieces, speed, appearance=appearance)
    clusters.sort(key=lambda c: -len(c.observations))
    players = identity._select_players(clusters, period)
    leftovers = [c for c in clusters if all(c is not p for p in players)]

    expected = max(int(round(artifacts.analysed_s * artifacts.sample_hz)), 1)
    total = sum(len(t) for t in tracklets)
    kept = sum(len(c.observations) for c in players)
    print(f"{len(tracklets)} tracce ({len(pieces) - len(tracklets)} divise per cambio di persona) → "
          f"{len(clusters)} cluster · "
          f"{kept} di {total} osservazioni ai 4 giocatori ({100 * kept / total:.0f}%)")
    print("giocatori tracciati: " + "  ".join(
        f"G{i + 1} {100 * len(c.observations) / expected:.0f}%" for i, c in enumerate(players)))

    # The uniform histogram is what the colour stage returns when it found no
    # usable pixel. Before kit colours included white, grey and black, that
    # was every player dressed in them — and colour told nobody apart.
    colours = [o.color for t in tracklets for o in t.observations]
    blind = sum(1 for c in colours if np.allclose(c, 1.0 / len(c), atol=1e-4))
    print(f"osservazioni senza colore utilizzabile: {blind} di {len(colours)} "
          f"({100 * blind / len(colours):.0f}%)")
    if blind > 0.2 * len(colours):
        print("  Il colore non distingue questi giocatori. Se la partita è stata analizzata")
        print("  prima che il bianco, il nero e il grigio venissero misurati, va rianalizzata.")

    # What told the players apart, and how far apart the two kinds of pair
    # are. The wider the gap, the more re-ID can be trusted on this match.
    if appearance is not None:
        gap = appearance.different_median - appearance.same_median
        print(f"aspetto: re-ID · stessa persona {appearance.same_median:.2f} · persone diverse "
              f"{appearance.different_median:.2f} · soglia {appearance.veto:.2f} "
              f"({appearance.positives} + {appearance.negatives} coppie)")
        if gap < 0.15:
            print("  Distanze vicine: il re-ID distingue poco questi giocatori (inquadratura")
            print("  lontana, poca risoluzione o divise molto simili).")
    else:
        print(f"aspetto: colore della divisa · {appearance_summary.get('reason', '')}")
    print()

    print(f"i {min(SHOWN, len(leftovers))} cluster esclusi più grandi, contro ciascun giocatore:")
    overlap_shares: list[float] = []
    for cluster in leftovers[:SHOWN]:
        covered = _covered_s(cluster)
        cells = []
        for player in players:
            reason, share = _reason(cluster, player, speed, period, appearance)
            cells.append(reason)
            if share is not None:
                overlap_shares.append(share)
        print(f"  {len(cluster.observations):>5} oss · {covered / 60:4.1f} min │ " + " │ ".join(cells))
    print()

    if overlap_shares:
        print(f"coppie bloccate perché viste insieme: {len(overlap_shares)} · "
              f"densità mediana {100 * statistics.median(overlap_shares):.0f}% "
              f"(soglia {100 * identity.COEXIST_DENSITY:.0f}%)")
        # Seen together with all four players means a fifth person — or four
        # clusters that are not four people. Which one is a count, not a guess:
        # a fifth person needs frames with more than four detections.
        per_frame = Counter(o.frame_index for _, o in read_observations(directory))
        crowded = sum(1 for n in per_frame.values() if n > identity.N_PLAYERS)
        share = crowded / max(len(per_frame), 1)
        print(f"  frame con più di {identity.N_PLAYERS} rilevazioni: {crowded} di {len(per_frame)} "
              f"({100 * share:.0f}%)")
        if share < 0.10:
            print("  Troppo pochi per una quinta persona: i cluster esclusi sono gli stessi")
            print("  quattro giocatori, mescolati. L'aspetto non basta a distinguerli.")
        else:
            print("  Una quinta persona è in campo o appena fuori per una parte rilevante")
            print("  della partita: controlla la calibrazione e il margine del campo.")
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


def _reason(cluster, player, speed: float, period: float, appearance) -> tuple[str, float | None]:
    if (cluster.start_s - player.end_s > identity.MAX_LINK_GAP_S
            or player.start_s - cluster.end_s > identity.MAX_LINK_GAP_S):
        return f"{'lontani':<17}", None
    density = _density(cluster, player, period)
    if identity._coexist(cluster, player, period):
        return f"insieme {100 * density:>3.0f}%".ljust(17), density
    look = identity.appearance_distance(cluster, player, appearance)
    if look > identity.MAX_COLOR_DISTANCE:
        label = "aspetto" if appearance is not None else "colore"
        return f"{label} {look:.2f}".ljust(17), None
    cost = identity._disjoint_link_cost(cluster, player, speed, period, appearance)
    if cost is None:
        return "salto".ljust(17), None
    return f"costo {cost:.2f}".ljust(17), None


if __name__ == "__main__":
    raise SystemExit(main())
