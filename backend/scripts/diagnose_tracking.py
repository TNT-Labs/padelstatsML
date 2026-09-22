#!/usr/bin/env python3
"""Explain why tracks die, from the artifacts of a run.

    python scripts/diagnose_tracking.py <match-id>

Fragmentation has two very different causes and they need opposite fixes:

  * the detector loses the player, and no track could have survived — the
    answer is recall (input size, confidence threshold, a larger model);
  * the detector sees the player, but the tracker refuses to re-associate
    the detection with the existing track — the answer is the association
    gate.

Guessing between them is how thresholds get tuned into a system that nobody
measured. This reads the stored observations and counts, for every track that
ends, whether another track starts soon after within reach — and what
association window that re-link would have needed.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.core.config import get_settings          # noqa: E402
from app.core.storage import artifacts_dir        # noqa: E402
from app.ml.artifacts import load_artifacts       # noqa: E402

# A successor further away in time than this is a different player, not the
# same one reappearing.
MAX_HANDOVER_S = 3.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match_id")
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args()

    settings = get_settings()
    directory = Path(args.artifacts) if args.artifacts else artifacts_dir(args.match_id)
    if not directory.exists():
        print(f"Artefatti non trovati in {directory}.", file=sys.stderr)
        return 1

    artifacts = load_artifacts(directory)
    tracklets = artifacts.tracklets
    if not tracklets:
        print("Nessuna traccia negli artefatti.", file=sys.stderr)
        return 1

    period = 1.0 / artifacts.sample_hz if artifacts.sample_hz else 0.2
    one_step_gate = settings.max_player_speed_ms * period + 0.6

    print(f"{len(tracklets)} tracce · campionamento {artifacts.sample_hz:.1f} Hz · "
          f"{artifacts.analysed_s / 60:.1f} min analizzati")
    lengths = sorted(len(t) for t in tracklets)
    print(f"osservazioni per traccia: mediana {statistics.median(lengths):.0f}, "
          f"p90 {lengths[int(len(lengths) * 0.9)]}, max {lengths[-1]}")
    print()

    starts = sorted(tracklets, key=lambda t: t.start_s)
    orphaned = 0
    reachable_now = 0
    needed_windows: list[float] = []
    gap_hist: Counter[int] = Counter()

    for track in tracklets:
        end_pos = np.array(track.observations[-1].foot_court)
        best: tuple[float, float] | None = None       # (gap, distance)
        for candidate in starts:
            gap = candidate.start_s - track.end_s
            if gap <= 0:
                continue
            if gap > MAX_HANDOVER_S:
                break
            distance = float(np.linalg.norm(np.array(candidate.observations[0].foot_court) - end_pos))
            if best is None or distance < best[1]:
                best = (gap, distance)

        if best is None:
            orphaned += 1
            continue

        gap, distance = best
        gap_hist[round(gap / period)] += 1
        if distance <= one_step_gate:
            reachable_now += 1
        # Window that would have been needed to accept this re-link.
        needed_windows.append(max(distance - 0.6, 0.0) / settings.max_player_speed_ms)

    with_successor = len(tracklets) - orphaned
    print(f"tracce senza successore entro {MAX_HANDOVER_S:.0f}s: {orphaned} "
          f"({100 * orphaned / len(tracklets):.0f}%)  → il giocatore è sparito davvero")
    print(f"tracce con un successore vicino:                {with_successor} "
          f"({100 * with_successor / len(tracklets):.0f}%)  → candidate a una ri-associazione")
    print()

    if with_successor:
        print(f"gate di un passo ({one_step_gate:.1f} m): {reachable_now} di {with_successor} "
              f"successori erano già raggiungibili ({100 * reachable_now / with_successor:.0f}%)")
        needed = sorted(needed_windows)
        print("finestra di associazione che sarebbe servita (secondi di movimento):")
        for label, q in (("mediana", 0.5), ("p75", 0.75), ("p90", 0.90), ("p99", 0.99)):
            print(f"  {label:>8}: {needed[int(q * (len(needed) - 1))]:.2f}s")
        print()
        print("distanza in campioni fra la morte di una traccia e la nascita della successiva:")
        for step, count in sorted(gap_hist.items())[:8]:
            print(f"  {step:>2} campioni ({step * period:.1f}s): {count}")

    print()
    print("Lettura: se la maggior parte dei successori era già raggiungibile con il")
    print("gate di un passo, il problema NON è il gate ma la resa del detector.")
    print("Se invece servivano finestre molto più lunghe, il gate era troppo stretto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
