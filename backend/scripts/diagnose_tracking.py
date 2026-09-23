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
from app.ml.artifacts import load_artifacts, resolve_artifacts_dir   # noqa: E402

# A successor further away in time than this is a different player, not the
# same one reappearing.
MAX_HANDOVER_S = 3.0


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
    if not artifacts.complete:
        print(
            "ATTENZIONE: questa analisi non è conclusa. Il file delle tracce è\n"
            "            parziale e i totali non sono ancora disponibili: i numeri\n"
            "            qui sotto riguardano solo la parte già elaborata.\n",
            file=sys.stderr,
        )

    tracklets = artifacts.tracklets
    if not tracklets:
        print("Nessuna traccia negli artefatti.", file=sys.stderr)
        return 1

    period = 1.0 / artifacts.sample_hz if artifacts.sample_hz else 0.2
    one_step_gate = settings.max_player_speed_ms * period + 0.6
    # The tracker can only re-associate while the old track is alive; past
    # its memory the split is the detector's, and identity linking's to mend.
    max_age = artifacts.meta.get("config", {}).get("track_max_age_s", settings.track_max_age_s)

    print(f"{len(tracklets)} tracce · campionamento {artifacts.sample_hz:.1f} Hz · "
          f"{artifacts.analysed_s / 60:.1f} min analizzati")
    lengths = sorted(len(t) for t in tracklets)
    print(f"osservazioni per traccia: mediana {statistics.median(lengths):.0f}, "
          f"p90 {lengths[int(len(lengths) * 0.9)]}, max {lengths[-1]}")
    print()

    starts = sorted(tracklets, key=lambda t: t.start_s)
    orphaned = 0
    within_age = 0
    reachable_one_step = 0
    reachable_per_track = 0
    needed_windows: list[float] = []
    box_changes: list[float] = []
    shrunk = grown = 0
    gap_hist: Counter[int] = Counter()

    for track in tracklets:
        last = track.observations[-1]
        end_pos = np.array(last.foot_court)
        best: tuple[float, float, float] | None = None    # (gap, distance, box ratio)
        for candidate in starts:
            gap = candidate.start_s - track.end_s
            if gap <= 0:
                continue
            if gap > MAX_HANDOVER_S:
                break
            first = candidate.observations[0]
            distance = float(np.linalg.norm(np.array(first.foot_court) - end_pos))
            if best is None or distance < best[1]:
                height_before = last.bbox[3] - last.bbox[1]
                height_after = first.bbox[3] - first.bbox[1]
                ratio = height_after / height_before if height_before > 1 else 1.0
                best = (gap, distance, ratio)

        if best is None:
            orphaned += 1
            continue

        gap, distance, ratio = best
        gap_hist[round(gap / period)] += 1
        needed_windows.append(max(distance - 0.6, 0.0) / settings.max_player_speed_ms)
        box_changes.append(abs(ratio - 1.0))
        if ratio < 0.8:
            shrunk += 1
        elif ratio > 1.25:
            grown += 1
        if gap > max_age + 1e-6:
            continue
        within_age += 1
        if distance <= one_step_gate:
            reachable_one_step += 1
        # Gate as the tracker computes it today: sized on the track's own age,
        # measured from the last observed position.
        if distance <= settings.max_player_speed_ms * max(gap, period) + 0.6:
            reachable_per_track += 1

    with_successor = len(tracklets) - orphaned
    print(f"tracce senza successore entro {MAX_HANDOVER_S:.0f}s: {orphaned} "
          f"({100 * orphaned / len(tracklets):.0f}%)  → il giocatore è sparito davvero")
    print(f"tracce con un successore vicino:                {with_successor} "
          f"({100 * with_successor / len(tracklets):.0f}%)  → candidate a una ri-associazione")
    print()

    if with_successor:
        beyond = with_successor - within_age
        label = f"nati entro la memoria della traccia ({max_age:.1f}s):"
        print(f"  {label:<46}{within_age:>5}  → la traccia era viva: toccava al tracker")
        print(f"  {'nati dopo:':<46}{beyond:>5}  → perdita del detector: tocca al")
        print(f"  {'':<46}{'':>5}    collegamento delle identità")
        print()

    if within_age:
        print(f"quante delle {within_age} ri-associazioni possibili accetta ciascun gate:")
        for label, count in (
            (f"un passo fisso ({one_step_gate:.1f} m):", reachable_one_step),
            ("età della traccia, dall'ultima posizione:", reachable_per_track),
        ):
            print(f"  {label:<46}{count:>5} ({100 * count / within_age:.0f}%)")
        print("  (se il gate le accetta e la traccia si è spezzata lo stesso, confronta")
        print("   col codice attuale: scripts/retrack.py)")
        print()

    if with_successor:

        needed = sorted(needed_windows)
        print("finestra di associazione che sarebbe servita (secondi di movimento):")
        for label, q in (("mediana", 0.5), ("p75", 0.75), ("p90", 0.90), ("p99", 0.99)):
            print(f"  {label:>8}: {needed[int(q * (len(needed) - 1))]:.2f}s")
        print()

        changes = sorted(box_changes)
        print("variazione di altezza del riquadro fra una traccia e la successiva:")
        for label, q in (("mediana", 0.5), ("p75", 0.75), ("p90", 0.90)):
            print(f"  {label:>8}: {100 * changes[int(q * (len(changes) - 1))]:.0f}%")
        print(f"  più basso di oltre il 20%: {shrunk} · più alto di oltre il 25%: {grown}")
        print("  (più basso: piedi coperti, il punto a terra sale; più alto: due giocatori")
        print("   fusi in un riquadro. In entrambi i casi cambia il riquadro, non il giocatore)")
        print()

        print("distanza in campioni fra la morte di una traccia e la nascita della successiva:")
        for step, count in sorted(gap_hist.items())[:8]:
            print(f"  {step:>2} campioni ({step * period:.1f}s): {count}")

    print()
    print("Lettura:")
    print("  · la maggior parte nasce dopo la memoria della traccia → è la resa del")
    print("    detector: input più grande, soglia più bassa, modello più grande")
    print("  · la maggior parte nasce mentre la traccia è viva ed è raggiungibile →")
    print("    è l'associazione: misura il codice attuale con scripts/retrack.py")
    print("  · molti riquadri più bassi al cambio di traccia → il punto a terra salta")
    print("    per occlusione: non è velocità del giocatore ma rumore di posizione")
    print()
    print("Attenzione: il successore è scelto come il più vicino nel tempo e nello")
    print("spazio, quindi in campo affollato può appartenere a un altro giocatore.")
    print("I numeri indicano dove guardare, non sostituiscono l'overlay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
