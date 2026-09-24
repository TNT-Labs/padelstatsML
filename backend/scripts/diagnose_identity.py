#!/usr/bin/env python3
"""Explain how the four players were found, and what may have mixed them.

    python scripts/diagnose_identity.py latest
    python scripts/diagnose_identity.py latest --collegamento

On a match with both pairs in view, identity finds players by their place
on court (app/ml/roles.py): the pair from its half, with the changeovers,
and the player from their side of the pair. This reports, per player, the
pair and side, how much of the match they are tracked for, where they stand
— x measured from their own side, so a revés player sits near 2-4 m and a
drive player near 6-8 m in either half — and the share of their detections
given to them by appearance against the side they stood on.

--collegamento reports the linking method instead (the fallback for videos
with a single pair): for the largest leftover clusters, against each of the
four players, the first rule that blocks the link:

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

Both end with the frames holding more than four detections, net of the
duplicate boxes of one player: the count that tells a fifth person apart.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.core.config import get_settings                               # noqa: E402
from app.ml import identity                                            # noqa: E402
from app.ml.artifacts import load_artifacts, read_observations, resolve_artifacts_dir  # noqa: E402
from app.ml.detect import part_of_another                              # noqa: E402
from app.ml.court import COURT_LENGTH_M, COURT_WIDTH_M                 # noqa: E402
from app.ml.roles import DRIVE, assign_roles                           # noqa: E402
from app.ml.tracking import Tracklet                                   # noqa: E402

SHOWN = 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "match_id",
        help="id della partita, un suo prefisso, oppure 'latest' per l'ultima analizzata",
    )
    parser.add_argument("--artifacts", default=None)
    parser.add_argument("--collegamento", action="store_true",
                        help="mostra il metodo di collegamento delle tracce anche se la partita usa i ruoli")
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
    expected = max(int(round(artifacts.analysed_s * artifacts.sample_hz)), 1)
    total = sum(len(t) for t in tracklets)
    print(f"{len(tracklets)} tracce ({len(pieces) - len(tracklets)} divise per cambio di persona) · "
          f"{total} osservazioni")

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

    if identity.roles_apply(pieces) and not args.collegamento:
        _report_roles(pieces, appearance, total, expected)
    else:
        _report_linking(pieces, appearance, speed, period, total, expected)
    _report_crowding(directory, artifacts)
    return 0


def _report_roles(pieces, appearance, total: int, expected: int) -> None:
    assignment = assign_roles(pieces, appearance)
    times = ", ".join(f"{int(t // 60)}:{int(t % 60):02d}" for t in assignment.changeovers)
    print(f"metodo: coppia e lato in campo · cambi di campo trovati: {len(assignment.changeovers)}"
          + (f" ({times})" if times else ""))
    print("  confrontali con il video: un cambio mancato scambia i numeri delle due coppie.")
    print()
    print("giocatori (x dal proprio lato: revés ~2-4 m, drive ~6-8 m, in entrambe le metà;")
    print("           rete: distanza mediana dalla rete, in metri):")
    print(f"  {'':>3} {'coppia':>7} {'lato':>6} {'tracciato':>9} {'x':>5} {'rete':>5}  {'fermo':>6}"
          f"  {'velocità':>9}  {'dall aspetto':>12}")
    kept = 0
    own_x: dict[tuple[int, int], float] = {}
    for n, (team, role) in enumerate(identity._ROLE_ORDER):
        group = assignment.players.get((team, role), [])
        if not group:
            print(f"  G{n + 1}  nessuna traccia")
            continue
        observations = identity.one_per_frame([rp.tracklet for rp in group])
        kept += len(observations)
        where = _whereabouts(observations)
        own = [o.foot_court[0] if o.foot_court[1] < COURT_LENGTH_M / 2 else COURT_WIDTH_M - o.foot_court[0]
               for o in observations]
        own_x[(team, role)] = float(np.median(own))
        from_net = float(np.median([abs(o.foot_court[1] - COURT_LENGTH_M / 2) for o in observations]))
        # Pieces whose side said the other role: given here by appearance.
        by_look = sum(len(rp.tracklet) for rp in group if (rp.side_score > 0) != (role == DRIVE))
        by_all = sum(len(rp.tracklet) for rp in group)
        print(f"  G{n + 1} {'vicina' if team == 0 else 'lontana':>7} {'drive' if role == DRIVE else 'revés':>6}"
              f" {100 * len(observations) / expected:8.0f}% {own_x[(team, role)]:5.1f} {from_net:5.1f}"
              f"  {100 * where['still']:5.0f}%  {where['kmh']:6.1f} km/h  {100 * by_look / max(by_all, 1):11.0f}%")
    print(f"  {kept} di {total} osservazioni ai 4 giocatori ({100 * kept / total:.0f}%): il resto sono frame"
          " contesi fra due pezzi dello stesso giocatore.")
    for team in (0, 1):
        pair = [own_x.get((team, role)) for role in (0, 1)]
        if None not in pair and abs(pair[1] - pair[0]) < 2.0:
            print(f"  Coppia {'vicina' if team == 0 else 'lontana'}: i due lati distano meno di 2 m. I compagni"
                  " non tengono un lato, o i loro pezzi sono mescolati.")
    print("  'dall aspetto' è la parte data al giocatore dall'aspetto contro il lato in cui stava:")
    print("  sono gli scambi di lato seguiti. Oltre il 30% l'aspetto decide più del lato.")
    print()


def _report_linking(pieces, appearance, speed: float, period: float, total: int, expected: int) -> None:
    clusters = identity._link_tracklets(pieces, speed, appearance=appearance)
    clusters.sort(key=lambda c: -len(c.observations))
    players = identity._select_players(clusters, period)
    leftovers = [c for c in clusters if all(c is not p for p in players)]

    kept = sum(len(c.observations) for c in players)
    print(f"metodo: collegamento delle tracce · {len(clusters)} cluster · "
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
            reason, share = _reason(cluster, player, speed, period, appearance)
            cells.append(reason)
            if share is not None:
                overlap_shares.append(share)
        print(f"  {len(cluster.observations):>5} oss · {covered / 60:4.1f} min │ " + " │ ".join(cells))
    print()

    # Where each of them stands and how much it moves. A player covers the
    # court at running pace; a coach, a spectator by the side glass or a
    # reflection in it stands still, near or past the lines.
    print("dove stanno e quanto si muovono (x 0-10 m, y 0-20 m; rete a y=10):")
    print(f"  {'':>12} {'x':>5} {'y':>5}  {'fuori linee':>11}  {'fermo':>6}  {'velocità':>9}")
    for label, cluster in [(f"G{i + 1}", c) for i, c in enumerate(players)] + \
            [(f"escluso {i + 1}", c) for i, c in enumerate(leftovers[:6])]:
        where = _whereabouts(cluster.observations)
        print(f"  {label:>12} {where['x']:5.1f} {where['y']:5.1f}  {100 * where['outside']:10.0f}%"
              f"  {100 * where['still']:5.0f}%  {where['kmh']:6.1f} km/h")
    print()

    if overlap_shares:
        print(f"coppie bloccate perché viste insieme: {len(overlap_shares)} · "
              f"densità mediana {100 * statistics.median(overlap_shares):.0f}% "
              f"(soglia {100 * identity.COEXIST_DENSITY:.0f}%)")
        print()


def _report_crowding(directory: Path, artifacts) -> None:
    """Frames with more than four detections, net of duplicate boxes.

    Seen together with all four players means a fifth person — or pieces
    that are not four people. Which one is a count, not a guess: a fifth
    person needs frames with more than four detections. The same player
    detected twice — a whole-body box and a box of part of them — is not
    one, so those duplicates are taken out of the count first.

    The count has to be read against the detector's recall: with five
    people present, five detections in one frame need all five seen at
    once. Each person is seen in a share r of the frames, so that happens
    in about r^5 of them — a few percent, not most."""
    stored = read_observations(directory)
    by_frame: dict[int, list] = {}
    for _, obs in stored:
        by_frame.setdefault(obs.frame_index, []).append(obs)
    frames = max(artifacts.frames_sampled or len(by_frame), 1)
    parts = 0
    people_per_frame = []
    for group in by_frame.values():
        duplicates = int(part_of_another([o.bbox for o in group]).sum()) if len(group) > 1 else 0
        parts += duplicates
        people_per_frame.append(len(group) - duplicates)
    crowded = sum(1 for n in people_per_frame if n > identity.N_PLAYERS)
    share = crowded / frames
    per_person = sum(people_per_frame) / frames / (identity.N_PLAYERS + 1)
    expected = per_person ** (identity.N_PLAYERS + 1)
    print("una quinta persona?")
    print(f"  rilevazioni doppie (riquadro di una parte di un giocatore): {parts} "
          f"({100 * parts / max(len(stored), 1):.1f}% delle osservazioni), escluse dal conteggio")
    print(f"  frame con più di {identity.N_PLAYERS} persone: {crowded} di {frames} ({100 * share:.1f}%)")
    print(f"  con una quinta persona sempre presente ci si aspetterebbe circa il "
          f"{100 * expected:.1f}% (ogni persona vista nel {100 * per_person:.0f}% dei frame)")
    if share >= 0.5 * expected:
        print("  Compatibile con una quinta persona in campo o appena fuori per buona parte")
        print("  della partita: i suoi pezzi finiscono a uno dei giocatori della sua metà.")
    else:
        print("  Troppo pochi per una quinta persona in campo per buona parte della partita.")


def _whereabouts(obs) -> dict:
    """Median position, share outside the lines, share of time standing
    still (under 1 km/h) and median speed while moving."""
    xy = np.array([o.foot_court for o in obs], dtype=float)
    outside = ((xy[:, 0] < 0) | (xy[:, 0] > 10) | (xy[:, 1] < 0) | (xy[:, 1] > 20)).mean()
    speeds = []
    for a, b in zip(obs, obs[1:]):
        dt = b.timestamp_s - a.timestamp_s
        if 0 < dt <= 0.5:
            speeds.append(float(np.hypot(b.foot_court[0] - a.foot_court[0],
                                         b.foot_court[1] - a.foot_court[1])) / dt * 3.6)
    speeds_arr = np.array(speeds) if speeds else np.zeros(1)
    moving = speeds_arr[speeds_arr >= 1.0]
    return {
        "x": float(np.median(xy[:, 0])),
        "y": float(np.median(xy[:, 1])),
        "outside": float(outside),
        "still": float((speeds_arr < 1.0).mean()),
        "kmh": float(np.median(moving)) if moving.size else 0.0,
    }


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
