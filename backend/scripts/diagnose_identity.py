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
from app.ml.artifacts import (  # noqa: E402
    completed_runs,
    load_artifacts,
    read_observations,
    resolve_artifacts_dir,
)
from app.ml.detect import part_of_another                              # noqa: E402
from app.ml.court import COURT_LENGTH_M, COURT_WIDTH_M                 # noqa: E402
from app.ml.roles import CHANGEOVER_PENALTY, DRIVE, TEAM_WINDOW_S, assign_roles  # noqa: E402
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
    if not artifacts.complete:
        others = [p.name[:8] for p in completed_runs(directory.parent) if p != directory]
        print(f"L'analisi {directory.name[:8]} è ancora in corso ({len(artifacts.tracklets)} tracce finora): "
              "riprova quando è finita.", file=sys.stderr)
        if others:
            print(f"Analisi concluse: {', '.join(others)} — passa l'id (basta il prefisso).", file=sys.stderr)
        return 1
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
        print(f"aspetto: re-ID · stessa persona {appearance.same_median:.2f}"
              f" · persone diverse {appearance.different_median:.2f}")
        print(f"  soglia {appearance.veto:.2f}"
              f" ({appearance.positives} + {appearance.negatives} coppie misurate)")
        if gap < 0.15:
            print("  Distanze vicine: il re-ID distingue poco questi giocatori")
            print("  (inquadratura lontana, poca risoluzione, divise simili).")
    else:
        print(f"aspetto: colore della divisa · {appearance_summary.get('reason', '')}")
    print()

    if identity.roles_apply(pieces) and not args.collegamento:
        _report_roles(pieces, appearance, total, expected)
    else:
        _report_linking(pieces, appearance, speed, period, total, expected)
    _report_crowding(directory, artifacts)
    if identity.roles_apply(pieces) and not args.collegamento:
        _report_behind_walls(tracklets, speed, expected)
    return 0


def _report_roles(pieces, appearance, total: int, expected: int) -> None:
    assignment = assign_roles(pieces, appearance)
    print("metodo: coppia e lato in campo")
    print(f"cambi di campo trovati: {len(assignment.changeovers)}")
    times = [_clock(t) for t in assignment.changeovers]
    for start in range(0, len(times), 8):
        print("  a " + ", ".join(times[start:start + 8]))
    print("  Confrontali con il video: un cambio mancato scambia i")
    print("  numeri delle due coppie da lì in poi.")
    names = {"colore": "colore divise", "reid": "re-ID"}
    for alternative in assignment.alternatives:
        chosen = " ← usato" if alternative is assignment.timeline else ""
        print(f"  {names[alternative.cue]:<13} nitidezza {alternative.clarity:6.1f}"
              f" · {len(alternative.changeovers)} cambi{chosen}")
    print()
    _report_timeline(assignment.timeline)

    print("giocatori")
    print("  x: mediana dal proprio lato (revés ~2-4 m, drive ~6-8 m)")
    print("  rete: distanza mediana dalla rete · aspetto: quota data")
    print("  dall'aspetto contro il lato in cui stava (scambi di lato)")
    print(f"  {'':>3} {'coppia':>7} {'lato':>5} {'tracc.':>6} {'x':>4} {'rete':>4}"
          f" {'fermo':>5} {'km/h':>4} {'aspetto':>7}")
    kept = 0
    own_x: dict[tuple[int, int], float] = {}
    for n, (team, role) in enumerate(identity._ROLE_ORDER):
        group = assignment.players.get((team, role), [])
        if not group:
            print(f"  G{n + 1}  nessuna traccia")
            continue
        observations = identity.one_per_frame([rp.tracklet for rp in group])
        if not observations:
            print(f"  G{n + 1}  tutte le rilevazioni contese")
            continue
        kept += len(observations)
        where = _whereabouts(observations)
        own_x[(team, role)] = float(np.median([_own_x(o) for o in observations]))
        from_net = float(np.median([abs(o.foot_court[1] - COURT_LENGTH_M / 2) for o in observations]))
        # Pieces whose side said the other role: given here by appearance.
        by_look = sum(len(rp.tracklet) for rp in group if (rp.side_score > 0) != (role == DRIVE))
        by_all = sum(len(rp.tracklet) for rp in group)
        print(f"  G{n + 1} {'vicina' if team == 0 else 'lontana':>7} {'drive' if role == DRIVE else 'revés':>5}"
              f" {100 * len(observations) / expected:5.0f}% {own_x[(team, role)]:4.1f} {from_net:4.1f}"
              f" {100 * where['still']:4.0f}% {where['kmh']:4.1f} {100 * by_look / max(by_all, 1):6.0f}%")
    print(f"  {kept} di {total} osservazioni ai 4 giocatori ({100 * kept / total:.0f}%)")
    for team in (0, 1):
        pair = [own_x.get((team, role)) for role in (0, 1)]
        if None not in pair and abs(pair[1] - pair[0]) < 2.0:
            print(f"  Coppia {'vicina' if team == 0 else 'lontana'}: i due lati distano meno di 2 m:")
            print("  i compagni non tengono un lato, o i loro pezzi sono mescolati.")
    print()
    _report_contested(assignment)


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _own_x(obs) -> float:
    """x from the player's own point of view: the far pair faces the camera."""
    x, y = obs.foot_court
    return x if y < COURT_LENGTH_M / 2 else COURT_WIDTH_M - x


def _report_timeline(timeline) -> None:
    """One character per window: which arrangement of the pairs was chosen,
    and whether the look of the two halves agreed. A missed changeover
    shows as a long run of '!' — strong evidence against the choice."""
    if not len(timeline.swapped):
        return
    typical = timeline.penalty / CHANGEOVER_PENALTY
    per_line = int(round(600 / TEAM_WINDOW_S))           # ten minutes a line
    marks = []
    for swapped, evidence in zip(timeline.swapped, timeline.evidence):
        letter = "B" if swapped else "A"
        if np.isnan(evidence):
            marks.append("?")
            continue
        # evidence > 0: the kick-off arrangement fits the look better.
        agrees = evidence * (-1 if swapped else 1)
        if agrees >= typical:
            marks.append(letter)
        elif agrees <= -typical:
            marks.append("!")
        else:
            marks.append(letter.lower())
    print("coppie nel tempo (un carattere ogni 20 s, uno spazio al minuto)")
    print("  A: coppie come all'inizio · B: coppie scambiate")
    print("  maiuscola: l'aspetto delle metà lo conferma · minuscola:")
    print("  prova debole · !: l'aspetto dice il contrario · ?: metà vuota")
    per_minute = int(round(60 / TEAM_WINDOW_S))
    for start in range(0, len(marks), per_line):
        chunk = marks[start:start + per_line]
        text = " ".join("".join(chunk[i:i + per_minute]) for i in range(0, len(chunk), per_minute))
        print(f"  {int(start * TEAM_WINDOW_S // 60):>3}' {text}")
    known = [m for m in marks if m != "?"]
    if known:
        strong = sum(1 for m in known if m.isupper())
        against = sum(1 for m in known if m == "!")
        print(f"  confermate {100 * strong / len(known):.0f}% · contrarie {100 * against / len(known):.0f}%"
              f" delle finestre")
        if strong < 0.3 * len(known):
            print("  L'aspetto distingue poco le due coppie: i cambi di campo")
            print("  possono sfuggire. Confronta con il video.")
    print()


def _report_contested(assignment) -> None:
    """Frames in which one player was given two detections, and what the
    second one is. A few tens of centimetres apart: the same person
    detected twice. Metres apart with the team-mate seen elsewhere in the
    same frame: three people on one half — a coach, a spectator, a
    reflection in the glass. Metres apart without the team-mate: most
    likely the team-mate, with a piece given to the wrong side."""
    frames_of = {key: {o.frame_index for rp in group for o in rp.tracklet.observations}
                 for key, group in assignment.players.items()}
    same = third = mate = 0
    contested = []
    for (team, role), group in assignment.players.items():
        partner_frames = frames_of.get((team, 1 - role), set())
        by_frame: dict[int, list] = {}
        for rp in group:
            for obs in rp.tracklet.observations:
                by_frame.setdefault(obs.frame_index, []).append(obs)
        for frame, found in by_frame.items():
            if len(found) < 2:
                continue
            contested.extend(found)
            a, b = found[0].foot_court, found[1].foot_court
            if np.hypot(a[0] - b[0], a[1] - b[1]) < 0.5:
                same += 1
            elif frame in partner_frames:
                third += 1
            else:
                mate += 1
    total = same + third + mate
    if not total:
        return
    print(f"frame contesi (due rilevazioni allo stesso giocatore): {total}")
    for label, count in (("stessa persona due volte (entro 0,5 m)", same),
                         ("terza persona nella metà (compagno visto)", third),
                         ("probabile compagno dal lato sbagliato", mate)):
        print(f"  {label:<44}{100 * count / total:4.0f}%")
    xy = np.array([o.foot_court for o in contested])
    outside = (xy[:, 0] < 0) | (xy[:, 0] > COURT_WIDTH_M) | (xy[:, 1] < 0) | (xy[:, 1] > COURT_LENGTH_M)
    behind = (xy[:, 1] < 0) | (xy[:, 1] > COURT_LENGTH_M)
    print(f"  rilevazioni contese fuori dalle linee: {100 * outside.mean():.0f}%"
          f" (dietro i fondi: {100 * behind.mean():.0f}%)")
    if third > max(same, mate):
        print("  Per lo più una persona in più in una metà: qualcuno in")
        print("  campo o appena fuori, o un riflesso nel vetro.")
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
    print(f"  rilevazioni doppie (parte di un giocatore): {parts}"
          f" ({100 * parts / max(len(stored), 1):.1f}%), escluse")
    print(f"  frame con più di {identity.N_PLAYERS} persone: {crowded} di {frames}"
          f" ({100 * share:.1f}%)")
    print(f"  attesi con una quinta persona sempre presente: {100 * expected:.1f}%")
    print(f"  (ogni persona è vista nel {100 * per_person:.0f}% dei frame)")
    if share >= 0.5 * expected:
        print("  Compatibile con una quinta persona in campo o appena fuori")
        print("  per buona parte della partita: vedi i frame contesi sopra.")
    else:
        print("  Troppo pochi per una quinta persona presente a lungo.")


# How far past the back walls a detection is tested for exclusion, in metres.
_BEHIND_MARGINS = (1.0, 0.5, 0.25)


def _behind_wall_m(obs) -> float:
    """How far behind the back glass the feet are: 0 on court. The back
    walls stand on the baselines, so a player cannot be there — only people
    behind the glass, reflections in it, or the foot point's error."""
    y = obs.foot_court[1]
    return max(-y, y - COURT_LENGTH_M, 0.0)


def _outside_side_m(obs) -> float:
    x = obs.foot_court[0]
    return max(-x, x - COURT_WIDTH_M, 0.0)


def _report_behind_walls(tracklets, speed: float, expected: int) -> None:
    """Where the detections outside the lines are, and what excluding the
    ones behind the back walls would do to the players — measured, before
    deciding a threshold."""
    observations = [o for t in tracklets for o in t.observations]
    total = len(observations)
    behind = np.array([_behind_wall_m(o) for o in observations])
    side = np.array([_outside_side_m(o) for o in observations])
    print()
    print("rilevazioni fuori dalle linee, per distanza in metri")
    for label, values in (("dietro fondi", behind), ("oltre lati", side)):
        cells = " · ".join(
            f"≤{hi:g} {100 * ((values > lo) & (values <= hi)).mean():.1f}%"
            for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5))
        )
        print(f"  {label:<13}{cells}")
    moving = _still_share(tracklets, lambda o: _behind_wall_m(o) > 0.5)
    on_court = _still_share(tracklets, lambda o: _behind_wall_m(o) == 0 and _outside_side_m(o) == 0)
    if moving is not None and on_court is not None:
        print(f"  ferme: {100 * moving:.0f}% di quelle oltre 0,5 m dai fondi,"
              f" {100 * on_court:.0f}% di quelle in campo")

    print("se si escludessero quelle oltre X m dietro i fondi")
    print("(prova: richiede circa un minuto)")
    print(f"  {'X':>10} {'cambi':>5} {'ai 4':>5} {'contese':>7}   tracciati G1-G4")
    for margin in (None, *_BEHIND_MARGINS):
        kept = tracklets if margin is None else [
            Tracklet(id=t.id, observations=[o for o in t.observations if _behind_wall_m(o) <= margin])
            for t in tracklets
        ]
        kept = [t for t in kept if len(t) >= 3]
        result = identity.resolve_players(kept, max_speed_ms=speed)
        attributed = sum(len(p) for p in result.players)
        tracked = " ".join(f"{100 * len(p) / expected:3.0f}%" for p in result.players)
        label = "nessuna" if margin is None else f"{margin:g} m"
        print(f"  {label:>10} {len(result.changeovers):>5} {100 * attributed / total:4.0f}%"
              f" {100 * result.observations_discarded / total:6.0f}%   {tracked}")
    print("  ai 4, contese: quote di tutte le rilevazioni, escluse comprese")


def _still_share(tracklets, selected) -> float | None:
    """Share of the selected detections standing still (under 1 km/h)."""
    still = count = 0
    for tracklet in tracklets:
        obs = tracklet.observations
        for a, b in zip(obs, obs[1:]):
            dt = b.timestamp_s - a.timestamp_s
            if not selected(b) or not 0 < dt <= 0.5:
                continue
            speed = float(np.hypot(b.foot_court[0] - a.foot_court[0], b.foot_court[1] - a.foot_court[1])) / dt
            count += 1
            still += speed * 3.6 < 1.0
    return still / count if count else None


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
