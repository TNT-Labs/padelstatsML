#!/usr/bin/env python3
"""Where the marked hits are lost: the ball, the hit models, the players.

    python training/hits/diagnose_hits.py RUN LABELS RUN LABELS [...]

The same pairs as train_hits.py, two matches or more: each match is read
with models trained on the others alone (a match never seen), and every
marked hit not found gets one cause, the first that applies:

1. ball not seen: no point of the ball's flights from 0.3 s before the hit
   to 0.45 s after (as late as a far hit is found);
2. ball seen on one side only: fewer than two points before, or after;
3. no player: the one who hit it has no box at that moment;
4. hit model: no candidate near it (probability under CANDIDATE_P);
5. decoding: a candidate was there, the sequence alternating between the
   halves left it out.

Then, of the hits found, how many have the half or the player wrong; of
the hits shown that are not real, how many are near a real one (a double:
a bounce, the glass) or near one missed (the right hit at the wrong time);
and how much of the ball is seen in the half second before and after the
hits, near and far half.

Needs scikit-learn, as train_hits.py.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_hits as th  # noqa: E402

from app.ml import ball_hits as bh  # noqa: E402

BEFORE_S, AFTER_S = 0.3, th.MATCH_S
COVER_S = 0.5
CAUSES = {
    "ball": "palla non vista",
    "one_side": "palla vista da un lato solo",
    "player": "giocatore senza riquadro",
    "model": "modello: nessun candidato",
    "decode": "decodifica: candidato scartato",
}


def diagnose(data: th.Data, model: bh.HitModel) -> list[dict]:
    """One row per marked hit (cause or found, half and player right) and
    per hit shown that matches none ("false")."""
    T = data.T
    rows = []
    for a, b in data.spans:
        lo, hi = bisect.bisect_left(T, a), bisect.bisect_right(T, b)
        cands = []
        if hi - lo >= 3:
            F = bh.point_features(T, data.X, data.Y, range(lo, hi), data.players, data.vp, data.height)
            cands = bh.candidates(model.hit.proba(F), model.far.proba(F), T[lo:hi])
        hits = bh.find_hits(T, data.X, data.Y, data.players, [{"start_s": a + bh.RALLY_LEAD_S, "end_s": b}],
                            data.vp, data.height, model)
        marked = [(t, p) for t, p in data.labels if a <= t <= b]
        matched: dict[int, bh.Hit] = {}
        for h in hits:                                   # as train_hits.measure pairs them
            near = [j for j, (t, _) in enumerate(marked) if j not in matched and abs(t - h.t) <= th.MATCH_S]
            if near:
                matched[min(near, key=lambda j: abs(marked[j][0] - h.t))] = h
        for j, (t, pid) in enumerate(marked):
            player = data.by_id.get(pid)
            far = player.far(t) if player else None
            before = int(((T >= t - BEFORE_S) & (T < t - 0.02)).sum())
            after = int(((T > t + 0.02) & (T <= t + AFTER_S)).sum())
            h = matched.get(j)
            if h is not None:
                cause = None
            elif before + after == 0:
                cause = "ball"
            elif before < 2 or after < 2:
                cause = "one_side"
            elif player is None or player.box(t) is None:
                cause = "player"
            elif not any(abs(c[0] - t) <= th.MATCH_S for c in cands):
                cause = "model"
            else:
                cause = "decode"
            rows.append({"t": t, "far": far, "cause": cause, "found": h is not None,
                         "half_ok": h is not None and h.far == far, "player_ok": h is not None and h.player == pid})
        for h in hits:
            if h not in matched.values():
                rows.append({"t": h.t, "false": True})
    # A hit shown that is not real: near a hit found (a double: a bounce,
    # the glass), near one missed (the right hit at the wrong time), or
    # away from every hit.
    marked = [r for r in rows if "cause" in r]
    for r in rows:
        if r.get("false"):
            near = min(marked, key=lambda m: abs(m["t"] - r["t"]), default=None)
            r["kind"] = ("alone" if near is None or abs(near["t"] - r["t"]) > 1.0
                         else "double" if near["found"] else "late")
    return rows


def coverage(data: th.Data, far: bool, fps: float) -> tuple[float, float, int]:
    """Share of frames with the ball in the COVER_S before and after the
    marked hits of one half, and how many hits."""
    before, after = [], []
    for t, pid in data.labels:
        player = data.by_id.get(pid)
        if player is None or player.far(t) != far:
            continue
        before.append(((data.T >= t - COVER_S) & (data.T < t)).sum() / (COVER_S * fps))
        after.append(((data.T > t) & (data.T <= t + COVER_S)).sum() / (COVER_S * fps))
    return (float(np.mean(before)) if before else 0.0, float(np.mean(after)) if after else 0.0, len(before))


def report(name: str, rows: list[dict]) -> None:
    marked = [r for r in rows if "cause" in r]
    false = [r for r in rows if r.get("false")]
    found = [r for r in marked if r["found"]]
    print(f"\n{name}: colpi segnati {len(marked)} · trovati {len(found)} · indicati non veri {len(false)}")
    for side, label in ((None, "tutti"), (False, "metà vicina"), (True, "metà lontana")):
        part = [r for r in marked if side is None or r["far"] is side]
        lost = Counter(r["cause"] for r in part if r["cause"])
        print(f"  persi, {label} ({len(part)} colpi): " + " · ".join(
            f"{CAUSES[c]} {lost.get(c, 0)}" for c in CAUSES))
    print(f"  dei trovati: metà campo sbagliata {sum(not r['half_ok'] for r in found)} · "
          f"giocatore sbagliato (metà giusta) {sum(r['half_ok'] and not r['player_ok'] for r in found)}")
    kinds = Counter(f["kind"] for f in false)
    double, late, alone = kinds["double"], kinds["late"], kinds["alone"]
    print(f"  non veri: entro 1 s da un colpo trovato {double} · entro 1 s da un colpo perso {late} · "
          f"lontani dai colpi {alone}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="+", metavar="RUN LABELS")
    args = parser.parse_args()
    if len(args.pairs) % 2 or len(args.pairs) < 4:
        raise SystemExit("Servono almeno due partite: coppie di analisi e colpi segnati.")
    with tempfile.TemporaryDirectory() as tmp:
        datas, names, fps = [], [], []
        for n in range(0, len(args.pairs), 2):
            run = Path(args.pairs[n])
            if run.is_file():
                folder = Path(tmp) / str(n)
                folder.mkdir()
                with tarfile.open(run) as archive:
                    archive.extractall(folder, filter="data")
                run = folder
            video = json.loads((run / "meta.json").read_text(encoding="utf-8")).get("video") or {}
            fps.append(float(video.get("fps") or 30.0))
            datas.append(th.Data(run, th.read_labels(args.pairs[n + 1])))
            names.append(Path(args.pairs[n]).name)
        every = []
        for d, data in enumerate(datas):
            model = th.loaded(th.as_model(*th.fit(datas, lambda i, k, d=d: i != d)))
            rows = diagnose(data, model)
            every += rows
            report(f"{names[d]} (modelli addestrati senza questa partita)", rows)
            for far in (False, True):
                b, a, n = coverage(data, far, fps[d])
                print(f"  palla vista, colpi {'lontani' if far else 'vicini'} ({n}): "
                      f"{b:.0%} dei fotogrammi nei {COVER_S} s prima, {a:.0%} dopo")
        report("Tutte le partite", every)


if __name__ == "__main__":
    main()
