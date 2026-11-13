#!/usr/bin/env python3
"""Train the hit models on hits marked in the app ("Segna i colpi").

    python training/hits/train_hits.py RUN LABELS [RUN LABELS ...] [--out backend/app/ml/hit_model.json]

RUN: the run's folder (meta.json, tracks.jsonl, ball.jsonl) or a .tgz of it,
as made on the Pi with
    docker exec padel-worker sh -c 'cd /data/artifacts/<id> && tar czf /data/run.tgz meta.json tracks.jsonl ball.jsonl result.json'
LABELS: the text "Copia i colpi segnati" gives (minute:seconds, name, player id).

Marked hits close together (under GROUP_GAP_S) make a group, about one
rally; each group is left out in turn, the models trained on the others
and measured on it. With more than one match, each match is also left out
whole: the models trained on the others, measured on a match they never
saw — lower, and what a new match should expect; that is the measure
stored in the model and shown in the app. Then the models are trained on every group
and written for app/ml/ball_hits.py.

Needs scikit-learn (pip install scikit-learn) on the computer that trains;
the Pi only reads the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml import ball_hits as bh  # noqa: E402
from app.ml.artifacts import load_artifacts, read_ball  # noqa: E402
from app.ml.identity import resolve_players  # noqa: E402

GROUP_GAP_S = 4.0
GROUP_MARGIN_S = 1.0
POSITIVE_S = 0.1          # ball points this close to a marked hit are hits
MATCH_S = 0.45            # a hit found this close to a marked one is it


def read_labels(path: str) -> list[tuple[float, int]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2 or ":" not in parts[0]:
            continue
        try:
            minutes, seconds = parts[0].split(":")
            out.append((int(minutes) * 60 + float(seconds), int(parts[-1])))
        except ValueError:
            continue
    return sorted(out)


def groups(labels):
    spans: list[list[float]] = []
    for t, _ in labels:
        if spans and t - spans[-1][1] < GROUP_GAP_S:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return [(a - GROUP_MARGIN_S, b + GROUP_MARGIN_S) for a, b in spans]


def gb(n, depth, **kw):
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(n_estimators=n, max_depth=depth, learning_rate=0.05, random_state=0, **kw)


def export_trees(clf, X) -> dict:
    trees = []
    for est in clf.estimators_[:, 0]:
        t = est.tree_
        trees.append({"left": t.children_left.tolist(), "right": t.children_right.tolist(),
                      "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                      "value": t.value[:, 0, 0].tolist()})
    rate = float(clf.learning_rate)
    raw = clf.decision_function(X[:1])[0]
    tree_sum = sum(float(est.predict(X[:1].astype(np.float32))[0]) for est in clf.estimators_[:, 0])
    out = {"init": float(raw - rate * tree_sum), "rate": rate, "trees": trees}
    check = bh.Trees(out).proba(X[:200])
    assert np.allclose(check, clf.predict_proba(X[:200])[:, 1], atol=1e-6), "esportazione degli alberi non fedele"
    return out


class Data:
    def __init__(self, run: Path, labels: list[tuple[float, int]]) -> None:
        artifacts = load_artifacts(run)
        video = artifacts.meta.get("video") or {}
        self.height = int(video.get("height") or artifacts.calibration.frame_size[1])
        speed = float((artifacts.meta.get("config") or {}).get("max_player_speed_ms") or 8.0)
        identity = resolve_players(artifacts.tracklets, max_speed_ms=speed)
        self.players = [bh.PlayerTrack.from_player(p) for p in identity.players if p.observations]
        self.by_id = {p.player_id: p for p in self.players}
        self.vp = bh.vanishing_point(artifacts.calibration)
        self.T, self.X, self.Y = bh.ball_points(read_ball(run), self.height)
        self.labels = labels
        self.spans = groups(labels)
        self.group = np.full(len(self.T), -1)
        for k, (a, b) in enumerate(self.spans):
            self.group[(self.T >= a) & (self.T <= b)] = k
        idx = np.where(self.group >= 0)[0]
        self.idx = idx
        self.F = bh.point_features(self.T, self.X, self.Y, idx, self.players, self.vp, self.height)
        self.g = self.group[idx]
        self.y = np.zeros(len(idx), int)
        self.far = np.zeros(len(idx), int)
        for t, pid in labels:
            near = np.abs(self.T[idx] - t) <= POSITIVE_S
            self.y[near] = 1
            self.far[near] = int(self.by_id[pid].far(t)) if pid in self.by_id else 0

    def partner_rows(self, hits):
        """(features, first partner hit it, group) for hits (t, pid): at the
        marked time and as late as a hit is found (far hits up to MATCH_S)."""
        rows = []
        for t0, pid in hits:
          for t in (t0, t0 + 0.15, t0 + 0.3):
            if pid not in self.by_id:
                continue
            side = bh.pair_at(self.players, t, self.by_id[pid].far(t0))
            if len(side) != 2 or pid not in (side[0].player_id, side[1].player_id):
                continue
            k = int(np.searchsorted([s[0] for s in self.spans], t0, side="right") - 1)
            rows.append((bh.partner_features(self.T, self.X, self.Y, side[0], side[1], t, self.height),
                         int(side[0].player_id == pid), k))
        return rows


def fit(datas: list[Data], keep) -> tuple:
    """The three models on the groups `keep(data index, group)` lets in."""
    from sklearn.linear_model import LogisticRegression

    F, y, far, Xp, yp = [], [], [], [], []
    for d, data in enumerate(datas):
        mask = np.array([keep(d, int(g)) for g in data.g], dtype=bool)
        F.append(data.F[mask]); y.append(data.y[mask]); far.append(data.far[mask])
        for features, first, k in data.partner_rows(data.labels):
            if keep(d, k):
                Xp.append(features); yp.append(first)
    F, y, far = np.concatenate(F), np.concatenate(y), np.concatenate(far)
    hit = gb(150, 3, subsample=0.8).fit(F, y)
    side = gb(100, 2).fit(F[y == 1], far[y == 1])
    partner = LogisticRegression(max_iter=2000).fit(np.array(Xp), np.array(yp))
    return hit, side, partner, F, y


def as_model(hit, side, partner, F, y, measured=None) -> dict:
    return {"version": 1, "features": bh.N_FEATURES, "measured": measured or {},
            "hit": export_trees(hit, F), "far": export_trees(side, F[y == 1]),
            "partner": {"coef": partner.coef_[0].tolist(), "intercept": float(partner.intercept_[0])}}


def loaded(model: dict) -> bh.HitModel:
    return bh.HitModel(bh.Trees(model["hit"]), bh.Trees(model["far"]), bh.Linear(model["partner"]), {})


def measure(data: Data, model: bh.HitModel, groups) -> list[int]:
    """[marked, found, shown, right half, right player] on `groups` of `data`."""
    marked_n = found = shown = right_half = right_player = 0
    for k in groups:
        a, b = data.spans[k]
        hits = bh.find_hits(data.T, data.X, data.Y, data.players, [{"start_s": a + bh.RALLY_LEAD_S, "end_s": b}],
                            data.vp, data.height, model)
        marked = [(t, p) for t, p in data.labels if a <= t <= b]
        marked_n += len(marked)
        used: set[int] = set()
        shown += len(hits)
        for h in hits:
            near = [j for j, (t, _) in enumerate(marked) if j not in used and abs(t - h.t) <= MATCH_S]
            if not near:
                continue
            j = min(near, key=lambda j: abs(marked[j][0] - h.t))
            used.add(j)
            found += 1
            pid = marked[j][1]
            if pid in data.by_id and data.by_id[pid].far(marked[j][0]) == h.far:
                right_half += 1
                right_player += h.player == pid
    return [marked_n, found, shown, right_half, right_player]


def shares(c: list[int]) -> dict:
    n, found, shown, half, player = c
    return {"marked": n, "found": round(found / max(n, 1), 3), "real": round(found / max(shown, 1), 3),
            "half": round(half / max(found, 1), 3), "player": round(player / max(found, 1), 3)}


def say(what: str, m: dict) -> None:
    print(f"{what}: colpi {m['marked']} · trovati {m['found']:.0%} · giusti {m['real']:.0%} · "
          f"metà campo {m['half']:.0%} · giocatore {m['player']:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="+", metavar="RUN LABELS",
                        help="una o più coppie: analisi (cartella o .tgz) e colpi segnati di quella partita")
    parser.add_argument("--out", default=str(ROOT / "backend" / "app" / "ml" / "hit_model.json"))
    args = parser.parse_args()
    if len(args.pairs) % 2:
        raise SystemExit("Servono coppie: analisi e colpi segnati, per ogni partita.")

    with tempfile.TemporaryDirectory() as tmp:
        datas, names = [], []
        for n in range(0, len(args.pairs), 2):
            run = Path(args.pairs[n])
            if run.is_file():
                folder = Path(tmp) / str(n)
                folder.mkdir()
                with tarfile.open(run) as archive:
                    archive.extractall(folder, filter="data")
                run = folder
            data = Data(run, read_labels(args.pairs[n + 1]))
            datas.append(data)
            names.append(Path(args.pairs[n]).name)
            print(f"{names[-1]}: colpi segnati {len(data.labels)} in {len(data.spans)} gruppi")

        # A match the models never saw: trained on the others, measured on it.
        unseen = [0] * 5
        if len(datas) > 1:
            for d, data in enumerate(datas):
                model = loaded(as_model(*fit(datas, lambda i, k, d=d: i != d)))
                counts = measure(data, model, range(len(data.spans)))
                unseen = [a + b for a, b in zip(unseen, counts)]
                say(f"Mai vista in addestramento · {names[d]}", shares(counts))
            say("Mai viste, insieme", shares(unseen))

        # Every group left out in turn, of all the matches together.
        total = [0] * 5
        for d, data in enumerate(datas):
            for k in range(len(data.spans)):
                model = loaded(as_model(*fit(datas, lambda i, g, d=d, k=k: (i, g) != (d, k))))
                total = [a + b for a, b in zip(total, measure(data, model, [k]))]
        measured = shares(total)
        say("Gruppi esclusi a turno, tutte le partite", measured)
        if len(datas) > 1:
            # What the app shows: how the models do on a match they never
            # saw, the lower and the truer of the two measures.
            measured = {**shares(unseen), "matches": len(datas), "unseen_match": True}

        out = as_model(*fit(datas, lambda i, k: True), measured=measured)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Scritto {args.out}")


if __name__ == "__main__":
    main()
