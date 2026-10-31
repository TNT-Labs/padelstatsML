#!/usr/bin/env python3
"""Train the hit models on hits marked in the app ("Segna i colpi").

    python training/hits/train_hits.py RUN LABELS [--out backend/app/ml/hit_model.json]

RUN: the run's folder (meta.json, tracks.jsonl, ball.jsonl) or a .tgz of it,
as made on the Pi with
    docker exec padel-worker sh -c 'cd /data/artifacts/<id> && tar czf /data/run.tgz meta.json tracks.jsonl ball.jsonl result.json'
LABELS: the text "Copia i colpi segnati" gives (minute:seconds, name, player id).

Marked hits close together (under GROUP_GAP_S) make a group, about one
rally; each group is left out in turn, the models trained on the others
and measured on it: the numbers printed, and stored in the model, are on
rallies the models never saw. Then the models are trained on every group
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


def fit(data: Data, train: np.ndarray, partner_rows):
    from sklearn.linear_model import LogisticRegression

    hit = gb(150, 3, subsample=0.8).fit(data.F[train], data.y[train])
    pos = train & (data.y == 1)
    far = gb(100, 2).fit(data.F[pos], data.far[pos])
    Xp = np.array([r[0] for r in partner_rows]); yp = np.array([r[1] for r in partner_rows])
    partner = LogisticRegression(max_iter=2000).fit(Xp, yp)
    return hit, far, partner


def evaluate(data: Data) -> dict:
    rows = data.partner_rows(data.labels)
    found = shown = right_half = right_player = 0
    for k, (a, b) in enumerate(data.spans):
        train = data.g != k
        hit, far, partner = fit(data, train, [r for r in rows if r[2] != k])
        model = bh.HitModel(bh.Trees(export_trees(hit, data.F[train])), bh.Trees(export_trees(far, data.F[train])),
                            bh.Linear({"coef": partner.coef_[0].tolist(), "intercept": float(partner.intercept_[0])}), {})
        hits = bh.find_hits(data.T, data.X, data.Y, data.players, [{"start_s": a + bh.RALLY_LEAD_S, "end_s": b}],
                            data.vp, data.height, model)
        marked = [(t, p) for t, p in data.labels if a <= t <= b]
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
    n = len(data.labels)
    return {"marked": n, "found": round(found / n, 3), "real": round(found / max(shown, 1), 3),
            "half": round(right_half / max(found, 1), 3), "player": round(right_player / max(found, 1), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("labels")
    parser.add_argument("--out", default=str(ROOT / "backend" / "app" / "ml" / "hit_model.json"))
    args = parser.parse_args()

    run = Path(args.run)
    with tempfile.TemporaryDirectory() as tmp:
        if run.is_file():
            with tarfile.open(run) as archive:
                archive.extractall(tmp, filter="data")
            run = Path(tmp)
        data = Data(run, read_labels(args.labels))
        print(f"Colpi segnati {len(data.labels)} in {len(data.spans)} gruppi · punti della palla {len(data.idx)}")
        measured = evaluate(data)
        print(f"Su scambi esclusi dall'addestramento: colpi trovati {measured['found']:.0%}, "
              f"giusti {measured['real']:.0%}, metà campo {measured['half']:.0%}, giocatore {measured['player']:.0%}")
        everything = np.ones(len(data.idx), bool)
        hit, far, partner = fit(data, everything, data.partner_rows(data.labels))
        out = {
            "version": 1,
            "features": bh.N_FEATURES,
            "measured": measured,
            "hit": export_trees(hit, data.F),
            "far": export_trees(far, data.F[data.y == 1]),
            "partner": {"coef": partner.coef_[0].tolist(), "intercept": float(partner.intercept_[0])},
        }
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Scritto {args.out}")


if __name__ == "__main__":
    main()
