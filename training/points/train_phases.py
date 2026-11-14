#!/usr/bin/env python3
"""Train the model that reads the hits as serve / in play / between points
(backend/app/ml/phases.py) on the points marked in the app ("Segna i punti").

    python training/points/train_phases.py \\
        --set primo   primo_run.tgz   training/points/labels/primo_set.txt \\
        --set secondo secondo_run.tgz training/points/labels/secondo_set.txt training/hits/labels/secondo_set.txt \\
        --set terzo   palla_run.tgz   training/points/labels/terzo_set.txt   training/hits/labels/terzo_set.txt

Each --set: a name, the run (folder or .tgz, as for training/hits), its
marked points, and its marked hits when there are any. Three sets or more.

The hits of each set are found as the analysis finds them, with hit models
that never saw that set (trained on the other sets with marked hits: for a
set without, on all of them) — the hits a new match will have. Each hit
in the stretch of marked points is labelled: the first of a point (within
SERVE_WITHIN_S of its marked serve) a serve, the others in play, those
outside every point a pause.

Measured with each set left out in turn: the models and the thresholds
(chosen on the other sets' predictions, each made by models that did not
see it) never saw it. That measure goes into the model and the app. Then
everything is trained on all the sets and written to phase_model.json.

Needs scikit-learn, as training/hits/train_hits.py.
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
sys.path.insert(0, str(ROOT / "training" / "hits"))
import train_hits as th  # noqa: E402  (puts backend/ on the path)

from app.core.config import get_settings  # noqa: E402
from app.ml import ball_hits as bh  # noqa: E402
from app.ml import phases as ph  # noqa: E402
from app.ml import pipeline as pl  # noqa: E402
from app.ml.artifacts import load_artifacts  # noqa: E402
from app.ml.point_eval import Span, compare, read_points  # noqa: E402
from app.ml.reprocess import pipeline_config, recompute  # noqa: E402

PAD_S = 0.7               # a hit this close outside a marked point belongs to it
SERVE_WITHIN_S = 1.5
OUTSIDE_S = 3.0           # hits this far beyond the marked stretch are not labelled
GRID = [(ts, tp, gap, lone) for ts in (0.4, 0.5, 0.6, 0.7) for tp in (0.5, 0.6, 0.7, 0.8, 1.01)
        for gap in (5.0, 6.0, 8.0, 99.0) for lone in (0.9,)]


class SetData:
    """A set's hits as the analysis finds them, their features and labels."""

    def __init__(self, name: str, run: Path, points_file: str, hit_model: bh.HitModel) -> None:
        self.name = name
        self.points = read_points(Path(points_file).read_text(encoding="utf-8"))
        seen: dict = {}
        original = pl.refine_rallies

        def spy(rallies, ball, net_distance=None, where=None, phases=None):
            seen.update(rallies=list(rallies), ball=ball, where=where)
            return original(rallies, ball, net_distance, where, None)
        pl.refine_rallies = spy
        try:
            config = pipeline_config(get_settings(), load_artifacts(run).meta)
            _, _, self.before, _ = recompute(run, None, config, hit_model=hit_model, phase_model=False)
        finally:
            pl.refine_rallies = original
        if "ball" not in seen:
            raise SystemExit(f"{name}: l'analisi non ha la palla, o i colpi non sono stati letti.")
        self.rallies, self.where = seen["rallies"], seen["where"]
        self.hits = bh.zone_hits(self.rallies, seen["ball"])
        self.F = ph.phase_features(self.hits, self.rallies, self.where)
        self.y = self._labels()

    def _labels(self) -> np.ndarray:
        """2 serve, 1 in play, 0 between points, -1 outside the marked stretch."""
        pts = self.points
        lo, hi = pts[0].start_s - OUTSIDE_S, pts[-1].end_s + OUTSIDE_S
        out = []
        for h in self.hits:
            if not lo <= h.t <= hi:
                out.append(-1)
                continue
            point = next((p for p in pts if p.start_s - PAD_S <= h.t <= p.end_s + PAD_S), None)
            if point is None:
                out.append(0)
                continue
            first = min(x.t for x in self.hits if point.start_s - PAD_S <= x.t <= point.end_s + PAD_S)
            out.append(2 if h.t == first and h.t - point.start_s <= SERVE_WITHIN_S else 1)
        return np.array(out, dtype=int)

    def spans(self, p_pause, p_serve, setting) -> list[Span]:
        return [Span(a, b + ph.TAIL_S, s) for a, b, _, s in ph.segment(self.hits, p_pause, p_serve, *setting)]


def gb(X, y):
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=0).fit(X, y)


def fit(sets: list[SetData]):
    X = np.vstack([s.F[s.y >= 0] for s in sets])
    y = np.concatenate([s.y[s.y >= 0] for s in sets])
    return gb(X, (y == 0).astype(int)), gb(X[y > 0], (y[y > 0] == 2).astype(int)), X, y


def predict(models, s: SetData):
    pause, serve = models[0], models[1]
    return pause.predict_proba(s.F)[:, 1], serve.predict_proba(s.F)[:, 1]


FALSE_COST = 0.5


def worth(summary: dict) -> float:
    """What a setting is chosen for: points right one to one, less
    FALSE_COST for each false rally (a split point already counts as not
    right)."""
    return summary["one_to_one"] - FALSE_COST * summary["false_rallies"]


def choose(sets: list[SetData]) -> tuple:
    """The setting best on these sets, each read by models that did not see it."""
    preds = {s.name: predict(fit([o for o in sets if o is not s]), s) for s in sets}
    return max(GRID, key=lambda g: sum(worth(compare(s.points, s.spans(*preds[s.name], g)).summary())
                                       for s in sets))


def summary_line(name: str, s: dict) -> str:
    return (f"{name}: giusti {s['one_to_one']}/{s['points']} · persi {s['lost']} · spezzati {s['split']} · "
            f"uniti ad altri {s['joined']} · scambi falsi {s['false_rallies']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="sets", action="append", nargs="+", required=True,
                        metavar="NOME RUN PUNTI [COLPI]")
    parser.add_argument("--out", default=str(ph.PHASE_MODEL_PATH))
    parser.add_argument("--false-cost", type=float, default=FALSE_COST,
                        help="quanto pesa uno scambio falso rispetto a un punto giusto, nella scelta delle soglie")
    args = parser.parse_args()
    globals()["FALSE_COST"] = args.false_cost
    if len(args.sets) < 3 or any(len(s) not in (3, 4) for s in args.sets):
        raise SystemExit("Servono almeno tre --set, ognuno: nome, analisi, punti segnati [, colpi segnati].")

    with tempfile.TemporaryDirectory() as tmp:
        runs = []
        for n, (name, run, *_rest) in enumerate(args.sets):
            path = Path(run)
            if path.is_file():
                folder = Path(tmp) / str(n)
                folder.mkdir()
                with tarfile.open(path) as archive:
                    archive.extractall(folder, filter="data")
                path = folder
            runs.append(path)
        hit_data = {s[0]: th.Data(runs[n], th.read_labels(s[3])) for n, s in enumerate(args.sets) if len(s) == 4}
        if not hit_data:
            raise SystemExit("Almeno un --set deve avere i colpi segnati: servono per trovare i colpi.")
        sets = []
        for n, (name, _run, points, *_rest) in enumerate(args.sets):
            others = [d for k, d in hit_data.items() if k != name]
            if not others:
                raise SystemExit(f"{name}: nessun altro set con colpi segnati per i modelli dei colpi.")
            hit_model = th.loaded(th.as_model(*th.fit(others, lambda i, k: True)))
            sets.append(SetData(name, runs[n], points, hit_model))
            y = sets[-1].y
            print(f"{name}: punti {len(sets[-1].points)} · colpi trovati nel tratto {int((y >= 0).sum())}: "
                  f"battute {int((y == 2).sum())}, in gioco {int((y == 1).sum())}, nelle pause {int((y == 0).sum())}")

        # Each set left out in turn: models and setting never saw it.
        keys = ("points", "one_to_one", "lost", "split", "joined", "false_rallies")
        total = dict.fromkeys(keys, 0)
        before = dict.fromkeys(keys, 0)
        for s in sets:
            others = [o for o in sets if o is not s]
            setting = choose(others)
            got = compare(s.points, s.spans(*predict(fit(others), s), setting)).summary()
            old = compare(s.points, [Span(r.start_s, r.end_s, r.serve) for r in s.before]).summary()
            print("  " + summary_line(f"{s.name}, mai visto", got) + f"   (prima: giusti {old['one_to_one']}, "
                  f"uniti {old['joined']}, falsi {old['false_rallies']})")
            for k in keys:
                total[k] += got[k]
                before[k] += old[k]
        print(summary_line("Tutti, ogni set mai visto", total))
        print(summary_line("Prima, senza questo modello", before))

        setting = choose(sets)
        pause, serve, X, y = fit(sets)
        out = {
            "version": 1, "features": ph.N_PHASE_FEATURES,
            "pause": th.export_trees(pause, X), "serve": th.export_trees(serve, X[y > 0]),
            "segment": dict(zip(("serve_threshold", "pause_threshold", "gap_s", "lone_serve"), setting)),
            "measured": {"sets": len(sets), **total, "before": before, "unseen_set": True},
        }
        print(f"Regolazione: battuta {setting[0]} · pausa {setting[1]} · buco {setting[2]} s · battuta sola {setting[3]}")
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Scritto {args.out}")


if __name__ == "__main__":
    main()
