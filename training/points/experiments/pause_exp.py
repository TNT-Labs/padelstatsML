"""Experiment: the hit model trained with the pauses between marked points as
negatives. Cross-set: measured on a set whose pauses (and hits) it never saw."""
import sys, pickle, copy
from pathlib import Path
import numpy as np
S = Path(__file__).parent  # cartella con runs/b, runs/c estratti e datas.pkl (vedi README)
sys.path.insert(0, "/home/user/padelstatsML/backend"); sys.path.insert(0, "/home/user/padelstatsML/training/hits")
import train_hits as th
from app.ml import ball_hits as bh
import app.ml.pipeline as pl
from app.ml import reprocess
from app.ml.pipeline import PipelineConfig
from app.ml.point_eval import read_points, compare, Span

ROOT = Path("/home/user/padelstatsML")
PAD = 0.7
datas = pickle.loads((S / "datas.pkl").read_bytes())       # terzo, secondo (hits marked)
cache = S / "primo_data.pkl"
if cache.exists():
    primo = pickle.loads(cache.read_bytes())
else:
    primo = th.Data(S / "runs/c", [])                      # no hits marked: pauses only
    cache.write_bytes(pickle.dumps(primo))
points = {"primo": read_points((ROOT / "training/points/labels/primo_set.txt").read_text()),
          "secondo": read_points((ROOT / "training/points/labels/secondo_set.txt").read_text())}
runs = {"primo": S / "runs/c", "secondo": S / "runs/b"}


def with_pauses(data, pts, every=1):
    """A copy of `data` whose rows include every ball point in the pauses
    between the marked points, as no hit, each pause a group of its own."""
    d = copy.copy(data)
    pauses = [(a.end_s + PAD, b.start_s - PAD) for a, b in zip(pts, pts[1:]) if b.start_s - a.end_s > 2 * PAD]
    idx = [i for i in range(len(d.T)) if any(a <= d.T[i] <= b for a, b in pauses) and d.group[i] < 0][::every]
    base = len(d.spans)
    g = np.array([base + next(k for k, (a, b) in enumerate(pauses) if a <= d.T[i] <= b) for i in idx], dtype=int)
    F = bh.point_features(d.T, d.X, d.Y, idx, d.players, d.vp, d.height) if idx else np.zeros((0, bh.N_FEATURES))
    d.F = np.vstack([data.F, F]) if len(data.F) else F
    d.g = np.concatenate([data.g, g]) if len(data.g) else g
    d.y = np.concatenate([data.y, np.zeros(len(idx), int)])
    d.far = np.concatenate([data.far, np.zeros(len(idx), int)])
    d.spans = list(data.spans) + pauses
    print(f"  pause aggiunte: {len(pauses)} ({sum(b - a for a, b in pauses):.0f} s, {len(idx)} punti della palla)")
    return d


def evaluate(name, model):
    cap = {}
    o_find, o_refine = bh.find_hits, pl.refine_rallies

    def find_spy(*a, **k):
        out = o_find(*a, **k); cap.setdefault("hits", []).extend(out); return out
    bh.find_hits = find_spy
    try:
        _, _, rallies, _ = reprocess.recompute(runs[name], None, PipelineConfig(detector_model="x"), hit_model=model)
    finally:
        bh.find_hits = o_find
    pts = points[name]
    s = compare(pts, [Span(r.start_s, r.end_s, r.serve) for r in rallies]).summary()
    hits = sorted({h.t for h in cap.get("hits", [])})
    inside = sum(1 for t in hits if any(p.start_s - PAD <= t <= p.end_s + PAD for p in pts))
    pause = sum(1 for t in hits if pts[0].start_s <= t <= pts[-1].end_s) - inside
    print(f"  giusti {s['one_to_one']:2d}/{s['points']} · persi {s['lost']} · spezzati {s['split']} · uniti {s['joined']:2d} · "
          f"falsi {s['false_rallies']} · inizio {s['start']['median_s']} (>2s {s['start']['far_off']}) · "
          f"fine {s['end']['median_s']} (>2s {s['end']['far_off']}) · battuta {s['serve_seen']}/{s['one_to_one']} · "
          f"colpi nei punti {inside}, nelle pause {pause}")


terzo, secondo = datas


def hits_on(data, model):
    c = th.measure(data, model, range(len(data.spans)))
    m = th.shares(c)
    return f"colpi segnati {m['marked']}: trovati {m['found']:.0%} · giusti {m['real']:.0%} · metà {m['half']:.0%} · giocatore {m['player']:.0%}"


for every in (1, 3, 10):
    print(f"\n=== pause, un punto della palla ogni {every} ===")
    print("primo (terzo + secondo + pause del secondo):")
    m = th.loaded(th.as_model(*th.fit([terzo, with_pauses(secondo, points["secondo"], every)], lambda i, k: True)))
    evaluate("primo", m)
    print("secondo (terzo + pause del primo):")
    m = th.loaded(th.as_model(*th.fit([terzo, with_pauses(primo, points["primo"], every)], lambda i, k: True)))
    evaluate("secondo", m)
    print("  " + hits_on(secondo, m))
print("\nriferimento, secondo con terzo solo:")
print("  " + hits_on(secondo, th.loaded(th.as_model(*th.fit([terzo], lambda i, k: True)))))
