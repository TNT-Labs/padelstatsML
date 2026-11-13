"""Experiment: each hit found is serve / play / pause; points from that.
Cross-set: trained on one set's marked points, measured on the other."""
import sys, pickle, math
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
sets = {
    "primo": (S / "runs/c", ROOT / "training/points/labels/primo_set.txt"),
    "secondo": (S / "runs/b", ROOT / "training/points/labels/secondo_set.txt"),
}
datas = pickle.loads((S / "datas.pkl").read_bytes())          # [terzo, secondo]
models = {"primo": None,                                         # in use: secondo + terzo
          "secondo": th.loaded(th.as_model(*th.fit(datas, lambda i, k: i != 1)))}


def capture(name):
    run, labels = sets[name]
    cap = {}
    o_find, o_where, o_refine = bh.find_hits, pl._where_players, pl.refine_rallies
    import app.ml.rallies as rl
    o_serves = rl.serve_starts

    def serves_spy(*a, **k):
        out = o_serves(*a, **k); cap["serves"] = out; return out
    rl.serve_starts = serves_spy

    def find_spy(*a, **k):
        out = o_find(*a, **k); cap.setdefault("hits", []).extend(out); return out

    def where_spy(paths):
        cap["paths"] = paths; w = o_where(paths); cap["where"] = w; return w

    def refine_spy(rallies, ball, nd=None):
        cap["mov"] = list(rallies); cap["ball"] = ball; cap["nd"] = nd
        out = o_refine(rallies, ball, nd); cap["refined"] = out[0]; return out
    bh.find_hits, pl._where_players, pl.refine_rallies = find_spy, where_spy, refine_spy
    try:
        reprocess.recompute(run, None, PipelineConfig(detector_model="x"), hit_model=models[name])
    finally:
        bh.find_hits, pl._where_players, pl.refine_rallies = o_find, o_where, o_refine
        rl.serve_starts = o_serves
    cap["hits"] = sorted({h.t: h for h in cap["hits"]}.values(), key=lambda h: h.t)
    cap["points"] = read_points(labels.read_text())
    return cap


def speed_stats(paths, a, b):
    vals = []
    for p in paths.values():
        m = (p.times >= a) & (p.times <= b)
        if m.any():
            vals.append(float(np.mean(p.speeds[m])))
    return (float(np.mean(vals)), float(np.max(vals))) if vals else (0.0, 0.0)


def features(cap):
    hits, paths, nd = cap["hits"], cap["paths"], cap["nd"]
    T = np.array([h.t for h in hits])
    mov = cap["mov"]
    rows = []
    for i, h in enumerate(hits):
        prev = h.t - T[i - 1] if i > 0 else 15.0
        nxt = T[i + 1] - h.t if i + 1 < len(T) else 15.0
        d = nd(h.player, h.t)
        b_mean, b_max = speed_stats(paths, h.t - 2.5, h.t - 0.3)
        a_mean, a_max = speed_stats(paths, h.t + 0.3, h.t + 2.5)
        back = sum(1 for pid in paths if (nd(pid, h.t) or 0) > 6.5)
        inside = [r for r in mov if r.start_s - 1 <= h.t <= r.end_s + 1]
        since = h.t - inside[0].start_s if inside else -5.0
        n_before = int(((T >= h.t - 6) & (T < h.t)).sum())
        n_after = int(((T > h.t) & (T <= h.t + 6)).sum())
        onsets = [b for _, b in cap.get("serves", [])]
        to_serve = min((h.t - o for o in onsets), key=abs, default=20.0)
        hp = paths.get(h.player)
        hs = 0.0
        if hp is not None:
            mm = (hp.times >= h.t - 1.5) & (hp.times <= h.t - 0.2)
            hs = float(np.mean(hp.speeds[mm])) if mm.any() else 0.0
        pos = cap["where"].position(h.player, h.t)
        lat = abs(pos[0] - 5.0) if pos else -1
        rows.append([min(prev, 15), min(nxt, 15), d if d is not None else -1, b_mean, b_max, a_mean, a_max,
                     back, float(h.far), float(bool(inside)), min(since, 30), n_before, n_after,
                     max(-20, min(20, to_serve)), hs, lat])
    return np.array(rows)


def labels(cap):
    pts, out = cap["points"], []
    lo, hi = pts[0].start_s - 3, pts[-1].end_s + 3
    for h in cap["hits"]:
        if not lo <= h.t <= hi:
            out.append(-1); continue
        inp = [p for p in pts if p.start_s - 0.7 <= h.t <= p.end_s + 0.7]
        if not inp:
            out.append(0)                                      # pause
        else:
            p = inp[0]
            first = min((x.t for x in cap["hits"] if p.start_s - 0.7 <= x.t <= p.end_s + 0.7))
            out.append(2 if h.t == first and h.t - p.start_s <= 1.5 else 1)   # serve / play
    return np.array(out)


def gb(X, y):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=0).fit(X, y)


LOG = lambda p: math.log(max(p, 1e-4))


def decode(pp, ps):
    """Viterbi over pause(0) / serve(2) / play(1). pp: P(pause), ps: P(serve | not pause)."""
    states = (0, 1, 2)
    trans = {(0, 0): 0.9, (0, 2): 0.08, (0, 1): 0.02,
             (2, 1): 0.85, (2, 0): 0.1, (2, 2): 0.05,
             (1, 1): 0.85, (1, 0): 0.1, (1, 2): 0.05}
    emit = lambda i, s: pp[i] if s == 0 else (1 - pp[i]) * (ps[i] if s == 2 else 1 - ps[i])
    V = [{s: LOG(emit(0, s)) + LOG({0: .5, 2: .4, 1: .1}[s]) for s in states}]
    back = [{}]
    for i in range(1, len(pp)):
        V.append({}); back.append({})
        for s in states:
            best = max(states, key=lambda r: V[i - 1][r] + LOG(trans[(r, s)]))
            V[i][s] = V[i - 1][best] + LOG(trans[(best, s)]) + LOG(emit(i, s))
            back[i][s] = best
    s = max(states, key=lambda r: V[-1][r])
    out = [s]
    for i in range(len(pp) - 1, 0, -1):
        s = back[i][s]; out.append(s)
    return out[::-1]


def spans_from(hits, states, tail=1.0, gap=None):
    spans, cur = [], None
    for h, s in zip(hits, states):
        if s == 0:
            if cur: spans.append(cur); cur = None
            continue
        if s == 2 or cur is None or (gap and h.t - cur[1] > gap):
            if cur: spans.append(cur)
            cur = [h.t, h.t, s == 2]
        else:
            cur[1] = h.t
    if cur: spans.append(cur)
    return [Span(a, b + tail, sv) for a, b, sv in spans]


def show(tag, pts, spans):
    s = compare(pts, spans).summary()
    print(f"  {tag:34s} giusti {s['one_to_one']:2d}/{s['points']} · persi {s['lost']} · spezzati {s['split']} · "
          f"uniti {s['joined']:2d} · falsi {s['false_rallies']} · inizio {s['start']['median_s']} "
          f"(>2s {s['start']['far_off']}) · fine {s['end']['median_s']} (>2s {s['end']['far_off']})")


caps = {n: capture(n) for n in sets}
F = {n: features(c) for n, c in caps.items()}
Y = {n: labels(c) for n, c in caps.items()}
for n in sets:
    y = Y[n]
    print(n, "colpi nel tratto", int((y >= 0).sum()), "pausa", int((y == 0).sum()), "gioco", int((y == 1).sum()),
          "battuta", int((y == 2).sum()))
for test, train in (("primo", "secondo"), ("secondo", "primo")):
    Xtr, ytr = F[train][Y[train] >= 0], Y[train][Y[train] >= 0]
    m_pause = gb(Xtr, (ytr == 0).astype(int))
    m_serve = gb(Xtr[ytr > 0], (ytr[ytr > 0] == 2).astype(int))
    c = caps[test]
    pp = m_pause.predict_proba(F[test])[:, 1]
    ps = m_serve.predict_proba(F[test])[:, 1]
    yt = Y[test]; m = yt >= 0
    from sklearn.metrics import roc_auc_score
    print(f"\n{test}: AUC pausa {roc_auc_score(yt[m] == 0, pp[m]):.2f} · AUC battuta (fra i colpi non di pausa) "
          f"{roc_auc_score(yt[m & (yt > 0)] == 2, ps[m & (yt > 0)]):.2f}")
    pred_pause = pp > 0.5
    print(f"\n{test} (modello addestrato su {train}): pausa giusta {np.mean(pred_pause[m] == (yt[m] == 0)):.0%} · "
          f"colpi di pausa riconosciuti {np.sum(pred_pause[m] & (yt[m] == 0))}/{np.sum(yt[m] == 0)} · "
          f"colpi in gioco scambiati per pausa {np.sum(pred_pause[m] & (yt[m] > 0))}/{np.sum(yt[m] > 0)}")
    st = decode(pp, ps)
    show("oggi (refine_rallies)", c["points"], [Span(r.start_s, r.end_s, r.serve) for r in c["refined"]])
    show("battuta/gioco/pausa", c["points"], spans_from(c["hits"], st))
    show("battuta/gioco/pausa + buco 6 s", c["points"], spans_from(c["hits"], st, gap=6.0))
    # oracle: perfect labels, how far the hits alone can go
    orc = [{0: 0, 1: 1, 2: 2}[v] if v >= 0 else 0 for v in yt]
    show("etichette perfette (limite)", c["points"], spans_from(c["hits"], orc))


def segment(hits, pp, ps, ts, tp, gap, tail=1.0):
    spans, cur = [], None
    for h, a, b in zip(hits, pp, ps):
        if a >= tp:                                     # a hit between points: ends the point
            if cur: spans.append(cur); cur = None
            continue
        if cur is None or b >= ts or h.t - cur[1] > gap:
            if cur: spans.append(cur)
            cur = [h.t, h.t, b >= ts]
        else:
            cur[1] = h.t
    if cur: spans.append(cur)
    return [Span(a, b + tail, sv) for a, b, sv in spans]


def score(pts, spans):
    s = compare(pts, spans).summary()
    return s["one_to_one"] - 0.5 * s["false_rallies"]


print("\n=== soglie scelte sul set di addestramento (previsioni fuori campione con 2 gruppi) ===")
from sklearn.model_selection import cross_val_predict
grid = [(ts, tp, g) for ts in (0.2, 0.3, 0.4, 0.5, 0.6) for tp in (0.5, 0.6, 0.7, 0.8, 1.01) for g in (5.0, 6.0, 8.0)]
for test, train in (("primo", "secondo"), ("secondo", "primo")):
    Xtr, ytr = F[train][Y[train] >= 0], Y[train][Y[train] >= 0]
    m_pause = gb(Xtr, (ytr == 0).astype(int))
    m_serve = gb(Xtr[ytr > 0], (ytr[ytr > 0] == 2).astype(int))
    # thresholds chosen on the training set, on its own predictions made out of fold (halves)
    ctr = caps[train]; mask = Y[train] >= 0
    tt_ = np.array([h.t for h in ctr["hits"]]); mid = np.median(tt_[mask])
    half = tt_ < mid
    pp_tr = np.zeros(len(F[train])); ps_tr = np.zeros(len(F[train]))
    for part in (half, ~half):
        fit_m = mask & ~part
        mp = gb(F[train][fit_m], (Y[train][fit_m] == 0).astype(int))
        ms = gb(F[train][fit_m & (Y[train] > 0)], (Y[train][fit_m & (Y[train] > 0)] == 2).astype(int))
        pp_tr[part] = mp.predict_proba(F[train][part])[:, 1]; ps_tr[part] = ms.predict_proba(F[train][part])[:, 1]
    best = max(grid, key=lambda g: score(ctr["points"], segment(ctr["hits"], pp_tr, ps_tr, *g)))
    c = caps[test]
    pp = m_pause.predict_proba(F[test])[:, 1]; ps = m_serve.predict_proba(F[test])[:, 1]
    print(f"\n{test} (addestrato su {train}, soglie battuta {best[0]} · pausa {best[1]} · buco {best[2]} s)")
    show("oggi (refine_rallies)", c["points"], [Span(r.start_s, r.end_s, r.serve) for r in c["refined"]])
    show("battuta/gioco/pausa, soglie", c["points"], segment(c["hits"], pp, ps, *best))
    show("solo buco (stesso valore)", c["points"], segment(c["hits"], np.zeros(len(pp)), np.zeros(len(ps)), 2, 2, best[2]))
