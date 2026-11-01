"""Hits from the ball: the stored models, the decoding, the pairs."""
from __future__ import annotations

import numpy as np

from app.ml import ball_hits as bh


def test_the_stored_models_read_the_features_the_code_makes():
    model = bh.load_model()
    assert model is not None
    F = np.zeros((3, bh.N_FEATURES))
    p = model.hit.proba(F)
    assert p.shape == (3,) and np.all((p >= 0) & (p <= 1))
    assert set(model.measured) >= {"found", "real", "half", "player"}


def test_trees_are_read_as_scikit_learn_builds_them():
    # One split on feature 1 at 0.5: left adds -1, right +1, at rate 0.5.
    trees = bh.Trees({"init": 0.0, "rate": 0.5, "trees": [
        {"left": [1, -1, -1], "right": [2, -1, -1], "feature": [1, -2, -2],
         "threshold": [0.5, -2.0, -2.0], "value": [0.0, -1.0, 1.0]}]})
    p = trees.proba(np.array([[0, 0.2], [0, 0.9]]))
    assert np.allclose(p, 1 / (1 + np.exp([0.5, -0.5])))


def test_the_hits_alternate_between_the_halves():
    # Strong candidates: near, far, near — and a far one too soon after.
    cands = [(1.0, 0.9, 0.1), (2.0, 0.8, 0.9), (2.2, 0.7, 0.9), (3.1, 0.9, 0.2)]
    out = bh.decode(cands)
    assert out == [(1.0, False), (2.0, True), (3.1, False)]
    assert bh.decode([(1.0, 0.05, 0.5)]) == []          # below the cost: no hit


def test_candidates_keep_the_likeliest_point_of_each_moment():
    t = np.array([1.0, 1.1, 1.2, 2.0])
    p = np.array([0.3, 0.6, 0.2, 0.05])
    assert [c[0] for c in bh.candidates(p, np.zeros(4), t)] == [1.1]


def _track(pid, team, far, box, seen=True):
    samples = [(t / 5, box, far) for t in range(50)] if seen else [(100.0, box, far)]
    return bh.PlayerTrack(pid, samples, team)


def test_the_pair_includes_a_partner_not_seen_at_that_moment():
    players = [_track(0, 0, False, (100, 500, 300, 1000)), _track(1, 0, False, (900, 500, 1100, 1000), seen=False),
               _track(2, 1, True, (700, 450, 740, 530)), _track(3, 1, True, (1000, 450, 1040, 530))]
    assert [p.player_id for p in bh.pair_at(players, 3.0, False)] == [0, 1]
    assert [p.player_id for p in bh.pair_at(players, 3.0, True)] == [2, 3]


def test_hits_are_read_from_a_rally_of_ball_points():
    players = [_track(0, 0, False, (100, 500, 400, 1000)), _track(1, 0, False, (1300, 500, 1600, 1000)),
               _track(2, 1, True, (700, 450, 740, 530)), _track(3, 1, True, (1000, 450, 1040, 530))]
    t = np.arange(0, 6, 1 / 30)
    # Back and forth between a near and a far player, a hit every 1.5 s.
    phase = (t % 3.0) / 3.0
    x = np.where(phase < 0.5, 250 + (720 - 250) * phase * 2, 720 - (720 - 250) * (phase - 0.5) * 2)
    y = np.where(phase < 0.5, 600 - 120 * phase * 2, 480 + 120 * (phase - 0.5) * 2)
    hits = bh.find_hits(t, x, y, players, [{"start_s": 0.5, "end_s": 6.0}], (888.0, 439.0), 1080, bh.load_model())
    assert all(0.0 <= h.t <= 6.0 and h.player in (0, 1, 2, 3) for h in hits)
    assert all(a.far != b.far for a, b in zip(hits, hits[1:]))      # the halves alternate


def _rally(i, a, b, serve=False):
    from app.ml.rallies import Rally

    return Rally(index=i, start_s=a, end_s=b, players_tracked=4, serve=serve)


def _refine(monkeypatch, rallies, times, back=lambda pid, t: 8.0):
    hits = [bh.Hit(t, k % 2 * 2, bool(k % 2)) for k, t in enumerate(times)]
    monkeypatch.setattr(bh, "find_hits", lambda *a, **k: hits)
    ball = bh.BallInput(np.zeros(3), np.zeros(3), np.zeros(3), [], (0.0, 0.0), 1080, None)
    return bh.refine_rallies(rallies, ball, back)


def test_a_rally_runs_from_the_serve_to_the_last_hit(monkeypatch):
    # Movement saw play from 9 s to 30 s; the ball was hit from 10 to 18 s.
    out, hits = _refine(monkeypatch, [_rally(0, 9.0, 30.0)], [10.0, 11.5, 13.0, 15.0, 18.0])
    assert [(r.start_s, r.end_s) for r in out] == [(10.0, 18.0 + bh.TAIL_S)]
    assert out[0].serve and len(hits) == 5


def test_hits_split_a_rally_and_join_two(monkeypatch):
    # One movement rally with two points in it (the ball idle 10 s), and a
    # point the movement cut in two (hits 2 s apart across the cut).
    out, _ = _refine(monkeypatch, [_rally(0, 0.0, 30.0), _rally(1, 40.0, 45.0), _rally(2, 46.0, 52.0)],
                     [1.0, 2.0, 3.0, 13.0, 14.0, 41.0, 43.0, 45.5, 47.0, 49.0])
    assert [(r.start_s, r.end_s) for r in out] == [(1.0, 4.0), (13.0, 15.0), (41.0, 50.0)]
    assert [r.index for r in out] == [0, 1, 2]


def test_a_rally_without_the_ball_stays_as_movement_found_it(monkeypatch):
    out, hits = _refine(monkeypatch, [_rally(0, 5.0, 12.0, serve=True), _rally(1, 20.0, 30.0)], [21.0, 23.0])
    assert [(r.start_s, r.end_s, r.serve) for r in out] == [(5.0, 12.0, True), (21.0, 24.0, True)]
    # A first hit from near the net is no serve.
    out, _ = _refine(monkeypatch, [_rally(0, 20.0, 30.0)], [21.0, 23.0], back=lambda pid, t: 2.0)
    assert not out[0].serve


def test_each_player_hits_from_somewhere_and_closes_rallies():
    result = {"rallies": [{"start_s": 10.0, "end_s": 20.0}, {"start_s": 30.0, "end_s": 40.0}],
              "per_player": {"0": {}, "2": {}}, "data_quality": {}}
    hits = [bh.Hit(10.0, 0, False), bh.Hit(11.0, 2, True), bh.Hit(30.0, 2, True), bh.Hit(31.0, 0, False)]
    dist = {0: 2.0, 2: 7.0}
    bh.hit_statistics(result, hits, lambda pid, t: dist[pid], lambda pid, t: (3.0, 10.0 - dist[pid]),
                      {"found": 0.8})
    assert result["rallies"][0]["hits"] == [[10.0, 0], [11.0, 2]]
    assert result["per_player"]["0"] == {"hits": 2, "hit_zones": {"net": 2, "mid": 0, "back": 0},
                                         "hit_spots": [[3.0, 8.0], [3.0, 8.0]], "closing_hits": 1}
    assert result["per_player"]["2"]["hit_zones"] == {"net": 0, "mid": 0, "back": 2}
    assert result["data_quality"]["ball"]["hits_measured"] == {"found": 0.8}
