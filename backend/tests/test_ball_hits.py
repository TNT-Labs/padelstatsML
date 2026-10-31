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
