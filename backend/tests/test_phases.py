"""The hits read as serve / in play / between points (app.ml.phases)."""
from __future__ import annotations

import json

import numpy as np

from app.ml import ball_hits as bh
from app.ml import phases as ph
from app.ml.smoothing import SmoothedPath


def _hits(times, player=0):
    return [bh.Hit(t, player, False) for t in times]


def test_the_stored_model_reads_the_features_the_code_makes():
    model = ph.load_phase_model()
    assert model is not None
    F = np.zeros((4, ph.N_PHASE_FEATURES))
    for trees in (model.pause, model.serve):
        p = trees.proba(F)
        assert p.shape == (4,) and np.all((p >= 0) & (p <= 1))
    assert 0 < model.serve_threshold <= 1 and model.gap_s > 0
    # Measured on sets it never saw, and against what was there before.
    assert model.measured["unseen_set"] and model.measured["points"] > 0
    assert set(model.measured["before"]) >= {"one_to_one", "joined", "false_rallies"}


def test_a_model_for_other_features_is_not_read(tmp_path):
    stored = json.loads(ph.PHASE_MODEL_PATH.read_text(encoding="utf-8"))
    stored["features"] = ph.N_PHASE_FEATURES + 1
    path = tmp_path / "phase.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    assert ph.load_phase_model(str(path)) is None
    assert ph.load_phase_model(str(tmp_path / "none.json")) is None


def test_pauses_end_a_point_and_serves_open_one():
    hits = _hits([1, 2, 3, 6, 7, 10, 11, 12, 30, 31])
    pause = [0, 0, 0, 0.9, 0.9, 0, 0, 0, 0, 0]
    serve = [0.9, 0, 0, 0, 0, 0.95, 0, 0.8, 0, 0]
    points = ph.segment(hits, pause, serve, serve_threshold=0.7, pause_threshold=0.5, gap_s=8.0, lone_serve=0.9)
    # 1-3; the two pause hits left out; 10-11 (12 is a serve: a new point,
    # alone and not sure enough to be an ace); 30-31 after a long gap.
    assert [(a, b, len(part), s) for a, b, part, s in points] == [
        (1, 3, 3, True), (10, 11, 2, True), (30, 31, 2, False)]


def test_a_lone_hit_is_a_point_only_as_a_sure_serve():
    hits = _hits([5, 40])
    points = ph.segment(hits, [0, 0], [0.95, 0.5], 0.7, 0.5, 8.0, 0.9)
    assert [(a, b, s) for a, b, _, s in points] == [(5, 5, True)]


class _Where:
    """Two players: 0 still at the back until 10 s then running, 1 at the net."""

    def __init__(self):
        t = np.arange(0, 20, 0.2)
        run = (t >= 10).astype(float)
        self.paths = {
            0: SmoothedPath(t, np.c_[np.full_like(t, 2.0), np.full_like(t, 18.0)],
                            np.c_[run * 3.0, np.zeros_like(t)], np.zeros(len(t), int), 0),
            1: SmoothedPath(t, np.c_[np.full_like(t, 5.0), np.full_like(t, 11.0)],
                            np.zeros((len(t), 2)), np.zeros(len(t), int), 0),
        }

    def position(self, pid, t):
        p = self.paths[pid]
        i = int(np.argmin(np.abs(p.times - t)))
        return float(p.positions[i, 0]), float(p.positions[i, 1])

    def net_distance(self, pid, t):
        return abs(self.position(pid, t)[1] - 10.0)


def test_the_features_say_what_happens_around_each_hit():
    from app.ml.rallies import Rally

    hits = [bh.Hit(10.0, 0, True), bh.Hit(12.0, 1, False)]
    rallies = [Rally(0, 9.5, 15.0, 4, serve=True)]
    F = ph.phase_features(hits, rallies, _Where())
    assert F.shape == (2, ph.N_PHASE_FEATURES)
    first = F[0]
    assert first[0] == ph.CAP_GAP_S and first[1] == 2.0         # no hit before, the next 2 s on
    assert first[2] == 8.0                                      # 8 m from the net
    assert first[3] == 0.0 and first[5] > 0                     # still before, running after
    assert first[7] == 1.0 and first[8] == 1.0                  # one at the back; far half
    assert first[9] == 1.0 and first[10] == 0.5                 # in play by movement, since 0.5 s
    assert first[13] == 0.5 and first[14] == 0.0                # 0.5 s after the serve formation; still
    assert first[15] == 3.0                                     # 3 m from the middle
    assert F[1][2] == 1.0 and F[1][11] == 1.0                   # at the net; one hit before


class _Stub:
    def __init__(self, values):
        self.values = values

    def proba(self, F):
        return np.asarray(self.values[: len(F)], dtype=float)


def test_the_rallies_are_the_points_read_from_the_hits(monkeypatch):
    from app.ml.rallies import Rally

    # Movement saw one long stretch of play holding two points and a pause.
    times = [10.0, 11.0, 12.0, 15.0, 16.0, 20.0, 21.5, 23.0]
    hits = [bh.Hit(t, k % 2, bool(k % 2)) for k, t in enumerate(times)]
    monkeypatch.setattr(bh, "find_hits", lambda *a, **k: hits)
    model = ph.PhaseModel(_Stub([0, 0, 0, 0.9, 0.9, 0, 0, 0]), _Stub([0.9, 0, 0, 0, 0, 0.9, 0, 0]),
                          serve_threshold=0.7, pause_threshold=0.5, gap_s=8.0, lone_serve=0.9)
    ball = bh.BallInput(np.zeros(3), np.zeros(3), np.zeros(3), [], (0.0, 0.0), 1080, None)
    rallies = [Rally(0, 9.0, 30.0, 3)]
    out, kept, dropped = bh.refine_rallies(rallies, ball, lambda pid, t: 8.0, _Where(), model)
    assert [(r.index, r.start_s, r.end_s, r.serve, r.players_tracked) for r in out] == [
        (0, 10.0, 12.0 + ph.TAIL_S, True, 3), (1, 20.0, 23.0 + ph.TAIL_S, True, 3)]
    assert [h.t for h in kept] == [10.0, 11.0, 12.0, 20.0, 21.5, 23.0]   # the pause's hits not counted
    assert dropped == 0
    # Without the model, the same hits make one rally, as before.
    out, kept, _ = bh.refine_rallies(rallies, ball, lambda pid, t: 8.0)
    assert [(r.start_s, r.end_s) for r in out] == [(10.0, 23.0 + bh.TAIL_S)] and len(kept) == 8

