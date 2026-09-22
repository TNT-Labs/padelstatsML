from __future__ import annotations

from app.ml.identity import resolve_players
from app.ml.tracking import Tracklet
from tests.synthetic import walk


def _tracklet(tid: int, observations) -> Tracklet:
    return Tracklet(id=tid, observations=observations)


def _four_players(calibration, t0: float = 0.0, t1: float = 40.0):
    """Two pairs holding their halves: near team at Y < 10, far team at Y > 10."""
    return [
        walk(calibration, 0, (2.0, 3.0), (3.5, 6.0), t0, t1),
        walk(calibration, 1, (8.0, 3.0), (6.5, 6.5), t0, t1),
        walk(calibration, 2, (2.5, 17.0), (4.0, 14.0), t0, t1),
        walk(calibration, 3, (7.5, 17.0), (6.0, 14.0), t0, t1),
    ]


def test_four_clean_tracklets_become_four_players(calibration):
    tracklets = [_tracklet(i, obs) for i, obs in enumerate(_four_players(calibration))]
    result = resolve_players(tracklets)

    assert len(result.players) == 4
    assert sorted(p.player_id for p in result.players) == [0, 1, 2, 3]
    assert [p.team for p in sorted(result.players, key=lambda p: p.player_id)] == [0, 0, 1, 1]


def test_a_player_split_by_an_occlusion_is_rejoined(calibration):
    """This is the failure the old 'top 4 track ids by frequency' could not
    handle: one person becomes two ids, and a real player drops out of the
    ranking entirely."""
    tracklets = [_tracklet(i, obs) for i, obs in enumerate(_four_players(calibration, 0.0, 20.0))]
    # Player 0 reappears after a 3 s occlusion, close to where they vanished.
    tracklets.append(_tracklet(10, walk(calibration, 0, (3.6, 6.2), (2.0, 4.0), 23.0, 40.0)))
    # Everyone else simply continues.
    for shirt, (start, end) in enumerate(
        [(None, None), ((6.5, 6.5), (8.0, 3.0)), ((4.0, 14.0), (2.5, 17.0)), ((6.0, 14.0), (7.5, 17.0))]
    ):
        if start is None:
            continue
        tracklets.append(_tracklet(20 + shirt, walk(calibration, shirt, start, end, 20.2, 40.0)))

    result = resolve_players(tracklets)
    assert len(result.players) == 4
    # The rejoined player must own both of their tracklets.
    merged = [p for p in result.players if len(p.source_tracklets) > 1]
    assert merged, "la tracklet interrotta non è stata ricongiunta"


def test_two_tracklets_overlapping_in_time_are_never_merged(calibration):
    """Two people cannot be one person, no matter how similar their shirts."""
    a = _tracklet(1, walk(calibration, 0, (2.0, 3.0), (3.0, 5.0), 0.0, 20.0))
    b = _tracklet(2, walk(calibration, 0, (8.0, 3.0), (7.0, 5.0), 0.0, 20.0))
    result = resolve_players([a, b])
    assert len(result.players) == 2
    assert all(len(p.source_tracklets) == 1 for p in result.players)


def test_teams_are_split_by_court_half(calibration):
    tracklets = [_tracklet(i, obs) for i, obs in enumerate(_four_players(calibration))]
    result = resolve_players(tracklets)
    for player in result.players:
        mean_y = sum(o.foot_court[1] for o in player.observations) / len(player.observations)
        assert (mean_y > 10.0) == (player.team == 1)


def test_player_ids_are_stable_across_runs(calibration):
    tracklets = [_tracklet(i, obs) for i, obs in enumerate(_four_players(calibration))]
    first = {p.player_id: p.mean_x() if hasattr(p, "mean_x") else p.observations[0].foot_court
             for p in resolve_players(tracklets).players}
    second = {p.player_id: p.mean_x() if hasattr(p, "mean_x") else p.observations[0].foot_court
              for p in resolve_players(list(reversed(tracklets))).players}
    assert first == second


def test_missing_players_are_reported_not_hidden(calibration):
    tracklets = [_tracklet(i, obs) for i, obs in enumerate(_four_players(calibration)[:2])]
    result = resolve_players(tracklets)
    assert len(result.players) == 2
    assert any("solo 2 giocatori" in w for w in result.warnings)


def test_no_tracklets_yields_a_warning_not_a_crash():
    result = resolve_players([])
    assert result.players == []
    assert result.warnings


# ── Regression tests for two linking defects found by the pipeline test ──────

def test_consecutive_tracklets_one_sample_apart_are_merged(calibration):
    """A track that dies on one sample and is reborn on the next is the most
    common fragmentation there is. An overlap tolerance applied in the wrong
    direction used to reject exactly this case, leaving every player split."""
    first = walk(calibration, 0, (2.0, 4.0), (3.0, 6.0), 0.0, 10.0)
    second = walk(calibration, 0, (3.0, 6.0), (2.0, 4.0), 10.2, 20.0)
    result = resolve_players([_tracklet(1, first), _tracklet(2, second)])

    assert len(result.players) == 1
    assert len(result.players[0].source_tracklets) == 2
    assert len(result.players[0].observations) == len(first) + len(second)


def test_team_mates_with_different_shirts_are_never_merged(calibration):
    """Two players of the same pair, tracked in different halves of the match.
    They never coexist in time and they are only five metres apart, so a cost
    function that lets the spatial term vanish over a long gap merges them —
    which silently deletes one real player from the results."""
    left = _tracklet(1, walk(calibration, 0, (2.5, 5.0), (2.6, 5.2), 0.0, 11.0))
    right = _tracklet(2, walk(calibration, 1, (7.5, 5.0), (7.6, 5.2), 19.0, 30.0))

    result = resolve_players([left, right])
    assert len(result.players) == 2
    assert all(len(p.source_tracklets) == 1 for p in result.players)


def test_identical_shirts_still_need_physical_reachability(calibration):
    """Same kit is not enough on its own: a jump no one could walk, outside
    the changeover window, must stay two separate tracks."""
    here = _tracklet(1, walk(calibration, 0, (1.0, 2.0), (1.2, 2.4), 0.0, 10.0))
    there = _tracklet(2, walk(calibration, 0, (9.0, 18.0), (8.8, 17.6), 200.0, 210.0))

    result = resolve_players([here, there])
    assert len(result.players) == 2


def test_linking_reports_progress(calibration):
    """Without this the progress bar sits at 89% for the whole stage, which on
    a heavily fragmented match is indistinguishable from a hang — exactly what
    it looked like on the first real Pi run."""
    tracklets = [_tracklet(i, obs) for i, obs in enumerate(_four_players(calibration, 0.0, 20.0))]
    tracklets.append(_tracklet(10, walk(calibration, 0, (3.6, 6.2), (2.0, 4.0), 23.0, 40.0)))

    seen: list[tuple[int, int]] = []
    resolve_players(tracklets, progress=lambda done, total: seen.append((done, total)))

    assert seen, "nessun avanzamento riportato durante il linking"
    assert all(total == len(tracklets) for _, total in seen)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_linking_stays_fast_when_tracking_fragments(calibration):
    """Real matches fragment far more than synthetic ones. The cost cache and
    the memoised colour signature keep this linear enough to finish in
    milliseconds; recomputing every pair after every merge made it cubic and
    took minutes on a Pi."""
    import time

    import numpy as np

    from app.ml.tracking import Observation

    rng = np.random.default_rng(0)
    tracklets = []
    timestamp = 0.0
    for track_id in range(300):
        colour = rng.random(32).astype(np.float32)
        colour /= colour.sum()
        observations = []
        for _ in range(20):
            observations.append(
                Observation(
                    frame_index=int(timestamp * 5),
                    timestamp_s=timestamp,
                    bbox=(0.0, 0.0, 10.0, 30.0),
                    foot_px=(100.0, 900.0),
                    foot_court=(float(rng.uniform(0, 10)), float(rng.uniform(0, 20))),
                    confidence=0.8,
                    color=colour,
                )
            )
            timestamp += 0.2
        tracklets.append(Tracklet(id=track_id, observations=observations))

    start = time.monotonic()
    result = resolve_players(tracklets)
    elapsed = time.monotonic() - start

    assert result.players
    # Generous: a Pi is several times slower than CI, and the pre-fix
    # implementation needed tens of seconds for this input on x86 alone.
    assert elapsed < 5.0, f"linking troppo lento: {elapsed:.1f}s per 300 tracce"
