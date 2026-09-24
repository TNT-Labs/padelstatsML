from __future__ import annotations

import numpy as np
import pytest

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
    resolve_players(tracklets, method="link", progress=lambda done, total: seen.append((done, total)))

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


def test_a_tracklet_inside_a_cluster_gap_can_still_join_it(calibration):
    """A cluster is a set of intervals, not one interval from its first
    observation to its last.

    Greedy linking takes the cheapest pair first, and that is often not the
    adjacent one: here the player leaves a spot, moves, and comes back, so
    the first and last tracklets meet at distance zero while the middle one
    is two metres away. Treating the merged pair as one block spanning the
    whole 30 s made the middle tracklet "coexist" with it: the player was
    split in two for good. On a real match this left 80% of the observations
    outside the four players.
    """
    first = _tracklet(1, walk(calibration, 0, (2.0, 3.0), (3.0, 4.0), 0.0, 10.0))
    middle = _tracklet(2, walk(calibration, 0, (3.0, 5.5), (3.0, 5.5), 12.0, 20.0))
    last = _tracklet(3, walk(calibration, 0, (3.0, 4.0), (2.0, 3.0), 22.0, 30.0))

    result = resolve_players([first, middle, last])

    assert len(result.players) == 1, f"giocatore diviso in {len(result.players)}"
    assert sorted(result.players[0].source_tracklets) == [1, 2, 3]
    assert result.observations_discarded == 0


def test_a_tracklet_coexisting_with_any_part_of_a_cluster_is_kept_out(calibration):
    """The other half of the interval semantics: filling a gap is allowed,
    overlapping any one of the cluster's pieces is not."""
    first = _tracklet(1, walk(calibration, 0, (2.0, 3.0), (3.0, 4.0), 0.0, 10.0))
    last = _tracklet(2, walk(calibration, 0, (3.0, 4.0), (2.0, 3.0), 22.0, 30.0))
    # Same shirt, sits in the gap but runs into the last piece by 3 s.
    intruder = _tracklet(3, walk(calibration, 0, (3.0, 5.0), (3.0, 5.0), 12.0, 25.0))

    result = resolve_players([first, last, intruder])

    owners = {tid: p.player_id for p in result.players for tid in p.source_tracklets}
    assert owners[1] == owners[2]
    assert owners[3] != owners[1]


def test_filling_a_gap_must_be_reachable_at_both_ends(calibration):
    """A tracklet in the gap joins only if the player could have walked in
    from the piece before it and out to the piece after it."""
    first = _tracklet(1, walk(calibration, 0, (2.0, 3.0), (2.0, 3.0), 0.0, 10.0))
    last = _tracklet(2, walk(calibration, 0, (2.0, 3.0), (2.0, 3.0), 20.4, 30.0))
    # 0.2 s after the first piece ends it is already 9 m away: nobody runs that.
    far = _tracklet(3, walk(calibration, 0, (8.0, 17.0), (8.0, 17.0), 10.2, 20.0))

    result = resolve_players([first, last, far])

    owners = {tid: p.player_id for p in result.players for tid in p.source_tracklets}
    assert owners[1] == owners[2]
    assert owners[3] != owners[1]


def test_a_tracklet_that_switched_player_is_split_between_them(calibration):
    """Two players cross; the tracker keeps the track but swaps the person.

    Unsplit, the mixed tracklet sits in one cluster while also coexisting
    with the other player's next piece, and one of the two players ends up
    broken in two. Split where the shirt changes, each half goes home.
    """
    a = _tracklet(1, walk(calibration, 0, (2.0, 4.0), (2.0, 5.0), 0.0, 10.0))
    b = _tracklet(2, walk(calibration, 0, (2.0, 5.0), (2.5, 5.0), 10.2, 20.0))
    mixed = _tracklet(3, walk(calibration, 0, (2.5, 5.0), (3.0, 5.0), 20.2, 25.0)
                      + walk(calibration, 1, (3.2, 5.0), (4.0, 6.0), 25.2, 30.0))
    c = _tracklet(4, walk(calibration, 1, (4.0, 4.0), (3.2, 5.0), 0.0, 25.0))
    e = _tracklet(5, walk(calibration, 0, (3.0, 5.0), (2.0, 4.0), 25.2, 40.0))
    d = _tracklet(6, walk(calibration, 1, (4.0, 6.0), (5.0, 6.0), 30.2, 40.0))

    result = resolve_players([a, b, mixed, c, e, d])

    assert len(result.players) == 2
    zero = next(p for p in result.players if p.owns(a.observations[0]))
    one = next(p for p in result.players if p.owns(c.observations[0]))
    assert zero is not one
    first_half, second_half = mixed.observations[:25], mixed.observations[-25:]
    assert all(zero.owns(o) for o in b.observations + e.observations + first_half)
    assert all(one.owns(o) for o in d.observations + second_half)
    assert result.observations_discarded == 0


def test_a_single_misplaced_end_sample_does_not_break_a_link(calibration):
    """Tracks tend to die on a bad box: the feet hidden, the foot point
    thrown metres away. The junction is judged on the median of the last
    samples, not on that one."""
    from dataclasses import replace

    first = walk(calibration, 0, (2.0, 4.0), (3.0, 6.0), 0.0, 10.0)
    first[-1] = replace(first[-1], foot_court=(3.0, 10.5))    # 4.5 m off in 0.2 s
    second = walk(calibration, 0, (3.0, 6.2), (2.0, 4.0), 10.2, 20.0)

    result = resolve_players([_tracklet(1, first), _tracklet(2, second)])
    assert len(result.players) == 1
    assert sorted(result.players[0].source_tracklets) == [1, 2]

    # And linked as a walk, not tolerated as a changeover: judged on the last
    # sample alone, the junction is a 4.5 m jump in 0.2 s, and only the
    # identical shirts save it — at the changeover penalty, which a real
    # competitor for the link would beat.
    from app.ml.identity import STRONG_COLOR_MATCH, _Cluster, _link_cost, _Segment

    def cluster(observations):
        return _Cluster(observations=observations, sources=[0], segments=[_Segment.of(observations)])

    cost = _link_cost(cluster(first), cluster(second), 8.0)
    assert cost is not None and cost < STRONG_COLOR_MATCH


def test_a_player_split_in_two_is_not_counted_twice(calibration):
    """Tracking of one player breaks for longer than linking can bridge, so
    that player becomes two clusters, each larger than the least-seen
    player's. The four largest would count the first player twice and drop
    the fourth. Four people on court coexist; the two halves of one person
    never do."""
    from app.ml.identity import MAX_LINK_GAP_S

    gap = MAX_LINK_GAP_S + 10.0
    end = 60.0 + gap + 60.0
    tracklets = [
        _tracklet(1, walk(calibration, 1, (8.0, 3.0), (7.0, 4.0), 0.0, end)),
        _tracklet(2, walk(calibration, 2, (2.5, 17.0), (3.5, 16.0), 0.0, end)),
        # Player 3, seen for a minute, lost for longer than linking bridges,
        # seen for another minute.
        _tracklet(3, walk(calibration, 3, (7.5, 17.0), (6.5, 16.0), 0.0, 60.0)),
        _tracklet(4, walk(calibration, 3, (6.5, 16.0), (7.5, 17.0), 60.0 + gap, end)),
        # Player 0 is barely seen: 20 s.
        _tracklet(5, walk(calibration, 0, (2.0, 3.0), (2.5, 4.0), 40.0, 60.0)),
    ]

    result = resolve_players(tracklets)
    sources = sorted(tuple(sorted(p.source_tracklets)) for p in result.players)
    assert (5,) in sources, f"il giocatore meno visto è stato escluso: {sources}"
    assert not ((3,) in sources and (4,) in sources), "lo stesso giocatore contato due volte"


def test_a_few_stray_shared_frames_do_not_keep_one_player_apart(calibration):
    """A player is tracked in three pieces. The middle one starts before the
    first ends: for a second two tracks followed the same person, handing
    the detection back and forth, and in two frames both held one — one of
    the two detections being somebody else.

    Two people on court are seen together in most samples; two frames out of
    six is one person with a stray detection. Any overlap used to count as
    coexistence, and those two frames kept the player split in two. They
    are merged now, and the two contested frames are dropped: nothing says
    which detection was the player."""
    first = walk(calibration, 0, (2.0, 3.0), (3.0, 5.0), 0.0, 10.0)
    middle = walk(calibration, 0, (3.0, 5.0), (3.0, 7.0), 9.0, 20.0)
    last = walk(calibration, 0, (3.0, 7.0), (2.0, 3.0), 20.2, 30.0)
    frames = lambda obs: {o.frame_index for o in obs}       # noqa: E731
    contested = {46, 48}
    # In the overlap (frames 45-50) the two tracks alternate...
    first = [o for o in first if o.frame_index < 45 or o.frame_index % 2 == 0]
    middle = [o for o in middle if o.frame_index > 50 or o.frame_index % 2 == 1 or o.frame_index in contested]
    assert frames(first) & frames(middle) == contested

    result = resolve_players([_tracklet(1, first), _tracklet(2, middle), _tracklet(3, last)])

    assert len(result.players) == 1
    assert sorted(result.players[0].source_tracklets) == [1, 2, 3]
    assert not frames(result.players[0].observations) & contested
    assert result.observations_discarded == 2 * len(contested)


def test_a_sliver_of_overlap_is_not_proof_of_two_people(calibration):
    """Two pieces of one player overlap for a single sample interval, and
    both frames in it are shared: a density of 100%, on two frames. That is
    a sliver, not two people seen together; it must not split the player."""
    first = walk(calibration, 0, (2.0, 3.0), (3.0, 5.0), 0.0, 10.0)
    middle = [o for o in walk(calibration, 0, (3.0, 5.0), (3.0, 7.0), 9.8, 20.0)]
    last = walk(calibration, 0, (3.0, 7.0), (2.0, 3.0), 20.2, 30.0)
    shared = {o.frame_index for o in first} & {o.frame_index for o in middle}
    assert len(shared) == 2

    result = resolve_players([_tracklet(1, first), _tracklet(2, middle), _tracklet(3, last)])

    assert len(result.players) == 1
    assert sorted(result.players[0].source_tracklets) == [1, 2, 3]


# ── Re-identification ────────────────────────────────────────────────────────

def _with_embeddings(observations, centroid, rng, noise=0.35):
    from dataclasses import replace

    out = []
    for obs in observations:
        vector = centroid + noise * rng.normal(size=centroid.shape) / np.sqrt(centroid.size)
        out.append(replace(obs, embedding=(vector / np.linalg.norm(vector)).astype(np.float32)))
    return out


def _four_players_crossing(calibration, with_embeddings: bool):
    """Two pairs, each in one kit, over a minute of play in 5 s pieces with a
    1 s gap between them. Every piece, team-mates swap sides: a player who
    came back to the middle from the left leaves it to the right. Space
    points each piece at the wrong successor, the kit cannot tell them
    apart; only their appearance can."""
    rng = np.random.default_rng(3)
    centroids = []
    for _ in range(4):
        c = rng.normal(size=64)
        centroids.append(c / np.linalg.norm(c))

    # (shirt, y, direction the player swings out to) for each player
    players = [(0, 5.0, -3.0), (0, 5.0, +3.0), (1, 15.0, -3.0), (1, 15.0, +3.0)]
    tracklets, truth = [], {}
    for segment in range(10):
        t0 = segment * 6.0
        for who, (shirt, y, swing) in enumerate(players):
            swing = swing if segment % 2 == 0 else -swing
            out_and_back = walk(calibration, shirt, (5.0, y), (5.0 + swing, y), t0, t0 + 2.4) + \
                walk(calibration, shirt, (5.0 + swing, y), (5.0, y), t0 + 2.6, t0 + 5.0)
            if with_embeddings:
                out_and_back = _with_embeddings(out_and_back, centroids[who], rng)
            tid = segment * 10 + who
            tracklets.append(_tracklet(tid, out_and_back))
            truth[tid] = who
    return tracklets, truth


def test_re_id_tells_apart_team_mates_in_the_same_kit(calibration):
    tracklets, truth = _four_players_crossing(calibration, with_embeddings=True)

    result = resolve_players(tracklets)

    assert result.appearance["cue"] == "reid"
    assert result.appearance["same"] < result.appearance["different"]
    assert len(result.players) == 4
    for player in result.players:
        people = {truth[tid] for tid in player.source_tracklets}
        assert len(people) == 1, f"un giocatore ne contiene {len(people)}: {sorted(player.source_tracklets)}"
    assert result.observations_discarded == 0


def test_without_re_id_the_match_says_so(calibration):
    tracklets, _ = _four_players_crossing(calibration, with_embeddings=False)
    result = resolve_players(tracklets)
    assert result.appearance["cue"] == "colore"
    assert "re-ID" in result.appearance["reason"]


def test_too_few_labelled_pairs_fall_back_to_colour(calibration):
    """Calibration needs examples of both kinds; a short clip does not have
    them, and a scale guessed from three pairs would be worse than colour."""
    from app.ml.identity import calibrate_appearance

    rng = np.random.default_rng(0)
    centroid = np.ones(64) / 8.0
    short = [_tracklet(1, _with_embeddings(walk(calibration, 0, (2.0, 3.0), (3.0, 4.0), 0.0, 4.0), centroid, rng))]
    appearance, summary = calibrate_appearance(short)
    assert appearance is None
    assert summary["cue"] == "colore" and "poche coppie" in summary["reason"]


def test_the_embedding_veto_lands_on_the_colour_veto():
    """Rescaled so that every threshold downstream applies unchanged."""
    from app.ml.identity import MAX_COLOR_DISTANCE, Appearance, _Cluster, appearance_distance

    appearance = Appearance(same_median=0.1, different_median=0.5, positives=30, negatives=30)
    assert appearance.veto == pytest.approx(0.3)

    def cluster(vector):
        return _Cluster(observations=[], sources=[], segments=[], embedding_sum=np.asarray(vector, float))

    a = cluster([1.0, 0.0])
    angle = np.arccos(1.0 - appearance.veto)
    b = cluster([np.cos(angle), np.sin(angle)])
    assert appearance_distance(a, b, appearance) == pytest.approx(MAX_COLOR_DISTANCE)
    assert appearance_distance(a, a, appearance) == pytest.approx(0.0)


def test_a_tracklet_that_switched_between_look_alikes_is_cut_by_re_id(calibration):
    """Same kit on both sides of the switch: colour sees one person, the
    embedding sees two."""
    from app.ml.identity import Appearance, _split_identity_switches

    rng = np.random.default_rng(1)
    a, b = np.zeros(64), np.zeros(64)
    a[0], b[1] = 1.0, 1.0
    first = _with_embeddings(walk(calibration, 0, (4.0, 5.0), (5.0, 5.0), 0.0, 4.0), a, rng, noise=0.2)
    second = _with_embeddings(walk(calibration, 0, (5.0, 5.2), (6.0, 5.2), 4.2, 8.0), b, rng, noise=0.2)
    appearance = Appearance(same_median=0.05, different_median=0.9, positives=30, negatives=30)

    assert len(_split_identity_switches(first + second)) == 1
    pieces = _split_identity_switches(first + second, appearance)
    assert [len(p) for p in pieces] == [len(first), len(second)]
