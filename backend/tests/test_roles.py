"""Players by their place on court: pair from the half, player from the side."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.ml.identity import resolve_players
from app.ml.roles import DRIVE, REVES
from app.ml.tracking import Tracklet
from tests.synthetic import make_observation

HZ = 5.0
# Where each person plays at kick-off: near revés, near drive, far drive, far
# revés — the order of the ids, left to right on screen, pair by pair.
HOMES = [(2.5, 6.0), (7.5, 6.0), (2.5, 14.0), (7.5, 14.0)]


def _play(calibration, duration_s, shirts, *, embeddings=None, home=None, breaks=(), piece_s=4.0):
    """Four people moving around their places, tracked in pieces of about
    `piece_s` seconds with a short gap between them — as fragmented as a
    real match. Every piece also ends at each of `breaks`, where the play
    changes (a changeover, two team-mates swapping sides).

    `home(k, t)` is where person k plays at time t; `embeddings` are their
    re-ID centroids, None for colour only. Returns the tracklets and the
    person behind each tracklet id."""
    home = home or (lambda k, t: HOMES[k])
    rng = np.random.default_rng(0)
    looks = {shirt: make_observation(calibration, (5.0, 5.0), 0.0, shirt) for shirt in set(shirts)}
    tracklets, truth = [], {}
    for k in range(4):
        # Stagger the gaps so the four people are not all lost at once.
        cuts = sorted({*np.arange(0.9 * k, duration_s, piece_s), *breaks, duration_s})
        for start, end in zip([0.0, *cuts[:-1]], cuts):
            observations = []
            for t in np.arange(start + 0.4, end, 1.0 / HZ):
                x0, y0 = home(k, t)
                if x0 is None:
                    continue                  # off the court, walking to the other end
                xy = (float(np.clip(x0 + 1.5 * np.sin(0.7 * t + k), 0.3, 9.7)),
                      float(y0 + 1.5 * np.sin(0.45 * t + 2 * k)))
                # Light, pose and the detector's box make every sample's
                # colour a little different.
                color = np.clip(looks[shirts[k]].color + rng.normal(0, 0.004, looks[shirts[k]].color.size), 0, None)
                obs = replace(looks[shirts[k]], frame_index=int(round(t * HZ)), timestamp_s=float(t),
                              foot_court=xy, color=(color / color.sum()).astype(np.float32))
                if embeddings is not None:
                    vector = embeddings[k] + 0.35 * rng.normal(size=embeddings[k].size) / np.sqrt(embeddings[k].size)
                    obs = replace(obs, embedding=(vector / np.linalg.norm(vector)).astype(np.float32))
                observations.append(obs)
            if len(observations) >= 3:
                tid = len(tracklets)
                tracklets.append(Tracklet(id=tid, observations=observations))
                truth[tid] = k
    return tracklets, truth


def _centroids(seed=3, size=64):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(4):
        c = rng.normal(size=size)
        out.append(c / np.linalg.norm(c))
    return out


def _people(result, truth):
    """The person behind each player, by id; fails if a player mixes two."""
    people = []
    for player in sorted(result.players, key=lambda p: p.player_id):
        found = {truth[tid] for tid in player.source_tracklets}
        assert len(found) == 1, f"il giocatore {player.player_id} unisce le persone {sorted(found)}"
        people.append(found.pop())
    return people


def test_team_mates_in_one_kit_are_told_apart_by_their_side(calibration):
    """All four in the same kit and no re-ID: nothing in their look tells
    anybody apart, yet each keeps a side of their pair."""
    tracklets, truth = _play(calibration, 120.0, shirts=[0, 0, 0, 0])

    result = resolve_players(tracklets)

    assert result.appearance["method"] == "ruoli"
    assert _people(result, truth) == [0, 1, 2, 3]
    players = sorted(result.players, key=lambda p: p.player_id)
    assert [p.team for p in players] == [0, 0, 1, 1]
    # The far pair faces the camera: their drive is on the left of the screen.
    assert [p.role for p in players] == [REVES, DRIVE, DRIVE, REVES]
    assert result.side_changes == 0


def test_pairs_keep_their_ids_across_a_changeover(calibration):
    """The pairs change ends after a minute. Positions alone would give the
    near half's ids to whoever stands there; the look of each half tells
    that the pairs swapped, and ids follow the people."""
    def home(k, t):
        if 60.0 <= t < 66.0:
            return None, None
        x, y = HOMES[k]
        return (10.0 - x, 20.0 - y) if t >= 66.0 else (x, y)

    tracklets, truth = _play(calibration, 130.0, shirts=[0, 0, 1, 1], home=home, breaks=(60.0, 66.0))

    result = resolve_players(tracklets)

    assert result.side_changes == 1
    assert _people(result, truth) == [0, 1, 2, 3]
    assert [p.team for p in sorted(result.players, key=lambda p: p.player_id)] == [0, 0, 1, 1]


def test_re_id_follows_team_mates_who_swap_sides(calibration):
    """For twenty seconds the near pair trade sides. By side alone they
    would trade numbers too; their look says who is who."""
    def home(k, t):
        x, y = HOMES[k]
        return (10.0 - x, y) if k < 2 and 40.0 <= t < 60.0 else (x, y)

    tracklets, truth = _play(calibration, 100.0, shirts=[0, 0, 0, 0], embeddings=_centroids(),
                             home=home, breaks=(40.0, 60.0))

    result = resolve_players(tracklets)

    assert result.appearance["cue"] == "reid"
    assert _people(result, truth) == [0, 1, 2, 3]


def test_kit_colour_follows_team_mates_who_swap_sides(calibration):
    """The same swap without re-ID, but each player in their own kit."""
    def home(k, t):
        x, y = HOMES[k]
        return (10.0 - x, y) if k < 2 and 40.0 <= t < 60.0 else (x, y)

    tracklets, truth = _play(calibration, 100.0, shirts=[0, 1, 2, 3], home=home, breaks=(40.0, 60.0))

    result = resolve_players(tracklets)

    assert result.appearance["cue"] == "colore"
    assert _people(result, truth) == [0, 1, 2, 3]


def test_a_changeover_is_not_invented_when_the_pairs_look_alike(calibration):
    """With four identical kits and no re-ID the halves always look the
    same: noise between windows must not read as the pairs changing ends."""
    tracklets, truth = _play(calibration, 600.0, shirts=[0, 0, 0, 0])

    result = resolve_players(tracklets)

    assert result.side_changes == 0
    assert _people(result, truth) == [0, 1, 2, 3]


def test_a_single_half_falls_back_to_linking(calibration):
    """Two people practising on one half have no pairs to find."""
    tracklets, _ = _play(calibration, 60.0, shirts=[0, 1, 2, 3])
    near = [t for t in tracklets if t.observations[0].foot_court[1] < 10.0]

    result = resolve_players(near)

    assert result.appearance["method"] == "collegamento"
    assert all(p.role is None for p in result.players)


def test_a_change_of_light_is_not_taken_for_two_people(calibration):
    """Halfway through, the light changes and every kit looks different —
    floodlights at dusk, a cloud. Split by look alone, the pieces fall into
    before and after, not into two people: team-mates seen together land
    in the same group, and that split must not be used."""
    tracklets, truth = _play(calibration, 120.0, shirts=[0, 0, 0, 0], breaks=(60.0,))
    later = make_observation(calibration, (5.0, 5.0), 0.0, 2).color
    for tracklet in tracklets:
        if tracklet.observations[0].timestamp_s >= 60.0:
            tracklet.observations[:] = [replace(o, color=later) for o in tracklet.observations]

    result = resolve_players(tracklets)

    assert _people(result, truth) == [0, 1, 2, 3]


def test_a_player_seen_without_their_team_mate_still_gets_a_side(calibration):
    """The near pair are never detected together — one is always hidden
    behind the other or out of frame. With no pair to order, each piece
    goes by the half of the court's width it stands in."""
    def home(k, t):
        if k < 2 and int(t // 5.0) % 2 != k:
            return None, None
        return HOMES[k]

    tracklets, truth = _play(calibration, 100.0, shirts=[0, 0, 1, 1], home=home,
                             breaks=tuple(np.arange(5.0, 100.0, 5.0)))

    result = resolve_players(tracklets)

    assert _people(result, truth) == [0, 1, 2, 3]
