"""Hits: where the ball's course turns at a player."""
from __future__ import annotations

from app.ml.ball_hits import PlayerBoxes, find_hits
from app.ml.ball_track import Flight

H = 1080
NEAR = PlayerBoxes(0, 0, [(t / 5, (800, 500, 960, 900)) for t in range(0, 100)])     # near court
FAR = PlayerBoxes(2, 1, [(t / 5, (900, 300, 940, 380)) for t in range(0, 100)])      # small, far
RALLY = [{"start_s": 1.0, "end_s": 15.0}]


def _flight(t0: float, n: int, x0: float, y0: float, dx: float, dy: float) -> Flight:
    f = Flight()
    for i in range(n):
        f.add(int(t0 * 30) + i, t0 + i / 30, x0 + dx * i, y0 + dy * i)
    return f


def test_a_flight_leaving_a_player_is_their_hit():
    flights = [_flight(2.0, 20, 900, 560, 0, -12),     # from the near player, towards the far side
               _flight(3.0, 20, 925, 330, 0, 12)]      # back from the far player
    hits = find_hits(flights, [NEAR, FAR], RALLY, H)
    assert [(h.t, h.player) for h in hits] == [(2.0, 0), (3.0, 2)]


def test_the_ball_at_the_feet_or_away_from_everyone_is_no_hit():
    flights = [_flight(2.0, 20, 880, 880, 3, -1),      # a bounce at the near player's feet
               _flight(4.0, 20, 300, 400, 5, 5)]       # nowhere near anyone
    assert find_hits(flights, [NEAR, FAR], RALLY, H) == []


def test_a_ball_hidden_at_the_stroke_is_hit_where_it_was_last_seen():
    flights = [_flight(2.0, 10, 1200, 300, -20, 20),    # coming to the near player...
               _flight(2.5, 20, 600, 300, -10, -10)]    # ...leaving well away from them
    # The first flight ends at about (1020, 480): beside the near player.
    hits = find_hits(flights, [NEAR, FAR], RALLY, H)
    assert [(h.player, h.t) for h in hits] == [(0, 2.3)]


def test_one_stroke_is_one_hit_and_hits_outside_the_rallies_do_not_count():
    flights = [_flight(2.0, 5, 900, 560, 0, -12), _flight(2.3, 20, 905, 520, 0, -12),
               _flight(20.0, 20, 900, 560, 0, -12)]
    hits = find_hits(flights, [NEAR, FAR], RALLY, H)
    assert [h.t for h in hits] == [2.0]
