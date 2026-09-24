"""Players by their place on court: which pair, and which side of the pair.

Why
---
Appearance could not tell team-mates apart on a real match: the re-ID model
put two different people only a little further apart than one person with
himself (0.29 against 0.09), and identity built on appearance alone mixed the
two players of each pair — each of the four "players" was half one person and
half the other. The pairs themselves were right.

Padel supplies what appearance lacks. A pair holds its half of the court,
changing ends only at changeovers. Within the pair each player keeps a side —
drive on the right, revés on the left, as they face the net — for most of a
match. So:

1. The pair is the half of the court, with changeovers found from the look
   of each half: averaged over two people and twenty seconds, that is far
   steadier than any one person's.
2. The player within the pair is the side: in every frame where both are
   seen, the one further left (from their own point of view) votes revés.
3. Appearance still counts — re-ID, or kit colour without it. A piece of
   track that clearly looks like the team-mate is given to the team-mate,
   however it voted on side: that is how a temporary swap of sides is
   followed, when appearance can see it.

Where two pieces given to one player share frames, one detection is
somebody else; identity drops those frames rather than guess. Flipping
the weaker piece to the team-mate instead was tried, and on simulated
matches it gave more detections to the wrong player, not fewer.

Limits, reported rather than hidden: when team-mates swap sides and
appearance cannot tell them apart, the swap is not seen, and their
numbers mix for its duration; and with four identical kits and no re-ID
a changeover cannot be seen either — the halves look the same.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.ml.court import COURT_LENGTH_M, COURT_WIDTH_M
from app.ml.tracking import Observation, Tracklet

# Windows over which the look of each half of the court is averaged.
TEAM_WINDOW_S = 20.0
# Cost of a changeover in the team timeline, in multiples of the typical
# window-to-window evidence, with an absolute floor for matches where the
# halves look alike and that typical value is itself noise.
CHANGEOVER_PENALTY = 3.0
CHANGEOVER_PENALTY_FLOOR = 0.1
# Roles need both pairs: each half must hold at least this share of the
# detections (not so on a practice session on one half).
MIN_HALF_SHARE = 0.1
# Weight of a frame where only one of the pair is seen: their absolute side
# is weaker evidence than the order of the two.
SINGLE_VOTE_WEIGHT = 0.5
# How strongly appearance can override the side, in units of the
# same-person noise measured on the match: at 3, a piece three times further
# from its side's look than noise explains is given to the team-mate.
LOOK_SCORE_LIMIT = 3.0
# Colour noise is measured on pieces of at least this many samples lasting
# at most this long, and only trusted from this many of them.
NOISE_MIN_OBSERVATIONS = 10
NOISE_MAX_DURATION_S = 20.0
NOISE_MIN_PIECES = 20
# Rounds of the alternating refinements (roles from looks, looks from roles).
REFINE_MAX_ROUNDS = 10
# A split by appearance alone is trusted as a start only if it keeps apart
# the pieces seen together for at least this share of the frames they share.
LOOK_SPLIT_MIN_APART = 0.8
# Two pieces sharing at least this many frames are two people.
SHARED_MIN_FRAMES = 3

REVES, DRIVE = 0, 1
# As the API and the web UI name them.
ROLE_NAMES = {DRIVE: "drive", REVES: "reves"}


@dataclass
class RolePiece:
    tracklet: Tracklet
    team: int = 0            # 0 = the pair on the near half at kick-off
    role: int = REVES
    side_score: float = 0.0  # +1 drive … -1 revés, from positions
    look_score: float = 0.0  # + looks like the drive player, - like revés
    look: np.ndarray | None = None   # mean appearance, unit length

    @property
    def preference(self) -> float:
        return self.side_score + self.look_score


@dataclass
class TeamTimeline:
    """Which pair was on which half, window by window."""
    swapped: np.ndarray          # per window: pairs on the ends opposite to kick-off
    # Per window, how much better the kick-off arrangement explains the look
    # of the two halves than the swapped one (swapped cost minus kept cost);
    # NaN where a half was empty. For the diagnostics.
    evidence: np.ndarray
    penalty: float               # price of one changeover, in the same units
    # When each changeover happened, to the detection: the windows only say
    # in which twenty seconds (see _refine_changeover).
    change_times: list[float] = field(default_factory=list)

    @property
    def changeovers(self) -> list[float]:
        return list(self.change_times)

    def swapped_at(self, t: float) -> bool:
        if not len(self.swapped):
            return False
        flips = bisect_right(self.change_times, t)
        return bool(self.swapped[0]) != (flips % 2 == 1)


@dataclass
class RoleAssignment:
    # (team, role) -> pieces
    players: dict[tuple[int, int], list[RolePiece]]
    timeline: TeamTimeline

    @property
    def changeovers(self) -> list[float]:
        """Times at which the pairs changed ends."""
        return self.timeline.changeovers


def roles_apply(pieces: list[Tracklet]) -> bool:
    """Whether the match has the shape roles need: people on both halves."""
    near = far = 0
    for piece in pieces:
        for obs in piece.observations:
            if obs.foot_court[1] < COURT_LENGTH_M / 2:
                near += 1
            else:
                far += 1
    total = near + far
    return total > 0 and min(near, far) >= MIN_HALF_SHARE * total


def assign_roles(pieces: list[Tracklet], appearance) -> RoleAssignment:
    """Give every piece of track a pair and a side. `appearance` is the
    match's calibrated re-ID (see identity.Appearance) or None."""
    reid = appearance is not None
    timeline = _team_timeline(pieces, reid)
    swapped_at = timeline.swapped_at

    role_pieces = [RolePiece(tracklet=p) for p in pieces]
    for rp in role_pieces:
        rp.team = _majority_team(rp.tracklet, swapped_at)
        rp.look = _mean_look(rp.tracklet.observations, reid)

    _side_votes(role_pieces, swapped_at)
    noise = float(appearance.same_median) if reid else _colour_noise(pieces)
    for team in (0, 1):
        members = [rp for rp in role_pieces if rp.team == team]
        if noise is None:
            _roles_from_side(members)
        else:
            _best_roles(members, max(noise, 1e-3))

    players: dict[tuple[int, int], list[RolePiece]] = defaultdict(list)
    for rp in role_pieces:
        players[(rp.team, rp.role)].append(rp)
    return RoleAssignment(players=dict(players), timeline=timeline)


def _best_roles(members: list[RolePiece], noise: float) -> None:
    """Roles within one pair from side and appearance together.

    Refining from the sides alone gets stuck where team-mates swap sides
    often: each role then starts as a mix of both people, the two looks
    are alike, and appearance can no longer pull them apart. So the
    refinement also starts from appearance alone — but only when that split
    puts pieces seen together in different groups, which two people are
    and which a split on noise, or on a few odd pieces, does not. Of the
    starts left, the one whose end both cues agree with more wins.
    """
    pairs = _shared_pairs(members)
    best: tuple[float, list[int]] | None = None
    for start in (_roles_from_side, _roles_from_look):
        if not start(members) or (start is _roles_from_look and not _separates(members, pairs)):
            continue
        for _ in range(REFINE_MAX_ROUNDS):
            _look_scores(members, noise)
            changed = False
            for rp in members:
                role = DRIVE if rp.preference > 0 else REVES
                changed |= role != rp.role
                rp.role = role
            if not changed:
                break
        _look_scores(members, noise)
        agreement = sum(len(rp.tracklet) * rp.preference * (1.0 if rp.role == DRIVE else -1.0)
                        for rp in members)
        if best is None or agreement > best[0]:
            best = (agreement, [rp.role for rp in members])
    if best is not None:
        for rp, role in zip(members, best[1]):
            rp.role = role
        _look_scores(members, noise)


def _roles_from_side(members: list[RolePiece]) -> bool:
    for rp in members:
        rp.role = DRIVE if rp.side_score > 0 else REVES
    return True


def _roles_from_look(members: list[RolePiece]) -> bool:
    """Two groups by appearance alone, named drive and revés by where their
    pieces stood. Length-weighted two-means, started from a split along
    the direction in which the looks vary most: the two people of a pair
    are tracked for about as long, so the start is balanced, and one odd
    piece cannot become a group of its own. False if no split was made."""
    _roles_from_side(members)
    looked = [rp for rp in members if rp.look is not None]
    if len(looked) < 2:
        return False
    looks = np.stack([rp.look for rp in looked])
    weights = np.array([len(rp.tracklet) for rp in looked], dtype=np.float64)
    centred = looks - np.average(looks, axis=0, weights=weights)
    direction = np.linalg.svd(centred * np.sqrt(weights)[:, None], full_matrices=False)[2][0]
    projection = centred @ direction
    order = np.argsort(projection)
    median = projection[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2.0)]]
    group = (projection > median).astype(int)

    for _ in range(REFINE_MAX_ROUNDS):
        if group.min() == group.max():
            return False
        centres = np.stack([(looks[group == g] * weights[group == g, None]).sum(axis=0) for g in (0, 1)])
        centres /= np.clip(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12, None)
        new_group = np.argmax(looks @ centres.T, axis=1)
        if np.array_equal(new_group, group):
            break
        group = new_group
    if group.min() == group.max():
        return False

    sides = weights * np.array([rp.side_score for rp in looked])
    drive_group = 0 if sides[group == 0].sum() >= sides[group == 1].sum() else 1
    for rp, g in zip(looked, group):
        rp.role = DRIVE if g == drive_group else REVES
    return True


def _separates(members: list[RolePiece], pairs: list[tuple[int, int, int]]) -> bool:
    """Whether the current roles keep apart the pieces seen together, for
    most of the frames they share."""
    total = sum(count for _, _, count in pairs)
    apart = sum(count for i, j, count in pairs if members[i].role != members[j].role)
    return total > 0 and apart >= LOOK_SPLIT_MIN_APART * total


def _look(obs: Observation, reid: bool) -> np.ndarray | None:
    """An observation's appearance as a unit vector: the re-ID embedding, or
    the square root of the colour histogram — whose dot product with another
    is their Bhattacharyya coefficient, so both cues share one geometry."""
    if reid:
        return obs.embedding
    total = float(obs.color.sum())
    return np.sqrt(np.clip(obs.color, 0.0, None) / total) if total > 0 else None


def _mean_look(observations: list[Observation], reid: bool) -> np.ndarray | None:
    vectors = [v for v in (_look(o, reid) for o in observations) if v is not None]
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else None


def _colour_noise(pieces: list[Tracklet]) -> float | None:
    """How far one person's colour drifts from itself: the median distance
    between the two halves of short pieces, which the tracker is unlikely to
    have swapped. The colour counterpart of the re-ID calibration."""
    distances = []
    for piece in pieces:
        obs = piece.observations
        if len(obs) < NOISE_MIN_OBSERVATIONS or obs[-1].timestamp_s - obs[0].timestamp_s > NOISE_MAX_DURATION_S:
            continue
        half = len(obs) // 2
        a, b = _mean_look(obs[:half], False), _mean_look(obs[half:], False)
        if a is not None and b is not None:
            distances.append(1.0 - float(a @ b))
    return float(np.median(distances)) if len(distances) >= NOISE_MIN_PIECES else None


# ── Pairs over time ──────────────────────────────────────────────────────────

def _team_timeline(pieces: list[Tracklet], reid: bool) -> TeamTimeline:
    """For each window, whether the pairs have swapped ends since kick-off.

    Each window gives the look of the near half and of the far half. Keeping
    the kick-off arrangement costs d(near, A) + d(far, B); swapped costs
    d(near, B) + d(far, A). A two-state dynamic programme with a price per
    changeover picks the cheapest history — changeovers only where the
    evidence holds for a while, none at all if the halves look alike.
    """
    observations = [o for p in pieces for o in p.observations]
    if not observations:
        return TeamTimeline(np.zeros(0, dtype=bool), np.zeros(0), 0.0)
    end = max(o.timestamp_s for o in observations)
    n_windows = int(end // TEAM_WINDOW_S) + 1

    near: list[list[np.ndarray]] = [[] for _ in range(n_windows)]
    far: list[list[np.ndarray]] = [[] for _ in range(n_windows)]
    for obs in observations:
        vector = _look(obs, reid)
        if vector is None:
            continue
        w = int(obs.timestamp_s // TEAM_WINDOW_S)
        (near if obs.foot_court[1] < COURT_LENGTH_M / 2 else far)[w].append(vector)

    def describe(vectors):
        if not vectors:
            return None
        mean = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else None

    near_d = [describe(v) for v in near]
    far_d = [describe(v) for v in far]
    both = [w for w in range(n_windows) if near_d[w] is not None and far_d[w] is not None]
    if not both:
        return TeamTimeline(np.zeros(0, dtype=bool), np.zeros(0), 0.0)

    def distance(a, b):
        return 1.0 - float(a @ b)

    team_a, team_b = near_d[both[0]], far_d[both[0]]
    swapped = np.zeros(n_windows, dtype=bool)
    for _ in range(5):
        keep_cost = np.zeros(n_windows)
        swap_cost = np.zeros(n_windows)
        for w in both:
            keep_cost[w] = distance(near_d[w], team_a) + distance(far_d[w], team_b)
            swap_cost[w] = distance(near_d[w], team_b) + distance(far_d[w], team_a)
        evidence = np.abs(keep_cost - swap_cost)[both]
        penalty = max(CHANGEOVER_PENALTY * float(np.median(evidence)), CHANGEOVER_PENALTY_FLOOR)
        swapped = _two_state_path(keep_cost, swap_cost, penalty)

        a_vectors = [near_d[w] if not swapped[w] else far_d[w] for w in both]
        b_vectors = [far_d[w] if not swapped[w] else near_d[w] for w in both]
        team_a, team_b = describe(a_vectors), describe(b_vectors)

    evidence_by_window = np.full(n_windows, np.nan)
    evidence_by_window[both] = (swap_cost - keep_cost)[both]

    change_times: list[float] = []
    for w in range(1, n_windows):
        if swapped[w] != swapped[w - 1]:
            # Between the previous changeover and the end of the next window.
            lo = max((w - 1) * TEAM_WINDOW_S, change_times[-1] if change_times else 0.0)
            hi = (w + 1) * TEAM_WINDOW_S
            change_times.append(_refine_changeover(
                observations, reid, team_a, team_b, lo, hi, bool(swapped[w - 1]), w * TEAM_WINDOW_S,
            ))
    return TeamTimeline(swapped=swapped, evidence=evidence_by_window, penalty=penalty,
                        change_times=change_times)


def _refine_changeover(
    observations: list[Observation], reid: bool, team_a: np.ndarray, team_b: np.ndarray,
    lo: float, hi: float, swapped_before: bool, fallback: float,
) -> float:
    """The moment, between `lo` and `hi`, at which the pairs changed ends.

    The windows place a changeover only to the twenty seconds, and every
    detection between the real moment and the window edge went to the other
    pair. Here each detection in the span votes by its own look — near the
    half's pair or the other — and the split is where the arrangement before
    and the one after, together, explain the detections best.
    """
    picked = [o for o in observations if lo <= o.timestamp_s < hi]
    picked = [(o, v) for o, v in ((o, _look(o, reid)) for o in picked) if v is not None]
    if not picked:
        return fallback
    picked.sort(key=lambda item: item[0].timestamp_s)
    times = np.array([o.timestamp_s for o, _ in picked])
    looks = np.stack([v for _, v in picked])
    near = np.array([o.foot_court[1] < COURT_LENGTH_M / 2 for o, _ in picked])
    to_a = 1.0 - looks @ team_a
    to_b = 1.0 - looks @ team_b
    # Kick-off arrangement: pair A on the near half. Swapped: pair B there.
    kept = np.where(near, to_a, to_b)
    swapped = np.where(near, to_b, to_a)
    before, after = (swapped, kept) if swapped_before else (kept, swapped)

    # cost[k]: the first k detections in the old arrangement, the rest in
    # the new one. Splits only between different instants.
    cost = np.concatenate([[0.0], np.cumsum(before)]) + np.concatenate([np.cumsum(after[::-1])[::-1], [0.0]])
    candidates = [k for k in range(len(times) + 1)
                  if k in (0, len(times)) or times[k] != times[k - 1]]
    k = min(candidates, key=lambda c: cost[c])
    if k == 0:
        return lo
    if k == len(times):
        return hi
    return float((times[k - 1] + times[k]) / 2.0)


def _two_state_path(keep_cost: np.ndarray, swap_cost: np.ndarray, penalty: float) -> np.ndarray:
    """Cheapest sequence of keep/swap states, paying `penalty` per change.
    Starts in the kick-off arrangement."""
    n = len(keep_cost)
    total = np.array([keep_cost[0], swap_cost[0] + penalty])
    back = np.zeros((n, 2), dtype=int)
    for w in range(1, n):
        stay_keep, from_swap = total[0], total[1] + penalty
        stay_swap, from_keep = total[1], total[0] + penalty
        back[w, 0] = 0 if stay_keep <= from_swap else 1
        back[w, 1] = 1 if stay_swap <= from_keep else 0
        total = np.array([min(stay_keep, from_swap) + keep_cost[w],
                          min(stay_swap, from_keep) + swap_cost[w]])
    path = np.zeros(n, dtype=bool)
    state = int(np.argmin(total))
    for w in range(n - 1, -1, -1):
        path[w] = bool(state)
        state = back[w, state]
    return path


def _team_of(obs: Observation, swapped_at) -> int:
    near = obs.foot_court[1] < COURT_LENGTH_M / 2
    return 0 if near != swapped_at(obs.timestamp_s) else 1


def _majority_team(tracklet: Tracklet, swapped_at) -> int:
    votes = sum(_team_of(o, swapped_at) for o in tracklet.observations)
    return 1 if votes * 2 > len(tracklet.observations) else 0


# ── Side within the pair ─────────────────────────────────────────────────────

def _own_side_x(obs: Observation) -> float:
    """x from the player's own point of view, facing the net: 0 on their
    left, COURT_WIDTH_M on their right. The far pair faces the camera, so
    their left is the camera's right."""
    x = obs.foot_court[0]
    return x if obs.foot_court[1] < COURT_LENGTH_M / 2 else COURT_WIDTH_M - x


def _side_votes(role_pieces: list[RolePiece], swapped_at) -> None:
    by_frame: dict[int, list[tuple[RolePiece, Observation]]] = defaultdict(list)
    for rp in role_pieces:
        for obs in rp.tracklet.observations:
            by_frame[obs.frame_index].append((rp, obs))

    votes: dict[int, float] = defaultdict(float)
    weights: dict[int, float] = defaultdict(float)
    for entries in by_frame.values():
        per_team: dict[int, list[tuple[RolePiece, Observation]]] = defaultdict(list)
        for rp, obs in entries:
            per_team[rp.team].append((rp, obs))
        for members in per_team.values():
            if len(members) == 2 and members[0][0] is not members[1][0]:
                # Both team-mates seen: the order decides, whatever the
                # absolute position.
                (a, oa), (b, ob) = sorted(members, key=lambda m: _own_side_x(m[1]))
                votes[id(a)] -= 1.0
                votes[id(b)] += 1.0
                weights[id(a)] += 1.0
                weights[id(b)] += 1.0
            elif len(members) == 1:
                rp, obs = members[0]
                votes[id(rp)] += SINGLE_VOTE_WEIGHT * (1.0 if _own_side_x(obs) > COURT_WIDTH_M / 2 else -1.0)
                weights[id(rp)] += SINGLE_VOTE_WEIGHT
            # Three or more of one pair in a frame: someone else is in it;
            # the frame says nothing reliable about sides.

    for rp in role_pieces:
        weight = weights.get(id(rp), 0.0)
        rp.side_score = votes.get(id(rp), 0.0) / weight if weight > 0 else 0.0


def _look_scores(role_pieces: list[RolePiece], noise: float) -> None:
    """Appearance's case for each piece: how much closer it looks to its pair's
    drive player than to the revés player, in units of same-person noise.
    Each player's look is the length-weighted mean of their current pieces,
    leaving the piece being judged out."""
    sums: dict[tuple[int, int], np.ndarray] = {}
    for rp in role_pieces:
        if rp.look is None:
            continue
        key = (rp.team, rp.role)
        weighted = rp.look * len(rp.tracklet)
        sums[key] = sums[key] + weighted if key in sums else weighted.copy()

    for rp in role_pieces:
        rp.look_score = 0.0
        if rp.look is None:
            continue
        looks = {}
        for role in (REVES, DRIVE):
            total = sums.get((rp.team, role))
            if total is None:
                break
            if role == rp.role:
                total = total - rp.look * len(rp.tracklet)
            norm = float(np.linalg.norm(total))
            if norm <= 0:
                break
            looks[role] = total / norm
        if len(looks) < 2:
            continue
        to_reves = 1.0 - float(rp.look @ looks[REVES])
        to_drive = 1.0 - float(rp.look @ looks[DRIVE])
        rp.look_score = float(np.clip((to_reves - to_drive) / noise, -LOOK_SCORE_LIMIT, LOOK_SCORE_LIMIT))


def _shared_pairs(members: list[RolePiece]) -> list[tuple[int, int, int]]:
    """(i, j, frames shared) for the pieces of one pair seen together in at
    least SHARED_MIN_FRAMES frames: two people."""
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, rp in enumerate(members):
        for obs in rp.tracklet.observations:
            by_frame[obs.frame_index].append(index)
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for present in by_frame.values():
        for n, i in enumerate(present):
            for j in present[n + 1:]:
                if i != j and members[i].team == members[j].team:
                    shared[(min(i, j), max(i, j))] += 1
    return [(i, j, count) for (i, j), count in shared.items() if count >= SHARED_MIN_FRAMES]
