"""Turn raw tracklets into four stable, team-assigned players.

The problem
-----------
The previous implementation was one line: keep the four `track_id`s that
appear in the most frames. With occlusions — and padel has constant occlusion,
four people in a 20x10 m box — a single player generates five to ten track ids
over a match. "Top four by frequency" therefore routinely returns three
fragments of the same person plus one of somebody else, and every per-player
number computed from it is meaningless.

The approach here
-----------------
1. Cut tracklets where the shirt changes: a track that switched person
   when two players crossed carries both, and would poison any cluster.
2. Link the pieces into clusters, using shirt colour, a physical
   reachability test at every hand-over, and coexistence: two clusters seen
   in the same frames while both on court are two people. A cluster is a set
   of pieces with gaps, and a piece can fill a gap. This is the step that
   repairs occlusions.
3. Take as players the four largest clusters that coexist with one another
   — four people on court at once — not simply the four largest.
4. Assign teams from which half of the court each cluster lives in. Padel
   teams hold their half for a whole game, so the side histogram is a strong
   and cheap team signal.
5. Number the players deterministically so the same person keeps the same id
   between re-runs: near team left-to-right = 0,1; far team = 2,3.

Known limitations, reported rather than hidden:

* players swap ends at changeovers. A swap that happens while both players
  are untracked can split one person into two clusters. `side_changes` and
  the warnings list expose this instead of silently degrading the numbers.
* shirt colour is the only appearance cue. Team-mates in the same kit are
  told apart only by where they are, so when the tracker swaps them while
  crossing, their numbers mix. On simulated matches: 91-98% of all
  detections go to the right player with four distinct kits; with two kits
  only 55-65% of those attributed do.
"""
from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.ml.court import COURT_LENGTH_M
from app.ml.tracking import Observation, Tracklet, observation_key

# (fusioni completate, totale tracce di partenza) — serve solo a far avanzare
# la barra durante uno stadio che altrimenti resta muto per minuti.
ProgressCallback = Callable[[int, int], None]

N_PLAYERS = 4

# A player may disappear behind an opponent for a few seconds; a changeover
# takes up to a minute. Beyond that, linking is guesswork.
MAX_LINK_GAP_S = 75.0
# Colour agreement good enough to link across a spatially implausible jump,
# which is what a changeover looks like when tracking is lost mid-walk.
STRONG_COLOR_MATCH = 0.22
# Hard veto: two tracklets whose shirts disagree this much are different
# people, however conveniently they line up in time and space. Without this
# veto a long gap makes the spatial term vanish and the blended cost lets any
# colour through — which merges one player's first half with a team-mate's
# second half and leaves a real player out of the top four entirely.
MAX_COLOR_DISTANCE = 0.45
# Distance scale for the spatial term. A fixed scale (rather than the
# reachable distance, which grows without bound with the gap) keeps "closer is
# better" meaningful for long gaps instead of making them free.
SPATIAL_SCALE_M = 6.0
# Overall cost above which two tracklets are left as separate people.
LINK_COST_THRESHOLD = 0.55
# Observations averaged to place the two ends of a piece. A track often dies
# on a bad box — feet hidden, foot point thrown a metre or more — so the very
# last observation is the least reliable one to judge a junction by.
ENDPOINT_SAMPLES = 3
# A piece of a tracklet shorter than this is not split off: three samples is
# the tracker's own minimum for a track worth keeping.
MIN_PIECE_OBSERVATIONS = 3
# Two clusters are two people when, during the time both are on court, they
# are detected in the same frame at least this often. It is a density, not a
# duration, because the two cases it separates differ in density: two
# players on court are seen together in most samples (0.6-0.8 measured),
# while two pieces of one player share a frame only where a tracklet briefly
# followed someone else (0-0.2). Any overlap at all used to count, and one
# such stray tracklet then kept two halves of a player apart for good.
COEXIST_DENSITY = 0.5
# Below this many shared frames the density is not evidence: two stray
# frames in a 0.6 s overlap already make 50%. Measured on simulated matches,
# requiring two let such a sliver keep a player's halves apart.
COEXIST_MIN_SHARED = 3


@dataclass
class PlayerTrack:
    player_id: int
    team: int                     # 0 = near half (Y < 10 m), 1 = far half
    observations: list[Observation]
    source_tracklets: list[int] = field(default_factory=list)
    _keys: set | None = field(default=None, repr=False, compare=False)

    def owns(self, obs: Observation) -> bool:
        """Whether this detection was attributed to this player. Asked per
        observation, not per tracklet id: a tracklet that switched from one
        player to another is split, and its pieces go to different players."""
        if self._keys is None:
            self._keys = {observation_key(o) for o in self.observations}
        return observation_key(obs) in self._keys

    @property
    def start_s(self) -> float:
        return self.observations[0].timestamp_s

    @property
    def end_s(self) -> float:
        return self.observations[-1].timestamp_s

    def __len__(self) -> int:
        return len(self.observations)


@dataclass
class IdentityResult:
    players: list[PlayerTrack]
    warnings: list[str]
    side_changes: int
    clusters_found: int
    observations_discarded: int


@dataclass(frozen=True)
class _Segment:
    """One piece of a tracklet inside a cluster: its time span, and where the
    player was at both ends, which is all linking needs to know.

    The ends are the median of the first and last few samples, each paired
    with the time of its middle sample so that position and time stay
    consistent for the reachability test."""
    start_s: float
    end_s: float
    head_s: float
    head_xy: tuple[float, float]
    tail_s: float
    tail_xy: tuple[float, float]

    @classmethod
    def of(cls, observations: list[Observation]) -> "_Segment":
        k = min(ENDPOINT_SAMPLES, len(observations))
        head, tail = observations[:k], observations[-k:]
        return cls(
            start_s=observations[0].timestamp_s,
            end_s=observations[-1].timestamp_s,
            head_s=head[k // 2].timestamp_s,
            head_xy=_median_xy(head),
            tail_s=tail[k // 2].timestamp_s,
            tail_xy=_median_xy(tail),
        )


def _median_xy(observations: list[Observation]) -> tuple[float, float]:
    xy = np.array([o.foot_court for o in observations], dtype=np.float64)
    median = np.median(xy, axis=0)
    return float(median[0]), float(median[1])


@dataclass
class _Cluster:
    observations: list[Observation]
    sources: list[int]
    # The cluster's pieces in time order. A cluster is a set of intervals with
    # gaps between them, not one block from its first observation to its
    # last: greedy linking often joins two pieces that are not adjacent, and a
    # tracklet that falls in the gap between them must still be able to join.
    segments: list[_Segment]
    # Cached colour signature. Computing it averages over every observation in
    # the cluster, and the linking loop compares each cluster against all the
    # others repeatedly — recomputing it per comparison made the stage cubic in
    # the number of tracklets. A merge builds a new _Cluster, so the cache
    # cannot go stale.
    _signature: np.ndarray | None = field(default=None, repr=False, compare=False)
    _sqrt_signature: np.ndarray | None = field(default=None, repr=False, compare=False)
    _frames: np.ndarray | None = field(default=None, repr=False, compare=False)

    def frames(self) -> np.ndarray:
        """Sorted frame indices of the cluster's observations."""
        if self._frames is None:
            self._frames = np.array(sorted({o.frame_index for o in self.observations}), dtype=np.int64)
        return self._frames

    @property
    def start_s(self) -> float:
        return self.observations[0].timestamp_s

    @property
    def end_s(self) -> float:
        return self.observations[-1].timestamp_s

    def signature(self) -> np.ndarray:
        if self._signature is None:
            weights = np.array([o.confidence for o in self.observations], dtype=np.float32)
            stack = np.stack([o.color for o in self.observations])
            total = float(weights.sum())
            sig = stack.mean(axis=0) if total <= 0 else (stack * weights[:, None]).sum(axis=0) / total
            norm = float(sig.sum())
            self._signature = sig / norm if norm > 0 else sig
        return self._signature

    def sqrt_signature(self) -> np.ndarray:
        """Element-wise root of the signature: the Bhattacharyya coefficient
        of two clusters is then one dot product, which matters because the
        linking loop compares clusters hundreds of thousands of times."""
        if self._sqrt_signature is None:
            self._sqrt_signature = np.sqrt(np.clip(self.signature(), 0.0, None))
        return self._sqrt_signature

    def mean_x(self) -> float:
        return float(np.mean([o.foot_court[0] for o in self.observations]))

    def far_fraction(self) -> float:
        ys = np.array([o.foot_court[1] for o in self.observations], dtype=np.float32)
        return float((ys > COURT_LENGTH_M / 2.0).mean())


def resolve_players(
    tracklets: list[Tracklet],
    max_speed_ms: float = 8.0,
    progress: ProgressCallback | None = None,
) -> IdentityResult:
    """Link tracklets into four players and assign teams and stable ids."""
    warnings: list[str] = []
    usable = [t for t in tracklets if len(t) >= 3]
    if not usable:
        return IdentityResult([], ["Nessun giocatore rilevato nel video."], 0, 0, 0)

    # A tracklet that switched person mid-way — two players crossing, the
    # tracker keeping the wrong one — carries both. Linking treats a cluster
    # as one person, so a single such tracklet makes two clusters of the same
    # player "coexist" and blocks them from ever merging. Cut it where the
    # shirt changes.
    pieces = [
        Tracklet(id=t.id, observations=part)
        for t in usable
        for part in _split_identity_switches(t.observations)
    ]
    clusters = _link_tracklets(pieces, max_speed_ms, progress=progress)
    clusters.sort(key=lambda c: -len(c.observations))

    # Counted from the input: frames dropped as shared while merging are
    # discarded observations too.
    total_obs = sum(len(t) for t in usable)
    selected = _select_players(clusters, _sample_period(usable))
    discarded = total_obs - sum(len(c.observations) for c in selected)

    if len(selected) < N_PLAYERS:
        warnings.append(
            f"Rilevati solo {len(selected)} giocatori invece di 4: "
            "controlla che tutto il campo sia inquadrato."
        )

    # A long tail of leftover clusters means the linking did not close the
    # fragmentation — the per-player numbers will under-report.
    if discarded > 0.20 * max(total_obs, 1):
        warnings.append(
            f"Il {discarded * 100 // max(total_obs, 1)}% delle rilevazioni non è stato "
            "attribuito a uno dei 4 giocatori: tracciamento frammentato."
        )

    players = _assign_teams_and_ids(selected, warnings)
    side_changes = _count_side_changes(players)
    if side_changes:
        warnings.append(
            f"Rilevati {side_changes} cambi di campo: le heatmap uniscono "
            "le posizioni prima e dopo il cambio."
        )

    return IdentityResult(
        players=players,
        warnings=warnings,
        side_changes=side_changes,
        clusters_found=len(clusters),
        observations_discarded=discarded,
    )


def _select_players(clusters: list[_Cluster], period_s: float) -> list[_Cluster]:
    """The four clusters that are the four players. `clusters` is sorted
    largest first.

    Four different people are on court at the same time, so the four players
    are clusters that coexist with one another. The four largest are not:
    a player whose tracking broke for longer than linking can bridge becomes
    two large clusters that never coexist, and "the four largest" then counts
    that player twice and leaves a real player out. Largest first, a cluster
    is taken only if it coexists with every one already taken; if that
    cannot fill four places, the rest are filled by size as before.
    """
    chosen: list[_Cluster] = []
    for cluster in clusters:
        if len(chosen) == N_PLAYERS:
            return chosen
        if all(_coexist(cluster, other, period_s) for other in chosen):
            chosen.append(cluster)
    for cluster in clusters:
        if len(chosen) == N_PLAYERS:
            break
        if all(cluster is not other for other in chosen):
            chosen.append(cluster)
    # Keep the largest-first order the rest of the stage expects.
    return sorted(chosen, key=lambda c: -len(c.observations))


# ── Identity switches ────────────────────────────────────────────────────────

def _split_identity_switches(observations: list[Observation]) -> list[list[Observation]]:
    """Cut a tracklet where it changes person.

    Finds the split point that makes the two sides' shirts most different,
    and cuts there if the two sides would be vetoed as different people were
    they separate tracklets — the same MAX_COLOR_DISTANCE linking uses, so
    this adds no threshold of its own. Recurses on both sides, for tracklets
    that switched more than once.

    Over-splitting is cheap: linking rejoins pieces of one person. Missing a
    switch is not: the mixed tracklet blocks two whole clusters from merging.
    Team-mates in the same kit cannot be told apart this way, and are left
    as they are.
    """
    n = len(observations)
    if n < 2 * MIN_PIECE_OBSERVATIONS:
        return [observations]

    weights = np.array([o.confidence for o in observations], dtype=np.float64)
    colors = np.stack([o.color for o in observations]).astype(np.float64) * weights[:, None]
    cumulative = np.cumsum(colors, axis=0)
    cuts = np.arange(MIN_PIECE_OBSERVATIONS, n - MIN_PIECE_OBSERVATIONS + 1)

    left = cumulative[cuts - 1]
    right = cumulative[-1] - left
    left = left / np.clip(left.sum(axis=1, keepdims=True), 1e-12, None)
    right = right / np.clip(right.sum(axis=1, keepdims=True), 1e-12, None)
    # Bhattacharyya distance of the two sides, as `histogram_distance`.
    distance = 1.0 - np.sqrt(np.clip(left, 0.0, None) * np.clip(right, 0.0, None)).sum(axis=1)

    best = int(np.argmax(distance))
    if distance[best] <= MAX_COLOR_DISTANCE:
        return [observations]
    cut = int(cuts[best])
    return (_split_identity_switches(observations[:cut])
            + _split_identity_switches(observations[cut:]))


# ── Linking ──────────────────────────────────────────────────────────────────

def _link_tracklets(
    tracklets: list[Tracklet],
    max_speed_ms: float,
    progress: ProgressCallback | None = None,
) -> list[_Cluster]:
    """Greedily merge the cheapest linkable pair until none is cheap enough.

    Pairwise costs are computed once and kept in a heap: a merge only adds
    the pairs of the new cluster, and entries of the two it consumed go
    stale. Checks run cheapest first — time span, coexistence, then cost.

    Coexistence is re-judged on the merged cluster rather than inherited from
    its parts. Inheriting it looked sound — if A was seen at the same instant
    as C, so is anything containing A — but it also inherits mistakes: one
    small piece of the wrong person linked into a player's cluster then kept
    that player's own remaining pieces out for good. Re-judged on the whole,
    the stray piece is outweighed; two genuinely different people stay apart
    because their density of shared frames does not dilute.
    """
    ordered = sorted(tracklets, key=lambda t: t.start_s)
    clusters: dict[int, _Cluster] = {
        index: _Cluster(
            observations=list(t.observations),
            sources=[t.id],
            segments=[_Segment.of(t.observations)],
        )
        for index, t in enumerate(ordered)
    }
    next_id = len(clusters)
    period_s = _sample_period(tracklets)

    # Every linkable pair, cheapest first. A pair's cost never changes while
    # both clusters exist, and ids are never reused, so an entry is stale
    # exactly when one of its clusters has been merged away.
    heap: list[tuple[float, int, int]] = []

    def add_pair(a: int, b: int) -> None:
        first, second = clusters[a], clusters[b]
        # No junction within MAX_LINK_GAP_S is possible when the spans are
        # further apart than that; nor can they coexist.
        if (first.start_s - second.end_s > MAX_LINK_GAP_S
                or second.start_s - first.end_s > MAX_LINK_GAP_S):
            return
        if _coexist(first, second, period_s):
            return
        cost = _disjoint_link_cost(first, second, max_speed_ms, period_s)
        if cost is not None:
            heapq.heappush(heap, (cost, min(a, b), max(a, b)))

    ids = list(clusters)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if clusters[b].start_s - clusters[a].end_s > MAX_LINK_GAP_S:
                break       # sorted by start: every later one is further still
            add_pair(a, b)

    total_candidates = len(clusters)
    while heap:
        best, first, second = heapq.heappop(heap)
        if first not in clusters or second not in clusters:
            continue
        if best > LINK_COST_THRESHOLD:
            break

        a, b = clusters.pop(first), clusters.pop(second)
        merged_id = next_id
        next_id += 1
        clusters[merged_id] = _Cluster(
            observations=_without_shared_frames(a, b),
            sources=a.sources + b.sources,
            segments=list(heapq.merge(a.segments, b.segments, key=lambda seg: seg.start_s)),
        )

        for other in list(clusters):
            if other != merged_id:
                add_pair(other, merged_id)

        if progress is not None:
            progress(total_candidates - len(clusters), total_candidates)

    return list(clusters.values())


def _link_cost(a: _Cluster, b: _Cluster, max_speed_ms: float) -> float | None:
    """Cost of declaring two clusters the same player. None = not linkable.

    Symmetric. The two clusters' pieces are laid out on one timeline; every
    point where the timeline passes from one cluster to the other is a
    junction, and each junction must be something one person could do.
    """
    period_s = _sample_period_of(a, b)
    if _coexist(a, b, period_s):
        return None     # seen at the same time, so they are different people
    return _disjoint_link_cost(a, b, max_speed_ms, period_s)


def _disjoint_link_cost(
    a: _Cluster, b: _Cluster, max_speed_ms: float, period_s: float,
) -> float | None:
    """`_link_cost` for two clusters already known not to coexist."""
    color = _color_distance(a, b)
    if color > MAX_COLOR_DISTANCE:
        return None     # different kit, therefore a different person

    spatial: list[float] = []
    unreachable = 0
    for gap, travel_s, distance in _junctions(a.segments, b.segments, period_s):
        if gap > MAX_LINK_GAP_S:
            continue    # too far apart in time to say anything either way
        if distance <= max_speed_ms * travel_s + 1.0:
            spatial.append(min(1.0, distance / SPATIAL_SCALE_M))
        else:
            unreachable += 1

    if not spatial and not unreachable:
        return None     # no junction close enough in time to judge

    if unreachable:
        # Physically unreachable, but that is exactly what a changeover looks
        # like when the walk itself was not tracked. Allow it only on a strong
        # appearance match, charge a penalty so a reachable link always wins,
        # and only once: a changeover is one crossing, and a piece that jumps
        # in and jumps back out is somebody else.
        if unreachable > 1 or color > STRONG_COLOR_MATCH:
            return None
        return color + 0.30

    return 0.6 * color + 0.4 * float(np.mean(spatial))


def _color_distance(a: _Cluster, b: _Cluster) -> float:
    """`histogram_distance` of the two signatures, as one dot product."""
    return float(np.clip(1.0 - float(a.sqrt_signature() @ b.sqrt_signature()), 0.0, 1.0))


def _coexist(a: _Cluster, b: _Cluster, period_s: float) -> bool:
    """Whether two clusters are two different people: seen in the same frame,
    repeatedly, while both are on court. See COEXIST_DENSITY."""
    overlap_s = _overlap_s(a.segments, b.segments)
    if overlap_s <= 0.0:
        return False
    shared = len(np.intersect1d(a.frames(), b.frames(), assume_unique=True))
    if shared < COEXIST_MIN_SHARED:
        return False
    # +1: an overlap of n periods spans n + 1 samples.
    samples = overlap_s / period_s + 1.0
    return shared >= COEXIST_DENSITY * samples


def _overlap_s(a: list[_Segment], b: list[_Segment]) -> float:
    """Seconds during which pieces of both clusters are on court. Both lists
    are sorted by start, and pieces of one cluster do not overlap."""
    total, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i].end_s, b[j].end_s) - max(a[i].start_s, b[j].start_s))
        if a[i].end_s < b[j].end_s:
            i += 1
        else:
            j += 1
    return total


def _without_shared_frames(a: _Cluster, b: _Cluster) -> list[Observation]:
    """Both clusters' observations in time order, minus the frames where both
    have one. Two linked clusters are one person, so a shared frame means
    one of the two detections belongs to somebody else — and nothing says
    which. Dropping both loses a sample; keeping the wrong one would put
    another player's position into this player's numbers."""
    shared = set(np.intersect1d(a.frames(), b.frames(), assume_unique=True).tolist())
    merged = heapq.merge(a.observations, b.observations, key=lambda o: o.timestamp_s)
    if not shared:
        return list(merged)
    return [o for o in merged if o.frame_index not in shared]


def _sample_period(tracklets: list[Tracklet]) -> float:
    """Sampling period of the run, from the tracklets themselves."""
    steps = [
        b.timestamp_s - a.timestamp_s
        for t in tracklets
        for a, b in zip(t.observations, t.observations[1:])
    ]
    positive = [s for s in steps if s > 1e-6]
    return float(np.median(positive)) if positive else 0.2


def _sample_period_of(a: _Cluster, b: _Cluster) -> float:
    return _sample_period([
        Tracklet(id=0, observations=a.observations),
        Tracklet(id=1, observations=b.observations),
    ])


def _junctions(
    a: list[_Segment], b: list[_Segment], period_s: float,
) -> list[tuple[float, float, float]]:
    """(gap, travel time, distance) at each hand-over between the two clusters.

    Pieces that overlap by more than one sample are not a hand-over — the
    later one starts before the earlier one ends — and are skipped: whether
    overlapping pieces are two people is `_coexist`'s judgement, made on the
    whole cluster. Judging them piece by piece, as a walk or by where both
    were at the same moment, let the few instants in which one track
    followed somebody else veto merges that dozens of hand-overs supported;
    measured on simulated matches it only lost correct attributions.

    Consecutive tracklets are the common case (a track dies on one sample and
    is reborn on the next), so the gap is clamped at zero rather than going
    negative inside the tolerated overlap.
    """
    timeline = sorted(
        [(seg, 0) for seg in a] + [(seg, 1) for seg in b],
        key=lambda item: item[0].start_s,
    )
    out: list[tuple[float, float, float]] = []
    for index, (before, owner_before) in enumerate(timeline):
        # The hand-over out of this piece is to the first piece that starts
        # once it has ended, stepping over any that overlap it: two tracks
        # briefly alternating on one player must not hide the piece that
        # really follows.
        for following in range(index + 1, len(timeline)):
            after, owner_after = timeline[following]
            if after.start_s < before.end_s - period_s - 1e-6:
                continue
            if owner_after != owner_before:
                gap = max(after.start_s - before.end_s, 0.0)
                # Movement is judged between the robust ends, over the time
                # that separates them — a little longer than the raw gap.
                travel_s = max(after.head_s - before.tail_s, gap)
                distance = float(np.hypot(after.head_xy[0] - before.tail_xy[0],
                                          after.head_xy[1] - before.tail_xy[1]))
                out.append((gap, travel_s, distance))
            break
    return out


# ── Teams and stable ids ─────────────────────────────────────────────────────

def _assign_teams_and_ids(clusters: list[_Cluster], warnings: list[str]) -> list[PlayerTrack]:
    if not clusters:
        return []

    ranked = sorted(clusters, key=lambda c: c.far_fraction())
    far_team = set(id(c) for c in ranked[-2:]) if len(ranked) >= 4 else set()

    if len(ranked) >= 4:
        near_max = ranked[1].far_fraction()
        far_min = ranked[2].far_fraction()
        if far_min - near_max < 0.15:
            warnings.append(
                "Separazione fra le due coppie poco netta: la squadra assegnata "
                "a ciascun giocatore potrebbe non essere corretta."
            )
    else:
        # Not enough clusters for a 2/2 split — fall back to the halfway line.
        far_team = set(id(c) for c in ranked if c.far_fraction() > 0.5)

    near = [c for c in clusters if id(c) not in far_team]
    far = [c for c in clusters if id(c) in far_team]
    near.sort(key=lambda c: c.mean_x())
    far.sort(key=lambda c: c.mean_x())

    players: list[PlayerTrack] = []
    for team, group in ((0, near), (1, far)):
        for cluster in group:
            # Pieces of one tracklet carry its id; list each tracklet once.
            sources = list(dict.fromkeys(cluster.sources))
            players.append(PlayerTrack(len(players), team, cluster.observations, sources))
    return players


def _count_side_changes(players: list[PlayerTrack], bin_seconds: float = 30.0) -> int:
    """Median number of half-court switches across the four players.

    Per-player flips are noisy (a player can stray past the net line), so the
    median across players is used: a real changeover moves everybody at once.
    """
    if not players:
        return 0

    flips: list[int] = []
    for player in players:
        times = np.array([o.timestamp_s for o in player.observations])
        sides = np.array(
            [1 if o.foot_court[1] > COURT_LENGTH_M / 2.0 else 0 for o in player.observations]
        )
        if len(times) < 2:
            flips.append(0)
            continue
        bins = ((times - times[0]) // bin_seconds).astype(int)
        majority = [
            int(round(float(sides[bins == b].mean()))) for b in np.unique(bins) if np.any(bins == b)
        ]
        flips.append(sum(1 for a, b in zip(majority, majority[1:]) if a != b))

    return int(np.median(flips)) if flips else 0
