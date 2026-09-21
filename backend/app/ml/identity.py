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
1. Link tracklets that cannot overlap in time into clusters, using shirt
   colour plus a physical reachability test. This is the step that repairs
   occlusions.
2. Keep the four longest clusters; they are the four players.
3. Assign teams from which half of the court each cluster lives in. Padel
   teams hold their half for a whole game, so the side histogram is a strong
   and cheap team signal.
4. Number the players deterministically so the same person keeps the same id
   between re-runs: near team left-to-right = 0,1; far team = 2,3.

Known limitation, reported rather than hidden: players swap ends at
changeovers. A swap that happens while both players are untracked can split
one person into two clusters. `side_changes` and the warnings list expose
this instead of silently degrading the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ml.court import COURT_LENGTH_M
from app.ml.tracking import Observation, Tracklet, histogram_distance

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
# How much genuine temporal overlap is tolerated before two tracklets are
# declared to be two different people. A tracker handing over between ids can
# briefly hold both, so one or two samples of overlap is normal; anything more
# means they coexist, and a person cannot coexist with themselves.
OVERLAP_TOLERANCE_S = 0.35


@dataclass
class PlayerTrack:
    player_id: int
    team: int                     # 0 = near half (Y < 10 m), 1 = far half
    observations: list[Observation]
    source_tracklets: list[int] = field(default_factory=list)

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


@dataclass
class _Cluster:
    observations: list[Observation]
    sources: list[int]

    @property
    def start_s(self) -> float:
        return self.observations[0].timestamp_s

    @property
    def end_s(self) -> float:
        return self.observations[-1].timestamp_s

    def signature(self) -> np.ndarray:
        weights = np.array([o.confidence for o in self.observations], dtype=np.float32)
        stack = np.stack([o.color for o in self.observations])
        total = float(weights.sum())
        sig = stack.mean(axis=0) if total <= 0 else (stack * weights[:, None]).sum(axis=0) / total
        norm = float(sig.sum())
        return sig / norm if norm > 0 else sig

    def mean_x(self) -> float:
        return float(np.mean([o.foot_court[0] for o in self.observations]))

    def far_fraction(self) -> float:
        ys = np.array([o.foot_court[1] for o in self.observations], dtype=np.float32)
        return float((ys > COURT_LENGTH_M / 2.0).mean())


def resolve_players(
    tracklets: list[Tracklet],
    max_speed_ms: float = 8.0,
) -> IdentityResult:
    """Link tracklets into four players and assign teams and stable ids."""
    warnings: list[str] = []
    usable = [t for t in tracklets if len(t) >= 3]
    if not usable:
        return IdentityResult([], ["Nessun giocatore rilevato nel video."], 0, 0, 0)

    clusters = _link_tracklets(usable, max_speed_ms)
    clusters.sort(key=lambda c: -len(c.observations))

    total_obs = sum(len(c.observations) for c in clusters)
    selected = clusters[:N_PLAYERS]
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


# ── Linking ──────────────────────────────────────────────────────────────────

def _link_tracklets(tracklets: list[Tracklet], max_speed_ms: float) -> list[_Cluster]:
    clusters = [
        _Cluster(observations=list(t.observations), sources=[t.id])
        for t in sorted(tracklets, key=lambda t: t.start_s)
    ]

    while True:
        best: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if i == j:
                    continue
                cost = _link_cost(clusters[i], clusters[j], max_speed_ms)
                if cost is None:
                    continue
                if best is None or cost < best[0]:
                    best = (cost, i, j)

        if best is None or best[0] > LINK_COST_THRESHOLD:
            break

        _, i, j = best
        merged = _Cluster(
            observations=sorted(
                clusters[i].observations + clusters[j].observations,
                key=lambda o: o.timestamp_s,
            ),
            sources=clusters[i].sources + clusters[j].sources,
        )
        for index in sorted((i, j), reverse=True):
            clusters.pop(index)
        clusters.append(merged)

    return clusters


def _link_cost(earlier: _Cluster, later: _Cluster, max_speed_ms: float) -> float | None:
    """Cost of declaring two clusters the same player. None = not linkable."""
    if earlier.end_s > later.start_s + OVERLAP_TOLERANCE_S:
        return None     # they coexist in time, so they are different people
    # Consecutive tracklets are the common case (a track dies on one sample
    # and is reborn on the next), so the gap is clamped at zero rather than
    # going negative inside the tolerated overlap.
    gap = max(later.start_s - earlier.end_s, 0.0)
    if gap > MAX_LINK_GAP_S:
        return None

    color = histogram_distance(earlier.signature(), later.signature())
    if color > MAX_COLOR_DISTANCE:
        return None     # different kit, therefore a different person

    tail = np.array(earlier.observations[-1].foot_court, dtype=np.float32)
    head = np.array(later.observations[0].foot_court, dtype=np.float32)
    distance = float(np.linalg.norm(head - tail))
    reachable = max_speed_ms * gap + 1.0

    if distance <= reachable:
        return 0.6 * color + 0.4 * min(1.0, distance / SPATIAL_SCALE_M)

    # Physically unreachable, but that is exactly what a changeover looks
    # like when the walk itself was not tracked. Allow it only on a strong
    # appearance match, and charge a penalty so a reachable link always wins.
    if color <= STRONG_COLOR_MATCH:
        return color + 0.30
    return None


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
    for cluster in near:
        players.append(PlayerTrack(len(players), 0, cluster.observations, cluster.sources))
    for cluster in far:
        players.append(PlayerTrack(len(players), 1, cluster.observations, cluster.sources))
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
