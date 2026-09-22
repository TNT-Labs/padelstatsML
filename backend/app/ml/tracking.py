"""Multi-player tracking in court space.

Design notes
------------
The previous version delegated tracking to `ultralytics.YOLO.track()`, which
associates boxes by IoU in *pixel* space. That is a poor fit here for two
reasons:

  1. We sample at ~5 Hz, not 30 Hz. A player running at 4 m/s moves ~0.8 m
     between samples, which on the far baseline is most of a bounding box —
     IoU is frequently zero between consecutive samples, so IoU-based
     association breaks exactly when the player is moving, i.e. when it
     matters.
  2. Pixel distance is not a physical quantity. A metre near the camera is
     four times more pixels than a metre on the far baseline, so a single
     pixel gate is either too loose in front or too tight at the back.

Associating in *metres* on the court floor fixes both: the gate becomes
`max_speed * dt`, a real physical constraint, uniform across the court.
A coarse HSV torso histogram is carried along as a second cue, which is what
later lets `identity` re-link a player across an occlusion or a changeover.

Tracks are deliberately allowed to die and be reborn. Producing many clean
tracklets and linking them afterwards with appearance evidence is far more
robust than forcing one tracker to survive every occlusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.ml.court import COURT_LENGTH_M, COURT_WIDTH_M, CourtCalibration
from app.ml.detect import Detection

# How far outside the painted court a detection may stand and still be a
# player. The glass walls are on the lines, so 1.5 m of slack covers players
# leaning into the wall plus homography error, without admitting spectators.
COURT_MARGIN_M = 1.5

# Colour histogram: 8 hue bins x 4 saturation bins. Coarse on purpose —
# padel kit is mostly flat colour and a fine histogram would key on lighting.
_HUE_BINS, _SAT_BINS = 8, 4
COLOR_DIM = _HUE_BINS * _SAT_BINS


@dataclass(frozen=True)
class Observation:
    frame_index: int
    timestamp_s: float
    bbox: tuple[float, float, float, float]
    foot_px: tuple[float, float]
    foot_court: tuple[float, float]
    confidence: float
    color: np.ndarray            # normalised histogram, shape (COLOR_DIM,)


@dataclass
class Tracklet:
    """A contiguous run of observations believed to be one player."""

    id: int
    observations: list[Observation] = field(default_factory=list)

    @property
    def start_s(self) -> float:
        return self.observations[0].timestamp_s

    @property
    def end_s(self) -> float:
        return self.observations[-1].timestamp_s

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def __len__(self) -> int:
        return len(self.observations)

    def color_signature(self) -> np.ndarray:
        """Confidence-weighted mean histogram over the tracklet."""
        if not self.observations:
            return np.zeros(COLOR_DIM, dtype=np.float32)
        weights = np.array([o.confidence for o in self.observations], dtype=np.float32)
        stack = np.stack([o.color for o in self.observations])
        total = float(weights.sum())
        if total <= 0:
            return stack.mean(axis=0)
        sig = (stack * weights[:, None]).sum(axis=0) / total
        norm = float(sig.sum())
        return sig / norm if norm > 0 else sig

    def mean_position(self) -> tuple[float, float]:
        pts = np.array([o.foot_court for o in self.observations], dtype=np.float32)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    def side_fraction_far(self) -> float:
        """Fraction of the time spent beyond the net (Y > 10 m)."""
        ys = np.array([o.foot_court[1] for o in self.observations], dtype=np.float32)
        return float((ys > COURT_LENGTH_M / 2.0).mean())


@dataclass
class _ActiveTrack:
    id: int
    observations: list[Observation]
    last_seen_s: float
    misses: int = 0

    def predict(self, now_s: float) -> np.ndarray:
        """Constant-velocity extrapolation of the court position."""
        last = self.observations[-1]
        pos = np.array(last.foot_court, dtype=np.float32)
        if len(self.observations) < 2:
            return pos
        prev = self.observations[-2]
        dt = last.timestamp_s - prev.timestamp_s
        if dt <= 1e-3:
            return pos
        vel = (np.array(last.foot_court) - np.array(prev.foot_court)) / dt
        # Clamp the extrapolation: a stale velocity must not fling the gate
        # across the court during a long occlusion.
        horizon = min(now_s - last.timestamp_s, 0.4)
        return pos + vel.astype(np.float32) * horizon

    def color_signature(self) -> np.ndarray:
        recent = self.observations[-12:]
        stack = np.stack([o.color for o in recent])
        sig = stack.mean(axis=0)
        norm = float(sig.sum())
        return sig / norm if norm > 0 else sig


class CourtTracker:
    """Greedy-optimal assignment tracker operating on court metres."""

    def __init__(
        self,
        calibration: CourtCalibration,
        max_speed_ms: float = 8.0,
        max_age_s: float = 1.2,
        min_observations: int = 3,
        max_players_per_frame: int = 6,
        color_weight: float = 0.35,
    ) -> None:
        self.calibration = calibration
        self.max_speed_ms = max_speed_ms
        self.max_age_s = max_age_s
        self.min_observations = min_observations
        self.max_players_per_frame = max_players_per_frame
        self.color_weight = color_weight

        self._active: list[_ActiveTrack] = []
        self._finished: list[Tracklet] = []
        self._next_id = 0
        self._last_ts: float | None = None

        # Diagnostics surfaced in data_quality
        self.frames_seen = 0
        self.detections_kept = 0
        self.detections_rejected_off_court = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[tuple[int, Observation]]:
        """Feed one sampled frame. Returns [(track_id, observation), ...]."""
        self.frames_seen += 1
        dt = timestamp_s - self._last_ts if self._last_ts is not None else 0.0
        self._last_ts = timestamp_s

        observations = self._to_observations(frame_index, timestamp_s, detections, frame)
        self._retire_stale(timestamp_s)

        if not observations:
            for track in self._active:
                track.misses += 1
            return []

        if not self._active:
            return [(self._spawn(obs), obs) for obs in observations]

        sample_period = dt if dt > 1e-3 else 0.2
        cost, feasible = self._build_cost(observations, timestamp_s, sample_period)

        rows, cols = linear_sum_assignment(cost)
        assigned_obs: set[int] = set()
        result: list[tuple[int, Observation]] = []

        for r, c in zip(rows, cols):
            if not feasible[r, c]:
                continue
            track = self._active[r]
            obs = observations[c]
            track.observations.append(obs)
            track.last_seen_s = timestamp_s
            track.misses = 0
            assigned_obs.add(c)
            result.append((track.id, obs))

        # Every track that took no detection this round ages by one sample.
        # Retirement itself is time-based (`max_age_s`), so this counter is a
        # diagnostic rather than a lifecycle input.
        for track in self._active:
            if track.last_seen_s != timestamp_s:
                track.misses += 1

        for idx, obs in enumerate(observations):
            if idx not in assigned_obs:
                result.append((self._spawn(obs), obs))

        return result

    def finish(self) -> list[Tracklet]:
        """Close all open tracks and return every tracklet worth keeping."""
        for track in list(self._active):
            self._close(track)
        self._active.clear()
        return [t for t in self._finished if len(t) >= self.min_observations]

    # ── Internals ────────────────────────────────────────────────────────────

    def _gate_distance(self, elapsed_s: float, sample_period_s: float) -> float:
        """Maximum plausible court displacement since a track was last seen.

        The window is measured per track, not per frame. A detector that
        misses a player for a few samples leaves that track stale, and during
        the gap the player keeps running: gating it on the sampling interval
        instead of on the track's own age rejects the perfectly valid
        re-association, kills the track and starts a new one. That is how four
        players become hundreds of tracklets.
        """
        step = max(elapsed_s, sample_period_s, 1e-3)
        return self.max_speed_ms * step + 0.6   # +0.6 m for homography noise

    def _build_cost(
        self,
        observations: list[Observation],
        now_s: float,
        sample_period_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_tracks, n_obs = len(self._active), len(observations)
        cost = np.full((n_tracks, n_obs), 1e6, dtype=np.float64)
        feasible = np.zeros((n_tracks, n_obs), dtype=bool)

        obs_pos = np.array([o.foot_court for o in observations], dtype=np.float32)
        obs_col = np.stack([o.color for o in observations])

        # Feasibility widens with a track's own age, but the spatial cost is
        # normalised by the one-step gate for every track: otherwise a stale
        # track, with its larger gate, would score any match as "close" and
        # outbid a fresh track that is genuinely next to the detection.
        cost_scale = self._gate_distance(sample_period_s, sample_period_s)

        for r, track in enumerate(self._active):
            predicted = track.predict(now_s)
            dists = np.linalg.norm(obs_pos - predicted[None, :], axis=1)
            track_col = track.color_signature()
            color_d = np.array(
                [histogram_distance(track_col, obs_col[i]) for i in range(n_obs)],
                dtype=np.float64,
            )
            gate = self._gate_distance(now_s - track.last_seen_s, sample_period_s)
            ok = dists <= gate
            spatial = np.clip(dists / cost_scale, 0.0, 1.0)
            blended = (1.0 - self.color_weight) * spatial + self.color_weight * color_d
            cost[r, ok] = blended[ok]
            feasible[r] = ok

        return cost, feasible

    def _to_observations(
        self,
        frame_index: int,
        timestamp_s: float,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[Observation]:
        if not detections:
            return []

        feet = np.array([d.foot_px for d in detections], dtype=np.float32)
        court = self.calibration.floor_points_to_court(feet)

        candidates: list[Observation] = []
        for det, pos in zip(detections, court):
            if not np.isfinite(pos).all() or not _inside_court(pos):
                self.detections_rejected_off_court += 1
                continue
            candidates.append(
                Observation(
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    bbox=det.bbox,
                    foot_px=det.foot_px,
                    foot_court=(float(pos[0]), float(pos[1])),
                    confidence=det.confidence,
                    color=_torso_histogram(frame, det.bbox),
                )
            )

        candidates.sort(key=lambda o: -o.confidence)
        kept = candidates[: self.max_players_per_frame]
        self.detections_kept += len(kept)
        return kept

    def _spawn(self, obs: Observation) -> int:
        track = _ActiveTrack(id=self._next_id, observations=[obs], last_seen_s=obs.timestamp_s)
        self._next_id += 1
        self._active.append(track)
        return track.id

    def _retire_stale(self, now_s: float) -> None:
        still_active: list[_ActiveTrack] = []
        for track in self._active:
            if now_s - track.last_seen_s > self.max_age_s:
                self._close(track)
            else:
                still_active.append(track)
        self._active = still_active

    def _close(self, track: _ActiveTrack) -> None:
        self._finished.append(Tracklet(id=track.id, observations=list(track.observations)))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _inside_court(position_m: np.ndarray) -> bool:
    x, y = float(position_m[0]), float(position_m[1])
    return (
        -COURT_MARGIN_M <= x <= COURT_WIDTH_M + COURT_MARGIN_M
        and -COURT_MARGIN_M <= y <= COURT_LENGTH_M + COURT_MARGIN_M
    )


def _torso_histogram(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Coarse HSV histogram of the player's shirt region.

    The window is the upper-middle of the bounding box, which is the shirt for
    a standing player and avoids the court surface leaking in at the edges.
    Very dark and very desaturated pixels are masked out so that shadow and
    white lines do not dominate the signature.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    cx1 = int(np.clip(x1 + 0.25 * (x2 - x1), 0, w - 1))
    cx2 = int(np.clip(x1 + 0.75 * (x2 - x1), 1, w))
    cy1 = int(np.clip(y1 + 0.15 * (y2 - y1), 0, h - 1))
    cy2 = int(np.clip(y1 + 0.55 * (y2 - y1), 1, h))

    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return np.full(COLOR_DIM, 1.0 / COLOR_DIM, dtype=np.float32)

    patch = frame[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 245))

    hist = cv2.calcHist([hsv], [0, 1], mask, [_HUE_BINS, _SAT_BINS], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 0:
        return np.full(COLOR_DIM, 1.0 / COLOR_DIM, dtype=np.float32)
    return (hist / total).flatten().astype(np.float32)


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance in [0, 1]; 0 = identical."""
    coefficient = float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
    return float(np.clip(1.0 - coefficient, 0.0, 1.0))
