"""Each player's path on the court, with the detector's jitter filtered out.

Why
---
Distance, speed and the rally boundaries used to be computed from the raw
foot positions, step by step. The bottom of a detector's box moves by a few
pixels from one frame to the next even when the player stands still, and a
few pixels are tens of centimetres on the near half and the better part of a
metre on the far one. Summed five times a second, that jitter is distance: a
player standing still for a minute "ran" 30 to 220 m, depending on the
camera and the half (measured on a simulated court, 2-3 px of jitter).

How
---
A constant-velocity Kalman filter per player, run forwards and then smoothed
backwards (Rauch-Tung-Striebel), on court metres. What makes it fit this
problem is the measurement noise: it is the pixel jitter carried through the
homography at the very spot of the detection, so a far-court detection —
where one pixel is many centimetres, mostly in depth — is trusted less, and
in the direction it is less precise. The path is then where the evidence and
plausible human motion agree.

Short gaps (a player hidden behind another for a second) are bridged by the
filter's prediction, so the distance run during them is no longer lost. A
jump no one could run is an outlier and is left out; several in a row mean
the track has moved to somebody else, and the path starts again from there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.court import CourtCalibration
from app.ml.tracking import Observation

# Jitter of the box's bottom edge, in pixels, as a share of the box height
# (bigger boxes jitter by more pixels), with a floor for small boxes.
JITTER_X = 0.015
JITTER_Y = 0.02
JITTER_MIN_PX = 1.5
# Spectral density of the random acceleration, (m/s^2)^2 per Hz: how much
# a player may change speed between two samples. Fitted on simulated rallies
# (tests/test_smoothing.py): lower lags the turns, higher lets jitter through.
ACCELERATION_PSD = 2.0
# A turn sharper than the motion model expects — a stop and back, a lunge —
# is retried with this much more acceleration before the detection is called
# an outlier: the filter follows the turn instead of losing the player.
MANEUVER_FACTOR = 30.0
# Gaps bridged by the prediction; a longer one starts a new path.
MAX_BRIDGE_S = 1.5
# An innovation beyond this Mahalanobis distance (chi-square, 2 dof, ~1 in
# 10 000) is not the same player moving.
OUTLIER_CHI2 = 18.4
# This many outliers in a row: the detections are someone else's from now
# on, or the player really is there — start a new path at them.
RESTART_AFTER = 3
# A step is never faster than this, bridged or not.
MAX_SPEED_MS = 8.0
# Below this filtered speed a player is standing: what is left of the jitter
# after filtering wanders at 0.1-0.2 m/s and would still add 10 m a minute.
# Walking between points is 0.8 m/s and more.
MIN_MOVING_MS = 0.3


@dataclass
class SmoothedPath:
    """One player's filtered path, at the times of their observations."""
    times: np.ndarray        # (n,)
    positions: np.ndarray    # (n, 2) metres
    velocities: np.ndarray   # (n, 2) m/s
    segment: np.ndarray      # (n,) path index: no distance between two paths
    outliers: int            # observations left out

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)

    def steps(self) -> tuple[np.ndarray, np.ndarray]:
        """(length, end time) of every step within a path; a step during
        which the player is standing (MIN_MOVING_MS) has no length."""
        if len(self.times) < 2:
            return np.zeros(0), np.zeros(0)
        same = self.segment[1:] == self.segment[:-1]
        lengths = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        speeds = self.speeds
        moving = np.maximum(speeds[1:], speeds[:-1]) >= MIN_MOVING_MS
        return np.where(moving, lengths, 0.0)[same], self.times[1:][same]


def smooth_path(observations: list[Observation], calibration: CourtCalibration) -> SmoothedPath:
    """Filter one player's observations (one per frame, in time order)."""
    n = len(observations)
    if n == 0:
        empty = np.zeros((0, 2))
        return SmoothedPath(np.zeros(0), empty, empty, np.zeros(0, dtype=int), 0)

    times = np.array([o.timestamp_s for o in observations], dtype=np.float64)
    measured = np.array([o.foot_court for o in observations], dtype=np.float64)
    noise = _measurement_noise(observations, calibration)

    keep = np.ones(n, dtype=bool)
    segment = np.zeros(n, dtype=int)
    # Forward pass, keeping what the backward pass needs.
    means = np.zeros((n, 4))
    covs = np.zeros((n, 4, 4))
    predicted_means = np.zeros((n, 4))
    predicted_covs = np.zeros((n, 4, 4))
    transitions = np.zeros((n, 4, 4))

    current = 0
    rejected_in_row = 0
    x, p = _start(measured[0], noise[0])
    means[0], covs[0] = x, p
    predicted_means[0], predicted_covs[0] = x, p
    transitions[0] = np.eye(4)
    last = 0                       # last index used by the filter
    for i in range(1, n):
        dt = times[i] - times[last]
        if dt > MAX_BRIDGE_S or dt <= 0:
            current += 1
            x, p = _start(measured[i], noise[i])
            transitions[i] = np.eye(4)
            predicted_means[i], predicted_covs[i] = x, p
            means[i], covs[i], segment[i], last, rejected_in_row = x, p, current, i, 0
            continue

        f = _transition(dt)
        x_pred = f @ x
        innovation = measured[i] - x_pred[:2]
        for factor in (1.0, MANEUVER_FACTOR):
            p_pred = f @ p @ f.T + _process_noise(dt) * factor
            s = p_pred[:2, :2] + noise[i]
            distance = float(innovation @ np.linalg.solve(s, innovation))
            if distance <= OUTLIER_CHI2:
                break
        jump = float(np.linalg.norm(measured[i] - x[:2])) / dt
        if distance > OUTLIER_CHI2 or jump > MAX_SPEED_MS + 3.0 * np.sqrt(np.trace(noise[i])) / dt:
            rejected_in_row += 1
            if rejected_in_row < RESTART_AFTER:
                keep[i] = False
                continue
            # Several in a row: start over from here, and give back the ones
            # just set aside to the new path.
            current += 1
            first = i - rejected_in_row + 1
            x, p = _start(measured[first], noise[first])
            for j in range(first, i + 1):
                keep[j] = True
                segment[j] = current
            means[first], covs[first] = x, p
            predicted_means[first], predicted_covs[first] = x, p
            transitions[first] = np.eye(4)
            last, rejected_in_row = first, 0
            for j in range(first + 1, i + 1):
                x, p = _update_step(j, last, x, p, times, measured, noise,
                                    means, covs, predicted_means, predicted_covs, transitions)
                segment[j], last = current, j
            continue

        rejected_in_row = 0
        k = p_pred[:, :2] @ np.linalg.inv(s)
        x = x_pred + k @ innovation
        p = (np.eye(4) - k[:, :2] @ np.eye(2, 4)) @ p_pred
        means[i], covs[i] = x, p
        predicted_means[i], predicted_covs[i] = x_pred, p_pred
        transitions[i] = f
        segment[i], last = current, i

    # Backward pass (Rauch-Tung-Striebel), path by path, kept samples only.
    kept = np.flatnonzero(keep)
    smoothed = means.copy()
    for seg in np.unique(segment[kept]):
        idx = kept[segment[kept] == seg]
        for a, b in zip(idx[-2::-1], idx[:0:-1]):
            gain = covs[a] @ transitions[b].T @ np.linalg.pinv(predicted_covs[b])
            smoothed[a] = means[a] + gain @ (smoothed[b] - predicted_means[b])

    return SmoothedPath(
        times=times[kept],
        positions=smoothed[kept, :2],
        velocities=smoothed[kept, 2:],
        segment=segment[kept],
        outliers=int(n - len(kept)),
    )


def _update_step(i, last, x, p, times, measured, noise, means, covs, predicted_means, predicted_covs, transitions):
    dt = times[i] - times[last]
    f = _transition(dt)
    x_pred = f @ x
    p_pred = f @ p @ f.T + _process_noise(dt)
    s = p_pred[:2, :2] + noise[i]
    k = p_pred[:, :2] @ np.linalg.inv(s)
    x = x_pred + k @ (measured[i] - x_pred[:2])
    p = (np.eye(4) - k[:, :2] @ np.eye(2, 4)) @ p_pred
    means[i], covs[i] = x, p
    predicted_means[i], predicted_covs[i] = x_pred, p_pred
    transitions[i] = f
    return x, p


def _start(position: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros((4, 4))
    p[:2, :2] = noise
    p[2:, 2:] = np.eye(2) * 4.0          # speed unknown: about +-2 m/s
    return np.array([position[0], position[1], 0.0, 0.0]), p


def _transition(dt: float) -> np.ndarray:
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    return f


def _process_noise(dt: float) -> np.ndarray:
    q = ACCELERATION_PSD
    block = np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]]) * q
    out = np.zeros((4, 4))
    out[np.ix_([0, 2], [0, 2])] = block
    out[np.ix_([1, 3], [1, 3])] = block
    return out


def _measurement_noise(observations: list[Observation], calibration: CourtCalibration) -> np.ndarray:
    """Covariance of each foot position in metres: the pixel jitter of its
    box, through the local Jacobian of the pixels-to-metres mapping at the
    foot (lens correction included)."""
    feet = np.array([o.foot_px for o in observations], dtype=np.float64)
    base = calibration.floor_points_to_court(feet).astype(np.float64)
    step = 1.0
    dx = (calibration.floor_points_to_court(feet + [step, 0.0]) - base) / step
    dy = (calibration.floor_points_to_court(feet + [0.0, step]) - base) / step
    out = np.zeros((len(observations), 2, 2))
    for i, obs in enumerate(observations):
        x1, y1, x2, y2 = obs.bbox
        height = max(y2 - y1, 1.0)
        sx = max(JITTER_X * height, JITTER_MIN_PX)
        sy = max(JITTER_Y * height, JITTER_MIN_PX)
        jac = np.column_stack([dx[i], dy[i]])
        out[i] = jac @ np.diag([sx * sx, sy * sy]) @ jac.T + np.eye(2) * 1e-4
    return out
