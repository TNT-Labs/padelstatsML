"""Re-run the tracker on a stored run, without the detector.

`tracks.jsonl` holds every observation the tracker was fed — projected to
court metres, colour histogram included, already filtered and capped exactly
as the live pass filtered them. Association needs nothing else, so it can be
repeated on a real match in seconds instead of repeating an hour of inference.

That makes a tracker change measurable before it ships: replay a match that
was analysed with the previous code and compare the stored tracklets with the
replayed ones (`scripts/retrack.py`).
"""
from __future__ import annotations

from collections import defaultdict

from app.ml.artifacts import LoadedArtifacts, read_observations
from app.ml.tracking import CourtTracker, Observation, Tracklet


def replay_tracking(
    artifacts: LoadedArtifacts,
    max_speed_ms: float | None = None,
    max_age_s: float | None = None,
) -> list[Tracklet]:
    """Tracklets the current tracker produces on the stored observations.

    Parameters default to the ones the run was made with, so that a
    difference in the output is a difference in the code.
    """
    meta = artifacts.meta
    config = meta.get("config", {})
    tracker = CourtTracker(
        calibration=artifacts.calibration,
        max_speed_ms=max_speed_ms if max_speed_ms is not None else config.get("max_player_speed_ms", 8.0),
        max_age_s=max_age_s if max_age_s is not None else config.get("track_max_age_s", 1.2),
        min_observations=int(meta.get("min_observations", 3)),
    )

    by_frame: dict[int, list[Observation]] = defaultdict(list)
    for _, obs in read_observations(artifacts.directory):
        by_frame[obs.frame_index].append(obs)
    if not by_frame:
        return []

    # Empty frames were not written, but the tracker must see them: the
    # sampling period is read from consecutive calls. Their timestamps come
    # from the stored ones — the fps in meta.json is rounded, the stored
    # timestamps are exact to the microsecond.
    last_index = max(by_frame)
    last_obs = by_frame[last_index][0]
    seconds_per_frame = last_obs.timestamp_s / last_index if last_index else 0.0
    step = int(meta.get("sample_step") or 1)
    indices = sorted(set(range(0, last_index + 1, step)) | set(by_frame))

    for index in indices:
        observations = by_frame.get(index, [])
        timestamp = observations[0].timestamp_s if observations else index * seconds_per_frame
        # The live tracker received them ordered by confidence.
        observations = sorted(observations, key=lambda o: -o.confidence)
        tracker.update_observations(timestamp, observations)

    return tracker.finish()
