"""The stored detections, placed with the match's current calibration.

The artifacts keep every detection's box and foot in pixels as well as in
metres. When the calibration of a match is corrected after its analysis —
refined from landmarks marked by hand — the metres change and the pixels do
not: projecting the pixels again gives the positions the new calibration
implies, without running the detector again. Who was associated with whom
(the tracklets) stays as the analysis found it.

The foot points are taken again from the boxes on the way (app.ml.footpoint:
a box cut by the net), with the new calibration — and for runs recorded
before that correction existed even when the calibration is the same.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.ml.artifacts import LoadedArtifacts
from app.ml.court import CourtCalibration, calibration_from_dict
from app.ml.footpoint import FOOT_MODEL_VERSION, FootModel
from app.ml.tracking import Tracklet


def with_calibration(artifacts: LoadedArtifacts, calibration_data: dict | None) -> LoadedArtifacts:
    """`artifacts` as seen through `calibration_data` (the match's own,
    `Match.calibration`): unchanged when it is the one of the analysis and
    the run's foot points are already corrected."""
    calibration = artifacts.calibration
    if calibration_data and calibration_data.get("corners_px"):
        stored = np.asarray(artifacts.calibration.corners_px, dtype=np.float64)
        current = np.asarray(calibration_data["corners_px"], dtype=np.float64)
        same_lens = abs(float(calibration_data.get("k1") or 0.0) - artifacts.calibration.k1) < 1e-9
        if not (stored.shape == current.shape and np.allclose(stored, current, atol=1e-3) and same_lens):
            calibration = calibration_from_dict(calibration_data)
    feet_current = artifacts.meta.get("foot_model") == FOOT_MODEL_VERSION
    if calibration is artifacts.calibration and feet_current:
        return artifacts
    # The whole match teaches the foot model at once: every box is placed
    # with what the match says about the camera, the first ones included.
    feet = FootModel(calibration).fit(o.bbox for t in artifacts.tracklets for o in t.observations)
    return replace(
        artifacts,
        calibration=calibration,
        tracklets=[reproject(t, calibration, feet) for t in artifacts.tracklets],
    )


def reproject(tracklet: Tracklet, calibration: CourtCalibration, feet: FootModel | None = None) -> Tracklet:
    """The tracklet's positions through `calibration`; with `feet`, the foot
    points taken again from the boxes first."""
    if not tracklet.observations:
        return tracklet
    if feet is not None:
        pixels = [feet.foot(o.bbox) for o in tracklet.observations]
    else:
        pixels = [o.foot_px for o in tracklet.observations]
    court = calibration.floor_points_to_court(np.array(pixels, dtype=np.float32))
    return Tracklet(id=tracklet.id, observations=[
        replace(o, foot_px=(float(p[0]), float(p[1])), foot_court=(float(c[0]), float(c[1])))
        for o, p, c in zip(tracklet.observations, pixels, court)
    ])


def recompute(directory, calibration_data: dict | None, config, max_speed_ms: float | None = None):
    """The post-detection stages on the stored run, through the match's
    current calibration: (result, identity, rallies, artifacts)."""
    from app.ml.artifacts import load_artifacts
    from app.ml.pipeline import analyse_tracklets, apply_run_quality

    artifacts = with_calibration(load_artifacts(directory), calibration_data)
    result, identity, rallies = analyse_tracklets(
        tracklets=artifacts.tracklets,
        calibration=artifacts.calibration,
        config=config,
        sample_hz=artifacts.sample_hz,
        frames_sampled=artifacts.frames_sampled,
        analysed_s=artifacts.analysed_s,
        start_s=float(artifacts.meta.get("start_s") or 0.0),
    )
    apply_run_quality(result, run_quality(artifacts))
    return result, identity, rallies, artifacts


def run_quality(artifacts) -> dict | None:
    """What only the decode pass measured: from meta.json; for runs
    recorded before it was kept there, from the result they produced."""
    stored = artifacts.meta.get("run_quality")
    if stored:
        return stored
    previous = (artifacts.result or {}).get("data_quality") or {}
    truncated = next((w for w in previous.get("warnings", []) if w.startswith("Analizzati ")), None)
    return {
        "variable_frame_rate": previous.get("variable_frame_rate",
                                            bool(artifacts.meta.get("variable_frame_rate"))),
        "scenery": (previous.get("tracking") or {}).get("scenery", 0),
        "far_pass": previous.get("far_pass"),
        "truncated": truncated,
    }


def pipeline_config(settings, meta: dict):
    """The analysis settings for re-running the stages after detection:
    the run's own speed limit, the current thresholds."""
    from app.ml.pipeline import PipelineConfig

    config = meta.get("config") or {}
    return PipelineConfig(
        detector_model=settings.detector_model,
        max_player_speed_ms=float(config.get("max_player_speed_ms") or settings.max_player_speed_ms),
        rally_speed_threshold_ms=settings.rally_speed_threshold_ms,
        rally_speed_adaptive=settings.rally_speed_adaptive,
        rally_min_duration_s=settings.rally_min_duration_s,
        rally_merge_gap_s=settings.rally_merge_gap_s,
    )


def same_slots(old: dict | None, new: dict) -> bool:
    """Whether every player id is the same pair and side as before: then
    the names given to the ids still name the same people."""
    def slots(per_player):
        return {str(k): ((v or {}).get("team"), (v or {}).get("role")) for k, v in per_player.items()}
    return bool(old) and slots(old) == slots(new)


def match_calibration(match_id: str) -> dict | None:
    """The match's current calibration, from the database (for scripts)."""
    from app.core.database import sync_session
    from app.models import Match

    with sync_session() as session:
        match = session.get(Match, match_id)
        return dict(match.calibration) if match is not None and match.calibration else None
