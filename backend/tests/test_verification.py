"""Accuracy check against points marked by hand."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.court import CalibrationSource, build_calibration
from app.ml.identity import PlayerTrack
from app.ml.rallies import Rally
from app.ml.smoothing import smooth_path
from app.ml.tracking import Observation
from app.ml.verification import LANDMARKS, calibration_errors, choose_frames, player_errors
from tests.test_court import COURT_CORNERS_M_LIST, FRAME, _pinhole

HZ = 5.0


def _camera(shift_near_px: float = 0.0):
    """A camera and a calibration from its corners; the two near corners
    placed `shift_near_px` too low, as a hurried hand would."""
    project = _pinhole(3.5, 12.0)
    corners = [list(project((x, y, 0.0))) for x, y in COURT_CORNERS_M_LIST]
    corners[2][1] += shift_near_px
    corners[3][1] += shift_near_px
    return project, build_calibration(corners, FRAME, CalibrationSource.MANUAL)


def test_a_right_calibration_lands_every_landmark_where_it_is():
    project, calib = _camera()
    marks = {lm.key: project((lm.x_m, lm.y_m, 0.0)) for lm in LANDMARKS}

    result = calibration_errors(calib, marks)

    assert result["all"]["count"] == len(LANDMARKS)
    assert result["all"]["max_m"] < 0.02


def test_a_misplaced_corner_shows_up_where_it_was_misplaced():
    project, calib = _camera(shift_near_px=25.0)
    marks = {lm.key: project((lm.x_m, lm.y_m, 0.0)) for lm in LANDMARKS}

    result = calibration_errors(calib, marks)

    assert result["by_half"]["vicina"]["mean_m"] > 0.2
    assert result["by_half"]["vicina"]["mean_m"] > result["by_half"]["lontana"]["mean_m"]
    # Unknown keys are ignored, not an error.
    assert calibration_errors(calib, {"altro": (1.0, 1.0)})["all"] == {"count": 0}


def _player(calib, project, pid, team, positions, start_frame=0):
    observations = []
    for k, (x, y) in enumerate(positions):
        foot = project((x, y, 0.0))
        court = calib.floor_points_to_court(np.array([foot], dtype=np.float32))[0]
        observations.append(Observation(
            start_frame + 6 * k, (start_frame + 6 * k) / 30.0, (foot[0] - 20, foot[1] - 120, foot[0] + 20, foot[1]),
            foot, (float(court[0]), float(court[1])), 0.9, np.full(72, 1 / 72, np.float32),
        ))
    return PlayerTrack(pid, team, observations, [pid])


def test_players_are_paired_with_the_marks_and_the_misses_counted():
    project, calib = _camera()
    homes = [(2.5, 5.0), (7.5, 5.0), (2.5, 15.0), (7.5, 15.0)]
    players = [_player(calib, project, i, i // 2, [homes[i]] * 20) for i in range(3)]
    paths = {p.player_id: smooth_path(p.observations, calib) for p in players}
    frame = players[0].observations[10].frame_index
    # Marked: the three placed, 10 cm off, and the fourth the analysis missed.
    feet = [project((x + 0.1, y, 0.0)) for x, y in homes]

    result = player_errors(calib, players, paths, [(frame, feet)])

    assert result["matched"] == 3 and result["missed"] == 1 and result["extra"] == 0
    assert result["raw"]["all"]["mean_m"] == pytest.approx(0.1, abs=0.03)
    assert result["smoothed"]["all"]["count"] == 3
    assert set(result["raw"]["by_half"]) == {"vicina", "lontana"}


def test_a_player_placed_where_nobody_is_counts_as_extra():
    project, calib = _camera()
    players = [_player(calib, project, 0, 0, [(5.0, 5.0)] * 10)]
    frame = players[0].observations[5].frame_index

    result = player_errors(calib, players, {}, [(frame, [project((5.0, 15.0, 0.0))])])

    assert result["matched"] == 0 and result["missed"] == 1 and result["extra"] == 1


def test_frames_are_spread_and_half_of_them_inside_rallies():
    project, calib = _camera()
    players = [_player(calib, project, i, i // 2, [(2.0 + 2 * i, 5.0 + 3 * i)] * 300) for i in range(4)]
    end = players[0].observations[-1].timestamp_s
    rallies = [Rally(0, 10.0, 20.0, 4), Rally(1, 40.0, 50.0, 4)]

    frames = choose_frames(players, rallies)

    assert len(frames) == 8
    assert len({f["frame_index"] for f in frames}) == 8
    assert sum(1 for f in frames if any(r.contains(f["t"]) for r in rallies)) == 4
    assert all(0.0 <= f["t"] <= end for f in frames)


@pytest.mark.asyncio
async def test_the_check_needs_a_completed_match_and_an_editor():
    from tests.test_auth import _match_of, _name
    from tests.conftest import signed_in

    owner = await signed_in(_name("u"))
    viewer = await signed_in(_name("vis"), "visualizzatore")
    async with owner, viewer:
        match_id = await _match_of(owner)
        assert (await owner.get(f"/api/matches/{match_id}/verification")).status_code == 409
        assert (await viewer.get(f"/api/matches/{match_id}/verification")).status_code == 403
        assert (await viewer.post(f"/api/matches/{match_id}/verification", json={})).status_code == 403


def test_landmarks_correct_a_calibration_made_on_misplaced_corners():
    from app.ml.verification import refine_calibration

    project, calib = _camera(shift_near_px=60.0)
    rng = np.random.default_rng(0)
    # Marked by hand: a few pixels off.
    marks = {lm.key: tuple(np.array(project((lm.x_m, lm.y_m, 0.0))) + rng.normal(0, 4, 2)) for lm in LANDMARKS}

    proposal = refine_calibration(calib, marks)

    # Measured on each landmark left out of the fit, not on the fit itself.
    assert proposal["before"]["mean_m"] > 0.5
    assert proposal["after"]["mean_m"] < 0.3
    true_corners = np.array([project((x, y, 0.0)) for x, y in COURT_CORNERS_M_LIST])
    assert np.abs(np.array(proposal["corners_px"]) - true_corners).max() < 25.0


def test_right_corners_are_not_made_worse():
    from app.ml.verification import refine_calibration

    project, calib = _camera()
    rng = np.random.default_rng(1)
    marks = {lm.key: tuple(np.array(project((lm.x_m, lm.y_m, 0.0))) + rng.normal(0, 4, 2)) for lm in LANDMARKS}

    proposal = refine_calibration(calib, marks)

    assert proposal["after"]["mean_m"] <= proposal["before"]["mean_m"] + 0.05


def test_too_few_landmarks_cannot_refine():
    from app.ml.court import CalibrationError
    from app.ml.verification import refine_calibration

    project, calib = _camera()
    marks = {lm.key: project((lm.x_m, lm.y_m, 0.0)) for lm in LANDMARKS[:3]}
    with pytest.raises(CalibrationError):
        refine_calibration(calib, marks)


def test_stored_detections_are_placed_again_with_a_corrected_calibration(tmp_path):
    from app.ml.artifacts import LoadedArtifacts
    from app.ml.reprocess import with_calibration
    from app.ml.tracking import Tracklet

    project, wrong = _camera(shift_near_px=60.0)
    _, right = _camera()
    player = _player(wrong, project, 0, 0, [(3.0, 4.0)] * 5)
    artifacts = LoadedArtifacts(tmp_path, {}, [Tracklet(0, player.observations)], wrong, None)
    assert abs(player.observations[0].foot_court[1] - 4.0) > 0.3     # placed wrong

    same = with_calibration(artifacts, {"corners_px": wrong.corners_px.tolist(), "frame_size": list(FRAME)})
    fixed = with_calibration(artifacts, {"corners_px": right.corners_px.tolist(), "frame_size": list(FRAME)})

    assert same is artifacts
    x, y = fixed.tracklets[0].observations[0].foot_court
    assert x == pytest.approx(3.0, abs=0.05) and y == pytest.approx(4.0, abs=0.05)
    assert fixed.tracklets[0].observations[0].foot_px == player.observations[0].foot_px


def test_the_lens_correction_is_found_from_the_landmarks():
    """A wide-angle lens close to the court: the landmarks along the bent
    lines show the bend, and the correction that explains it is kept."""
    from app.ml.court import distort_points
    from app.ml.verification import refine_calibration

    k1 = 0.2
    project = _pinhole(2.0, 3.0)
    seen = lambda x, y: distort_points(np.array([project((x, y, 0.0))]), k1, FRAME)[0]
    calib = build_calibration([seen(x, y) for x, y in COURT_CORNERS_M_LIST], FRAME, CalibrationSource.MANUAL)
    rng = np.random.default_rng(0)
    marks = {lm.key: tuple(seen(lm.x_m, lm.y_m) + rng.normal(0, 3, 2)) for lm in LANDMARKS}

    proposal = refine_calibration(calib, marks)

    assert proposal["k1"] == pytest.approx(k1, abs=0.06)
    assert proposal["after"]["mean_m"] < proposal["before"]["mean_m"] - 0.05


def test_no_lens_correction_where_there_is_no_bend():
    from app.ml.verification import refine_calibration

    project, calib = _camera()
    rng = np.random.default_rng(2)
    marks = {lm.key: tuple(np.array(project((lm.x_m, lm.y_m, 0.0))) + rng.normal(0, 3, 2)) for lm in LANDMARKS}

    assert refine_calibration(calib, marks)["k1"] == 0.0
