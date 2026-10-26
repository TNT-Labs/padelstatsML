"""Recognising a camera position already calibrated (app/ml/camera_match.py)."""
import cv2
import numpy as np

from app.ml.camera_match import match_camera, transfer

FRAME = (1024, 576)
CORNERS = [[300, 270], [650, 270], [1300, 700], [-300, 700]]


def _scene(seed: int) -> np.ndarray:
    """A picture full of fixed structure, like fences and a roof."""
    rng = np.random.default_rng(seed)
    image = np.full((FRAME[1], FRAME[0], 3), 90, np.uint8)
    for _ in range(400):
        x, y = rng.integers(0, FRAME[0]), rng.integers(0, FRAME[1])
        w, h = rng.integers(8, 60), rng.integers(8, 60)
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return image


def _moved(image: np.ndarray, degrees: float, dx: float, dy: float):
    m = cv2.getRotationMatrix2D((FRAME[0] / 2, FRAME[1] / 2), degrees, 1.0)
    m[:, 2] += [dx, dy]
    return cv2.warpAffine(image, m, FRAME), m


def test_the_same_camera_nudged_is_recognised_and_its_points_follow():
    reference = _scene(1)
    image, m = _moved(reference, 1.5, 12, -8)
    found = match_camera(reference, image, FRAME, CORNERS)
    assert found is not None and found.same_position
    points = np.array([[215.0, 322.0], [720.0, 335.0], [467.0, 452.0]])
    expected = cv2.transform(points.reshape(1, -1, 2), m)[0]
    assert np.abs(transfer(points, found, 0.0, FRAME) - expected).max() < 1.0


def test_a_camera_moved_a_lot_is_not_applied_by_itself():
    reference = _scene(1)
    image, _ = _moved(reference, 6, 60, 30)
    found = match_camera(reference, image, FRAME, CORNERS)
    assert found is not None and not found.same_position


def test_another_place_is_not_taken_for_this_one():
    found = match_camera(_scene(1), _scene(2), FRAME, CORNERS)
    assert found is None or not found.same_position
