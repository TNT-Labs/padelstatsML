"""Each player's picture, for putting a name to them.

The picture used to be the single best-looking crop among all the tracks a
player was built from: the largest, most confident box. That rewards the
extreme sample, and a player's samples are not all theirs — a few around a
changeover or a crossing belong to somebody else. On a real match two
players got a picture of the other pair.

So a picture is chosen after identity, among the samples typical of the
player — kit colour and re-ID close to the player's own average — and alone
in the frame, with no other player's box over it. Size and confidence only
rank those.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.ml.identity import PlayerTrack
from app.ml.roles import _look
from app.ml.tracking import Observation, observation_key

CROP_W, CROP_H = 160, 240
CROP_PAD = 20
CROP_JPEG_QUALITY = 85
# A sample is typical when it looks at least as much like the player as
# this share of the player's own samples does.
TYPICAL_SHARE = 0.5
# Another player's box covering more than this share of the sample's box
# puts two people in the picture.
MAX_OVERLAP = 0.05


def crop_score(obs: Observation) -> float | None:
    """How good a picture this detection makes, or None if it cannot make
    one: confident, and large up to the size where a face is legible."""
    x1, y1, x2, y2 = obs.bbox
    box_h, box_w = y2 - y1, x2 - x1
    if box_h < 40 or box_w < 12:
        return None
    if not 1.2 <= box_h / max(box_w, 1e-3) <= 5.0:
        return None
    return obs.confidence * min(box_h / 200.0, 1.0)


def choose_pictures(
    players: list[PlayerTrack], candidates: dict[int, list[Observation]] | None = None,
) -> dict[int, Observation]:
    """The sample to picture for each player, among `candidates` (all the
    player's samples by default). Typical and alone if any is; typical if
    not; otherwise simply the best-looking."""
    boxes_by_frame: dict[int, list[tuple[tuple, tuple]]] = defaultdict(list)
    for player in players:
        for obs in player.observations:
            boxes_by_frame[obs.frame_index].append((observation_key(obs), obs.bbox))

    chosen: dict[int, Observation] = {}
    for player in players:
        pool = candidates.get(player.player_id, []) if candidates is not None else player.observations
        scored = [(score, obs) for obs in pool if (score := crop_score(obs)) is not None]
        if not scored:
            continue
        typical = _typical(player, [obs for _, obs in scored])
        alone = [_alone(obs, boxes_by_frame) for _, obs in scored]
        for keep in (
            lambda i: typical[i] and alone[i],
            lambda i: typical[i],
            lambda i: True,
        ):
            ranked = [(score, i) for i, (score, _) in enumerate(scored) if keep(i)]
            if ranked:
                chosen[player.player_id] = scored[max(ranked)[1]][1]
                break
    return chosen


def _typical(player: PlayerTrack, samples: list[Observation]) -> list[bool]:
    """Whether each sample looks like the player, by every cue there is."""
    typical = [True] * len(samples)
    for reid in (False, True):
        own = [v for v in (_look(o, reid) for o in player.observations) if v is not None]
        if len(own) < 2:
            continue
        mean = np.mean(own, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 0:
            continue
        mean /= norm
        limit = float(np.quantile([1.0 - float(v @ mean) for v in own], TYPICAL_SHARE))
        for i, obs in enumerate(samples):
            look = _look(obs, reid)
            if look is None or 1.0 - float(look @ mean) > limit:
                typical[i] = False
    return typical


def _alone(obs: Observation, boxes_by_frame: dict) -> bool:
    x1, y1, x2, y2 = obs.bbox
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    own = observation_key(obs)
    for key, (ox1, oy1, ox2, oy2) in boxes_by_frame.get(obs.frame_index, []):
        if key == own:
            continue
        overlap = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
        if overlap / area > MAX_OVERLAP:
            return False
    return True


def thumbnail(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> bytes | None:
    """The player framed at the thumbnail's own 2:3 proportions, then scaled.

    The box is widened or heightened around its centre to 2:3 before resizing
    — stretching the box itself squashed or elongated the player.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    box_w, box_h = (x2 - x1) + 2 * CROP_PAD, (y2 - y1) + 2 * CROP_PAD
    target = CROP_W / CROP_H
    if box_w / box_h < target:
        box_w = box_h * target
    else:
        box_h = box_w / target
    cx1 = int(np.clip(cx - box_w / 2.0, 0, width - 1))
    cy1 = int(np.clip(cy - box_h / 2.0, 0, height - 1))
    cx2 = int(np.clip(cx + box_w / 2.0, 1, width))
    cy2 = int(np.clip(cy + box_h / 2.0, 1, height))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    resized = cv2.resize(frame[cy1:cy2, cx1:cx2], (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def pictures_from_video(video_path: str | Path, picks: dict[int, Observation]) -> dict[int, bytes]:
    """Cut the chosen samples out of the video, reading it in order from the
    start — the way the analysis numbered its frames. Seeking straight to a
    frame number can land a few frames off on some videos, and a player in
    motion would then be cut off."""
    wanted: dict[int, list[tuple[int, Observation]]] = defaultdict(list)
    for player_id, obs in picks.items():
        wanted[obs.frame_index].append((player_id, obs))
    out: dict[int, bytes] = {}
    if not wanted:
        return out
    last = max(wanted)
    capture = cv2.VideoCapture(str(video_path))
    try:
        index = 0
        while index <= last and capture.grab():
            if index in wanted:
                ok, frame = capture.retrieve()
                if ok:
                    for player_id, obs in wanted[index]:
                        picture = thumbnail(frame, obs.bbox)
                        if picture is not None:
                            out[player_id] = picture
            index += 1
    finally:
        capture.release()
    return out
