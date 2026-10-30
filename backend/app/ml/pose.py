"""The players' posture: 17 points of the body (COCO order) per person.

Computed only with the AI accelerator (app.ml.detect_hailo), by a YOLOv8
pose network: on the CPU it would cost as much as the whole analysis again.
Each detection is cropped into a tile of a mosaic the network looks at
(mosaic, poses_in_tiles), so the pose found there is that player's; it
travels with the detection as Observation.keypoints: stored in the artifacts,
drawn on the video, and the material for strokes (overheads, forehands,
backhands) once checked against the video.

The network's outputs are the raw heads of YOLOv8 on three grids (strides
8, 16, 32 for a 640 input): per cell 64 values of box distribution (4 sides
x 16 bins), the person score, 51 keypoint values (17 x, y, visibility).
decode_yolov8_pose follows Hailo's own Python post-processing
(Hailo-Application-Code-Examples, pose_estimation_utils.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

KEYPOINT_NAMES = (
    "naso", "occhio sx", "occhio dx", "orecchio sx", "orecchio dx",
    "spalla sx", "spalla dx", "gomito sx", "gomito dx", "polso sx", "polso dx",
    "anca sx", "anca dx", "ginocchio sx", "ginocchio dx", "caviglia sx", "caviglia dx",
)
# Limbs drawn on the video: pairs of keypoint indices.
SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
)
REG_MAX = 16                 # bins per box side
BOX_CHANNELS = 4 * REG_MAX
KEYPOINT_CHANNELS = 51

SCORE_THRESHOLD = 0.3
# In a tile somebody is known to be: a lower score is enough.
TILE_SCORE_THRESHOLD = 0.15
# The pose network sees up to 4 players at once, each a tile of a 2 x 2
# mosaic: a far player 40 px tall in the frame is 320 px there. Over the
# whole frame the network found the near players and missed the far ones.
GRID = 2
# Room around each player's box in the tile, for raised arms and racket.
TILE_MARGIN = 1.3
# Players given a pose per frame, most confident detections first: two
# mosaics at most.
MAX_POSED = 8
NMS_IOU = 0.6
MAX_POSES = 20


def decode_yolov8_pose(outputs: dict[str, np.ndarray], imgsz: tuple[int, int],
                       score_threshold: float = SCORE_THRESHOLD, iou: float = NMS_IOU,
                       max_poses: int = MAX_POSES) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """(box x1 y1 x2 y2, score, keypoints (17, 3)) in the network's input
    pixels, from its raw outputs (each H x W x C, float)."""
    heads: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for array in outputs.values():
        array = np.asarray(array, dtype=np.float32)
        array = array.reshape(array.shape[-3:]) if array.ndim > 3 else array
        height, width, channels = array.shape
        kind = "box" if channels == BOX_CHANNELS else "kpt" if channels == KEYPOINT_CHANNELS else "cls"
        heads.setdefault((height, width), {})[kind] = array
    boxes, scores, keypoints = [], [], []
    for (height, width), head in sorted(heads.items(), reverse=True):
        if not {"box", "kpt", "cls"} <= head.keys():
            continue
        stride = imgsz[0] / height
        cls = head["cls"][..., 0].reshape(-1)
        # Probabilities from the chip, or the logits before the sigmoid.
        if cls.size and (cls.min() < 0.0 or cls.max() > 1.0):
            cls = 1.0 / (1.0 + np.exp(-cls))
        keep = np.flatnonzero(cls >= score_threshold)
        if not keep.size:
            continue
        rows, cols = np.divmod(keep, width)
        cx, cy = (cols + 0.5) * stride, (rows + 0.5) * stride
        distribution = head["box"].reshape(-1, 4, REG_MAX)[keep]
        distribution = np.exp(distribution - distribution.max(axis=-1, keepdims=True))
        distribution /= distribution.sum(axis=-1, keepdims=True)
        distance = (distribution * np.arange(REG_MAX)).sum(axis=-1) * stride     # left, top, right, bottom
        boxes.append(np.stack([cx - distance[:, 0], cy - distance[:, 1],
                               cx + distance[:, 2], cy + distance[:, 3]], axis=1))
        scores.append(cls[keep])
        raw = head["kpt"].reshape(-1, 17, 3)[keep].copy()
        raw[..., 0] = stride * (2.0 * raw[..., 0] - 0.5) + cx[:, None]
        raw[..., 1] = stride * (2.0 * raw[..., 1] - 0.5) + cy[:, None]
        raw[..., 2] = 1.0 / (1.0 + np.exp(-raw[..., 2]))
        keypoints.append(raw)
    if not boxes:
        return []
    boxes_all, scores_all, kpts_all = np.concatenate(boxes), np.concatenate(scores), np.concatenate(keypoints)
    order = np.argsort(-scores_all)[: max_poses * 10]
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_all[order]]
    kept = cv2.dnn.NMSBoxes(xywh, scores_all[order].tolist(), score_threshold, iou)
    kept = order[np.asarray(kept).reshape(-1)[:max_poses]] if len(kept) else []
    return [(boxes_all[i], float(scores_all[i]), kpts_all[i]) for i in kept]


@dataclass(frozen=True)
class Tile:
    """Where one detection's crop sits in the mosaic: frame point
    (x, y) -> mosaic (x0 + (x - cx0) * scale, y0 + (y - cy0) * scale)."""
    x0: float
    y0: float
    cx0: float
    cy0: float
    scale: float

    def to_frame(self, points: np.ndarray) -> np.ndarray:
        out = np.array(points, dtype=np.float32, copy=True)
        out[..., 0] = (out[..., 0] - self.x0) / self.scale + self.cx0
        out[..., 1] = (out[..., 1] - self.y0) / self.scale + self.cy0
        return out

    def contains(self, x: float, y: float, side: float) -> bool:
        return self.x0 <= x < self.x0 + side and self.y0 <= y < self.y0 + side


def mosaic(frame: np.ndarray, boxes, imgsz: tuple[int, int], fill: int = 114) -> tuple[np.ndarray, list[Tile]]:
    """The network's input with up to GRID x GRID players, each cropped
    square around its box (TILE_MARGIN) and scaled to fill its tile; RGB."""
    side = imgsz[0] // GRID
    picture = np.full((imgsz[0], imgsz[1], 3), fill, dtype=np.uint8)
    tiles: list[Tile] = []
    for k, (x1, y1, x2, y2) in enumerate(list(boxes)[: GRID * GRID]):
        row, col = divmod(k, GRID)
        crop = max(x2 - x1, y2 - y1, 8.0) * TILE_MARGIN
        cx0, cy0 = (x1 + x2) / 2 - crop / 2, (y1 + y2) / 2 - crop / 2
        scale = side / crop
        x0, y0 = col * side, row * side
        matrix = np.float32([[scale, 0, x0 - cx0 * scale], [0, scale, y0 - cy0 * scale]])
        warped = cv2.warpAffine(frame, matrix, (imgsz[1], imgsz[0]), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(fill, fill, fill))
        picture[y0:y0 + side, x0:x0 + side] = warped[y0:y0 + side, x0:x0 + side]
        tiles.append(Tile(float(x0), float(y0), float(cx0), float(cy0), float(scale)))
    return np.ascontiguousarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)), tiles


def poses_in_tiles(decoded, tiles: list[Tile], imgsz: tuple[int, int]) -> list[np.ndarray | None]:
    """For each tile the keypoints, in frame pixels, of the most confident
    pose centred in it; None where there is none."""
    side = imgsz[0] // GRID
    best: list[tuple[float, np.ndarray] | None] = [None] * len(tiles)
    for box, score, kpts in decoded:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for k, tile in enumerate(tiles):
            if tile.contains(cx, cy, side) and (best[k] is None or score > best[k][0]):
                best[k] = (score, kpts)
    return [tiles[k].to_frame(b[1]) if b is not None else None for k, b in enumerate(best)]
