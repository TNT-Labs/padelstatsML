"""The players' posture: 17 points of the body (COCO order) per person.

Computed only with the AI accelerator (app.ml.detect_hailo), by a YOLOv8
pose network: on the CPU it would cost as much as the whole analysis again.
Each pose is attached to the detection whose box it overlaps (attach_poses)
and travels with it as Observation.keypoints: stored in the artifacts,
drawn on the video, and the material for strokes (overheads, forehands,
backhands) once checked against the video.

The network's outputs are the raw heads of YOLOv8 on three grids (strides
8, 16, 32 for a 640 input): per cell 64 values of box distribution (4 sides
x 16 bins), the person score, 51 keypoint values (17 x, y, visibility).
decode_yolov8_pose follows Hailo's own Python post-processing
(Hailo-Application-Code-Examples, pose_estimation_utils.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

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
NMS_IOU = 0.6
MAX_POSES = 20
# A pose belongs to a detection when their boxes overlap this much.
ATTACH_IOU = 0.45


@dataclass(frozen=True)
class Pose:
    bbox: tuple[float, float, float, float]   # frame pixels
    score: float
    keypoints: np.ndarray                     # (17, 3): x, y in frame pixels, visibility 0-1


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


def to_frame(decoded, ratio: float, pad: tuple[float, float]) -> list[Pose]:
    """Poses in frame pixels from the network's letterboxed input."""
    out = []
    for box, score, kpts in decoded:
        frame_box = ((box[[0, 2]] - pad[0]) / ratio, (box[[1, 3]] - pad[1]) / ratio)
        points = kpts.copy()
        points[:, 0] = (points[:, 0] - pad[0]) / ratio
        points[:, 1] = (points[:, 1] - pad[1]) / ratio
        out.append(Pose((float(frame_box[0][0]), float(frame_box[1][0]),
                         float(frame_box[0][1]), float(frame_box[1][1])), score, points.astype(np.float32)))
    return out


def attach_poses(detections: list, poses: list[Pose], min_iou: float = ATTACH_IOU) -> list:
    """The detections, each with the keypoints of the pose that overlaps its
    box best (one pose per detection, the best overlaps first)."""
    if not poses or not detections:
        return detections
    pairs = sorted(
        ((_iou(d.bbox, p.bbox), i, j) for i, d in enumerate(detections) for j, p in enumerate(poses)),
        reverse=True,
    )
    out, used_d, used_p = list(detections), set(), set()
    for overlap, i, j in pairs:
        if overlap < min_iou:
            break
        if i in used_d or j in used_p:
            continue
        used_d.add(i)
        used_p.add(j)
        out[i] = replace(detections[i], keypoints=poses[j].keypoints)
    return out


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0
