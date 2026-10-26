"""Find the net and the near service line in a picture of the court.

What a padel court shows reliably, from any camera behind the back glass:

* the net's tape, the longest thick white line in the picture, ending at
  the two posts;
* the near service line, a long white line wall to wall;
* the centre service line, from the service line up to the net and beyond;
* the net's bottom cord, a dark line crossing the centre line on the floor.

From these, the four handles of the "net_service" reference (the foot of the
two posts, the ends of the near service line):

1. White line evidence (top-hat of the darkest colour channel) and its long
   straight segments.
2. The centre line: the longest near-vertical white line in the middle of
   the picture. The service line: the long near-horizontal white line
   through its lower end — the T. The tape: the longest near-horizontal
   white line crossing the centre line above the T.
3. The net's foot on the centre line: the dark dip in the centre line
   between the tape and the T, where the bottom cord crosses it.
4. The net's foot: the line through the dip along the tape; the posts stand
   on it below the tape's ends. The dip is the projective midpoint of the
   posts (the centre line meets the net half way across): that fixes where
   the court's width runs to infinity.
5. The homography from the posts, the T and that vanishing point.

The check that makes it trustworthy: the service line found must run along
the court's service line from one edge of the picture to the other. Nothing
in the steps above forces it, so a wrong line anywhere shows up there. The
proposal carries that error; the calibration page shows it and the user
confirms.

A camera far to one side, a court with other markings, a net hidden by
people: no proposal, rather than a wrong one.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.ml.court import NET_Y_M, SERVICE_LINE_OFFSET_M, REFERENCE_POINTS_M, NET_HEIGHT_CENTER_M

_NEAR_SERVICE = NET_Y_M - SERVICE_LINE_OFFSET_M
# Evidence above this (scaled to its 99.5th percentile) is line.
LINE_THRESHOLD = 0.45
# Segments at least this share of the picture's width.
MIN_SEGMENT = 0.08
# The centre line's foot (the T) within this share of the height of the
# service line; segments within it of a line belong to it.
ON_LINE = 0.015
# The tape at least this share of the width long.
MIN_TAPE = 0.25
# The dark cord: at least this much darker than the centre line around it.
MIN_DIP = 0.2
# The service line found lies this close to the court's, in metres, at both
# edges of the picture, for a proposal; confidence falls to 0 there.
MAX_LINE_ERROR_M = 0.5
# The net's height between tape and cord, over the width scale across the
# net: about 1 for a camera looking roughly level; far from it, wrong lines.
NET_SCALE_RANGE = (0.45, 2.0)


@dataclass(frozen=True)
class NetServiceProposal:
    handles: np.ndarray          # 4x2, REFERENCE_POINTS_M["net_service"] order
    confidence: float            # 0..1
    line_error_m: float          # how far the service line found strays from the court's


def background(frames: list[np.ndarray]) -> np.ndarray:
    """The court without players: the median of frames spread over the video."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def detect_net_service(image: np.ndarray) -> NetServiceProposal | None:
    h, w = image.shape[:2]
    evidence = _evidence(image)
    horizontal, vertical = _segments(evidence, w, h)
    centre = _centre_line(vertical, w, h)
    if centre is None:
        return None
    a, b, top, bottom = centre                       # x = a*y + b, rows top..bottom
    service = _service_line(horizontal, a, b, bottom, h)
    if service is None:
        return None
    t = _intersect(service, (1.0, -a, -b))           # x - a*y - b = 0
    if t is None:
        return None
    tape = _tape(horizontal, a, b, t, w, h)
    if tape is None:
        return None
    tape_line, x_left, x_right = tape
    tape_at_centre = _intersect(tape_line, (1.0, -a, -b))
    if tape_at_centre is None or tape_at_centre[1] >= t[1]:
        return None
    # The net is about as tall in the picture as 0.88 m across it: the tape
    # spans the 10 m between the posts.
    expected = (x_right - x_left) / 10.0 * NET_HEIGHT_CENTER_M
    foot_row = _cord_row(image, a, b, tape_at_centre[1], t[1], expected)
    if foot_row is None:
        return None
    foot = np.array([a * foot_row + b, foot_row])
    # The net's foot runs as the tape does (both across the court, close in
    # the picture); the posts stand on it below the tape's ends.
    base = np.cross(np.r_[foot, 1.0], np.r_[foot + [tape_line[1], -tape_line[0]], 1.0])
    posts = [_intersect(base, (1.0, 0.0, -x)) for x in (x_left, x_right)]
    if any(p is None for p in posts):
        return None
    # The court's depth: the side walls. The turf ends against them, and
    # that edge runs from each post towards the camera; where it meets the
    # centre line is where the court's length runs to infinity. A side
    # whose edge does not run through its post (a bench, the floor of the
    # next court) gives nothing.
    sides = _side_lines(image, posts, foot, t, base)
    if not sides:
        return None
    # The post stands where its wall meets the net's foot: sharper than the
    # tape's end, which the post's own width blurs.
    if "left" in sides:
        posts[0] = _intersect(base, sides["left"])
    if "right" in sides:
        posts[1] = _intersect(base, sides["right"])
    if any(p is None for p in posts):
        return None
    centre_line = np.array([1.0, -a, -b])
    depth_points = [np.cross(side, centre_line) for side in sides.values()]
    depth = depth_points[0] / np.linalg.norm(depth_points[0])

    court = np.array([[0.0, NET_Y_M, 1.0], [10.0, NET_Y_M, 1.0], [5.0, _NEAR_SERVICE, 1.0], [0.0, 1.0, 0.0]])
    picture = np.array([[*posts[0], 1.0], [*posts[1], 1.0], [*t, 1.0], depth])
    homography = _dlt(court, picture)
    if homography is None:
        return None
    inverse = np.linalg.inv(homography)

    def to_court(point):
        q = inverse @ np.r_[point, 1.0]
        return None if abs(q[2]) < 1e-12 else q[:2] / q[2]

    # The checks — none of them is forced by the steps above:
    # the centre line meets the net half way across;
    errors = []
    centre = to_court(foot)
    if centre is None:
        return None
    errors.append(abs(centre[0] - 5.0))
    # the service line found runs along the court's, edge to edge;
    for x in (0.0, w - 1.0):
        point = to_court((x, -(service[0] * x + service[2]) / service[1]))
        if point is None:
            return None
        errors.append(abs(point[1] - _NEAR_SERVICE))
    # the other side wall, when found, agrees.
    if len(sides) == 2:
        other = list(sides.values())[1]
        for y in (_NEAR_SERVICE, NET_Y_M - 1.0):
            p = homography @ np.array([10.0 if "right" in sides else 0.0, y, 1.0])
            errors.append(_distance(other, p[:2] / p[2]) / max(per_metre_at(homography, y), 1e-9))
    line_error = float(max(errors))
    # The net's height in the picture against the width scale there.
    per_metre = float(np.linalg.norm(np.subtract(posts[1], posts[0]))) / 10.0
    net_scale = (foot_row - tape_at_centre[1]) / max(per_metre * NET_HEIGHT_CENTER_M, 1e-9)
    if line_error > MAX_LINE_ERROR_M or not NET_SCALE_RANGE[0] <= net_scale <= NET_SCALE_RANGE[1]:
        return None

    handles = []
    for x, y in REFERENCE_POINTS_M["net_service"]:
        p = homography @ np.array([x, y, 1.0])
        handles.append(p[:2] / p[2])
    handles = np.array(handles)
    if not np.isfinite(handles).all():
        return None
    confidence = float(np.clip(1.0 - line_error / MAX_LINE_ERROR_M, 0.0, 1.0))
    return NetServiceProposal(handles, round(confidence, 2), round(line_error, 2))


# ── Steps ────────────────────────────────────────────────────────────────────

def _evidence(image: np.ndarray) -> np.ndarray:
    """Thin bright structures: white top-hat of the darkest colour channel
    (a white line is bright in all three, the turf is not)."""
    darkest = image.min(axis=2) if image.ndim == 3 else image
    size = max(9, int(round(image.shape[0] * 0.025)) | 1)
    tophat = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    tophat = cv2.GaussianBlur(tophat.astype(np.float32), (0, 0), 1.0)
    return tophat / max(float(np.percentile(tophat, 99.5)), 1.0)


def _segments(evidence: np.ndarray, w: int, h: int):
    binary = (evidence > LINE_THRESHOLD).astype(np.uint8) * 255
    found = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=40,
                            minLineLength=int(MIN_SEGMENT * w), maxLineGap=8)
    horizontal, vertical = [], []
    for x1, y1, x2, y2 in ([] if found is None else found[:, 0, :].astype(np.float64)):
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) < 0.25 * abs(dx):
            horizontal.append((x1, y1, x2, y2))
        elif abs(dx) < 0.35 * abs(dy):
            vertical.append((x1, y1, x2, y2))
    return horizontal, vertical


def _centre_line(vertical, w: int, h: int):
    """The centre service line: the near-vertical white segments in the
    middle of the picture, grouped by where they cross the lower half; the
    group with the most length. (a, b, top row, bottom row), x = a*y + b."""
    middle = [s for s in vertical if 0.2 * w < (s[0] + s[2]) / 2 < 0.8 * w and max(s[1], s[3]) > 0.4 * h]
    if not middle:
        return None
    row = 0.7 * h

    def x_at(s, y):
        x1, y1, x2, y2 = s
        return x1 + (x2 - x1) * (y - y1) / (y2 - y1)

    groups: list[list] = []
    for s in sorted(middle, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        for g in groups:
            if abs(x_at(s, row) - x_at(g[0], row)) < 0.02 * w:
                g.append(s)
                break
        else:
            groups.append([s])
    best = max(groups, key=lambda g: sum(np.hypot(s[2] - s[0], s[3] - s[1]) for s in g))
    ys = np.array([y for s in best for y in (s[1], s[3])])
    xs = np.array([x for s in best for x in (s[0], s[2])])
    a, b = np.polyfit(ys, xs, 1)
    return float(a), float(b), float(ys.min()), float(ys.max())


def _fit(segments) -> np.ndarray:
    """Homogeneous line (a, b, c), a*x + b*y + c = 0, through the segments'
    points, longer segments weighing more."""
    points = []
    for x1, y1, x2, y2 in segments:
        n = max(int(np.hypot(x2 - x1, y2 - y1) / 10), 2)
        points.extend(zip(np.linspace(x1, x2, n), np.linspace(y1, y2, n)))
    vx, vy, x0, y0 = cv2.fitLine(np.array(points, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    line = np.array([vy, -vx, vx * y0 - vy * x0], dtype=np.float64)
    return line / np.hypot(line[0], line[1])


def _distance(line: np.ndarray, point) -> float:
    return abs(line[0] * point[0] + line[1] * point[1] + line[2])


def _service_line(segments, a: float, b: float, bottom: float, h: int) -> np.ndarray | None:
    """The near service line: of the near-horizontal white lines crossing
    the centre line close to its lower end (the T), the one with the most
    length — the T's foot is the service line's lower edge, a few pixels
    below its middle."""
    centre = np.array([1.0, -a, -b])
    crossings = []
    for s in segments:
        p = _intersect(_fit([s]), centre)
        if p is not None and bottom - 0.05 * h <= p[1] <= bottom + ON_LINE * h:
            crossings.append((p[1], s))
    if not crossings:
        return None
    groups: list[list] = []
    for y, s in sorted(crossings, key=lambda c: c[0]):
        if groups and y - groups[-1][-1][0] < ON_LINE * h:
            groups[-1].append((y, s))
        else:
            groups.append([(y, s)])
    best = max(groups, key=lambda g: sum(abs(s[2] - s[0]) for _, s in g))
    return _fit([s for _, s in best])


def _tape(horizontal, a: float, b: float, t, w: int, h: int):
    """The longest near-horizontal white line crossing the centre line above
    the T: its line and its ends' columns."""
    candidates = []
    for s in horizontal:
        x1, y1, x2, y2 = s
        y_mid = (y1 + y2) / 2
        if y_mid >= t[1] - 0.04 * h:
            continue
        cx = a * y_mid + b
        if not min(x1, x2) - 0.02 * w <= cx <= max(x1, x2) + 0.02 * w:
            continue
        candidates.append(s)
    if not candidates:
        return None
    longest = max(candidates, key=lambda s: abs(s[2] - s[0]))
    line = _fit([longest])
    group = [s for s in candidates if max(_distance(line, s[:2]), _distance(line, s[2:])) < 0.01 * h]
    line = _fit(group)
    x_left = min(min(s[0], s[2]) for s in group)
    x_right = max(max(s[0], s[2]) for s in group)
    if x_right - x_left < MIN_TAPE * w:
        return None
    return line, float(x_left), float(x_right)


def _cord_row(image: np.ndarray, a: float, b: float, tape_row: float, t_row: float,
              expected: float) -> float | None:
    """Row of the net's bottom cord on the centre line: the darkest dip of
    the line where the net's foot is expected, `expected` rows below the
    tape give or take — the dense mesh right under the tape is dark too."""
    darkest = image.min(axis=2).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    rows = np.arange(int(tape_row + NET_SCALE_RANGE[0] * expected),
                     int(min(tape_row + NET_SCALE_RANGE[1] * expected, t_row - 4)))
    if len(rows) < 8:
        return None
    columns = a * rows + b
    profile = np.array([
        darkest[r, max(int(round(c)) - 1, 0): int(round(c)) + 2].max() for r, c in zip(rows, columns)
    ])
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 3), 0).ravel()
    level = float(np.median(profile))
    i = int(np.argmin(profile))
    if level <= 0 or (level - profile[i]) / level < MIN_DIP:
        return None
    return float(rows[i])


def per_metre_at(homography: np.ndarray, y: float) -> float:
    """Pixels per metre across the court at court row y."""
    a = homography @ np.array([4.5, y, 1.0])
    b = homography @ np.array([5.5, y, 1.0])
    return float(np.linalg.norm(a[:2] / a[2] - b[:2] / b[2]))


# The turf: this close to the colour around the T (CIELAB distance).
TURF_DISTANCE = 28.0
# Edge points this close to a side wall line belong to it, in pixels.
SIDE_TOLERANCE_PX = 3.0
SIDE_MIN_POINTS = 4
# A side wall's edge meets the net's foot this close to the tape's end, as
# a share of the tape's length: a little inside it, or further out — the
# tape is found shorter when something crosses it near a post, never longer.
POST_INSIDE = 0.03
POST_OUTSIDE = 0.15


def _side_lines(image: np.ndarray, posts, foot, t, base: np.ndarray) -> dict[str, np.ndarray]:
    """The side walls' foot as lines through the posts, where the turf's
    edge shows it: {"left": line, "right": line}, either may be missing."""
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    row = int((foot[1] + t[1]) / 2)
    col = int(round((foot[0] + t[0]) / 2))
    patch = lab[max(row - 10, 0):row + 10, max(col - 25, 0):col + 25].reshape(-1, 3)
    turf = (np.linalg.norm(lab - np.median(patch, axis=0), axis=2) < TURF_DISTANCE).astype(np.uint8)
    # Painted lines and glass joints are thin: they do not end the turf.
    turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((1, 11), np.uint8))
    turf = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((11, 1), np.uint8))

    found: dict[str, np.ndarray] = {}
    for name, post, step in (("left", posts[0], -1), ("right", posts[1], 1)):
        points = []
        for r in range(int(post[1]) + 6, h, 6):
            x, run = int(round(foot[0] + (r - foot[1]) * (t[0] - foot[0]) / max(t[1] - foot[1], 1e-9))), 0
            while 0 <= x < w:
                run = run + 1 if not turf[r, x] else 0
                if run >= 6:
                    points.append((x - step * 5.0, float(r)))
                    break
                x += step
        line = _edge_line(np.array(points, np.float64))
        if line is None:
            continue
        meets = _intersect(line, base)
        span = float(np.linalg.norm(np.subtract(posts[1], posts[0])))
        if meets is None:
            continue
        outward = (meets[0] - post[0]) * step         # beyond the tape's end: positive
        if -POST_INSIDE * span <= outward <= POST_OUTSIDE * span:
            found[name] = line
    return found


def _edge_line(points: np.ndarray) -> np.ndarray | None:
    """The line most edge points lie on (RANSAC on pairs of them), refined
    on them; None when too few agree."""
    if len(points) < SIDE_MIN_POINTS:
        return None
    best: np.ndarray | None = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = points[j] - points[i]
            if np.linalg.norm(d) < 8:
                continue
            normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            inliers = points[np.abs((points - points[i]) @ normal) < SIDE_TOLERANCE_PX]
            if best is None or len(inliers) > len(best):
                best = inliers
    if best is None or len(best) < SIDE_MIN_POINTS or np.ptp(best[:, 1]) < 24:
        return None
    vx, vy, x0, y0 = cv2.fitLine(best.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    line = np.array([vy, -vx, vx * y0 - vy * x0], dtype=np.float64)
    line /= np.hypot(line[0], line[1])
    return line


def _intersect(l1, l2):
    p = np.cross(np.asarray(l1, np.float64), np.asarray(l2, np.float64))
    if abs(p[2]) < 1e-12:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Homography from four homogeneous correspondences (points at infinity
    allowed): src -> dst."""
    rows = []
    for (x, y, z), (u, v, s) in zip(src, dst):
        rows.append([0, 0, 0, -s * x, -s * y, -s * z, v * x, v * y, v * z])
        rows.append([s * x, s * y, s * z, 0, 0, 0, -u * x, -u * y, -u * z])
    _, sv, vt = np.linalg.svd(np.asarray(rows))
    if sv[-2] < 1e-9 * sv[0]:
        return None
    return vt[-1].reshape(3, 3)
