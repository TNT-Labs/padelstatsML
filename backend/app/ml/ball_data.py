"""Ball annotations: the data to train a ball detector on the user's videos.

No ball detector off the shelf follows a padel ball in these videos: a few
pixels across in the far half, a smear at speed, often behind the glass or
a player. A network of the TrackNet family — three consecutive frames in,
a heat map of where the ball is out — learns it from a few thousand frames
marked by hand. This module is the bookkeeping for that:

- clips: short runs of consecutive frames (CLIP_FRAMES, about two seconds)
  chosen inside the rallies of an analysed match, spread over the whole
  match, so that the marked frames show play from every part of it;
- frames: each clip's frames cut out of the video once and kept as JPEG,
  full size, for the person marking the ball to zoom into;
- labels: per frame, where the ball is, in video pixels, or that it is not
  in sight (behind a player, out of the picture);
- export: the labelled clips as a zip for training (frames scaled down to
  EXPORT_WIDTH, labels.csv).

Frames are found by number with OpenCV's seek. On some videos a seek lands
a frame or two off; within a clip the frames are read in order, so a clip
is always consecutive frames, which is what training needs, and the labels
belong to the very images stored here.

Everything lives under DATA_DIR/ball/<match id>/.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# About two seconds of play at 30 fps.
CLIP_FRAMES = 60
# Clips proposed for a match: 40 x 60 = 2400 frames, more than enough for a
# first model; the person marks as many as they like.
CLIPS_PER_MATCH = 40
# Clips start this far into a rally: the serve is the most varied moment,
# but players walking into place before it are not.
INTO_RALLY_S = 0.5
JPEG_QUALITY = 90
EXPORT_WIDTH = 640

NOT_VISIBLE = "none"


@dataclass(frozen=True)
class Clip:
    id: int
    start_frame: int
    frames: int
    start_s: float


def ball_dir(data_dir: Path, match_id: str) -> Path:
    return Path(data_dir) / "ball" / match_id


def choose_clips(rallies: list[dict], fps: float, count: int = CLIPS_PER_MATCH,
                 length: int = CLIP_FRAMES) -> list[Clip]:
    """`count` clips inside the rallies (dicts with start_s, end_s), spread
    over the match: each rally long enough gets clips in proportion to its
    length, placed evenly along it."""
    need_s = length / fps + INTO_RALLY_S
    usable = [r for r in rallies if r["end_s"] - r["start_s"] >= need_s]
    if not usable or fps <= 0:
        return []
    total = sum(r["end_s"] - r["start_s"] for r in usable)
    starts: list[float] = []
    for rally in usable:
        span = rally["end_s"] - rally["start_s"] - need_s
        n = max(1, round(count * (rally["end_s"] - rally["start_s"]) / total))
        for k in range(n):
            t = rally["start_s"] + INTO_RALLY_S + span * (k + 0.5) / n
            starts.append(t)
    # Too many from the rounding: keep an even spread of them.
    if len(starts) > count:
        keep = np.linspace(0, len(starts) - 1, count).round().astype(int)
        starts = [starts[i] for i in keep]
    starts.sort()
    clips: list[Clip] = []
    last_end = -1
    for t in starts:
        start = int(round(t * fps))
        if start <= last_end:                 # overlapping a clip already chosen
            continue
        clips.append(Clip(len(clips), start, length, round(start / fps, 2)))
        last_end = start + length - 1
    return clips


class BallData:
    """The clips, frames and labels of one match."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    # ── Clips ────────────────────────────────────────────────────────────────
    def clips(self) -> list[Clip] | None:
        path = self.directory / "clips.json"
        if not path.exists():
            return None
        return [Clip(**c) for c in json.loads(path.read_text(encoding="utf-8"))]

    def save_clips(self, clips: list[Clip]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "clips.json").write_text(json.dumps([asdict(c) for c in clips]), encoding="utf-8")

    def clip(self, clip_id: int) -> Clip | None:
        return next((c for c in self.clips() or [] if c.id == clip_id), None)

    # ── Frames ───────────────────────────────────────────────────────────────
    def frame_path(self, clip: Clip, offset: int) -> Path:
        return self.directory / "frames" / f"{clip.start_frame:07d}" / f"{offset:03d}.jpg"

    def has_frames(self, clip: Clip) -> bool:
        return self.frame_path(clip, clip.frames - 1).exists()

    def extract(self, clip: Clip, video: Path) -> int:
        """Cut the clip's frames out of `video`; the number written (fewer at
        the end of the video)."""
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise OSError(f"Video non leggibile: {video}")
        written = 0
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
            folder = self.frame_path(clip, 0).parent
            folder.mkdir(parents=True, exist_ok=True)
            for offset in range(clip.frames):
                ok, image = capture.read()
                if not ok or image is None:
                    break
                cv2.imwrite(str(self.frame_path(clip, offset)), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                written += 1
        finally:
            capture.release()
        return written

    # ── Labels ───────────────────────────────────────────────────────────────
    def labels(self) -> dict[str, list[float] | str]:
        """"<frame number>" -> [x, y] in video pixels, or NOT_VISIBLE."""
        path = self.directory / "labels.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def update_labels(self, changes: dict[str, list[float] | str | None]) -> dict:
        labels = self.labels()
        for frame, value in changes.items():
            if value is None:
                labels.pop(frame, None)
            else:
                labels[frame] = value
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / "labels.json.part"
        tmp.write_text(json.dumps(labels, sort_keys=True), encoding="utf-8")
        tmp.replace(self.directory / "labels.json")
        return labels

    # ── Export ───────────────────────────────────────────────────────────────
    def export_zip(self, match_id: str, frame_size: tuple[int, int], target: Path) -> Path:
        """The labelled frames for training: each labelled clip's frames
        scaled to EXPORT_WIDTH, and labels.csv with one row per frame —
        clip, frame number, file, visible, x, y (scaled pixels; empty when
        not visible or not marked)."""
        labels = self.labels()
        width, height = frame_size
        scale = EXPORT_WIDTH / float(width)
        out_h = int(round(height * scale))
        rows = io.StringIO()
        writer = csv.writer(rows)
        writer.writerow(["clip", "frame", "file", "visible", "x", "y"])
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
            for clip in self.clips() or []:
                frames = [clip.start_frame + i for i in range(clip.frames)]
                if not any(str(f) in labels for f in frames) or not self.has_frames(clip):
                    continue
                for offset, frame in enumerate(frames):
                    image = cv2.imread(str(self.frame_path(clip, offset)))
                    if image is None:
                        continue
                    small = cv2.resize(image, (EXPORT_WIDTH, out_h), interpolation=cv2.INTER_AREA)
                    ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    name = f"frames/{match_id[:8]}_{clip.id:03d}/{offset:03d}.jpg"
                    archive.writestr(name, jpeg.tobytes())
                    label = labels.get(str(frame))
                    if isinstance(label, list):
                        writer.writerow([f"{match_id[:8]}_{clip.id:03d}", frame, name, 1,
                                         round(label[0] * scale, 1), round(label[1] * scale, 1)])
                    else:
                        # Not visible, or not marked: unknown frames still
                        # carry the sequence; training leaves them out.
                        writer.writerow([f"{match_id[:8]}_{clip.id:03d}", frame, name,
                                         0 if label == NOT_VISIBLE else "", "", ""])
            archive.writestr("labels.csv", rows.getvalue())
            archive.writestr("meta.json", json.dumps({
                "match": match_id, "source_size": [width, height], "size": [EXPORT_WIDTH, out_h],
                "clip_frames": CLIP_FRAMES,
            }))
        return target
