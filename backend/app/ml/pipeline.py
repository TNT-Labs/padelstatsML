"""Single-pass analysis pipeline.

The whole video is decoded exactly once. Every stage that needs pixels —
detection, colour signatures, player thumbnails — consumes the frame while it
is already in memory. The previous pipeline decoded the file five times, which
on a Pi 5 reading from USB storage cost more than the neural network itself.

Stages:
    1. probe + detector load                 0 ->  3 %
    2. single decode pass (detect + track)   3 -> 88 %
    3. tracklet linking / player identity   88 -> 92 %
    4. rally segmentation                   92 -> 95 %
    5. Tier 1 metrics                       95 -> 99 %
    6. player thumbnails (already buffered) 99 -> 100 %

Only step 2 is expensive; the progress bar is therefore meaningful, and the
message carries a live ETA because a 60-minute match takes about an hour.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.ml.court import CourtCalibration
from app.ml.detect import PersonDetector
from app.ml.identity import resolve_players
from app.ml.metrics import MetricsInput, compute_metrics
from app.ml.rallies import detect_rallies
from app.ml.tracking import CourtTracker, Observation
from app.ml.video import FrameSampler, probe

ProgressCallback = Callable[[int, str], None]

_CROP_W, _CROP_H = 160, 240
_CROP_PAD = 20
# Upper bound on buffered thumbnail candidates. Each is ~115 KB, so this caps
# thumbnail memory at ~4 MB regardless of how fragmented tracking gets.
_MAX_CROP_CANDIDATES = 32


@dataclass
class PipelineConfig:
    detector_model: str
    detector_imgsz: int = 480
    detector_conf: float = 0.30
    detector_iou: float = 0.55
    inference_threads: int = 0
    sample_hz: float = 5.0
    max_analysis_minutes: int = 120
    max_player_speed_ms: float = 8.0
    track_max_age_s: float = 1.2
    rally_speed_threshold_ms: float = 1.1
    rally_min_duration_s: float = 3.0
    rally_merge_gap_s: float = 1.5


@dataclass
class _CropCandidate:
    track_id: int
    score: float
    image: np.ndarray = field(repr=False)


class AnalysisPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        video_path: str | Path,
        calibration: CourtCalibration,
        progress: ProgressCallback | None = None,
    ) -> dict:
        def report(percent: int, message: str) -> None:
            if progress:
                progress(percent, message)

        # ── 1. Setup ─────────────────────────────────────────────────────────
        report(1, "Lettura video…")
        info = probe(video_path)

        report(2, "Caricamento modello…")
        detector = PersonDetector(
            model_path=self.config.detector_model,
            imgsz=self.config.detector_imgsz,
            conf_threshold=self.config.detector_conf,
            iou_threshold=self.config.detector_iou,
            threads=self.config.inference_threads,
        )

        max_duration_s = self.config.max_analysis_minutes * 60.0
        sampler = FrameSampler(
            video_path,
            target_hz=self.config.sample_hz,
            info=info,
            max_duration_s=max_duration_s,
        )
        tracker = CourtTracker(
            calibration=calibration,
            max_speed_ms=self.config.max_player_speed_ms,
            max_age_s=self.config.track_max_age_s,
        )
        roi = calibration.roi_px()

        # ── 2. Single decode pass ────────────────────────────────────────────
        report(3, "Analisi dei frame…")
        expected = sampler.expected_samples
        crops: dict[int, _CropCandidate] = {}
        frames_sampled = 0
        last_timestamp = 0.0
        started = time.monotonic()

        for frame in sampler:
            detections = detector.detect(frame.image, roi=roi)
            tracked = tracker.update(
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                detections=detections,
                frame=frame.image,
            )
            _collect_crops(crops, tracked, frame.image)

            frames_sampled += 1
            last_timestamp = frame.timestamp_s

            if frames_sampled % 40 == 0:
                report(
                    _pass_progress(frames_sampled, expected),
                    _pass_message(frames_sampled, expected, started, last_timestamp),
                )

        tracklets = tracker.finish()
        analysed_s = max(last_timestamp, 0.0)
        report(88, f"{len(tracklets)} tracce raccolte su {frames_sampled} frame")

        if not tracklets:
            raise RuntimeError(
                "Nessun giocatore rilevato. Verifica che la calibrazione del campo "
                "corrisponda al video e che i giocatori siano visibili."
            )

        # ── 3. Identity ──────────────────────────────────────────────────────
        report(89, "Ricostruzione identità giocatori…")
        identity = resolve_players(tracklets, max_speed_ms=self.config.max_player_speed_ms)
        if not identity.players:
            raise RuntimeError(
                "Impossibile ricostruire i giocatori dalle tracce rilevate."
            )
        report(92, f"{len(identity.players)} giocatori identificati")

        # ── 4. Rallies ───────────────────────────────────────────────────────
        report(93, "Segmentazione degli scambi…")
        rallies = detect_rallies(
            identity.players,
            sample_hz=sampler.effective_hz,
            speed_threshold_ms=self.config.rally_speed_threshold_ms,
            min_duration_s=self.config.rally_min_duration_s,
            merge_gap_s=self.config.rally_merge_gap_s,
            max_speed_ms=self.config.max_player_speed_ms,
        )
        report(95, f"{len(rallies)} scambi individuati")

        # ── 5. Metrics ───────────────────────────────────────────────────────
        report(96, "Calcolo statistiche…")
        result = compute_metrics(
            MetricsInput(
                players=identity.players,
                rallies=rallies,
                identity=identity,
                calibration=calibration,
                sample_hz=sampler.effective_hz,
                frames_sampled=frames_sampled,
                analysed_s=analysed_s,
                max_speed_ms=self.config.max_player_speed_ms,
                detector_model=Path(self.config.detector_model).name,
            )
        )

        if info.frame_count and analysed_s + 1.0 < info.duration_s:
            result["data_quality"]["warnings"].append(
                f"Analizzati i primi {analysed_s / 60:.0f} minuti su "
                f"{info.duration_s / 60:.0f} (limite MAX_ANALYSIS_MINUTES)."
            )

        # ── 6. Thumbnails ────────────────────────────────────────────────────
        report(99, "Estrazione anteprime giocatori…")
        result["player_crops_data"] = _encode_player_crops(identity.players, crops)

        report(100, "Analisi completata")
        return result


# ── Progress helpers ─────────────────────────────────────────────────────────

def _pass_progress(done: int, expected: int) -> int:
    if expected <= 0:
        return 45          # unknown length: park in the middle rather than lie
    return int(3 + 85 * min(1.0, done / expected))


def _pass_message(done: int, expected: int, started: float, video_s: float) -> str:
    elapsed = time.monotonic() - started
    base = f"Analisi frame · {video_s / 60:.0f} min di video elaborati"
    if expected <= 0 or done < 20 or elapsed < 5:
        return base
    remaining = (elapsed / done) * max(expected - done, 0)
    return f"{base} · {remaining / 60:.0f} min rimanenti"


# ── Thumbnails ───────────────────────────────────────────────────────────────

def _collect_crops(
    crops: dict[int, _CropCandidate],
    tracked: list[tuple[int, Observation]],
    frame: np.ndarray,
) -> None:
    """Keep the single best-looking crop per track while the frame is hot.

    Score favours confident detections of a large, upright box: those are the
    unoccluded frames, which is exactly what makes a usable thumbnail.
    """
    height, width = frame.shape[:2]
    for track_id, obs in tracked:
        x1, y1, x2, y2 = obs.bbox
        box_h, box_w = y2 - y1, x2 - x1
        if box_h < 40 or box_w < 12:
            continue
        aspect = box_h / max(box_w, 1e-3)
        if not (1.2 <= aspect <= 5.0):
            continue
        score = obs.confidence * min(box_h / 200.0, 1.0)

        existing = crops.get(track_id)
        if existing is not None and existing.score >= score:
            continue
        if existing is None and len(crops) >= _MAX_CROP_CANDIDATES:
            weakest = min(crops.values(), key=lambda c: c.score)
            if weakest.score >= score:
                continue
            crops.pop(weakest.track_id, None)

        cx1 = int(np.clip(x1 - _CROP_PAD, 0, width - 1))
        cy1 = int(np.clip(y1 - _CROP_PAD, 0, height - 1))
        cx2 = int(np.clip(x2 + _CROP_PAD, 1, width))
        cy2 = int(np.clip(y2 + _CROP_PAD, 1, height))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue
        crops[track_id] = _CropCandidate(track_id, score, frame[cy1:cy2, cx1:cx2].copy())


def _encode_player_crops(players, crops: dict[int, _CropCandidate]) -> dict[int, bytes]:
    """Pick the best buffered crop for each player and JPEG-encode it."""
    out: dict[int, bytes] = {}
    for player in players:
        candidates = [crops[tid] for tid in player.source_tracklets if tid in crops]
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c.score)
        resized = cv2.resize(best.image, (_CROP_W, _CROP_H), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out[player.player_id] = buffer.tobytes()
    return out
