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

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ml.artifacts import ArtifactWriter
from app.ml.court import CourtCalibration
from app.ml.detect import PersonDetector
from app.ml.identity import IdentityResult, resolve_players
from app.ml.player_boxes import save_player_boxes
from app.ml.metrics import MetricsInput, compute_metrics
from app.ml.rallies import Rally, detect_rallies
from app.ml.thumbnails import choose_pictures, crop_score, thumbnail
from app.ml.reid import ReidEmbedder
from app.ml.tracking import CourtTracker, Observation, Tracklet
from app.ml.video import FrameSampler, VideoInfo, probe

logger = logging.getLogger("padel.pipeline")

ProgressCallback = Callable[[int, str], None]

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
    track_min_observations: int = 3
    rally_speed_threshold_ms: float = 1.1
    rally_min_duration_s: float = 3.0
    rally_merge_gap_s: float = 1.5
    # Optional: without it identity falls back to kit colour.
    reid_model: str | None = None


@dataclass
class _CropCandidate:
    track_id: int
    score: float
    # Already a finished thumbnail: resized and JPEG-encoded, ~8 KB. Keeping
    # one per track costs ~10-20 MB on a fragmented match; keeping raw crops
    # did not fit, which is why there used to be a cap of 32 for the whole
    # match — and it left the players far from the camera without a picture.
    jpeg: bytes = field(repr=False)
    # The detection the crop was cut from: a tracklet split by identity can
    # belong to two players, so the crop goes to whoever owns this sample.
    observation: Observation | None = field(default=None, repr=False)


class AnalysisPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        video_path: str | Path,
        calibration: CourtCalibration,
        progress: ProgressCallback | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> dict:
        """Analyse a video.

        `artifacts_dir`, when given, receives the raw per-frame observations
        plus the run's metadata. Inference is the only expensive stage, so
        keeping its output makes every downstream threshold re-tunable in
        under a second (see scripts/retune.py) and lets the debug overlay be
        rendered without running the detector again.
        """
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
        embedder = self._embedder()

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
            min_observations=self.config.track_min_observations,
            embedder=embedder,
        )
        roi = calibration.roi_px()

        # ── 2. Single decode pass ────────────────────────────────────────────
        report(3, "Analisi dei frame…")
        expected = sampler.expected_samples
        crops: dict[int, _CropCandidate] = {}
        frames_sampled = 0
        last_timestamp = 0.0
        started = time.monotonic()

        writer: ArtifactWriter | None = None
        try:
            if artifacts_dir is not None:
                writer = ArtifactWriter(artifacts_dir).__enter__()
                # Scritto subito, prima di qualunque osservazione: chi legge la
                # cartella mentre l'analisi gira trova un meta.json coerente
                # con il tracks.jsonl in corso, marcato come incompleto.
                writer.write_meta(self._meta(video_path, info, calibration, sampler))

            for frame in sampler:
                detections = detector.detect(frame.image, roi=roi)
                tracked = tracker.update(
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    detections=detections,
                    frame=frame.image,
                )
                _collect_crops(crops, tracked, frame.image)
                if writer is not None:
                    writer.write_observations(tracked)

                frames_sampled += 1
                last_timestamp = frame.timestamp_s

                if frames_sampled % 40 == 0:
                    report(
                        _pass_progress(frames_sampled, expected),
                        _pass_message(frames_sampled, expected, started, last_timestamp),
                    )

            tracklets = tracker.finish()
            analysed_s = max(last_timestamp, 0.0)

            if writer is not None:
                # Written before the downstream stages run, so a crash in
                # identity or metrics still leaves a replayable artifact set.
                writer.write_meta(
                    self._meta(
                        video_path=video_path,
                        info=info,
                        calibration=calibration,
                        sampler=sampler,
                        frames_sampled=frames_sampled,
                        analysed_s=analysed_s,
                        inference_seconds=time.monotonic() - started,
                    )
                )
        finally:
            if writer is not None:
                writer.__exit__(None, None, None)

        report(88, f"{len(tracklets)} tracce raccolte su {frames_sampled} frame")

        if not tracklets:
            raise RuntimeError(
                "Nessun giocatore rilevato. Verifica che la calibrazione del campo "
                "corrisponda al video e che i giocatori siano visibili."
            )

        # ── 3-5. Identity, rallies, metrics ──────────────────────────────────
        report(89, f"Ricostruzione identità da {len(tracklets)} tracce…")

        def identity_progress(done: int, total: int) -> None:
            # 89 -> 92 while tracklets are being linked. Without this the bar
            # sits at 89% for the whole stage, which on a heavily fragmented
            # match is indistinguishable from a hang.
            share = done / total if total else 0.0
            report(89 + int(3 * min(share, 1.0)), f"Ricostruzione identità · {done}/{total} tracce unite")

        result, identity, rallies = analyse_tracklets(
            tracklets=tracklets,
            calibration=calibration,
            config=self.config,
            sample_hz=sampler.effective_hz,
            frames_sampled=frames_sampled,
            analysed_s=analysed_s,
            identity_progress=identity_progress,
        )
        report(95, f"{len(identity.players)} giocatori · {len(rallies)} scambi")

        if info.frame_count and analysed_s + 1.0 < info.duration_s:
            result["data_quality"]["warnings"].append(
                f"Analizzati i primi {analysed_s / 60:.0f} minuti su "
                f"{info.duration_s / 60:.0f} (limite MAX_ANALYSIS_MINUTES)."
            )

        # ── 6. Thumbnails ────────────────────────────────────────────────────
        report(99, "Estrazione anteprime giocatori…")
        result["player_crops_data"] = _encode_player_crops(identity.players, crops)

        if artifacts_dir is not None:
            ArtifactWriter(artifacts_dir).write_result(result)
            try:
                save_player_boxes(artifacts_dir, identity, sampler.effective_hz,
                                  {"width": info.width, "height": info.height}, result["per_player"])
            except Exception as exc:              # noqa: BLE001
                # Only the web player's overlay needs it, and the API can
                # rebuild it: never fail an hour of analysis over it.
                logger.warning("Riquadri per il video non salvati: %s", exc)

        report(100, "Analisi completata")
        return result

    def _embedder(self) -> ReidEmbedder | None:
        """The re-ID model if installed. Its absence is not an error — the
        analysis runs as before, telling players apart by kit colour only —
        but it is logged, and recorded in meta.json, so a weak identity result
        can be traced to it."""
        if not self.config.reid_model:
            return None
        path = Path(self.config.reid_model)
        if not path.exists():
            logger.warning(
                "Modello re-ID assente (%s): identità dal solo colore della divisa. "
                "Esportalo con scripts/export_reid_onnx.py.", path,
            )
            return None
        return ReidEmbedder(path, threads=self.config.inference_threads)

    def _meta(
        self,
        video_path: str | Path,
        info: VideoInfo,
        calibration: CourtCalibration,
        sampler: FrameSampler,
        frames_sampled: int | None = None,
        analysed_s: float | None = None,
        inference_seconds: float | None = None,
    ) -> dict:
        """Run metadata. Called twice: once when the decode pass starts, with
        only what is already known, and once when it ends with the totals."""
        return {
            # False until the decode pass finishes. A reader that ignores this
            # would pair a previous run's totals with a partial track file.
            "complete": frames_sampled is not None,
            "video": {
                "name": Path(video_path).name,
                "fps": round(info.fps, 3),
                "frame_count": info.frame_count,
                "width": info.width,
                "height": info.height,
                "duration_s": round(info.duration_s, 2),
            },
            "calibration": calibration.to_dict(),
            "sample_hz": sampler.effective_hz,
            "sample_step": sampler.step,
            "frames_sampled": frames_sampled,
            "analysed_s": round(analysed_s, 2) if analysed_s is not None else None,
            "min_observations": self.config.track_min_observations,
            "inference_seconds": (
                round(inference_seconds, 1) if inference_seconds is not None else None
            ),
            "config": {
                "detector_model": Path(self.config.detector_model).name,
                "detector_imgsz": self.config.detector_imgsz,
                "detector_conf": self.config.detector_conf,
                "detector_iou": self.config.detector_iou,
                # Which appearance cue identity had: the re-ID model, or None
                # for kit colour only.
                "reid_model": (
                    Path(self.config.reid_model).name
                    if self.config.reid_model and Path(self.config.reid_model).exists()
                    else None
                ),
                "max_player_speed_ms": self.config.max_player_speed_ms,
                "track_max_age_s": self.config.track_max_age_s,
                "rally_speed_threshold_ms": self.config.rally_speed_threshold_ms,
                "rally_min_duration_s": self.config.rally_min_duration_s,
                "rally_merge_gap_s": self.config.rally_merge_gap_s,
            },
        }


def analyse_tracklets(
    tracklets: list[Tracklet],
    calibration: CourtCalibration,
    config: PipelineConfig,
    sample_hz: float,
    frames_sampled: int,
    analysed_s: float,
    identity_progress: Callable[[int, int], None] | None = None,
) -> tuple[dict, IdentityResult, list[Rally]]:
    """Everything after tracking: identity, rallies, metrics.

    Split out of `run` so that replaying stored artifacts exercises exactly
    the production code path. A tuning script that reimplemented these steps
    would drift from the pipeline and measure the wrong system.
    """
    identity = resolve_players(
        tracklets, max_speed_ms=config.max_player_speed_ms, progress=identity_progress
    )
    if not identity.players:
        raise RuntimeError("Impossibile ricostruire i giocatori dalle tracce rilevate.")

    rallies = detect_rallies(
        identity.players,
        sample_hz=sample_hz,
        speed_threshold_ms=config.rally_speed_threshold_ms,
        min_duration_s=config.rally_min_duration_s,
        merge_gap_s=config.rally_merge_gap_s,
        max_speed_ms=config.max_player_speed_ms,
    )

    result = compute_metrics(
        MetricsInput(
            players=identity.players,
            rallies=rallies,
            identity=identity,
            calibration=calibration,
            sample_hz=sample_hz,
            frames_sampled=frames_sampled,
            analysed_s=analysed_s,
            max_speed_ms=config.max_player_speed_ms,
            detector_model=Path(config.detector_model).name,
        )
    )
    return result, identity, rallies


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
    """Keep the single best-looking crop of every track while the frame is hot.

    The comparison is within a track only, so a player always far from the
    camera, whose boxes are small, still gets their best picture. Which
    track's crop becomes the player's picture is decided after identity
    (see thumbnails.choose_pictures).
    """
    for track_id, obs in tracked:
        score = crop_score(obs)
        if score is None:
            continue
        existing = crops.get(track_id)
        if existing is not None and existing.score >= score:
            continue
        jpeg = thumbnail(frame, obs.bbox)
        if jpeg is not None:
            crops[track_id] = _CropCandidate(track_id, score, jpeg, obs)


def _encode_player_crops(players, crops: dict[int, _CropCandidate]) -> dict[int, bytes]:
    """Each player's picture, among the crops of the tracks they were built
    from: a sample typical of the player and alone in the frame, rather
    than simply the best-looking one."""
    owned: dict[int, list[_CropCandidate]] = {}
    for player in players:
        owned[player.player_id] = [
            crops[tid] for tid in dict.fromkeys(player.source_tracklets)
            if tid in crops and crops[tid].observation is not None and player.owns(crops[tid].observation)
        ]
    chosen = choose_pictures(players, {pid: [c.observation for c in cands] for pid, cands in owned.items()})
    out: dict[int, bytes] = {}
    for player_id, obs in chosen.items():
        for candidate in owned[player_id]:
            if candidate.observation is obs:
                out[player_id] = candidate.jpeg
                break
    return out
