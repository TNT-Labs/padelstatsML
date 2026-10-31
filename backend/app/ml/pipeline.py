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

from app.ml.accel import AcceleratorError, accelerator_mode, hailo_status, pose_models
from app.ml.artifacts import ArtifactWriter
from app.ml.court import CourtCalibration
from app.ml.detect import DetectorError, PersonDetector, merge_detections, within_crop
from app.ml.detect_hailo import HailoPersonDetector, HailoPoseEstimator, HailoRuntimeError
from app.ml.footpoint import FOOT_MODEL_VERSION
from app.ml.identity import IdentityResult, resolve_players
from app.ml.player_boxes import save_player_boxes
from app.ml.metrics import MetricsInput, compute_metrics
from app.ml.smoothing import smooth_path
from app.ml.strokes import detect_strokes
from app.ml.rallies import Rally, activity_threshold, detect_rallies
from app.ml.thumbnails import choose_pictures, crop_score, thumbnail
from app.ml.reid import ReidEmbedder
from app.ml.tracking import CourtTracker, Observation, Tracklet
from app.ml.video import FrameSampler, VideoInfo, probe, video_background

logger = logging.getLogger("padel.pipeline")

# Automatic second look at the far half: when a player on the far baseline
# reaches the network shorter than this. Below it the nano model misses most
# of them: on a real match, low camera, 22 px tall, they came out at
# confidence 0.05; in the second look at 0.45.
FAR_PASS_AUTO_PX = 32.0

# Set when the accelerator failed in the middle of an analysis: the worker's
# next analyses run on the CPU until it is restarted, instead of failing on
# it again.
_ACCELERATOR_FAILED: str | None = None

ProgressCallback = Callable[[int, str], None]

@dataclass
class PipelineConfig:
    detector_model: str
    detector_imgsz: int = 480
    detector_conf: float = 0.30
    # Below detector_conf, down to this: only to extend a live track (see
    # CourtTracker.confirm_conf).
    detector_conf_weak: float = 0.12
    detector_iou: float = 0.55
    inference_threads: int = 0
    sample_hz: float = 5.0
    max_analysis_minutes: int = 120
    max_player_speed_ms: float = 8.0
    track_max_age_s: float = 1.2
    track_min_observations: int = 3
    rally_speed_threshold_ms: float = 1.1
    # True: the match chooses its threshold, at most rally_speed_threshold_ms
    # (rallies.activity_threshold). False: exactly rally_speed_threshold_ms.
    rally_speed_adaptive: bool = True
    rally_min_duration_s: float = 3.0
    rally_merge_gap_s: float = 3.0
    # A second look of the detector at the far half alone, where players are
    # smallest: more of them found, at the cost of a second inference per
    # sampled frame (see CourtCalibration.far_roi_px).
    # None: automatic, when the far players are too small for the network.
    detector_far_pass: bool | None = False
    # Optional: without it identity falls back to kit colour.
    reid_model: str | None = None
    # Where the detector runs: "auto" (the Hailo accelerator when usable),
    # "cpu" or "hailo" (see app/ml/accel.py), and the accelerator's network.
    accelerator: str = "cpu"
    hailo_model: str | None = None
    # The players' posture on the accelerator ("auto" / "off"), and a pose
    # network of one's own (see app/ml/pose.py).
    pose: str = "off"
    hailo_pose_model: str | None = None
    # Posture rate (per second) between the samples too, stored in
    # poses.jsonl: 0, only at the samples (see Settings.pose_hz).
    pose_hz: float = 0.0


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
        start_s: float = 0.0,
    ) -> dict:
        """Analyse a video, from `start_s` seconds in.

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
        detector, accelerator_note = self._detector()
        poser, pose_note = self._poser(detector)
        posed_frames = 0
        embedder = self._embedder()

        max_duration_s = self.config.max_analysis_minutes * 60.0
        sampler = FrameSampler(
            video_path,
            target_hz=self.config.sample_hz,
            info=info,
            max_duration_s=max_duration_s,
            start_s=start_s,
            # The posture between the samples too, when asked (POSE_HZ),
            # stored for a model of strokes: the samples' own analysis is
            # unchanged.
            pose_hz=self.config.pose_hz if poser is not None else 0.0,
        )
        if sampler.start_frame:
            report(2, f"Salto al minuto {_clock(sampler.start_s)}, inizio dell'analisi…")
        # The empty court: detections that are part of it (a banner with a
        # figure, a post) are scenery, from the first frame on.
        scenery = video_background(video_path, start_s=start_s)
        tracker = CourtTracker(
            calibration=calibration,
            scenery=scenery,
            max_speed_ms=self.config.max_player_speed_ms,
            max_age_s=self.config.track_max_age_s,
            min_observations=self.config.track_min_observations,
            embedder=embedder,
            confirm_conf=self.config.detector_conf,
        )
        roi = calibration.roi_px()
        far_roi = calibration.far_roi_px() if self.config.detector_far_pass is not False else None
        if (self.config.detector_far_pass is None and far_roi is not None
                and calibration.far_player_input_px(max(getattr(detector, "imgsz", (480, 480)))) >= FAR_PASS_AUTO_PX):
            far_roi = None
        far_added = 0

        # ── 2. Single decode pass ────────────────────────────────────────────
        report(3, "Analisi dei frame…")
        expected = sampler.expected_samples
        crops: dict[int, _CropCandidate] = {}
        frames_sampled = 0
        last_timestamp = sampler.start_s
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
                if not frame.main:
                    if poser is not None:
                        boxes = tracker.current_boxes(frame.timestamp_s)
                        try:
                            points = poser.points_for(frame.image, [box for _, box in boxes])
                        except HailoRuntimeError as exc:
                            logger.error("%s · proseguo senza postura", exc)
                            _close(poser)
                            poser = None
                            pose_note = (f"La postura si è interrotta al minuto {_clock(frame.timestamp_s)}: "
                                         "da lì in poi non è stata stimata.")
                            continue
                        found = [(track_id, p) for (track_id, _), p in zip(boxes, points) if p is not None]
                        if writer is not None:
                            writer.write_poses(frame.index, frame.timestamp_s, found)
                    continue
                try:
                    detections, added = _look(detector, frame.image, roi, far_roi)
                except HailoRuntimeError as exc:
                    # An hour of analysis is not thrown away for it: the rest
                    # of the video on the CPU, and the note says from when.
                    _accelerator_failed(str(exc))
                    logger.error("%s · proseguo sul processore", exc)
                    _close(poser)
                    poser = None
                    _close(detector)
                    detector = self._cpu_detector()
                    accelerator_note = (
                        f"L'acceleratore si è bloccato al minuto {_clock(frame.timestamp_s)}: "
                        "il resto del video è stato analizzato sul processore.")
                    detections, added = _look(detector, frame.image, roi, far_roi)
                far_added += added
                if poser is not None:
                    try:
                        posed = poser.estimate_for(frame.image, detections)
                    except HailoRuntimeError as exc:
                        logger.error("%s · proseguo senza postura", exc)
                        _close(poser)
                        poser = None
                        pose_note = (f"La postura si è interrotta al minuto {_clock(frame.timestamp_s)}: "
                                     "da lì in poi non è stata stimata.")
                    else:
                        posed_frames += any(d.keypoints is not None for d in posed)
                        detections = posed
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
                        _pass_message(frames_sampled, expected, started, last_timestamp - sampler.start_s),
                    )

            tracklets = tracker.finish()
            # How much of the video was analysed: from the start, not from 0.
            analysed_s = max(last_timestamp - sampler.start_s, 0.0)
            run_quality = {
                # What ran the detector: the accelerator's board and network,
                # or the CPU model; and why not the accelerator, if it should
                # have been.
                "detector": _detector_name(detector) or Path(self.config.detector_model).name,
                "accelerator_note": accelerator_note,
                # The players' posture: the network, and the frames where
                # someone's was found.
                "pose": ({"model": poser.name if poser is not None else None, "frames": posed_frames,
                          "dense_hz": round(sampler.info.fps / sampler.pose_step, 1) if sampler.pose_step else None}
                         if posed_frames or poser is not None else None),
                "pose_note": pose_note,
                "variable_frame_rate": sampler.variable_frame_rate,
                # Detections recognised as part of the empty court and left out.
                "scenery": tracker.detections_rejected_scenery,
                # How much the second look added: tells whether it is worth its time.
                "far_pass": ({"active": far_roi is not None, "added": far_added}
                             if self.config.detector_far_pass is not False else None),
                "truncated": (
                    f"Analizzati {analysed_s / 60:.0f} minuti su "
                    f"{(info.duration_s - sampler.start_s) / 60:.0f} (limite MAX_ANALYSIS_MINUTES)."
                    if info.frame_count and sampler.start_s + analysed_s + 1.0 < info.duration_s else None
                ),
            }

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
                        run_quality=run_quality,
                    )
                )
        finally:
            if writer is not None:
                writer.__exit__(None, None, None)
            _close(poser)
            _close(detector)

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
            start_s=sampler.start_s,
        )
        report(95, f"{len(identity.players)} giocatori · {len(rallies)} scambi")

        apply_run_quality(result, run_quality)

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

    def _detector(self):
        """(detector, note): on the accelerator when usable, on the CPU
        otherwise; the note says why the accelerator was not used when it is
        there but could not be."""
        mode = accelerator_mode(self.config.accelerator)
        status = hailo_status(mode, self.config.hailo_model)
        if mode == "hailo" and not status.usable:
            raise AcceleratorError(f"ACCELERATOR=hailo ma l'acceleratore è {status.reason}.")
        if mode == "auto" and _ACCELERATOR_FAILED:
            return self._cpu_detector(), (
                f"Acceleratore non usato: si era bloccato in un'analisi precedente ({_ACCELERATOR_FAILED}). "
                "Riavvia il worker per riprovarlo.")
        if not status.usable:
            # Not fitted, or turned off: the CPU, as always, without a note.
            quiet = status.device is None
            return self._cpu_detector(), None if quiet else f"Acceleratore {status.reason}: analisi sul processore."
        try:
            detector = HailoPersonDetector(
                status.model,
                alternatives=status.models,
                conf_threshold=min(self.config.detector_conf, self.config.detector_conf_weak),
                iou_threshold=self.config.detector_iou,
            )
        except DetectorError as exc:
            if mode == "hailo":
                raise AcceleratorError(str(exc)) from exc
            logger.warning("%s · analisi sul processore", exc)
            return self._cpu_detector(), f"{exc}: analisi sul processore."
        logger.info("Rilevatore sull'acceleratore: %s", detector.name)
        return detector, None

    def _poser(self, detector):
        """(pose estimator, note): only beside a detector on the accelerator,
        sharing its board; None with the reason when it cannot be."""
        if self.config.pose == "off" or not getattr(detector, "on_accelerator", False):
            return None, None
        models = pose_models(self.config.hailo_pose_model)
        if not models:
            return None, None
        try:
            poser = HailoPoseEstimator(models[0], alternatives=models)
        except DetectorError as exc:
            logger.warning("%s", exc)
            return None, f"{exc}: analisi senza postura."
        logger.info("Postura sull'acceleratore: %s", poser.name)
        return poser, None

    def _cpu_detector(self) -> PersonDetector:
        return PersonDetector(
            model_path=self.config.detector_model,
            imgsz=self.config.detector_imgsz,
            conf_threshold=min(self.config.detector_conf, self.config.detector_conf_weak),
            iou_threshold=self.config.detector_iou,
            threads=self.config.inference_threads,
        )

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
        run_quality: dict | None = None,
    ) -> dict:
        """Run metadata. Called twice: once when the decode pass starts, with
        only what is already known, and once when it ends with the totals."""
        return {
            # What only the decode pass knows, for the reliability panel: kept
            # here so a recomputation (refresh, verification) reports it too.
            "run_quality": run_quality,
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
            "start_s": round(sampler.start_s, 3),
            # Timestamps are the video's own; a variable rate is recorded so a
            # drift seen in an older run can be explained.
            "variable_frame_rate": sampler.variable_frame_rate,
            "frames_sampled": frames_sampled,
            "analysed_s": round(analysed_s, 2) if analysed_s is not None else None,
            "min_observations": self.config.track_min_observations,
            # Foot points of boxes cut by the net already corrected in
            # tracks.jsonl (app.ml.footpoint); older runs are corrected on load.
            "foot_model": FOOT_MODEL_VERSION,
            "inference_seconds": (
                round(inference_seconds, 1) if inference_seconds is not None else None
            ),
            "config": {
                "detector_model": Path(self.config.detector_model).name,
                "detector_imgsz": self.config.detector_imgsz,
                "accelerator": accelerator_mode(self.config.accelerator),
                "pose": self.config.pose,
                "detector_conf": self.config.detector_conf,
                "detector_conf_weak": self.config.detector_conf_weak,
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
                "rally_speed_adaptive": self.config.rally_speed_adaptive,
                "detector_far_pass": self.config.detector_far_pass,
                "rally_min_duration_s": self.config.rally_min_duration_s,
                "rally_merge_gap_s": self.config.rally_merge_gap_s,
            },
        }


def apply_run_quality(result: dict, run_quality: dict | None) -> None:
    """Put into the result's data_quality what only the decode pass knows
    (see AnalysisPipeline._meta): the stages after it cannot measure it."""
    if not run_quality:
        return
    quality = result["data_quality"]
    quality["variable_frame_rate"] = bool(run_quality.get("variable_frame_rate"))
    quality.setdefault("tracking", {})["scenery"] = int(run_quality.get("scenery") or 0)
    if run_quality.get("far_pass"):
        quality["far_pass"] = run_quality["far_pass"]
    if run_quality.get("truncated"):
        quality["warnings"].append(run_quality["truncated"])
    if run_quality.get("detector"):
        quality["detector_model"] = run_quality["detector"]
    if run_quality.get("pose"):
        quality["pose"] = run_quality["pose"]
    if run_quality.get("pose_note"):
        quality["warnings"].append(run_quality["pose_note"])
    if run_quality.get("accelerator_note"):
        quality["warnings"].append(run_quality["accelerator_note"])


def analyse_tracklets(
    tracklets: list[Tracklet],
    calibration: CourtCalibration,
    config: PipelineConfig,
    sample_hz: float,
    frames_sampled: int,
    analysed_s: float,
    identity_progress: Callable[[int, int], None] | None = None,
    start_s: float = 0.0,
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

    # Each player's path with the detector's jitter filtered out: what
    # distance, speed and the rally boundaries are measured on.
    paths = {p.player_id: smooth_path(p.observations, calibration) for p in identity.players}
    # The configured threshold is the ceiling; the match picks its own below
    # it (see rallies.activity_threshold).
    threshold = activity_threshold(
        identity.players, sample_hz, config.rally_speed_threshold_ms,
        max_speed_ms=config.max_player_speed_ms, paths=paths,
    ) if config.rally_speed_adaptive else config.rally_speed_threshold_ms
    rallies = detect_rallies(
        identity.players,
        sample_hz=sample_hz,
        speed_threshold_ms=threshold,
        min_duration_s=config.rally_min_duration_s,
        merge_gap_s=config.rally_merge_gap_s,
        max_speed_ms=config.max_player_speed_ms,
        paths=paths,
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
            paths=paths,
        )
    )
    # High strokes, where the posture was estimated (app.ml.strokes).
    if any(o.keypoints is not None for p in identity.players for o in p.observations):
        for player_id, strokes in detect_strokes(identity.players, paths, rallies).items():
            if str(player_id) in result["per_player"]:
                result["per_player"][str(player_id)]["strokes"] = strokes
    result["data_quality"]["analysis_start_s"] = round(start_s, 1)
    result["data_quality"]["rally_speed_threshold_ms"] = round(threshold, 2)
    # Counted on the players' samples, so a recomputed run reports it too:
    # weak detections that kept a track alive, and boxes cut by the net whose
    # feet were placed from the rest of the box (below the box's bottom).
    samples = [o for p in identity.players for o in p.observations]
    result["data_quality"]["tracking"] = {
        "weak_detections": sum(o.confidence < config.detector_conf for o in samples),
        "feet_below_net": sum(o.foot_px[1] > o.bbox[3] + 0.5 for o in samples),
    }
    return result, identity, rallies


def _look(detector, image, roi, far_roi) -> tuple[list, int]:
    """The detections of one frame — the court, and the far half closer
    when far_roi is set — and how many the closer look added."""
    detections = detector.detect(image, roi=roi)
    if far_roi is None:
        return detections, 0
    seen = len(detections)
    closer = within_crop(detector.detect(image, roi=far_roi), far_roi, image.shape)
    detections = merge_detections(detections, closer)
    return detections, max(len(detections) - seen, 0)


def _accelerator_failed(reason: str) -> None:
    global _ACCELERATOR_FAILED
    _ACCELERATOR_FAILED = reason


def _detector_name(detector) -> str | None:
    name = getattr(detector, "name", None)
    if name:
        return name
    path = getattr(detector, "model_path", None)
    return Path(path).name if path else None


def _close(detector) -> None:
    if detector is None:
        return
    close = getattr(detector, "close", None)
    if close is not None:
        close()


# ── Progress helpers ─────────────────────────────────────────────────────────

def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


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
