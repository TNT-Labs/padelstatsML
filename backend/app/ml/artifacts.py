"""Per-stage artifacts: make an analysis inspectable and re-runnable.

Why this exists
---------------
Inference is the expensive part of the pipeline — about an hour for an hour
of video on a Pi 5. Every other stage (identity linking, rally segmentation,
metrics) is milliseconds. Without artifacts, changing a single threshold in
`identity.py` means re-running the whole hour to see the effect, which is how
the previous version ended up with a dozen commits titled "improve accuracy"
and no way to tell whether any of them did.

Writing the raw observations once turns threshold tuning into a sub-second
loop (`scripts/retune.py`) and makes the debug overlay possible without
touching the detector at all (`scripts/overlay.py`).

Layout, one directory per match:

    meta.json        video properties, calibration, config, stage timings
    tracks.jsonl     one line per observation, streamed during the pass
    result.json      the final metrics payload

`tracks.jsonl` is the only large file and the only one that cannot be
recomputed: measured at ~240 bytes per observation, a 60-minute match with
four players sampled at 5 Hz costs about 17 MB. Set KEEP_ARTIFACTS=false to
turn it off, at the price of losing the tuning loop and the overlay.

On precision: the record is lossy by design (see `_PRECISION_*` below), so a
replay reproduces the original metrics to well within the precision the UI
displays — not bit-for-bit. The tolerances are chosen so that no stored value
can flip a threshold decision; `test_pipeline.py` pins them.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable

import numpy as np

from app.ml.court import CalibrationSource, CourtCalibration, build_calibration
from app.ml.tracking import Observation, Tracklet

SCHEMA_VERSION = 1


class ArtifactWriter:
    """Streams observations to disk while the pipeline runs.

    Used as a context manager so a crashed analysis still leaves behind the
    frames it did process — which is exactly when you most want to look.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._tracks: IO[str] | None = None
        self._count = 0

    def __enter__(self) -> "ArtifactWriter":
        # A re-analysis truncates tracks.jsonl and starts refilling it, so any
        # meta.json or result.json left by the previous run would describe a
        # file that no longer matches them. Reading the directory mid-run then
        # mixes two generations — a completed run's frame count next to a few
        # percent of the new run's observations — which is a very convincing
        # way to reach the wrong conclusion. Clear them first; the pipeline
        # writes a fresh meta.json immediately, and the result at the end.
        for stale in ("meta.json", "result.json"):
            (self.directory / stale).unlink(missing_ok=True)
        self._tracks = (self.directory / "tracks.jsonl").open("w", encoding="utf-8")
        return self

    def __exit__(self, *_exc) -> None:
        if self._tracks is not None:
            self._tracks.close()
            self._tracks = None

    @property
    def observations_written(self) -> int:
        return self._count

    def write_meta(self, meta: dict) -> None:
        payload = {"schema_version": SCHEMA_VERSION, **meta}
        (self.directory / "meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def write_result(self, result: dict) -> None:
        # player_crops_data holds raw JPEG bytes and is not JSON-serialisable;
        # the thumbnails are persisted separately by the worker.
        payload = {k: v for k, v in result.items() if k != "player_crops_data"}
        (self.directory / "result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def write_observations(self, tracked: Iterable[tuple[int, Observation]]) -> None:
        if self._tracks is None:
            raise RuntimeError("ArtifactWriter va usato come context manager")
        for track_id, obs in tracked:
            self._tracks.write(json.dumps(_encode(track_id, obs), separators=(",", ":")))
            self._tracks.write("\n")
            self._count += 1


@dataclass(frozen=True)
class LoadedArtifacts:
    directory: Path
    meta: dict
    tracklets: list[Tracklet]
    calibration: CourtCalibration
    result: dict | None

    @property
    def complete(self) -> bool:
        """False while the decode pass is still running.

        Artifacts written before this flag existed were only ever saved at the
        end of the pass, so their absence means complete.
        """
        return bool(self.meta.get("complete", True))

    @property
    def sample_hz(self) -> float:
        return float(self.meta.get("sample_hz", 5.0))

    @property
    def analysed_s(self) -> float:
        return float(self.meta.get("analysed_s", 0.0))

    @property
    def frames_sampled(self) -> int:
        return int(self.meta.get("frames_sampled", 0))

    def observations_by_frame(self) -> dict[int, list[tuple[int, Observation]]]:
        """Frame index -> [(track_id, observation), ...], for the overlay."""
        out: dict[int, list[tuple[int, Observation]]] = defaultdict(list)
        for tracklet in self.tracklets:
            for obs in tracklet.observations:
                out[obs.frame_index].append((tracklet.id, obs))
        return out


def load_artifacts(directory: str | Path) -> LoadedArtifacts:
    """Rebuild tracklets and calibration from a previous run."""
    path = Path(directory)
    meta_file = path / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json non trovato in {path}")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    version = meta.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Artefatti in formato v{version}, atteso v{SCHEMA_VERSION}: rianalizza la partita."
        )

    calibration_data = meta["calibration"]
    calibration = build_calibration(
        corners_px=calibration_data["corners_px"],
        frame_size=tuple(calibration_data["frame_size"]),
        source=CalibrationSource(calibration_data.get("source", "manual")),
        net_px=calibration_data.get("net_px"),
    )

    # The pipeline drops tracks shorter than this; replaying must drop the
    # same ones or the tuning loop measures a different system.
    min_observations = int(meta.get("min_observations", 3))
    tracklets = _read_tracklets(path / "tracks.jsonl", min_observations)

    result_file = path / "result.json"
    result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None

    return LoadedArtifacts(
        directory=path,
        meta=meta,
        tracklets=tracklets,
        calibration=calibration,
        result=result,
    )


# ── Encoding ─────────────────────────────────────────────────────────────────

# Stored precision, chosen per field by what depends on it downstream.
#
#   timestamp — speeds are step/dt, and dt is ~0.2 s at 5 Hz, so an error of
#     1 ms in a timestamp is a 0.5% error in a speed. At 29.97 fps the
#     timestamps are not round numbers, so this needs microseconds, not
#     milliseconds. This was worth more than it costs: six decimals removes
#     the largest source of replay drift.
#   foot_court — feeds distance, speed and the heatmap. Five decimals is
#     10 micrometres, far below the homography's own error.
#   foot_px / bbox — only the overlay and the thumbnails read these; a tenth
#     of a pixel is plenty.
#   colour histogram — compared against a 0.45 veto and a 0.55 link cost, so
#     four decimals sits four orders of magnitude below any decision
#     boundary. It is also 32 numbers per observation, i.e. most of the file.
_PRECISION_TIME = 6
_PRECISION_COURT = 5
_PRECISION_PIXEL = 1
_PRECISION_COLOR = 4


def _encode(track_id: int, obs: Observation) -> dict:
    """Compact record. Keys are short because this file has one line per
    observation — roughly 100k lines for a full match."""
    return {
        "f": obs.frame_index,
        "t": round(obs.timestamp_s, _PRECISION_TIME),
        "id": track_id,
        "b": [round(v, _PRECISION_PIXEL) for v in obs.bbox],
        "p": [round(v, _PRECISION_PIXEL) for v in obs.foot_px],
        "c": [round(v, _PRECISION_COURT) for v in obs.foot_court],
        "q": round(obs.confidence, 3),
        "h": [round(float(v), _PRECISION_COLOR) for v in obs.color],
    }


def _decode(record: dict) -> tuple[int, Observation]:
    return int(record["id"]), Observation(
        frame_index=int(record["f"]),
        timestamp_s=float(record["t"]),
        bbox=tuple(float(v) for v in record["b"]),      # type: ignore[arg-type]
        foot_px=tuple(float(v) for v in record["p"]),   # type: ignore[arg-type]
        foot_court=tuple(float(v) for v in record["c"]),  # type: ignore[arg-type]
        confidence=float(record["q"]),
        color=np.asarray(record["h"], dtype=np.float32),
    )


def _read_tracklets(path: Path, min_observations: int) -> list[Tracklet]:
    if not path.exists():
        raise FileNotFoundError(f"tracks.jsonl non trovato in {path.parent}")

    grouped: dict[int, list[Observation]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                track_id, obs = _decode(json.loads(line))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number} record illeggibile: {exc}") from exc
            grouped[track_id].append(obs)

    tracklets: list[Tracklet] = []
    for track_id, observations in grouped.items():
        if len(observations) < min_observations:
            continue
        observations.sort(key=lambda o: o.timestamp_s)
        tracklets.append(Tracklet(id=track_id, observations=observations))

    tracklets.sort(key=lambda t: t.start_s)
    return tracklets
