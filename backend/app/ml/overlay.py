"""Debug overlay: draw what the pipeline actually saw.

Numbers always look plausible. A distance of 2 480 m and a heatmap that
favours the net look exactly the same whether the tracking was right or
whether two players were swapped for half the match. The only way to tell is
to look at the frames.

This module renders each sampled frame with:
  * the calibrated court projected back into the image (outline, net, service
    lines) — if this does not sit on the real court, nothing else matters;
  * every tracked player, coloured by the canonical id the identity stage
    assigned, with speed and track id;
  * a top-down mini-map showing the same four players in court metres, which
    is where an identity swap becomes obvious;
  * a rally banner and a timeline of the segmented points.

It reads stored artifacts, so it never runs the detector: rendering a clip is
seconds, not an hour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ml.artifacts import LoadedArtifacts
from app.ml.court import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    SERVICE_LINE_OFFSET_M,
    CourtCalibration,
)
from app.ml.identity import PlayerTrack
from app.ml.rallies import Rally
from app.ml.tracking import observation_key
from app.ml.video import FrameSampler

# BGR, matching the four player colours used by the web UI.
PLAYER_BGR = [(68, 68, 239), (246, 130, 59), (11, 158, 245), (246, 92, 139)]
UNASSIGNED_BGR = (150, 150, 150)
COURT_BGR = (235, 235, 235)
NET_BGR = (0, 235, 255)

MINIMAP_W = 150
MINIMAP_MARGIN = 16
MINIMAP_ALPHA = 0.78          # blended, so it never hides part of the court
BANNER_H = 34
LABEL_LINE_H = 18


@dataclass
class OverlayContext:
    """Everything the renderer needs, precomputed once."""

    calibration: CourtCalibration
    players: list[PlayerTrack]
    rallies: list[Rally]
    sample_hz: float
    observations_by_frame: dict[int, list]
    # Per detection, not per track id: identity may split a tracklet that
    # switched person, and give its pieces to different players.
    owner: dict[tuple[int, tuple[float, float]], int]
    speeds: dict[tuple[int, int], float]      # (track_id, frame_index) -> m/s
    analysed_s: float


def build_context(
    artifacts: LoadedArtifacts,
    players: list[PlayerTrack],
    rallies: list[Rally],
) -> OverlayContext:
    owner = {
        observation_key(obs): player.player_id
        for player in players
        for obs in player.observations
    }

    speeds: dict[tuple[int, int], float] = {}
    for tracklet in artifacts.tracklets:
        observations = tracklet.observations
        for previous, current in zip(observations, observations[1:]):
            dt = current.timestamp_s - previous.timestamp_s
            if dt <= 1e-3:
                continue
            step = float(
                np.hypot(
                    current.foot_court[0] - previous.foot_court[0],
                    current.foot_court[1] - previous.foot_court[1],
                )
            )
            speeds[(tracklet.id, current.frame_index)] = step / dt

    return OverlayContext(
        calibration=artifacts.calibration,
        players=players,
        rallies=rallies,
        sample_hz=artifacts.sample_hz,
        observations_by_frame=artifacts.observations_by_frame(),
        owner=owner,
        speeds=speeds,
        analysed_s=artifacts.analysed_s,
    )


def render_frame(frame: np.ndarray, frame_index: int, timestamp_s: float, ctx: OverlayContext) -> np.ndarray:
    canvas = frame.copy()
    _draw_court(canvas, ctx.calibration)

    entries = ctx.observations_by_frame.get(frame_index, [])
    for track_id, obs in entries:
        player_id = ctx.owner.get(observation_key(obs))
        color = PLAYER_BGR[player_id % 4] if player_id is not None else UNASSIGNED_BGR
        _draw_player(canvas, track_id, player_id, obs, color, ctx)

    _draw_minimap(canvas, entries, ctx)
    _draw_banner(canvas, timestamp_s, len(entries), ctx)
    _draw_timeline(canvas, timestamp_s, ctx)
    return canvas


# ── Drawing primitives ───────────────────────────────────────────────────────

def _draw_court(canvas: np.ndarray, calibration: CourtCalibration) -> None:
    """Project the court model back onto the image.

    This is the single most informative element: if the drawn net does not
    lie on the real net, the calibration is wrong and every metre-based
    number downstream is wrong with it.
    """
    lines_m = [
        ([[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M], [0, 0]], COURT_BGR, 2),
        ([[0, NET_Y_M], [COURT_WIDTH_M, NET_Y_M]], NET_BGR, 3),
        ([[0, NET_Y_M - SERVICE_LINE_OFFSET_M], [COURT_WIDTH_M, NET_Y_M - SERVICE_LINE_OFFSET_M]], COURT_BGR, 1),
        ([[0, NET_Y_M + SERVICE_LINE_OFFSET_M], [COURT_WIDTH_M, NET_Y_M + SERVICE_LINE_OFFSET_M]], COURT_BGR, 1),
        ([[COURT_WIDTH_M / 2, NET_Y_M - SERVICE_LINE_OFFSET_M], [COURT_WIDTH_M / 2, NET_Y_M + SERVICE_LINE_OFFSET_M]], COURT_BGR, 1),
    ]
    height, width = canvas.shape[:2]
    for points_m, color, thickness in lines_m:
        projected = calibration.court_to_pixels(np.array(points_m, dtype=np.float32))
        if not np.isfinite(projected).all():
            continue
        # Clamp generously rather than dropping: a court line leaving the
        # frame is normal on a wide camera position.
        pts = np.clip(projected, [-4 * width, -4 * height], [4 * width, 4 * height])
        cv2.polylines(canvas, [pts.astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def _draw_player(canvas, track_id: int, player_id: int | None, obs, color, ctx: OverlayContext) -> None:
    height, width = canvas.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in obs.bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    foot = (int(obs.foot_px[0]), int(obs.foot_px[1]))
    cv2.circle(canvas, foot, 4, color, -1)

    speed = ctx.speeds.get((track_id, obs.frame_index))
    label = f"G{player_id + 1}" if player_id is not None else f"t{track_id}?"
    detail = f"{obs.foot_court[0]:.1f},{obs.foot_court[1]:.1f}m"
    if speed is not None:
        detail += f"  {speed:.1f}m/s"

    # Players on the far baseline sit near the top of the frame, where a label
    # above the box would be hidden by the banner. Flip it inside the box
    # instead of letting it clip: an unreadable id defeats the whole overlay.
    above = y1 - LABEL_LINE_H - 6
    if above - LABEL_LINE_H < BANNER_H:
        first_y, second_y = y1 + LABEL_LINE_H, y1 + 2 * LABEL_LINE_H
    else:
        first_y, second_y = above, y1 - 6

    # Keep the text inside the frame when the player is at the right edge.
    text_x = min(x1, width - 170)
    text_x = max(text_x, 4)
    second_y = min(second_y, height - 6)
    first_y = min(first_y, height - 6)

    _label(canvas, label, (text_x, first_y), color, scale=0.6, thickness=2)
    _label(canvas, detail, (text_x, second_y), color, scale=0.42, thickness=1)


def _draw_minimap(canvas: np.ndarray, entries, ctx: OverlayContext) -> None:
    """Top-down court in metres.

    An identity swap is nearly invisible in the image — two boxes exchange
    colour for a moment — but on the mini-map a player jumping across the net
    is immediate.
    """
    height, width = canvas.shape[:2]
    map_h = MINIMAP_W * 2
    x0 = width - MINIMAP_W - MINIMAP_MARGIN
    y0 = MINIMAP_MARGIN
    if x0 < 0 or y0 + map_h > height:
        return

    region = canvas[y0: y0 + map_h, x0: x0 + MINIMAP_W]
    panel = region.copy()
    cv2.rectangle(panel, (0, 0), (MINIMAP_W - 1, map_h - 1), (40, 60, 45), -1)
    cv2.rectangle(panel, (0, 0), (MINIMAP_W - 1, map_h - 1), COURT_BGR, 1)
    net_y = int(map_h * (1 - NET_Y_M / COURT_LENGTH_M))
    cv2.line(panel, (0, net_y), (MINIMAP_W, net_y), NET_BGR, 1)

    for track_id, obs in entries:
        px = int(obs.foot_court[0] / COURT_WIDTH_M * MINIMAP_W)
        py = int((1 - obs.foot_court[1] / COURT_LENGTH_M) * map_h)
        if not (0 <= px < MINIMAP_W and 0 <= py < map_h):
            continue        # outside the court: already flagged by the tracker
        player_id = ctx.owner.get(observation_key(obs))
        color = PLAYER_BGR[player_id % 4] if player_id is not None else UNASSIGNED_BGR
        cv2.circle(panel, (px, py), 5, color, -1)
        cv2.circle(panel, (px, py), 5, (255, 255, 255), 1)

    # Blended rather than pasted: on a real recording the court fills the
    # frame, and an opaque panel would hide the very thing being checked.
    cv2.addWeighted(panel, MINIMAP_ALPHA, region, 1.0 - MINIMAP_ALPHA, 0, region)


def _draw_banner(canvas: np.ndarray, timestamp_s: float, tracked: int, ctx: OverlayContext) -> None:
    rally = next((r for r in ctx.rallies if r.contains(timestamp_s)), None)
    minutes, seconds = divmod(int(timestamp_s), 60)

    text = f"{minutes:02d}:{seconds:02d}  ·  {tracked} giocatori tracciati  ·  {ctx.sample_hz:.1f} Hz"
    if rally is not None:
        elapsed = timestamp_s - rally.start_s
        text += f"  ·  SCAMBIO #{rally.index + 1} ({elapsed:.1f}s / {rally.duration_s:.1f}s)"

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (25, 25, 25), -1)
    color = (120, 255, 120) if rally is not None else (220, 220, 220)
    cv2.putText(canvas, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _draw_timeline(canvas: np.ndarray, timestamp_s: float, ctx: OverlayContext) -> None:
    """Rally segments across the whole match, with the playhead."""
    if ctx.analysed_s <= 0:
        return
    height, width = canvas.shape[:2]
    bar_h = 14
    top = height - bar_h - 8
    cv2.rectangle(canvas, (8, top), (width - 8, top + bar_h), (40, 40, 40), -1)

    span = width - 16
    for rally in ctx.rallies:
        x1 = 8 + int(span * rally.start_s / ctx.analysed_s)
        x2 = 8 + int(span * rally.end_s / ctx.analysed_s)
        cv2.rectangle(canvas, (x1, top), (max(x2, x1 + 1), top + bar_h), (90, 200, 110), -1)

    playhead = 8 + int(span * min(timestamp_s / ctx.analysed_s, 1.0))
    cv2.line(canvas, (playhead, top - 3), (playhead, top + bar_h + 3), (0, 220, 255), 2)


def _label(canvas, text: str, origin, color, scale: float, thickness: int) -> None:
    """Text with a dark outline, readable over both the court and the walls."""
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ── Rendering entry points ───────────────────────────────────────────────────

def render_video(
    video_path: str | Path,
    ctx: OverlayContext,
    out_path: str | Path,
    start_s: float = 0.0,
    end_s: float | None = None,
    playback_fps: float = 10.0,
    scale: float = 1.0,
    progress: callable | None = None,
) -> int:
    """Render an annotated MP4. Returns the number of frames written.

    The source is re-sampled at the same rate the analysis used, so frame
    indices line up with the stored observations exactly — no interpolation,
    no guessing.
    """
    sampler = FrameSampler(video_path, target_hz=ctx.sample_hz, max_duration_s=end_s)

    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        for frame in sampler:
            if frame.timestamp_s < start_s:
                continue
            if end_s is not None and frame.timestamp_s > end_s:
                break

            canvas = render_frame(frame.image, frame.index, frame.timestamp_s, ctx)
            if scale != 1.0:
                canvas = cv2.resize(
                    canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )

            if writer is None:
                height, width = canvas.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Impossibile scrivere {out_path}: codec mp4v non disponibile."
                    )
            writer.write(canvas)
            written += 1
            if progress and written % 50 == 0:
                progress(written, frame.timestamp_s)
    finally:
        if writer is not None:
            writer.release()
    return written


def render_stills(
    video_path: str | Path,
    ctx: OverlayContext,
    out_dir: str | Path,
    count: int = 12,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[Path]:
    """Write `count` annotated JPEGs spread over the clip.

    Useful when no video codec is available, and faster to skim than a video
    when all you want is a sanity check.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    sampler = FrameSampler(video_path, target_hz=ctx.sample_hz, max_duration_s=end_s)
    frames = [f for f in sampler if f.timestamp_s >= start_s and (end_s is None or f.timestamp_s <= end_s)]
    if not frames:
        return []

    # Prefer frames where the pipeline saw players: a still of an empty court
    # tells you nothing.
    populated = [f for f in frames if ctx.observations_by_frame.get(f.index)]
    chosen_pool = populated or frames
    step = max(1, len(chosen_pool) // count)
    chosen = chosen_pool[::step][:count]

    written: list[Path] = []
    for frame in chosen:
        canvas = render_frame(frame.image, frame.index, frame.timestamp_s, ctx)
        path = directory / f"t{int(frame.timestamp_s):05d}s_f{frame.index:07d}.jpg"
        if cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            written.append(path)
    return written
