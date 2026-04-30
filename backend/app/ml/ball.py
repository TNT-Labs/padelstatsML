"""Ball tracking via TrackNetV2 (when weights available) or MOG2 fallback.

TrackNetV2 architecture: VGG-16 style encoder + U-Net decoder with skip connections.
  Input:  [B, 9, H, W]  — 3 consecutive RGB frames stacked channel-wise
  Output: [B, 1, H, W]  — probability heatmap; argmax → ball pixel location

Weight compatibility: matches the architecture in
  - michele98/ball_tracking_padel
  - SamuReyes/TrackNetV2-padel

Fallback (no weights): MOG2 background subtraction + circularity filter.
Accuracy is lower but produces usable detections on static-camera videos,
so the rest of the pipeline (events, shots, stats) still runs.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_W = 640
INPUT_H = 360


@dataclass
class BallDetection:
    frame_idx: int
    pos_px: tuple[float, float] | None  # None = ball not visible in this frame
    confidence: float


# ── TrackNetV2 full architecture ────────────────────────────────────────────

class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _EncBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_convs: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [_ConvBNReLU(in_ch, out_ch)]
        for _ in range(n_convs - 1):
            layers.append(_ConvBNReLU(out_ch, out_ch))
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class _DecBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_ch + skip_ch, out_ch),
            _ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # interpolate handles odd spatial sizes that break plain Upsample
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class TrackNetV2(nn.Module):
    """Full TrackNetV2 encoder-decoder for ball heatmap prediction.

    Architecture matches the VGG-16 encoder / U-Net decoder family used by
    padel-specific fine-tuned weights. load_state_dict(strict=False) handles
    minor naming differences between repos.
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 1) -> None:
        super().__init__()
        # Encoder — 5 VGG-style blocks
        self.enc1 = _EncBlock(in_channels, 64,  n_convs=2)
        self.enc2 = _EncBlock(64,          128, n_convs=2)
        self.enc3 = _EncBlock(128,         256, n_convs=3)
        self.enc4 = _EncBlock(256,         512, n_convs=3)
        self.enc5 = _EncBlock(512,         512, n_convs=3)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            _ConvBNReLU(512, 512),
            _ConvBNReLU(512, 512),
        )

        # Decoder — skip connections from encoder
        self.dec5 = _DecBlock(512, 512, 512)   # concat with enc5 (512)
        self.dec4 = _DecBlock(512, 512, 256)   # concat with enc4 (512)
        self.dec3 = _DecBlock(256, 256, 128)   # concat with enc3 (256)
        self.dec2 = _DecBlock(128, 128, 64)    # concat with enc2 (128)
        self.dec1 = _DecBlock(64,  64,  64)    # concat with enc1 (64)

        self.head = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        s4 = self.enc4(F.max_pool2d(s3, 2))
        s5 = self.enc5(F.max_pool2d(s4, 2))
        b  = self.bottleneck(F.max_pool2d(s5, 2))
        d  = self.dec5(b,  s5)
        d  = self.dec4(d,  s4)
        d  = self.dec3(d,  s3)
        d  = self.dec2(d,  s2)
        d  = self.dec1(d,  s1)
        return self.head(d)


# ── MOG2 background subtraction fallback ────────────────────────────────────

class _MOG2BallDetector:
    """Ball detector using MOG2 background subtraction + contour analysis.

    No weights required. Works on static-camera padel videos where the ball
    is the dominant fast-moving object. False-positive rate is higher than
    TrackNetV2 but produces enough valid detections to drive event detection.
    """

    # Ball size bounds in pixels at processing resolution (640×360)
    _MIN_AREA = 6
    _MAX_AREA = 420
    _MIN_CIRCULARITY = 0.30
    _CONF_THRESHOLD = 0.42

    def track_video(self, video_path: str) -> Iterator[BallDetection]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        proc_w = min(orig_w, INPUT_W)
        proc_h = min(orig_h, INPUT_H)
        sx, sy = orig_w / proc_w, orig_h / proc_h

        bg = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=36, detectShadows=False
        )

        # Warm-up: feed ~2 s of frames so background model stabilises
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        warmup = int(fps * 2)
        for _ in range(warmup):
            ok, frame = cap.read()
            if not ok:
                break
            bg.apply(cv2.resize(frame, (proc_w, proc_h)))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            small = cv2.resize(frame, (proc_w, proc_h))
            fg    = bg.apply(small)
            fg    = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open)
            fg    = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_pos: tuple[float, float] | None = None
            best_score = 0.0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self._MIN_AREA or area > self._MAX_AREA:
                    continue
                perim = cv2.arcLength(cnt, True)
                if perim < 1:
                    continue
                circ = 4 * np.pi * area / (perim ** 2)
                if circ < self._MIN_CIRCULARITY:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] < 1:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                x0, y0, bw, bh = cv2.boundingRect(cnt)
                roi = small[y0: y0 + bh, x0: x0 + bw]
                brightness = float(roi.mean()) / 255.0 if roi.size else 0.5
                # Score: circularity (shape) + brightness (ball is usually bright)
                score = circ * 0.65 + brightness * 0.35
                if score > best_score:
                    best_score = score
                    best_pos = (cx * sx, cy * sy)

            if best_pos and best_score > self._CONF_THRESHOLD:
                yield BallDetection(
                    frame_idx=frame_idx,
                    pos_px=best_pos,
                    confidence=float(min(best_score, 0.85)),
                )
            else:
                yield BallDetection(frame_idx=frame_idx, pos_px=None, confidence=0.0)

        cap.release()


# ── Public BallTracker ───────────────────────────────────────────────────────

class BallTracker:
    def __init__(self, weights_path: str | None, device: str = "cpu"):
        self.device = device
        self._model: TrackNetV2 | None = None
        self._ort_session = None
        self._ort_input_name: str = "input"

        if not (weights_path and Path(weights_path).exists()):
            return

        # Prefer ONNX: onnxruntime is 3-5× faster than PyTorch on ARM64 CPU.
        # Export once with: python scripts/export_onnx.py --tracknet
        onnx_path = Path(weights_path).with_suffix(".onnx")
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                self._ort_session = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                )
                self._ort_input_name = self._ort_session.get_inputs()[0].name
                print(f"[BallTracker] ONNX runtime: {onnx_path.name}")
                return
            except Exception as exc:
                print(f"[BallTracker] ONNX load failed: {exc} — trying PyTorch")

        try:
            model = TrackNetV2().to(device)
            state = torch.load(weights_path, map_location=device, weights_only=True)
            if isinstance(state, dict):
                for key in ("model", "state_dict", "net"):
                    if key in state:
                        state = state[key]
                        break
            model.load_state_dict(state, strict=False)
            model.eval()
            self._model = model
            print(f"[BallTracker] PyTorch model loaded: {Path(weights_path).name}")
        except Exception as exc:
            print(f"[BallTracker] Weight load failed: {exc} — using MOG2 fallback")

    @property
    def using_neural_model(self) -> bool:
        return self._model is not None or self._ort_session is not None

    def track_video(self, video_path: str, stride: int = 1) -> Iterator[BallDetection]:
        """Track ball across all frames.

        Args:
            stride: run neural inference every Nth frame; frames in between
                    are emitted as (pos_px=None) and filled by the Kalman
                    smoother.  stride=2 halves processing time with minimal
                    accuracy loss on typical 25-30fps padel videos.
        """
        if not self.using_neural_model:
            yield from _MOG2BallDetector().track_video(video_path)
            return

        from collections import deque

        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sx, sy = orig_w / INPUT_W, orig_h / INPUT_H

        buf: deque[np.ndarray] = deque(maxlen=3)
        frame_num = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            small = cv2.resize(frame, (INPUT_W, INPUT_H))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            buf.append(rgb)

            # Skip inference on non-stride frames or until buffer is full
            if len(buf) < 3 or frame_num % stride != 0:
                yield BallDetection(frame_idx=frame_idx, pos_px=None, confidence=0.0)
                frame_num += 1
                continue

            # Stack: (H,W,3)×3 → (H,W,9)
            x = np.concatenate(list(buf), axis=2)

            if self._ort_session is not None:
                inp = x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,9,H,W)
                heatmap = self._ort_session.run(
                    None, {self._ort_input_name: inp}
                )[0][0, 0]
            else:
                inp_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    heatmap = self._model(inp_t)[0, 0].cpu().numpy()  # type: ignore[union-attr]

            conf = float(heatmap.max())
            if conf < 0.5:
                yield BallDetection(frame_idx=frame_idx, pos_px=None, confidence=conf)
            else:
                mask = heatmap > 0.3
                ys, xs = np.where(mask)
                w = heatmap[mask]
                xp = float((xs * w).sum() / w.sum())
                yp = float((ys * w).sum() / w.sum())
                yield BallDetection(
                    frame_idx=frame_idx,
                    pos_px=(xp * sx, yp * sy),
                    confidence=conf,
                )
            frame_num += 1

        cap.release()


def smooth_trajectory(detections: list[BallDetection], max_gap: int = 5) -> list[BallDetection]:
    """Kalman-filter smoothing + gap filling for ball trajectory.

    State [x, y, vx, vy] with constant-velocity model.  Valid detections
    update the filter; frames with no detection up to max_gap are filled
    with the Kalman prediction (conf=0.3).  Falls back to linear
    interpolation when filterpy is unavailable.
    """
    if not detections:
        return detections

    try:
        from filterpy.kalman import KalmanFilter as KF
    except ImportError:
        return _quadratic_interpolate(detections, max_gap)

    first = next((d for d in detections if d.pos_px is not None), None)
    if first is None:
        return detections

    kf = KF(dim_x=4, dim_z=2)
    dt = 1.0
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]], dtype=np.float32)
    kf.H  = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float32)
    kf.R  = np.eye(2, dtype=np.float32) * 25.0   # measurement noise (~5 px std)
    kf.Q  = np.eye(4, dtype=np.float32) * 0.8    # process noise
    kf.P *= 100.0
    kf.x  = np.array([first.pos_px[0], first.pos_px[1], 0.0, 0.0], dtype=np.float32)

    out = list(detections)
    misses = 0
    for i, det in enumerate(out):
        kf.predict()
        if det.pos_px is not None:
            kf.update(np.array([det.pos_px[0], det.pos_px[1]], dtype=np.float32))
            misses = 0
            out[i] = BallDetection(
                frame_idx=det.frame_idx,
                pos_px=(float(kf.x[0]), float(kf.x[1])),
                confidence=det.confidence,
            )
        elif misses < max_gap:
            misses += 1
            out[i] = BallDetection(
                frame_idx=det.frame_idx,
                pos_px=(float(kf.x[0]), float(kf.x[1])),
                confidence=0.3,
            )
        else:
            misses += 1
    return out


def _quadratic_interpolate(detections: list[BallDetection], max_gap: int) -> list[BallDetection]:
    """Quadratic (degree-2) gap filling when filterpy is unavailable.

    For each gap fits separate polynomials x(t) and y(t) to the known
    positions surrounding the gap (up to 5 frames on each side).  A
    degree-2 polynomial follows the parabolic arc of a ball under gravity
    far better than a straight line.  Falls back to linear if fewer than
    3 surrounding points are available.
    """
    _CTX = 5   # context frames to use on each side of a gap

    out = list(detections)
    n, i = len(out), 0
    while i < n:
        if out[i].pos_px is None:
            j = i + 1
            while j < n and out[j].pos_px is None:
                j += 1
            if (j - i) <= max_gap and i > 0 and j < n:
                pre = [
                    (out[k].frame_idx, out[k].pos_px)
                    for k in range(max(0, i - _CTX), i)
                    if out[k].pos_px is not None
                ]
                post = [
                    (out[k].frame_idx, out[k].pos_px)
                    for k in range(j, min(n, j + _CTX))
                    if out[k].pos_px is not None
                ]
                known = pre + post
                gap_frames = [out[idx].frame_idx for idx in range(i, j)]

                if len(known) >= 3:
                    ts = np.array([p[0] for p in known], dtype=np.float64)
                    xs = np.array([p[1][0] for p in known], dtype=np.float64)
                    ys = np.array([p[1][1] for p in known], dtype=np.float64)
                    cx = np.polyfit(ts, xs, 2)
                    cy = np.polyfit(ts, ys, 2)
                    for idx, t in zip(range(i, j), gap_frames):
                        out[idx] = BallDetection(
                            frame_idx=out[idx].frame_idx,
                            pos_px=(float(np.polyval(cx, t)), float(np.polyval(cy, t))),
                            confidence=0.3,
                        )
                elif len(known) >= 2:
                    # linear fallback when context is too sparse for quadratic
                    t0, p0 = known[0]
                    t1, p1 = known[-1]
                    span = max(t1 - t0, 1)
                    for idx, t in zip(range(i, j), gap_frames):
                        r = (t - t0) / span
                        out[idx] = BallDetection(
                            frame_idx=out[idx].frame_idx,
                            pos_px=(p0[0] * (1 - r) + p1[0] * r,
                                    p0[1] * (1 - r) + p1[1] * r),
                            confidence=0.3,
                        )
            i = j
        else:
            i += 1
    return out
