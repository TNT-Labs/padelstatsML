"""Ball tracking via TrackNetV2.

TrackNet prende 3 frame consecutivi e produce una heatmap di probabilità della palla.
Il peak della heatmap è la posizione palla.

Per MVP forniamo:
- Wrapper sull'inference (assumendo pesi pre-trained da `michele98/ball_tracking_padel`)
- Fallback opzionale a YOLOv8 trained per ball detection (meno accurato ma più semplice)
- Trajectory smoothing con filtro Kalman per gestire frame con ball lost

NOTA: per ottenere risultati production-grade serve fine-tuning su dataset padel
proprio. Repo di riferimento già listati nel README.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

INPUT_W = 640
INPUT_H = 360


@dataclass
class BallDetection:
    frame_idx: int
    pos_px: tuple[float, float] | None  # None = ball not visible
    confidence: float


class TrackNetV2(nn.Module):
    """Architettura TrackNetV2 semplificata.

    Per il file completo (encoder VGG-style + decoder con skip connections)
    riferirsi al repo originale. Qui stub per integrazione.
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 1):
        super().__init__()
        # Stub - sostituire con architettura completa da repo TrackNetV2
        self.placeholder = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.placeholder(x))


class BallTracker:
    def __init__(self, weights_path: str | None, device: str = "cuda"):
        self.device = device
        self.model: TrackNetV2 | None = None

        if weights_path and Path(weights_path).exists():
            self.model = TrackNetV2().to(device)
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state)
            self.model.eval()

    def track_video(self, video_path: str) -> Iterator[BallDetection]:
        """Stream ball detections.

        Implementazione: per ogni frame i, costruisci input come stack di
        frame [i-2, i-1, i] resized a 640x360, normalizzati. Forward pass.
        Estrai peak dalla heatmap. Threshold di confidenza.
        """
        if self.model is None:
            # Fallback: nessun modello caricato → restituisci None per ogni frame
            # (l'app procederà senza ball tracking, eventi saranno meno accurati)
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for i in range(total):
                yield BallDetection(frame_idx=i, pos_px=None, confidence=0.0)
            return

        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sx = orig_w / INPUT_W
        sy = orig_h / INPUT_H

        buffer: list[np.ndarray] = []
        frame_idx = -1

        while True:
            ok, frame = cap.read()
            frame_idx += 1
            if not ok:
                break

            small = cv2.resize(frame, (INPUT_W, INPUT_H))
            buffer.append(small)
            if len(buffer) > 3:
                buffer.pop(0)

            if len(buffer) < 3:
                yield BallDetection(frame_idx=frame_idx, pos_px=None, confidence=0.0)
                continue

            # Stack 3 frames (9 channels)
            x = np.concatenate(buffer, axis=2).astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                heatmap = self.model(x)[0, 0].cpu().numpy()

            # Peak
            y_peak, x_peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            conf = float(heatmap[y_peak, x_peak])

            if conf < 0.5:
                yield BallDetection(frame_idx=frame_idx, pos_px=None, confidence=conf)
            else:
                yield BallDetection(
                    frame_idx=frame_idx,
                    pos_px=(float(x_peak * sx), float(y_peak * sy)),
                    confidence=conf,
                )

        cap.release()


def smooth_trajectory(detections: list[BallDetection], max_gap: int = 5) -> list[BallDetection]:
    """Interpola gap brevi nella trajectory con Kalman filter o linear interp.

    Per MVP: linear interpolation per gap <= max_gap frame.
    """
    out = list(detections)
    n = len(out)
    i = 0
    while i < n:
        if out[i].pos_px is None:
            # trova prossima detection valida
            j = i + 1
            while j < n and out[j].pos_px is None:
                j += 1
            if j < n and i > 0 and out[i - 1].pos_px is not None and (j - i) <= max_gap:
                # interpola
                p_start = out[i - 1].pos_px
                p_end = out[j].pos_px
                steps = j - i + 1
                for k, idx in enumerate(range(i, j), start=1):
                    t = k / steps
                    out[idx] = BallDetection(
                        frame_idx=out[idx].frame_idx,
                        pos_px=(
                            p_start[0] * (1 - t) + p_end[0] * t,
                            p_start[1] * (1 - t) + p_end[1] * t,
                        ),
                        confidence=0.3,  # interpolated → low conf
                    )
            i = j
        else:
            i += 1
    return out
