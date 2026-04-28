"""Player tracking: YOLOv8 person detection + ByteTrack per ID consistenti.

Output: per ogni frame, lista di (track_id, bbox, foot_position_px).
Ultralytics ha ByteTrack integrato via `model.track()`, ottimo per MVP.

Filtri MVP:
- Solo classe 'person' (COCO id=0)
- Solo detection dentro al campo (proiettate via omografia)
- Limite 4 giocatori (i primi 4 track_id stabili)
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from app.ml.court import CourtCalibration


@dataclass
class PlayerDetection:
    frame_idx: int
    track_id: int
    bbox_px: tuple[float, float, float, float]  # x1, y1, x2, y2
    foot_px: tuple[float, float]  # piedi = mid-bottom della bbox
    foot_court: tuple[float, float]  # in court coords (m)
    confidence: float


class PlayerTracker:
    def __init__(self, weights: str, device: str = "cuda"):
        self.model = YOLO(weights)
        self.device = device

    def track_video(
        self, video_path: str, calibration: CourtCalibration, stride: int = 1
    ) -> Iterator[list[PlayerDetection]]:
        """Stream player detections frame-by-frame.

        Args:
            stride: process 1 frame ogni N (1 = tutti). Per video lunghi
                    stride=2 dimezza tempo con minima perdita di accuracy.

        Yields:
            Lista di PlayerDetection per il frame corrente.
        """
        # tracker="bytetrack.yaml" usa config default ottimizzata
        results = self.model.track(
            source=video_path,
            stream=True,
            persist=True,
            classes=[0],  # only person
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
            vid_stride=stride,
            conf=0.35,
            iou=0.5,
        )

        court_polygon = calibration.corners_px

        for frame_idx, r in enumerate(results):
            if r.boxes is None or r.boxes.id is None:
                yield []
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            detections: list[PlayerDetection] = []
            for bbox, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = bbox
                foot_px = ((x1 + x2) / 2, y2)

                # Filter: solo se i piedi sono nel campo (con margine)
                if not _point_in_polygon_with_margin(foot_px, court_polygon, margin=50):
                    continue

                foot_court = calibration.pixel_to_court(np.array([foot_px]))[0]

                detections.append(
                    PlayerDetection(
                        frame_idx=frame_idx * stride,
                        track_id=int(tid),
                        bbox_px=(float(x1), float(y1), float(x2), float(y2)),
                        foot_px=(float(foot_px[0]), float(foot_px[1])),
                        foot_court=(float(foot_court[0]), float(foot_court[1])),
                        confidence=float(conf),
                    )
                )

            yield detections


def _point_in_polygon_with_margin(point: tuple[float, float], polygon: np.ndarray, margin: float) -> bool:
    """Allarga il poligono di `margin` pixel poi verifica contenimento."""
    # Espandi dal centroide
    centroid = polygon.mean(axis=0)
    expanded = polygon + (polygon - centroid) * (margin / np.linalg.norm(polygon - centroid, axis=1, keepdims=True).clip(min=1))
    return cv2.pointPolygonTest(expanded.astype(np.float32), point, False) >= 0


def normalize_player_ids(
    all_detections: list[list[PlayerDetection]],
) -> dict[int, int]:
    """Mappa i track_id YOLO ai 4 player canonici (0..3).

    Strategia: i 4 track_id che appaiono in più frame = i 4 giocatori.
    In partita reale, ID switch sono frequenti (occlusioni). Per MVP
    semplifichiamo: top-4 per frequenza.

    In produzione: usare team-side classification (chi sta in quale metà
    campo) + appearance features (re-id) per stabilizzare IDs.
    """
    counts: dict[int, int] = {}
    for frame_dets in all_detections:
        for d in frame_dets:
            counts[d.track_id] = counts.get(d.track_id, 0) + 1

    top4 = sorted(counts.items(), key=lambda x: -x[1])[:4]
    return {tid: i for i, (tid, _) in enumerate(top4)}
