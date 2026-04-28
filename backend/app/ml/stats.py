"""Aggregazione finale delle statistiche per giocatore.

Input: tutte le tracce + eventi classificati.
Output: dict serializzabile coerente con MatchStatsRead schema.

Stats calcolate:
1. Distanza percorsa (m) — somma dei movimenti tra frame consecutivi (in court coords)
2. Heatmap — istogramma 2D delle posizioni (downsampled a griglia 1m x 1m)
3. Vincenti / errori — euristica: HIT seguito da fine rally (ball lost o out)
   è vincente per chi ha colpito (se la palla esce dal campo = errore di chi colpisce
   se invece il rally finisce con la palla che rimbalza 2 volte = vincente di chi ha colpito ultimo)
4. Conteggio per tipo di colpo

NOTA winners/errors: euristica imperfetta nel MVP. Per accuracy alta serve
ground truth (regola: fuori = errore di chi ha colpito ultimo nel proprio campo,
doppio rimbalzo = vincente per chi colpiva). Questo è facile in tennis con linee
chiare; in padel più complesso per i muri.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from app.ml.court import CourtCalibration
from app.ml.events import Event, EventType
from app.ml.players import PlayerDetection
from app.ml.shots import ClassifiedShot, ShotType


def compute_stats(
    player_detections: list[list[PlayerDetection]],
    classified_shots: list[ClassifiedShot],
    rallies: list[tuple[int, int, int | None]],  # (start_frame, end_frame, last_hit_player_id)
    id_mapping: dict[int, int],  # YOLO track_id -> canonical 0..3
    calibration: CourtCalibration,
    fps: float,
) -> dict:
    n_players = 4

    # 1. Distanza percorsa: integra movimenti del foot in court coords
    distance_m: dict[int, float] = defaultdict(float)
    last_pos: dict[int, tuple[float, float]] = {}

    for frame_dets in player_detections:
        for d in frame_dets:
            canonical_id = id_mapping.get(d.track_id)
            if canonical_id is None:
                continue
            if canonical_id in last_pos:
                lx, ly = last_pos[canonical_id]
                cx, cy = d.foot_court
                step = float(np.hypot(cx - lx, cy - ly))
                # Filtra step impossibili (>5m per frame = teleport, scartare)
                if step < 5.0:
                    distance_m[canonical_id] += step
            last_pos[canonical_id] = d.foot_court

    # 2. Heatmaps: istogramma 2D 10x20 celle (1m x 1m)
    heatmaps_acc: dict[int, np.ndarray] = {i: np.zeros((10, 20)) for i in range(n_players)}
    for frame_dets in player_detections:
        for d in frame_dets:
            canonical_id = id_mapping.get(d.track_id)
            if canonical_id is None:
                continue
            cx, cy = d.foot_court
            ix = int(np.clip(cx, 0, 9.99))
            iy = int(np.clip(cy, 0, 19.99))
            heatmaps_acc[canonical_id][ix, iy] += 1

    # Normalize heatmaps to weights summing to 1
    heatmaps_serialized: dict[str, list[list[float]]] = {}
    for pid, hm in heatmaps_acc.items():
        total = hm.sum()
        if total == 0:
            heatmaps_serialized[str(pid)] = []
            continue
        normalized = hm / total
        # Sparse format: solo celle non-zero come [x, y, weight]
        points = []
        for ix in range(10):
            for iy in range(20):
                w = normalized[ix, iy]
                if w > 0.001:  # threshold di rumore
                    points.append([float(ix + 0.5), float(iy + 0.5), float(w)])
        heatmaps_serialized[str(pid)] = points

    # 3. Conteggio colpi per tipo
    shots_by_player: dict[int, dict[str, int]] = {
        i: {st.value: 0 for st in ShotType} for i in range(n_players)
    }
    for cs in classified_shots:
        if cs.event.player_id is None:
            continue
        canonical_id = id_mapping.get(cs.event.player_id)
        if canonical_id is None:
            continue
        shots_by_player[canonical_id][cs.shot_type.value] += 1

    # 4. Winners/errors da rallies
    # Euristica MVP: vincente = ultimo hit del rally se rally finisce dentro al campo
    # (richiede ground truth migliore; placeholder semplificato)
    winners: dict[int, int] = defaultdict(int)
    errors: dict[int, int] = defaultdict(int)
    for _, _, last_player in rallies:
        if last_player is None:
            continue
        canonical_id = id_mapping.get(last_player)
        if canonical_id is None:
            continue
        # Placeholder: 50/50 winner/error. Sostituire con logica reale.
        winners[canonical_id] += 1

    # Build per_player dict
    per_player: dict[str, dict] = {}
    for pid in range(n_players):
        per_player[str(pid)] = {
            "distance_m": round(distance_m[pid], 2),
            "winners": winners[pid],
            "errors": errors[pid],
            "shots": shots_by_player[pid],
        }

    return {
        "per_player": per_player,
        "heatmaps": heatmaps_serialized,
        "rallies": [
            {"start_frame": s, "end_frame": e, "last_player": id_mapping.get(p) if p else None}
            for s, e, p in rallies
        ],
        "court_calibration": calibration.to_dict(),
    }


def detect_rallies(events: list[Event], min_rally_gap_frames: int = 60) -> list[tuple[int, int, int | None]]:
    """Segmenta eventi in rally separati da pause.

    Un rally è una sequenza di HIT/BOUNCE con gap < min_rally_gap_frames tra eventi.
    """
    if not events:
        return []
    sorted_evs = sorted(events, key=lambda e: e.frame_idx)
    rallies: list[tuple[int, int, int | None]] = []
    start = sorted_evs[0].frame_idx
    last = sorted_evs[0].frame_idx
    last_hitter: int | None = sorted_evs[0].player_id if sorted_evs[0].type == EventType.HIT else None

    for ev in sorted_evs[1:]:
        if ev.frame_idx - last > min_rally_gap_frames:
            rallies.append((start, last, last_hitter))
            start = ev.frame_idx
            last_hitter = None
        last = ev.frame_idx
        if ev.type == EventType.HIT:
            last_hitter = ev.player_id

    rallies.append((start, last, last_hitter))
    return rallies
