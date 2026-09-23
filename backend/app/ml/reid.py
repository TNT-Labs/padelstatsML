"""Person re-identification embeddings over onnxruntime (CPU).

Identity used to rest on the colour of the kit alone, and on a real match
that could not tell the four players apart: team-mates in the same kit, or
kits of similar dark and light tones. A re-identification network describes
a person's whole appearance — build, hair, shoes, how the kit sits — as a
vector in which the same person lands close to themselves across frames.

OSNet x0_25 (see scripts/export_reid_onnx.py) costs about 0.08 GFLOPs per
player, a few percent of the detector. Everything that can run without it
still does: with no model, identity falls back to kit colour.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.ml.detect import _physical_cores

# ImageNet normalisation, which OSNet was trained with.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Below this many pixels on either side a crop describes nothing.
_MIN_CROP_PX = 8


class ReidError(RuntimeError):
    pass


class ReidEmbedder:
    def __init__(self, model_path: str | Path, threads: int = 0) -> None:
        path = Path(model_path)
        if not path.exists():
            raise ReidError(f"Modello re-ID non trovato: {path}")
        if path.suffix.lower() != ".onnx":
            raise ReidError(f"Atteso un modello .onnx, ricevuto '{path.suffix}'.")

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads if threads > 0 else min(_physical_cores(), 4)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        _, _, height, width = model_input.shape
        if not isinstance(height, int) or not isinstance(width, int):
            raise ReidError(f"{path.name}: dimensioni di input non fisse ({model_input.shape}).")
        self.input_size = (int(height), int(width))
        self.model_name = path.name

    def embed(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[np.ndarray | None]:
        """One unit-length embedding per box, or None where the crop is too
        small to describe anybody. All crops of a frame go in one batch."""
        crops: list[np.ndarray] = []
        slots: list[int] = []
        for index, bbox in enumerate(bboxes):
            crop = self._prepare(frame, bbox)
            if crop is not None:
                crops.append(crop)
                slots.append(index)

        out: list[np.ndarray | None] = [None] * len(bboxes)
        if not crops:
            return out
        vectors = self._session.run(None, {self._input_name: np.stack(crops)})[0]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)
        for slot, vector in zip(slots, vectors):
            out[slot] = vector
        return out

    def _prepare(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        ix1, iy1 = int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1))
        ix2, iy2 = int(np.clip(x2, 1, w)), int(np.clip(y2, 1, h))
        if ix2 - ix1 < _MIN_CROP_PX or iy2 - iy1 < _MIN_CROP_PX:
            return None
        height, width = self.input_size
        crop = cv2.resize(frame[iy1:iy2, ix1:ix2], (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)
