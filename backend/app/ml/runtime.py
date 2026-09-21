"""CPU runtime tuning for the Raspberry Pi 5 (4x Cortex-A76, no GPU).

Thread oversubscription is the classic way to lose 30% of throughput on a
4-core board: onnxruntime, OpenCV and OpenBLAS each default to spawning one
thread per core, and then fight each other. Everything is pinned to the same
budget here, before those libraries are imported.
"""
from __future__ import annotations

import os


def configure_cpu(n_threads: int = 0) -> int:
    """Pin every native thread pool to the same budget. Returns the budget."""
    if n_threads <= 0:
        try:
            n_threads = len(os.sched_getaffinity(0))   # type: ignore[attr-defined]
        except (AttributeError, OSError):
            n_threads = os.cpu_count() or 4
        n_threads = min(n_threads, 4)

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n_threads)

    try:
        import cv2
        # OpenCV is used for decode, resize and colour conversion, which run
        # between inference calls — it gets the same budget, not a share.
        cv2.setNumThreads(n_threads)
    except ImportError:
        pass

    return n_threads
