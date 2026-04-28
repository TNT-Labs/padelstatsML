"""CPU inference optimization for ARM/x86 hosts without a GPU.

Call configure_for_cpu() once at worker startup. On Pi 5 (Cortex-A76, 4 cores)
this is the single biggest latency win: PyTorch defaults to creating too many
threads and they compete for the 4 physical cores.
"""
from __future__ import annotations

import os


def configure_for_cpu(n_threads: int | None = None) -> None:
    """Tune PyTorch and OpenCV thread counts for CPU-only inference.

    Args:
        n_threads: number of intra-op threads.  Defaults to env var
                   TORCH_NUM_THREADS, then os.cpu_count(), capped at 4.
    """
    # Resolve thread count before importing torch (env vars read at import time)
    if n_threads is None:
        n_threads = int(os.environ.get("TORCH_NUM_THREADS", "") or 0) or min(os.cpu_count() or 4, 4)

    # Set env vars so any subsequently imported native lib also picks them up
    os.environ["OMP_NUM_THREADS"]        = str(n_threads)
    os.environ["MKL_NUM_THREADS"]        = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"]    = str(n_threads)

    try:
        import torch
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
        # Disable gradient tracking globally — inference-only worker
        torch.set_grad_enabled(False)
        # oneDNN (ARM NEON / AVX optimizations) — enabled by default in PyTorch 2+
        # but make it explicit
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
    except ImportError:
        pass

    try:
        import cv2
        cv2.setNumThreads(n_threads)
    except ImportError:
        pass
