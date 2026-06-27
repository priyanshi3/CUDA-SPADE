"""
engine.py — Detection engine.

Tries to load the compiled spade_cuda GPU module.
If it is not found (module not built, or no NVIDIA GPU), falls back to
equivalent NumPy/pandas CPU implementations so the demo always runs.
"""

import os
import sys
import glob
import time

import numpy as np


# ── GPU module discovery ───────────────────────────────────────────────────────

def _load_gpu_module():
    # Walk up from this file to the repo root, then search the build tree
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pyd in glob.glob(
            os.path.join(repo, "build", "**", "spade_cuda*.pyd"),
            recursive=True):
        sys.path.insert(0, os.path.dirname(pyd))
        # Python 3.8+ no longer searches PATH for extension-module DLL deps;
        # os.add_dll_directory() is the correct way to register CUDA's DLLs.
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.dirname(pyd))
            for cuda_bin in [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
            ]:
                if os.path.isdir(cuda_bin):
                    os.add_dll_directory(cuda_bin)
        break
    try:
        import spade_cuda
        return spade_cuda
    except ImportError:
        return None


_cuda = _load_gpu_module()


def is_gpu_available() -> bool:
    return _cuda is not None


# ── CPU fallbacks ─────────────────────────────────────────────────────────────
# Used automatically when the GPU module is not available.

def _cpu_zscore(data: np.ndarray, window: int, threshold: float) -> dict:
    import pandas as pd
    t0   = time.perf_counter()
    s    = pd.Series(data)
    mean = s.rolling(window, center=True, min_periods=1).mean()
    std  = s.rolling(window, center=True, min_periods=1).std(ddof=0).fillna(0)
    z    = (s - mean).abs() / (std + 1e-12)
    flags = (z > threshold).astype(np.int32).to_numpy()
    return {
        "num_anomalies": int(flags.sum()),
        "processing_ms": (time.perf_counter() - t0) * 1000.0,
        "flags":         flags,
    }


def _cpu_fft(data: np.ndarray, sample_rate_hz: float,
             threshold_multiplier: float) -> dict:
    t0       = time.perf_counter()
    n        = len(data)
    spectrum = np.fft.rfft(data)
    power    = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    sorted_p = np.sort(power[1:])
    median   = float(sorted_p[len(sorted_p) // 2])
    thr      = threshold_multiplier * (median + 1e-12)
    flags    = (power > thr).astype(np.int32)
    flags[0] = 0   # skip DC bin
    masked   = power[1:] * flags[1:]
    peak_bin = int(np.argmax(masked)) + 1 if masked.any() else 0
    return {
        "num_anomalous_bins": int(flags.sum()),
        "processing_ms":      (time.perf_counter() - t0) * 1000.0,
        "peak_freq_hz":       float(peak_bin * sample_rate_hz / n),
        "bin_flags":          flags,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_zscore(data: np.ndarray, window: int, threshold: float) -> dict:
    if _cuda is not None:
        return _cuda.detect_anomalies(data, window=window, threshold=threshold)
    return _cpu_zscore(data, window, threshold)


def run_fft(data: np.ndarray, sample_rate_hz: float,
            threshold_multiplier: float) -> dict:
    if _cuda is not None:
        return _cuda.detect_spectral(data,
                                     sample_rate_hz=sample_rate_hz,
                                     threshold_multiplier=threshold_multiplier)
    return _cpu_fft(data, sample_rate_hz, threshold_multiplier)
