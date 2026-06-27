"""
benchmark.py — CPU vs GPU anomaly detection benchmark

Compares pandas rolling Z-score (CPU) against spade_cuda (GPU) at four
data scales.  Prints a timing table and saves a speedup chart.

Run from the repo root after building:
    python benchmarks/benchmark.py

The spade_cuda module must be built first:
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
"""

import sys
import os
import time
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt

# -- Locate the compiled spade_cuda module -------------------------------------
# CMake puts the .pyd in build/python/Release/ or build/python/Debug/
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pyd in glob.glob(os.path.join(_repo, "build", "python", "**", "spade_cuda*.pyd"),
                      recursive=True):
    _pyd_dir = os.path.dirname(_pyd)
    sys.path.insert(0, _pyd_dir)
    # Python 3.8+ does not search PATH for DLL dependencies of extension modules.
    # We must explicitly register CUDA's DLL directory so the loader can find
    # cufft64_12.dll and friends before attempting the import.
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_pyd_dir)
        for _cuda_bin in [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
        ]:
            if os.path.isdir(_cuda_bin):
                os.add_dll_directory(_cuda_bin)
    break

try:
    import spade_cuda
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("[!] spade_cuda module not found.")
    print("    Build it with:")
    print("      cmake -B build -DCMAKE_BUILD_TYPE=Release")
    print("      cmake --build build --config Release")
    print("    CPU-only benchmarks will run.\n")


# -- CPU baseline: pandas rolling Z-score -------------------------------------
# This is what a data scientist would write without GPU acceleration.
# pandas rolling is implemented in C (via Cython), so this is fast CPU code.
def cpu_zscore(data: np.ndarray, window: int, threshold: float) -> np.ndarray:
    s         = pd.Series(data)
    roll_mean = s.rolling(window, center=True, min_periods=1).mean()
    roll_std  = s.rolling(window, center=True, min_periods=1).std(ddof=0).fillna(0.0)
    z         = (s - roll_mean).abs() / (roll_std + 1e-12)
    return (z > threshold).astype(np.int32).to_numpy()


# -- Synthetic sensor data -----------------------------------------------------
def make_sensor_data(n: int, seed: int = 42) -> np.ndarray:
    """Temperature sensor: 65 °C baseline, sinusoidal drift, Gaussian noise."""
    rng  = np.random.default_rng(seed)
    t    = np.arange(n, dtype=np.float32)
    data = (65.0 + 0.5 * np.sin(t / 200.0) + rng.normal(0, 0.3, n)).astype(np.float32)
    # Inject obvious spikes so we can sanity-check results
    spike_indices = [n // 4, n // 2, 3 * n // 4]
    for idx in spike_indices:
        data[idx] = 90.0
    return data


# -- Benchmark runner ----------------------------------------------------------
WINDOW    = 100
THRESHOLD = 3.0
SIZES     = [10_000, 100_000, 1_000_000, 10_000_000]
REPEATS   = 3          # average over multiple runs to reduce noise


def time_fn(fn, repeats=REPEATS):
    """Return (mean_ms, result) averaged over `repeats` calls."""
    result = None
    total  = 0.0
    for _ in range(repeats):
        t0     = time.perf_counter()
        result = fn()
        total += (time.perf_counter() - t0) * 1000.0
    return total / repeats, result


def run_benchmarks():
    rows = []   # collected for the table

    print("CUDA-SPADE  vs  CPU (pandas rolling Z-score)")
    print("=" * 70)
    if not HAS_GPU:
        print("  GPU column skipped — build spade_cuda first.\n")

    for n in SIZES:
        data = make_sensor_data(n)

        # -- CPU ---------------------------------------------------------------
        cpu_ms, cpu_flags = time_fn(
            lambda: cpu_zscore(data, WINDOW, THRESHOLD))

        if HAS_GPU:
            # -- GPU: kernel time only (reported by CUDA events) ---------------
            # This excludes Python overhead and PCIe transfer time.
            gpu_result = spade_cuda.detect_anomalies(data, WINDOW, THRESHOLD)
            kernel_ms  = gpu_result["processing_ms"]

            # -- GPU: total wall time (including PCIe transfers + Python overhead)
            gpu_total_ms, gpu_result = time_fn(
                lambda: spade_cuda.detect_anomalies(data, WINDOW, THRESHOLD))

            speedup_total  = cpu_ms / gpu_total_ms
            speedup_kernel = cpu_ms / kernel_ms

            row = {
                "n":              n,
                "cpu_ms":         cpu_ms,
                "gpu_total_ms":   gpu_total_ms,
                "gpu_kernel_ms":  kernel_ms,
                "speedup_total":  speedup_total,
                "speedup_kernel": speedup_kernel,
                "gpu_anomalies":  gpu_result["num_anomalies"],
                "cpu_anomalies":  int(cpu_flags.sum()),
            }
        else:
            row = {
                "n":              n,
                "cpu_ms":         cpu_ms,
                "gpu_total_ms":   None,
                "gpu_kernel_ms":  None,
                "speedup_total":  None,
                "speedup_kernel": None,
                "gpu_anomalies":  None,
                "cpu_anomalies":  int(cpu_flags.sum()),
            }

        rows.append(row)

        # Print row
        if HAS_GPU:
            print(f"  N={n:>10,} | CPU {cpu_ms:7.1f} ms | "
                  f"GPU total {gpu_total_ms:6.1f} ms | "
                  f"kernel {kernel_ms:5.2f} ms | "
                  f"speedup {speedup_total:5.1f}x")
        else:
            print(f"  N={n:>10,} | CPU {cpu_ms:7.1f} ms")

    print("=" * 70)
    _print_table(rows)
    return rows


def _print_table(rows):
    print()
    print("Summary table (wall-clock speedup = CPU time / GPU total time)")
    print(f"  {'N':>12} | {'CPU (ms)':>10} | {'GPU total':>10} | "
          f"{'GPU kernel':>10} | {'Speedup':>8}")
    print("  " + "-" * 60)
    for r in rows:
        if r["gpu_total_ms"] is not None:
            print(f"  {r['n']:>12,} | {r['cpu_ms']:>10.1f} | "
                  f"{r['gpu_total_ms']:>10.1f} | "
                  f"{r['gpu_kernel_ms']:>10.2f} | "
                  f"{r['speedup_total']:>7.1f}x")
        else:
            print(f"  {r['n']:>12,} | {r['cpu_ms']:>10.1f} | "
                  f"{'N/A':>10} | {'N/A':>10} | {'N/A':>8}")
    print()


def plot_results(rows):
    ns      = [r["n"] for r in rows]
    cpu_ms  = [r["cpu_ms"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -- Left: raw timing ------------------------------------------------------
    ax = axes[0]
    ax.loglog(ns, cpu_ms, "o-", color="#E24B4A", linewidth=2,
              markersize=7, label="CPU (pandas rolling)")

    if HAS_GPU and rows[0]["gpu_total_ms"] is not None:
        gpu_total  = [r["gpu_total_ms"]  for r in rows]
        gpu_kernel = [r["gpu_kernel_ms"] for r in rows]
        ax.loglog(ns, gpu_total,  "s-", color="#378ADD", linewidth=2,
                  markersize=7, label="GPU total (incl. transfer)")
        ax.loglog(ns, gpu_kernel, "^--", color="#5CB85C", linewidth=1.5,
                  markersize=6, label="GPU kernel only")

    ax.set_xlabel("Number of samples",  fontsize=11)
    ax.set_ylabel("Time (ms, log scale)", fontsize=11)
    ax.set_title("Processing time vs dataset size", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # -- Right: speedup --------------------------------------------------------
    ax = axes[1]
    if HAS_GPU and rows[0]["speedup_total"] is not None:
        speedups = [r["speedup_total"] for r in rows]
        bars = ax.bar([str(f"{n:,}") for n in ns], speedups,
                      color="#378ADD", alpha=0.85)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1,
                   label="Break-even (1×)")
        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, f"{sp:.1f}×",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Speedup vs CPU (higher = better)", fontsize=11)
        ax.set_title("GPU speedup (wall-clock including PCIe transfer)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "GPU data not available\n(build spade_cuda first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("Number of samples", fontsize=11)

    plt.suptitle(
        "CUDA-SPADE: GPU-accelerated Z-score anomaly detection\n"
        "RTX 4060 Laptop GPU  vs  pandas rolling (CPU)",
        fontsize=12
    )
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "speedup_chart.png")
    plt.savefig(out, dpi=150)
    print(f"Chart saved → {out}")


if __name__ == "__main__":
    rows = run_benchmarks()
    plot_results(rows)
