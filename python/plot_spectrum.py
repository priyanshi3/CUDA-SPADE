"""
plot_spectrum.py — Visualize healthy vs bearing-fault vibration spectrum

Run AFTER generate_data.py has created sensor_data.csv:
    python python/generate_data.py
    python python/plot_spectrum.py

What it shows:
    Left panel:  Power spectrum of a healthy vibration segment.
                 You see only the expected motor harmonics (50, 100, 150 Hz).
    Right panel: Power spectrum during a bearing fault (samples 30000-34095).
                 An extra spike appears at 235 Hz — the BPFO (ball pass
                 frequency outer race).  This is what cuFFT detects.

Why this matters:
    In raw time-domain data the bearing fault looks like a slightly higher
    vibration amplitude — easy to miss with Z-score alone.
    In the frequency domain the 235 Hz spike is unmistakable.
    That's the advantage of spectral analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def power_spectrum(signal: np.ndarray, sample_rate_hz: float):
    """Returns (frequencies_hz, power) for the one-sided spectrum."""
    n       = len(signal)
    fft_out = np.fft.rfft(signal)
    # Normalise by n so amplitude is independent of window length
    power   = (np.abs(fft_out) ** 2) / n
    freqs   = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    return freqs, power


def annotate_freq(ax, freq_hz: float, label: str, color: str = "gray"):
    """Draw a labelled vertical line at a known frequency."""
    ax.axvline(freq_hz, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
    # Place text just above the bottom of the visible y-range
    y_min, y_max = ax.get_ylim()
    ax.text(freq_hz + 2, y_min * 10, label,
            fontsize=7, color=color, va="bottom")


def main():
    csv_path = Path(__file__).parent.parent / "sensor_data.csv"
    if not csv_path.exists():
        print(f"[!] sensor_data.csv not found at {csv_path}")
        print("    Run:  python python/generate_data.py  first.")
        return

    print(f"Loading {csv_path} ...")
    df        = pd.read_csv(csv_path)
    vibration = df["vibration"].values

    SAMPLE_RATE = 1000.0   # vibration sensor runs at 1 kHz
    SEG_LEN     = 4096     # samples per analysis window (~4 seconds)

    # ── Healthy segment: before any fault (samples 0–4095) ───────────────────
    seg_healthy = vibration[0:SEG_LEN]
    freqs_h, power_h = power_spectrum(seg_healthy, SAMPLE_RATE)

    # ── Fault segment: bearing fault injected at sample 30000 ─────────────────
    FAULT_START = 30_000
    seg_fault   = vibration[FAULT_START : FAULT_START + SEG_LEN]
    freqs_f, power_f = power_spectrum(seg_fault, SAMPLE_RATE)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax_h, ax_f) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, freqs, power, title, line_color in [
        (ax_h, freqs_h, power_h,
         "Healthy vibration (samples 0–4095)",         "#378ADD"),
        (ax_f, freqs_f, power_f,
         f"Bearing fault (samples {FAULT_START:,}–{FAULT_START+SEG_LEN-1:,})", "#E24B4A"),
    ]:
        ax.semilogy(freqs, power + 1e-10, color=line_color, linewidth=0.7)
        ax.set_xlim(0, 400)
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("Power (log scale)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25)

    # Mark known healthy harmonics on both panels
    for ax in (ax_h, ax_f):
        for freq, lbl in [(50, "50 Hz\nmotor"), (100, "100 Hz\n2nd"), (150, "150 Hz\n3rd")]:
            annotate_freq(ax, freq, lbl, color="gray")

    # Highlight the BPFO fault frequency only on the right panel
    ax_f.axvline(235, color="#FF8C00", linestyle="-", linewidth=1.5,
                 label="235 Hz BPFO (bearing fault)")
    # Find the power at the 235 Hz bin to place the label near the peak
    bpfo_bin   = int(round(235 * SEG_LEN / SAMPLE_RATE))
    bpfo_power = power_f[bpfo_bin]
    ax_f.text(237, bpfo_power * 0.15,
              "235 Hz\nBPFO\n← fault here",
              fontsize=8, color="#FF8C00", fontweight="bold")
    ax_f.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        "cuFFT spectral analysis — healthy vs. bearing fault\n"
        "(bearing faults are invisible in time-domain but obvious in frequency-domain)",
        fontsize=11
    )
    plt.tight_layout()

    out_path = Path(__file__).parent.parent / "spectrum_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
