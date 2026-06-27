"""
generate_data.py — Synthetic MES/SCADA sensor data generator

Generates realistic industrial sensor streams with injected anomalies.
Uses your MES/SCADA domain knowledge to make the data believable.

Output: sensor_data.csv with columns:
    timestamp, temperature, vibration, pressure, is_anomaly

Run: python python/generate_data.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

def generate_temperature(n: int, sample_rate_hz: float = 10.0) -> np.ndarray:
    """
    Simulates a temperature sensor on an industrial motor.
    - Baseline: 65°C (motor running warm)
    - Daily cycle: ±3°C (room temperature effect)
    - Gaussian noise: ±0.5°C (sensor noise)
    """
    t = np.arange(n) / sample_rate_hz
    baseline   = 65.0
    daily_wave = 3.0 * np.sin(2 * np.pi * t / (24 * 3600))
    noise      = np.random.normal(0, 0.5, n)
    return baseline + daily_wave + noise


def generate_vibration(n: int, sample_rate_hz: float = 1000.0) -> np.ndarray:
    """
    Simulates a vibration sensor on a rotating machine (e.g. pump, compressor).
    - Fundamental frequency: 50 Hz (motor running at 3000 RPM)
    - Harmonics: 100 Hz, 150 Hz (normal for rotating machinery)
    - Noise floor: small random vibration
    This is what a healthy machine looks like in the frequency domain.
    """
    t = np.arange(n) / sample_rate_hz
    fundamental = 0.5  * np.sin(2 * np.pi * 50  * t)   # 50 Hz
    harmonic2   = 0.2  * np.sin(2 * np.pi * 100 * t)   # 2nd harmonic
    harmonic3   = 0.08 * np.sin(2 * np.pi * 150 * t)   # 3rd harmonic
    noise       = np.random.normal(0, 0.02, n)
    return fundamental + harmonic2 + harmonic3 + noise


def generate_pressure(n: int) -> np.ndarray:
    """
    Simulates a pressure sensor in a hydraulic system.
    - Baseline: 6.0 bar (system operating pressure)
    - Slow drift: ±0.2 bar (pump wear over time)
    - Noise: ±0.05 bar (sensor + fluid turbulence)
    """
    baseline = 6.0
    drift    = 0.2 * np.sin(np.linspace(0, 2 * np.pi, n))
    noise    = np.random.normal(0, 0.05, n)
    return baseline + drift + noise


def inject_anomalies(data: np.ndarray, sensor_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Injects realistic industrial fault patterns.
    Returns (modified_data, anomaly_mask).
    """
    n = len(data)
    anomaly_mask = np.zeros(n, dtype=int)
    data = data.copy()

    if sensor_name == "temperature":
        # Fault type: overheating spike (e.g. cooling fan failure)
        # Real pattern: rapid rise then gradual cooling
        for start in [10_000, 45_000, 78_000]:
            width = 300
            end   = min(start + width, n)
            spike = 20.0 * np.exp(-np.linspace(0, 3, end - start))
            data[start:end] += spike
            anomaly_mask[start:end] = 1

        # Fault type: stuck sensor (value frozen — common SCADA fault)
        stuck_start = 25_000
        stuck_end   = 25_200
        data[stuck_start:stuck_end] = data[stuck_start]
        anomaly_mask[stuck_start:stuck_end] = 1

    elif sensor_name == "vibration":
        # Fault type: bearing fault — new frequency appears at 235 Hz
        # This is a classic BPFO (ball pass frequency outer race) signature
        fault_start = 30_000
        fault_end   = min(fault_start + 5000, n)
        t = np.arange(fault_end - fault_start) / 1000.0
        bearing_fault = 0.4 * np.sin(2 * np.pi * 235 * t)
        data[fault_start:fault_end] += bearing_fault
        anomaly_mask[fault_start:fault_end] = 1

        # Fault type: imbalance — fundamental amplitude increases suddenly
        imbalance_start = 60_000
        imbalance_end   = min(imbalance_start + 3000, n)
        t = np.arange(imbalance_end - imbalance_start) / 1000.0
        data[imbalance_start:imbalance_end] += 1.5 * np.sin(2 * np.pi * 50 * t)
        anomaly_mask[imbalance_start:imbalance_end] = 1

    elif sensor_name == "pressure":
        # Fault type: pressure drop (pipe leak or valve failure)
        for start in [15_000, 55_000]:
            end = min(start + 500, n)
            data[start:end] -= 2.5   # sudden 2.5 bar drop
            anomaly_mask[start:end] = 1

        # Fault type: pressure surge (water hammer)
        surge_pos = 35_000
        if surge_pos < n:
            data[surge_pos] += 8.0   # 8 bar spike — dangerous
            anomaly_mask[surge_pos]  = 1

    return data, anomaly_mask


def main():
    N = 100_000   # 100K samples per sensor
    print(f"Generating {N:,} samples per sensor...")

    # Generate clean signals
    temperature = generate_temperature(N, sample_rate_hz=10.0)
    vibration   = generate_vibration(N, sample_rate_hz=1000.0)
    pressure    = generate_pressure(N)

    # Inject faults
    temperature, temp_faults = inject_anomalies(temperature, "temperature")
    vibration,   vib_faults  = inject_anomalies(vibration,  "vibration")
    pressure,    pres_faults  = inject_anomalies(pressure,   "pressure")

    # Combined anomaly mask (any sensor anomalous = anomaly)
    any_fault = np.clip(temp_faults + vib_faults + pres_faults, 0, 1)

    print(f"Anomaly rate: {any_fault.mean()*100:.1f}% ({any_fault.sum():,} samples)")

    # Save to CSV
    out_path = Path(__file__).parent.parent / "sensor_data.csv"
    df = pd.DataFrame({
        "sample_idx":  np.arange(N),
        "temperature": temperature,
        "vibration":   vibration,
        "pressure":    pressure,
        "is_anomaly":  any_fault,
    })
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # Quick plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, col, name in zip(axes,
                             ["temperature", "vibration", "pressure"],
                             ["Temperature (°C)", "Vibration (g)", "Pressure (bar)"]):
        x = df["sample_idx"]
        ax.plot(x, df[col], linewidth=0.4, color="#378ADD", label=name)
        fault_x = x[df["is_anomaly"] == 1]
        fault_y = df[col][df["is_anomaly"] == 1]
        ax.scatter(fault_x, fault_y, color="#E24B4A", s=1, label="anomaly", zorder=3)
        ax.set_ylabel(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Sample index")
    plt.suptitle("Synthetic industrial sensor data (red = injected anomalies)", fontsize=12)
    plt.tight_layout()
    plot_path = out_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
