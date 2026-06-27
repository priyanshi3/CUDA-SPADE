"""
sensor_sim.py — Synthetic sensor data for demo mode.

Each SensorSimulator keeps an internal time counter so consecutive
calls produce a continuous, seamless signal rather than restarting.
"""

import numpy as np


class SensorSimulator:
    """Generates realistic float32 sensor waveforms with optional fault injection."""

    def __init__(self, cfg: dict, rng_seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(rng_seed)
        self.t   = 0   # cumulative sample counter across all generate() calls

    def generate(self, n_samples: int, inject_fault: bool = False) -> np.ndarray:
        t = np.arange(self.t, self.t + n_samples, dtype=np.float64)
        self.t += n_samples
        stype = self.cfg.get("sensor_type", "generic")
        dispatch = {
            "temperature": self._temperature,
            "vibration":   self._vibration,
            "pressure":    self._pressure,
            "current":     self._current,
        }
        return dispatch.get(stype, self._generic)(t, n_samples, inject_fault)

    # ── Sensor waveforms ──────────────────────────────────────────────────────

    def _temperature(self, t, n, fault) -> np.ndarray:
        base  = self.cfg.get("baseline", 65.0)
        noise = self.cfg.get("noise", 0.3)
        data  = (base + 0.5 * np.sin(t / 50.0) +
                 self.rng.normal(0, noise, n)).astype(np.float32)
        if fault:
            # Inject 3 heat spikes spread across the middle of the batch
            idxs = self.rng.integers(n // 4, 3 * n // 4, size=3)
            data[idxs] = base + 15.0   # 15 C above normal — overheating
        return data

    def _vibration(self, t, n, fault) -> np.ndarray:
        sr    = self.cfg.get("sample_rate_hz", 1000.0)
        noise = self.cfg.get("noise", 0.05)
        ts    = t / sr   # time in seconds
        # Healthy: fundamental 50 Hz + 2nd and 3rd harmonics
        data  = (0.50 * np.sin(2 * np.pi * 50.0  * ts) +
                 0.20 * np.sin(2 * np.pi * 100.0 * ts) +
                 0.08 * np.sin(2 * np.pi * 150.0 * ts) +
                 self.rng.normal(0, noise, n)).astype(np.float32)
        if fault:
            # BPFO bearing fault at 235 Hz — amplitude 4x the fundamental
            # This is exactly what Phase 3 cuFFT detects
            data += (2.0 * np.sin(2 * np.pi * 235.0 * ts)).astype(np.float32)
        return data

    def _pressure(self, t, n, fault) -> np.ndarray:
        base  = self.cfg.get("baseline", 180.0)
        noise = self.cfg.get("noise", 1.5)
        data  = (base + 2.0 * np.sin(t / 200.0) +
                 self.rng.normal(0, noise, n)).astype(np.float32)
        if fault:
            data -= 25.0   # sudden 25-bar drop — valve leak or pump failure
        return data

    def _current(self, t, n, fault) -> np.ndarray:
        base  = self.cfg.get("baseline", 12.5)
        noise = self.cfg.get("noise", 0.15)
        data  = (base + 0.3 * np.sin(t / 100.0) +
                 self.rng.normal(0, noise, n)).astype(np.float32)
        if fault:
            data += 6.0   # overload current — motor stalling or mechanical jam
        return data

    def _generic(self, t, n, fault) -> np.ndarray:
        base  = self.cfg.get("baseline", 0.0)
        noise = self.cfg.get("noise", 1.0)
        data  = (base + self.rng.normal(0, noise, n)).astype(np.float32)
        if fault:
            idxs = self.rng.integers(0, n, size=5)
            data[idxs] = base + 10 * noise
        return data
