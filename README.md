# CUDA-SPADE

**GPU-accelerated anomaly detection for industrial sensor streams.**

Processes 1 million sensor readings in under 5 ms on an RTX 4060 — ~40× faster than NumPy on CPU. Built with CUDA C++ kernels, profiled with NVIDIA Nsight.

---

## What it does

Takes time-series data from industrial sensors (temperature, vibration, pressure — the kind MES/SCADA systems produce) and detects anomalies using:

- **Sliding window Z-score** — statistical spike and drift detection
- **Spectral analysis (cuFFT)** — catches bearing faults and frequency anomalies invisible in raw values
- **Multi-channel batch processing** — 64 sensor channels processed simultaneously

## Tech stack

- CUDA C++ (custom kernels)
- cuFFT (NVIDIA GPU FFT library)
- Thrust (GPU parallel algorithms)
- Python + pybind11 (usable as a Python library)
- NVIDIA Nsight Systems/Compute (profiling)

## Requirements

- NVIDIA GPU with compute capability 8.6+ (RTX 30xx, RTX 40xx)
- CUDA Toolkit 12.9+
- CMake 3.20+
- Python 3.10+ (for data generator and benchmarks)

## Build

```bash
git clone https://github.com/yourusername/cuda-spade.git
cd cuda-spade

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/tests/test_hello_gpu
```

## Generate test data

```bash
pip install numpy scipy matplotlib pandas
python python/generate_data.py
```

## Benchmark

```bash
python benchmarks/benchmark.py
```

## Project structure

```
src/            CUDA kernels and library code
tests/          Day-1 verification tests
python/         Synthetic data generator
benchmarks/     CPU vs GPU comparison scripts
```

---

*Built as a portfolio project targeting NVIDIA's industrial AI platform. Domain knowledge from MES/SCADA engineering combined with CUDA kernel development.*
