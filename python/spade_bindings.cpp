// spade_bindings.cpp — pybind11 bridge between CUDA library and Python
//
// How it works:
//   pybind11 wraps C++ functions so Python can call them like normal Python
//   functions.  A numpy array is just a struct with a data pointer and a size;
//   pybind11 exposes that pointer so we can hand it directly to CUDA.
//
// Build: cmake --build build --config Release  (after running cmake -B build)
// Use:
//   import sys; sys.path.insert(0, "build/python/Release")
//   import spade_cuda
//   result = spade_cuda.detect_anomalies(data, window=100, threshold=3.0)
//   print(result["num_anomalies"], result["processing_ms"])

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "spade.h"

namespace py = pybind11;

// Shorthand for a contiguous C-order float32 array.
// py::array::forcecast means pybind11 will convert float64 → float32 if needed.
using FloatArr = py::array_t<float, py::array::c_style | py::array::forcecast>;

// ── detect_anomalies ──────────────────────────────────────────────────────────
// Wraps spade_zscore_detect (Phase 2).
// Returns a Python dict so the caller doesn't need to pre-allocate output arrays.
py::dict detect_anomalies(FloatArr sensor_data, int window, float threshold)
{
    auto info = sensor_data.request();
    if (info.ndim != 1)
        throw std::runtime_error("sensor_data must be a 1-D array");

    const int n = static_cast<int>(info.size);

    // Allocate the output array on the Python heap so it's a normal numpy array
    py::array_t<int> flags(n);
    auto flags_info = flags.request();

    AnomalyResult result;
    {
        // Release the Python GIL (Global Interpreter Lock) while the GPU runs.
        // Without this, no other Python thread can run during the GPU call —
        // even though the CPU is just waiting for the GPU to finish.
        py::gil_scoped_release release;
        result = spade_zscore_detect(
            static_cast<const float*>(info.ptr),
            static_cast<int*>(flags_info.ptr),
            n, window, threshold);
    }

    py::dict d;
    d["flags"]         = flags;           // np.ndarray[int32], length n
    d["num_anomalies"] = result.num_anomalies;
    d["processing_ms"] = result.processing_ms;
    return d;
}

// ── detect_spectral ───────────────────────────────────────────────────────────
// Wraps spade_fft_detect (Phase 3).
py::dict detect_spectral(FloatArr sensor_data,
                          float sample_rate_hz,
                          float threshold_multiplier)
{
    auto info = sensor_data.request();
    if (info.ndim != 1)
        throw std::runtime_error("sensor_data must be a 1-D array");

    const int n        = static_cast<int>(info.size);
    const int num_bins = n / 2 + 1;

    py::array_t<int> bin_flags(num_bins);
    auto bf_info = bin_flags.request();
    float peak_freq_hz = -1.0f;

    SpectralResult result;
    {
        py::gil_scoped_release release;
        result = spade_fft_detect(
            static_cast<const float*>(info.ptr),
            static_cast<int*>(bf_info.ptr),
            n, sample_rate_hz, threshold_multiplier, &peak_freq_hz);
    }

    py::dict d;
    d["bin_flags"]          = bin_flags;           // np.ndarray[int32], length n/2+1
    d["num_anomalous_bins"] = result.num_anomalous_bins;
    d["processing_ms"]      = result.processing_ms;
    d["peak_freq_hz"]       = peak_freq_hz;
    return d;
}

// ── detect_anomalies_fast ──────────────────────────────────────────────────────
// Wraps spade_zscore_streamed (Phase 4): shared memory kernel + CUDA streams.
py::dict detect_anomalies_fast(FloatArr sensor_data,
                                int   window,
                                float threshold,
                                int   num_streams)
{
    auto info = sensor_data.request();
    if (info.ndim != 1)
        throw std::runtime_error("sensor_data must be a 1-D array");

    const int n = static_cast<int>(info.size);

    py::array_t<int> flags(n);
    auto flags_info = flags.request();

    AnomalyResult result;
    {
        py::gil_scoped_release release;
        result = spade_zscore_streamed(
            static_cast<const float*>(info.ptr),
            static_cast<int*>(flags_info.ptr),
            n, window, threshold, num_streams);
    }

    py::dict d;
    d["flags"]         = flags;
    d["num_anomalies"] = result.num_anomalies;
    d["processing_ms"] = result.processing_ms;
    return d;
}

// ── Module definition ─────────────────────────────────────────────────────────
PYBIND11_MODULE(spade_cuda, m)
{
    m.doc() = "CUDA-SPADE: GPU-accelerated industrial sensor anomaly detection";

    m.def("detect_anomalies", &detect_anomalies,
          py::arg("sensor_data"),
          py::arg("window")    = 100,
          py::arg("threshold") = 3.0f,
          "Z-score sliding window anomaly detection on GPU (Phase 2 kernel).\n"
          "Returns dict: flags[n], num_anomalies, processing_ms");

    m.def("detect_spectral", &detect_spectral,
          py::arg("sensor_data"),
          py::arg("sample_rate_hz")       = 1000.0f,
          py::arg("threshold_multiplier") = 5.0f,
          "Frequency-domain anomaly detection via cuFFT (Phase 3).\n"
          "Returns dict: bin_flags[n/2+1], num_anomalous_bins, processing_ms, peak_freq_hz");

    m.def("detect_anomalies_fast", &detect_anomalies_fast,
          py::arg("sensor_data"),
          py::arg("window")      = 100,
          py::arg("threshold")   = 3.0f,
          py::arg("num_streams") = 3,
          "Shared memory kernel + CUDA streams (Phase 4).\n"
          "Returns dict: flags[n], num_anomalies, processing_ms");
}
