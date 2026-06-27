#pragma once
#include <cstdint>

// ── Result struct returned by anomaly detection ───────────────────────────────
struct AnomalyResult {
    int    num_anomalies;   // total flagged points
    float  processing_ms;  // GPU kernel time in milliseconds
};

// ── Phase 1: basics ───────────────────────────────────────────────────────────

// Print device info (GPU name, memory, compute capability).
// Call this first thing — confirms your GPU is visible.
void spade_print_device_info();

// Add two float arrays on the GPU: out[i] = a[i] + b[i]
// Your first kernel. Shows GPU parallelism working.
void spade_vector_add(const float* a, const float* b, float* out, int n);

// ── Phase 2: anomaly detection ────────────────────────────────────────────────

// Z-score sliding window anomaly detector.
// For each point i, computes mean and std of the surrounding `window` values,
// then flags i as anomalous if |z-score| > threshold.
// anomaly_flags[i] = 1 if anomalous, 0 otherwise.
AnomalyResult spade_zscore_detect(
    const float* sensor_data,   // input: n sensor readings on HOST (CPU)
    int*         anomaly_flags, // output: n flags on HOST (CPU), 1 = anomaly
    int          n,             // number of samples
    int          window,        // sliding window size (e.g. 100)
    float        threshold      // z-score threshold (e.g. 3.0)
);

// ── Phase 3: cuFFT spectral anomaly detection ─────────────────────────────────

// Holds results from the frequency-domain detector.
struct SpectralResult {
    int   num_anomalous_bins; // how many frequency bins were flagged
    float processing_ms;      // total GPU time (FFT + power kernel), in ms
};

// Frequency-domain anomaly detection using cuFFT.
//
// Converts n time-domain samples → power spectrum on the GPU (via real-to-complex
// FFT), then flags any bin whose power exceeds threshold_multiplier × median power.
//
// This catches bearing faults (sharp spike at the BPFO frequency ~235 Hz) and other
// narrowband disturbances that are invisible in the raw time series but obvious as
// a spectral spike.
//
// Output:
//   bin_flags[k] = 1 if the k-th frequency bin is anomalous.  k = 0 … n/2.
//   *peak_freq_hz = frequency (Hz) of the single highest-power anomalous bin,
//                  or -1.0 if nothing was flagged.
//
// Caller must allocate bin_flags with at least (n/2 + 1) elements.
SpectralResult spade_fft_detect(
    const float* sensor_data,          // HOST: n time-domain samples
    int*         bin_flags,            // HOST: (n/2+1) anomaly flags (caller allocates)
    int          n,                    // number of samples (power-of-two preferred)
    float        sample_rate_hz,       // e.g. 1000.0 for the vibration sensor
    float        threshold_multiplier, // flag bins above X × median power (e.g. 5.0)
    float*       peak_freq_hz          // OUTPUT: Hz of loudest anomalous bin (or -1)
);
