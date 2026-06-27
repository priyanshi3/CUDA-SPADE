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
