// test_hello_gpu.cu
//
// Run this to confirm:
//   1. CUDA is installed correctly
//   2. Your GPU is detected
//   3. The vector addition kernel produces correct output
//   4. The z-score kernel flags known anomalies
//   5. The cuFFT kernel detects a bearing fault at 235 Hz
//
// Run after building:   ./tests/test_hello_gpu (Linux)
//                       tests\test_hello_gpu.exe (Windows)

#include "spade.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

// Simple test helper — prints PASS or FAIL
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (cond) { printf("  PASS: %s\n", msg); }                   \
        else       { printf("  FAIL: %s\n", msg); }                   \
    } while(0)

// ── Test 1: GPU info ──────────────────────────────────────────────────────────
void test_device_info() {
    printf("\n[Test 1] GPU device info\n");
    spade_print_device_info();
    printf("  (If you see your GPU name above, this test passed)\n");
}

// ── Test 2: Vector addition ───────────────────────────────────────────────────
void test_vector_add() {
    printf("\n[Test 2] Vector addition on GPU\n");

    const int n = 1'000'000;  // 1 million elements
    std::vector<float> a(n), b(n), out(n);

    // Fill with known values: a[i] = i, b[i] = 2*i, expected out[i] = 3*i
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(2 * i);
    }

    spade_vector_add(a.data(), b.data(), out.data(), n);

    // Verify results
    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = 3.0f * i;
        if (fabsf(out[i] - expected) > 1e-3f) {
            printf("  Mismatch at i=%d: got %.2f, expected %.2f\n", i, out[i], expected);
            correct = false;
            break;
        }
    }

    CHECK(correct, "1M element vector add — all values correct");
}

// ── Test 3: Z-score anomaly detection ────────────────────────────────────────
void test_zscore_detect() {
    printf("\n[Test 3] Z-score anomaly detection\n");

    const int n = 10'000;
    std::vector<float> data(n);
    std::vector<int>   flags(n);

    // Simulate a temperature sensor: baseline 25.0°C, noise ±0.5°C
    // Then inject 3 obvious spikes at known positions
    for (int i = 0; i < n; i++) {
        data[i] = 25.0f + 0.3f * sinf((float)i / 50.0f);  // gentle sinusoidal
    }
    // Inject spikes — these should be flagged as anomalies
    int spike_positions[] = { 1000, 4500, 8200 };
    for (int p : spike_positions) {
        data[p] = 35.0f;  // 10°C spike — way outside normal range
    }

    AnomalyResult result = spade_zscore_detect(
        data.data(), flags.data(), n,
        /*window=*/100, /*threshold=*/3.0f
    );

    printf("  GPU kernel time : %.3f ms\n", result.processing_ms);
    printf("  Total anomalies : %d\n", result.num_anomalies);

    // Check that each injected spike was detected
    bool spike_1000 = (flags[1000] == 1);
    bool spike_4500 = (flags[4500] == 1);
    bool spike_8200 = (flags[8200] == 1);

    CHECK(spike_1000, "Spike at position 1000 detected");
    CHECK(spike_4500, "Spike at position 4500 detected");
    CHECK(spike_8200, "Spike at position 8200 detected");
    CHECK(result.num_anomalies < 100, "False positive rate reasonable (<100 flags in 10K points)");
}

// ── Test 4: cuFFT bearing fault detection ────────────────────────────────────
//
// We inject a 235 Hz bearing fault into a healthy vibration signal and verify
// the FFT detector finds it as the loudest spectral peak.
//
// Key design choice — why n=1000, sample_rate=1000?
//   Bin width = sample_rate / n = 1.0 Hz.
//   Every integer-Hz frequency lands EXACTLY on a bin centre.
//   A sinusoid that fits perfectly into the FFT window contributes power to
//   ONE bin only (no spectral leakage).  This gives a clean, predictable test:
//   we inject 4 frequencies → exactly 4 bins get flagged.
//
//   With non-power-of-two n the FFT uses mixed-radix decomposition; cuFFT
//   handles this automatically.
void test_fft_detect() {
    printf("\n[Test 4] cuFFT spectral bearing fault detection\n");

    // 1000 samples at 1000 Hz = 1 second.  Bin width = 1 Hz.
    const int   n              = 1000;
    const float sample_rate_hz = 1000.0f;
    const float PI             = 3.14159265358979f;

    std::vector<float> data(n);

    // Healthy vibration: 50 Hz fundamental + 100 Hz and 150 Hz harmonics.
    // All three are integer Hz → land exactly on bins 50, 100, 150.
    for (int i = 0; i < n; i++) {
        float t = (float)i / sample_rate_hz;
        data[i] = 0.50f * sinf(2.0f * PI * 50.0f  * t)
                + 0.20f * sinf(2.0f * PI * 100.0f * t)
                + 0.08f * sinf(2.0f * PI * 150.0f * t);
    }

    // Inject BPFO bearing fault at exactly 235 Hz (bin 235).
    // Amplitude 2.0 → power = 4.0, which is 16× the 50 Hz fundamental (0.5² = 0.25).
    // The 235 Hz bin will dominate the spectrum and be returned as peak_freq.
    for (int i = 0; i < n; i++) {
        float t = (float)i / sample_rate_hz;
        data[i] += 2.0f * sinf(2.0f * PI * 235.0f * t);
    }

    int num_bins = n / 2 + 1;   // 501 bins for n=1000
    std::vector<int> bin_flags(num_bins, 0);
    float peak_freq = 0.0f;

    SpectralResult result = spade_fft_detect(
        data.data(), bin_flags.data(), n,
        sample_rate_hz,
        /*threshold_multiplier=*/5.0f,
        &peak_freq
    );

    printf("  GPU time (FFT + power kernel): %.3f ms\n", result.processing_ms);
    printf("  Anomalous bins               : %d / %d\n",
           result.num_anomalous_bins, num_bins);
    printf("  Peak anomaly frequency       : %.1f Hz\n", peak_freq);

    // For n=1000, sample_rate=1000: bin k = frequency in Hz exactly.
    // bin 235 IS the 235 Hz BPFO bearing fault frequency.
    // The three meaningful checks are:
    //   1. Something was flagged at all.
    //   2. The loudest flagged bin is the 235 Hz BPFO (not one of the healthy harmonics).
    //   3. The 235 Hz bin itself is specifically flagged (direct confirmation).
    bool found_anomalies    = (result.num_anomalous_bins > 0);
    bool peak_near_bpfo     = (fabsf(peak_freq - 235.0f) < 2.0f);
    bool bpfo_bin_flagged   = (bin_flags[235] == 1);

    CHECK(found_anomalies,  "Spectral anomalies detected");
    CHECK(peak_near_bpfo,   "Peak anomaly at ~235 Hz (bearing fault BPFO)");
    CHECK(bpfo_bin_flagged, "Bin 235 (= 235 Hz BPFO) is explicitly flagged");
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    printf("=========================================\n");
    printf("  CUDA-SPADE — verification tests\n");
    printf("=========================================\n");

    test_device_info();
    test_vector_add();
    test_zscore_detect();
    test_fft_detect();

    printf("\n=========================================\n");
    printf("  Done. If all tests PASS, you are ready.\n");
    printf("=========================================\n\n");
    return 0;
}
