// test_hello_gpu.cu
//
// Your "day one" test. Run this to confirm:
//   1. CUDA is installed correctly
//   2. Your GPU is detected
//   3. The vector addition kernel produces correct output
//   4. The z-score kernel flags known anomalies
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

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    printf("=========================================\n");
    printf("  CUDA-SPADE — Day 1 verification tests\n");
    printf("=========================================\n");

    test_device_info();
    test_vector_add();
    test_zscore_detect();

    printf("\n=========================================\n");
    printf("  Done. If all tests PASS, you are ready.\n");
    printf("=========================================\n\n");
    return 0;
}
