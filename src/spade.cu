// spade.cu — CUDA-SPADE: GPU-accelerated sensor anomaly detection
//
// HOW TO READ THIS FILE:
//   Functions marked __global__ run ON THE GPU (called "kernels").
//   Functions without __global__ run on the CPU.
//   The GPU executes a kernel across thousands of threads simultaneously —
//   each thread handles one element of the array independently.

#include "spade.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// ── Helper macro ──────────────────────────────────────────────────────────────
// Wraps every CUDA API call so errors are caught immediately with file/line info.
// Usage: CUDA_CHECK(cudaMalloc(...));
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ═════════════════════════════════════════════════════════════════════════════
// PHASE 1 — Device info + vector addition
// ═════════════════════════════════════════════════════════════════════════════

void spade_print_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("No CUDA-capable GPU found!\n");
        return;
    }

    for (int d = 0; d < device_count; d++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        printf("GPU %d: %s\n", d, prop.name);
        printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.1f GB\n",
               (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors    : %d\n", prop.multiProcessorCount);
        printf("  Max threads/block  : %d\n", prop.maxThreadsPerBlock);
        printf("  Warp size          : %d\n", prop.warpSize);
    }
}

// ── Kernel: vector addition ───────────────────────────────────────────────────
//
// __global__ means this function runs on the GPU across many threads.
//
// When you call this kernel with <<<blocks, threads>>>, CUDA launches
// (blocks × threads) copies of this function simultaneously.
// Each copy has a unique threadIdx.x and blockIdx.x, which together
// give it a unique global index i — so each thread processes one element.
//
// Example: n = 1,000,000 elements
//   threads_per_block = 256
//   blocks = ceil(1M / 256) = 3907
//   Total threads = 3907 × 256 ≈ 1M — each handles one array position
//
__global__ void kernel_vector_add(const float* a, const float* b,
                                   float* out, int n) {
    // Compute this thread's unique global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: last block may have more threads than remaining elements
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void spade_vector_add(const float* a_host, const float* b_host,
                       float* out_host, int n) {
    size_t bytes = n * sizeof(float);

    // Step 1: Allocate memory on the GPU ("device memory")
    float *a_dev, *b_dev, *out_dev;
    CUDA_CHECK(cudaMalloc(&a_dev,   bytes));
    CUDA_CHECK(cudaMalloc(&b_dev,   bytes));
    CUDA_CHECK(cudaMalloc(&out_dev, bytes));

    // Step 2: Copy input data from CPU → GPU
    CUDA_CHECK(cudaMemcpy(a_dev, a_host, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_dev, b_host, bytes, cudaMemcpyHostToDevice));

    // Step 3: Launch the kernel
    //   <<<number_of_blocks, threads_per_block>>>
    //   256 threads/block is a common default — fills one warp (32) × 8
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;  // ceiling division
    kernel_vector_add<<<blocks, threads>>>(a_dev, b_dev, out_dev, n);

    // Step 4: Wait for GPU to finish (kernels are async by default)
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Copy results back from GPU → CPU
    CUDA_CHECK(cudaMemcpy(out_host, out_dev, bytes, cudaMemcpyDeviceToHost));

    // Step 6: Free GPU memory (just like free() for malloc)
    CUDA_CHECK(cudaFree(a_dev));
    CUDA_CHECK(cudaFree(b_dev));
    CUDA_CHECK(cudaFree(out_dev));
}

// ═════════════════════════════════════════════════════════════════════════════
// PHASE 2 — Z-score sliding window anomaly detection
// ═════════════════════════════════════════════════════════════════════════════
//
// The algorithm:
//   For each point i in the sensor stream:
//     1. Look at the window of `window` points centered on i
//     2. Compute mean and std deviation of those points
//     3. z-score = (data[i] - mean) / std
//     4. If |z-score| > threshold → flag as anomaly
//
// The GPU advantage: all N points are processed simultaneously.
// On CPU, this is O(N × window). On GPU, it's roughly O(window) because
// each of the N threads handles one point independently.

__global__ void kernel_zscore_detect(
    const float* data,       // input sensor readings (on GPU)
    int*         flags,      // output anomaly flags (on GPU)
    int          n,          // total number of samples
    int          window,     // sliding window size
    float        threshold   // z-score threshold for anomaly
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Determine the window boundaries around point i
    int half   = window / 2;
    int start  = max(0, i - half);
    int end    = min(n - 1, i + half);
    int count  = end - start + 1;

    // Pass 1: compute mean of window
    float sum = 0.0f;
    for (int j = start; j <= end; j++) {
        sum += data[j];
    }
    float mean = sum / count;

    // Pass 2: compute standard deviation
    float sq_sum = 0.0f;
    for (int j = start; j <= end; j++) {
        float diff = data[j] - mean;
        sq_sum += diff * diff;
    }
    float std_dev = sqrtf(sq_sum / count);

    // Compute z-score and flag anomaly
    // Guard against division by zero (std ≈ 0 means constant signal)
    float z_score = 0.0f;
    if (std_dev > 1e-6f) {
        z_score = fabsf(data[i] - mean) / std_dev;
    }

    flags[i] = (z_score > threshold) ? 1 : 0;
}

AnomalyResult spade_zscore_detect(
    const float* data_host,
    int*         flags_host,
    int          n,
    int          window,
    float        threshold
) {
    size_t data_bytes  = n * sizeof(float);
    size_t flags_bytes = n * sizeof(int);

    // Allocate GPU memory
    float* data_dev;
    int*   flags_dev;
    CUDA_CHECK(cudaMalloc(&data_dev,  data_bytes));
    CUDA_CHECK(cudaMalloc(&flags_dev, flags_bytes));

    // Copy sensor data to GPU
    CUDA_CHECK(cudaMemcpy(data_dev, data_host, data_bytes, cudaMemcpyHostToDevice));

    // Time the kernel using CUDA events (more accurate than CPU timers)
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    CUDA_CHECK(cudaEventRecord(start_ev));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    kernel_zscore_detect<<<blocks, threads>>>(data_dev, flags_dev, n, window, threshold);

    CUDA_CHECK(cudaEventRecord(stop_ev));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    // Copy results back
    CUDA_CHECK(cudaMemcpy(flags_host, flags_dev, flags_bytes, cudaMemcpyDeviceToHost));

    // Count anomalies on CPU (small operation, not worth a kernel)
    int num_anomalies = 0;
    for (int i = 0; i < n; i++) {
        num_anomalies += flags_host[i];
    }

    // Cleanup
    CUDA_CHECK(cudaFree(data_dev));
    CUDA_CHECK(cudaFree(flags_dev));
    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));

    return { num_anomalies, elapsed_ms };
}

// ═════════════════════════════════════════════════════════════════════════════
// PHASE 3 — cuFFT spectral anomaly detection
// ═════════════════════════════════════════════════════════════════════════════
//
// Why FFT for anomaly detection?
//   The Z-score kernel catches spikes in raw amplitude.
//   A bearing fault changes the vibration FREQUENCY CONTENT — a new peak
//   appears at the BPFO frequency (~235 Hz for this machine).  In raw data
//   that's nearly invisible; in the frequency domain it's an unmistakable spike.
//
// cuFFT R2C (real-to-complex) pipeline:
//   Input:  n floats  (time domain)
//   Output: n/2+1 complex bins  (frequency domain)
//   Bin k → frequency = k × sample_rate_hz / n
//
// After the FFT we convert each complex bin to its POWER (re² + im²) with a
// small custom kernel, then find the median power on the CPU and flag any bin
// that exceeds threshold_multiplier × median.

// ── cuFFT error checker (mirrors CUDA_CHECK) ──────────────────────────────────
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult _r = (call);                                                \
        if (_r != CUFFT_SUCCESS) {                                              \
            fprintf(stderr, "cuFFT error at %s:%d — code %d\n",                \
                    __FILE__, __LINE__, (int)_r);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── Kernel: complex spectrum → power per bin ──────────────────────────────────
//
// cuFFT outputs cufftComplex = {float x (real), float y (imag)} for each bin.
// Power = re² + im²  (squared magnitude — proportional to signal energy).
// One thread per output bin — perfectly parallel, no dependencies.
__global__ void kernel_compute_power(
    const cufftComplex* spectrum,  // (n/2+1) complex bins from cuFFT
    float*              power,     // (n/2+1) power values to fill
    int                 num_bins
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_bins) return;
    float re = spectrum[k].x;
    float im = spectrum[k].y;
    power[k] = re * re + im * im;
}

// ── Host wrapper ───────────────────────────────────────────────────────────────
SpectralResult spade_fft_detect(
    const float* data_host,
    int*         bin_flags,
    int          n,
    float        sample_rate_hz,
    float        threshold_multiplier,
    float*       peak_freq_hz
) {
    int    num_bins   = n / 2 + 1;
    size_t data_bytes = (size_t)n        * sizeof(float);
    size_t comp_bytes = (size_t)num_bins * sizeof(cufftComplex);
    size_t powr_bytes = (size_t)num_bins * sizeof(float);

    // ── Step 1: Allocate GPU memory ───────────────────────────────────────────
    float*        data_dev;
    cufftComplex* spec_dev;
    float*        powr_dev;
    CUDA_CHECK(cudaMalloc(&data_dev, data_bytes));
    CUDA_CHECK(cudaMalloc(&spec_dev, comp_bytes));
    CUDA_CHECK(cudaMalloc(&powr_dev, powr_bytes));

    // ── Step 2: Copy time-domain data to GPU ──────────────────────────────────
    CUDA_CHECK(cudaMemcpy(data_dev, data_host, data_bytes, cudaMemcpyHostToDevice));

    // ── Step 3: Create cuFFT plan ─────────────────────────────────────────────
    // cufftPlan1d(plan, signal_length, transform_type, batch_count)
    // CUFFT_R2C = Real-to-Complex: takes n floats, outputs n/2+1 complex values.
    // The "1" at the end means we process 1 signal (not a batch).
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_R2C, 1));

    // ── Step 4: Time FFT + power kernel together using CUDA events ────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    // ── Step 5: Execute FFT — data_dev (float*) → spec_dev (complex*) ─────────
    // cufftExecR2C fills spec_dev with (n/2+1) frequency-domain complex values.
    CUFFT_CHECK(cufftExecR2C(plan, (cufftReal*)data_dev, spec_dev));

    // ── Step 6: Convert complex bins → scalar power ───────────────────────────
    int threads = 256;
    int blocks  = (num_bins + threads - 1) / threads;
    kernel_compute_power<<<blocks, threads>>>(spec_dev, powr_dev, num_bins);

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    // ── Step 7: Copy power spectrum back to CPU ────────────────────────────────
    // num_bins is at most a few thousand — copying it is fast.
    std::vector<float> power_host(num_bins);
    CUDA_CHECK(cudaMemcpy(power_host.data(), powr_dev, powr_bytes, cudaMemcpyDeviceToHost));

    // ── Step 8: Compute median power (baseline noise floor) ───────────────────
    // Median is more robust than mean — a few loud peaks don't skew it.
    // We sort a copy so the original order is preserved for flagging.
    std::vector<float> sorted_power = power_host;
    std::sort(sorted_power.begin(), sorted_power.end());
    float median_power    = sorted_power[num_bins / 2];
    float threshold_power = threshold_multiplier * (median_power + 1e-12f);

    // ── Step 9: Flag anomalous bins and find the loudest one ──────────────────
    // Skip bin 0 (DC component = signal mean, not meaningful for vibration).
    int   num_anomalous = 0;
    float peak_power    = -1.0f;
    int   peak_bin      = -1;

    bin_flags[0] = 0;  // always ignore DC
    for (int k = 1; k < num_bins; k++) {
        if (power_host[k] > threshold_power) {
            bin_flags[k] = 1;
            num_anomalous++;
            if (power_host[k] > peak_power) {
                peak_power = power_host[k];
                peak_bin   = k;
            }
        } else {
            bin_flags[k] = 0;
        }
    }

    if (peak_freq_hz) {
        *peak_freq_hz = (peak_bin >= 0)
            ? (float)peak_bin * sample_rate_hz / (float)n
            : -1.0f;
    }

    // ── Step 10: Cleanup ──────────────────────────────────────────────────────
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(data_dev));
    CUDA_CHECK(cudaFree(spec_dev));
    CUDA_CHECK(cudaFree(powr_dev));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return { num_anomalous, elapsed_ms };
}
