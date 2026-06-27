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
