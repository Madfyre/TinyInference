// Benchmark driver for the INT8 weight quantization kernel in kernels/05-quantization.
//
// Per row r:  scale_out[r] = 127 / max(1e-5, max_c |W[r][c] + S[c]|)
//             Wq[r][c]     = RoundNearest((W[r][c] + S[c]) * scale_out[r])
//
// The kernel is memory bound, so the figure of merit is achieved bandwidth
// relative to the device peak rather than FLOP/s.
//
// Build:
//   nvcc -std=c++17 -O3 -arch=sm_80 \
//        -I../common -I../kernels/05-quantization \
//        bench_quantization.cu ../kernels/05-quantization/quantization.cu \
//        ../common/cuda_helpers.cpp -o bench_quantization

#include "quantization.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (expr);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                         cudaGetErrorString(err_));                                         \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

namespace {

constexpr int kWarmupIters = 10;
constexpr int kTimedIters = 100;
constexpr float kMinScale = 1e-5f;

void ReferenceQuantize(int rows, int cols, const std::vector<float>& w,
                       const std::vector<float>& s, std::vector<int8_t>& out,
                       std::vector<float>& out_scales) {
    for (int r = 0; r < rows; ++r) {
        float max_abs = kMinScale;
        for (int c = 0; c < cols; ++c) {
            max_abs = std::max(max_abs, std::fabs(w[static_cast<size_t>(r) * cols + c] + s[c]));
        }
        const float scale = 127.0f / max_abs;
        out_scales[r] = scale;
        for (int c = 0; c < cols; ++c) {
            const float v = (w[static_cast<size_t>(r) * cols + c] + s[c]) * scale;
            out[static_cast<size_t>(r) * cols + c] = static_cast<int8_t>(std::nearbyint(v));
        }
    }
}

template <typename Fn>
float TimeMs(Fn&& fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < kWarmupIters; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kTimedIters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / kTimedIters;
}

void RunCase(int rows, int cols, double peak_gbs) {
    const size_t n = static_cast<size_t>(rows) * cols;

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> w(n), s(cols);
    for (auto& v : w) v = dist(rng);
    for (auto& v : s) v = dist(rng);

    float *d_w = nullptr, *d_s = nullptr, *d_scales = nullptr;
    int8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_w, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_scales, rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_w, w.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s, s.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    auto run = [&]() {
        Quantization(static_cast<size_t>(rows), static_cast<size_t>(cols), d_w, d_s,
                     static_cast<size_t>(cols), static_cast<size_t>(cols), d_out, d_scales);
    };

    // ---- correctness ----
    run();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<int8_t> got(n), want(n);
    std::vector<float> got_scales(rows), want_scales(rows);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_scales.data(), d_scales, rows * sizeof(float),
                          cudaMemcpyDeviceToHost));
    ReferenceQuantize(rows, cols, w, s, want, want_scales);

    int max_int_diff = 0;
    for (size_t i = 0; i < n; ++i) {
        max_int_diff = std::max(max_int_diff, std::abs(static_cast<int>(got[i]) -
                                                       static_cast<int>(want[i])));
    }
    double max_scale_rel = 0.0;
    for (int r = 0; r < rows; ++r) {
        const double denom = std::fabs(want_scales[r]) > 0.0 ? std::fabs(want_scales[r]) : 1.0;
        max_scale_rel = std::max(max_scale_rel, std::fabs(got_scales[r] - want_scales[r]) / denom);
    }
    // Allow an off-by-one on ties: the reference rounds half-to-even in host code.
    const bool ok = max_int_diff <= 1 && max_scale_rel < 1e-5;

    // ---- timing ----
    const float ms = TimeMs(run);

    // Bytes moved: read W and S, write Wq and the per-row scales.
    const double bytes = static_cast<double>(n) * sizeof(float) + cols * sizeof(float) +
                         static_cast<double>(n) * sizeof(int8_t) + rows * sizeof(float);
    const double gbs = bytes / (ms * 1e-3) / 1e9;

    std::printf("%6d %6d | %8.3f | %8.1f GB/s | %6.1f%% peak | %s (max |diff| %d)\n", rows, cols,
                ms, gbs, peak_gbs > 0.0 ? 100.0 * gbs / peak_gbs : 0.0, ok ? "PASS" : "FAIL",
                max_int_diff);

    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scales));
}

}  // namespace

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Theoretical peak: memory clock is in kHz, bus width in bits, DDR -> factor 2.
    const double peak_gbs =
        2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    std::printf("GPU: %s (SM %d.%d), theoretical peak %.0f GB/s\n\n", prop.name, prop.major,
                prop.minor, peak_gbs);

    std::printf("  rows   cols |    ms    |    bandwidth   |   of peak  | check\n");
    std::printf("--------------+----------+----------------+------------+------\n");

    RunCase(4096, 4096, peak_gbs);
    RunCase(8192, 8192, peak_gbs);
    RunCase(4096, 11008, peak_gbs);  // Llama-style MLP weight shape
    RunCase(11008, 4096, peak_gbs);

    return 0;
}
