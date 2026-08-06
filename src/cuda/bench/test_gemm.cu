// Correctness tests for the FP16 tiled GEMM in kernels/04-gemm-v1.
//
// Two independent references are used. Small shapes are checked against a CPU
// implementation accumulating in float; large shapes, where a CPU reference would be
// slow, are cross-checked against cuBLAS. Device allocations are pitched, so every case
// also exercises the kernel's handling of strides wider than the logical matrix.
//
// Build (adjust -arch to your GPU):
//   nvcc -std=c++17 -O3 -arch=sm_80 -I../common -I../kernels/04-gemm-v1 \
//        test_gemm.cu ../common/cuda_helpers.cpp -lcublas -o test_gemm

#include "bench_utils.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

HostMatrix ReferenceGemm(const HostMatrix& a, const HostMatrix& b, const HostMatrix& c,
                         float alpha, float beta) {
    HostMatrix expected(c.rows, c.cols, c.layout);
    for (size_t row = 0; row < c.rows; ++row) {
        for (size_t col = 0; col < c.cols; ++col) {
            float accumulator = 0.0f;
            for (size_t inner = 0; inner < a.cols; ++inner) {
                accumulator += __half2float(a.At(row, inner)) * __half2float(b.At(inner, col));
            }
            expected.At(row, col) =
                __float2half(alpha * accumulator + beta * __half2float(c.At(row, col)));
        }
    }
    return expected;
}

// Reports the largest deviation relative to the magnitude of the reference.
double MaxRelativeError(const HostMatrix& got, const HostMatrix& expected) {
    double max_abs_diff = 0.0;
    double reference_magnitude = 0.0;
    for (size_t index = 0; index < got.data.size(); ++index) {
        const double lhs = __half2float(got.data[index]);
        const double rhs = __half2float(expected.data[index]);
        max_abs_diff = std::max(max_abs_diff, std::fabs(lhs - rhs));
        reference_magnitude = std::max(reference_magnitude, std::fabs(rhs));
    }
    return reference_magnitude > 0.0 ? max_abs_diff / reference_magnitude : max_abs_diff;
}

void Report(const std::string& name, double relative_error, double tolerance) {
    const bool passed = relative_error < tolerance;
    if (!passed) {
        ++g_failures;
    }
    std::printf("  %-38s %-6s (rel err %.3e, tol %.0e)\n", name.c_str(), passed ? "PASS" : "FAIL",
                relative_error, tolerance);
}

// Runs the kernel once and returns the result on the host.
HostMatrix RunKernel(const HostMatrix& a_host, const HostMatrix& b_host,
                     const HostMatrix& c_host, float alpha, float beta) {
    DeviceMatrix a = a_host.ToGPU();
    DeviceMatrix b = b_host.ToGPU();
    DeviceMatrix c = c_host.ToGPU();
    DeviceMatrix d = AllocLike(c);

    GEMM(a, b, c, d, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    HostMatrix result = HostMatrix::FromGPU(d);

    FreeDeviceMatrix(a);
    FreeDeviceMatrix(b);
    FreeDeviceMatrix(c);
    FreeDeviceMatrix(d);
    return result;
}

void CheckAgainstCpu(size_t m, size_t n, size_t k, float alpha, float beta) {
    std::mt19937 generator(42);
    HostMatrix a = GenerateRandomMatrix(m, k, MatrixLayout::RowMajor, generator);
    HostMatrix b = GenerateRandomMatrix(k, n, MatrixLayout::ColMajor, generator);
    HostMatrix c = GenerateRandomMatrix(m, n, MatrixLayout::ColMajor, generator);

    HostMatrix got = RunKernel(a, b, c, alpha, beta);
    HostMatrix expected = ReferenceGemm(a, b, c, alpha, beta);

    char label[128];
    std::snprintf(label, sizeof(label), "cpu  %4zux%4zux%4zu a=%.1f b=%.1f", m, n, k, alpha, beta);
    // FP16 storage with float accumulation: a few units in the last place are expected.
    Report(label, MaxRelativeError(got, expected), 2e-2);
}

void CheckAgainstCublas(size_t m, size_t n, size_t k, float alpha, float beta,
                        cublasHandle_t handle) {
    std::mt19937 generator(7);
    HostMatrix a_host = GenerateRandomMatrix(m, k, MatrixLayout::RowMajor, generator);
    HostMatrix b_host = GenerateRandomMatrix(k, n, MatrixLayout::ColMajor, generator);
    HostMatrix c_host = GenerateRandomMatrix(m, n, MatrixLayout::ColMajor, generator);

    HostMatrix got = RunKernel(a_host, b_host, c_host, alpha, beta);

    DeviceMatrix a = a_host.ToGPU();
    DeviceMatrix b = b_host.ToGPU();
    DeviceMatrix c = c_host.ToGPU();
    DeviceMatrix d = AllocLike(c);

    // cuBLAS accumulates into its output, so seed it with C and let beta apply there.
    CopyDeviceMatrix(c, d);
    CublasGemm(handle, a, b, d, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    HostMatrix expected = HostMatrix::FromGPU(d);

    FreeDeviceMatrix(a);
    FreeDeviceMatrix(b);
    FreeDeviceMatrix(c);
    FreeDeviceMatrix(d);

    char label[128];
    std::snprintf(label, sizeof(label), "cublas %4zux%4zux%4zu a=%.1f b=%.1f", m, n, k, alpha,
                  beta);
    Report(label, MaxRelativeError(got, expected), 3e-2);
}

}  // namespace

int main() {
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf("GPU: %s (SM %d.%d)\n\n", properties.name, properties.major, properties.minor);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::printf("Small shapes against a CPU reference:\n");
    CheckAgainstCpu(32, 32, 32, 1.0f, 0.0f);
    CheckAgainstCpu(64, 64, 64, 1.0f, 0.5f);
    CheckAgainstCpu(128, 96, 64, 1.0f, 0.0f);
    // Deliberately not multiples of the 32-wide tile, to hit the boundary paths.
    CheckAgainstCpu(33, 47, 65, 1.0f, 0.25f);
    CheckAgainstCpu(17, 5, 129, 2.0f, -1.0f);
    CheckAgainstCpu(1, 1, 512, 1.0f, 0.0f);
    CheckAgainstCpu(256, 1, 256, 1.0f, 0.0f);
    CheckAgainstCpu(1, 256, 256, 1.0f, 0.0f);

    std::printf("\nLarge shapes against cuBLAS:\n");
    CheckAgainstCublas(512, 512, 512, 1.0f, 0.0f, handle);
    CheckAgainstCublas(1024, 1024, 1024, 1.0f, 0.5f, handle);
    CheckAgainstCublas(1000, 1000, 1000, 1.0f, 0.0f, handle);
    CheckAgainstCublas(2048, 512, 1024, 1.0f, 0.0f, handle);

    CUBLAS_CHECK(cublasDestroy(handle));

    std::printf("\n%s\n", g_failures == 0 ? "All checks passed." : "SOME CHECKS FAILED.");
    return g_failures == 0 ? 0 : 1;
}
