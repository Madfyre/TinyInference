// Throughput benchmark for the FP16 tiled GEMM in kernels/04-gemm-v1, measured against
// cuBLAS on identical inputs and layouts.
//
// Timing a kernel repeatedly on one buffer set measures the L2-resident case and
// flatters the result. Following the same approach as the course harness, this driver
// allocates enough distinct input sets to exceed the L2 cache and rotates through them,
// so consecutive iterations mostly miss in L2. The same rotation is applied to the
// cuBLAS baseline, keeping the comparison fair.
//
// Build (adjust -arch to your GPU):
//   nvcc -std=c++17 -O3 -arch=sm_80 -I../common -I../kernels/04-gemm-v1 \
//        bench_gemm.cu ../common/cuda_helpers.cpp -lcublas -o bench_gemm

#include "bench_utils.h"

#include <cuda_helpers.h>

#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int kWarmupIters = 5;
constexpr int kTimedIters = 50;

struct Workspace {
    DeviceMatrix a;
    DeviceMatrix b;
    DeviceMatrix c;
    DeviceMatrix d;
};

size_t BytesPerWorkspace(const Workspace& workspace) {
    return (workspace.a.rows * workspace.a.stride + workspace.b.cols * workspace.b.stride +
            workspace.c.cols * workspace.c.stride + workspace.d.cols * workspace.d.stride) *
           sizeof(__half);
}

// Builds enough independent workspaces to overflow the L2 cache, so that rotating
// through them during timing keeps the inputs cold.
std::vector<Workspace> BuildWorkspaces(size_t m, size_t n, size_t k) {
    std::mt19937 generator(42);
    const size_t l2_bytes = GetL2CacheSizeBytes();

    std::vector<Workspace> workspaces;
    size_t allocated_bytes = 0;
    while (allocated_bytes < l2_bytes || workspaces.size() < 2) {
        HostMatrix a_host = GenerateRandomMatrix(m, k, MatrixLayout::RowMajor, generator);
        HostMatrix b_host = GenerateRandomMatrix(k, n, MatrixLayout::ColMajor, generator);
        HostMatrix c_host = GenerateRandomMatrix(m, n, MatrixLayout::ColMajor, generator);

        Workspace workspace{.a = a_host.ToGPU(),
                            .b = b_host.ToGPU(),
                            .c = c_host.ToGPU(),
                            .d = DeviceMatrix{}};
        workspace.d = AllocLike(workspace.c);

        allocated_bytes += BytesPerWorkspace(workspace);
        workspaces.push_back(workspace);

        // Guard against pathological memory use on very large shapes.
        if (workspaces.size() >= 32) {
            break;
        }
    }
    return workspaces;
}

void ReleaseWorkspaces(std::vector<Workspace>& workspaces) {
    for (Workspace& workspace : workspaces) {
        FreeDeviceMatrix(workspace.a);
        FreeDeviceMatrix(workspace.b);
        FreeDeviceMatrix(workspace.c);
        FreeDeviceMatrix(workspace.d);
    }
    workspaces.clear();
}

// Runs `fn(workspace)` over the rotating workspace set and returns mean ms per call.
template <typename Fn>
float TimeMs(std::vector<Workspace>& workspaces, Fn&& fn) {
    size_t index = 0;
    auto step = [&]() {
        fn(workspaces[index]);
        index = (index + 1) % workspaces.size();
    };

    for (int iter = 0; iter < kWarmupIters; ++iter) {
        step();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kTimedIters; ++iter) {
        step();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / kTimedIters;
}

void RunCase(size_t m, size_t n, size_t k, float alpha, float beta, cublasHandle_t handle) {
    std::vector<Workspace> workspaces = BuildWorkspaces(m, n, k);

    const float mine_ms = TimeMs(workspaces, [&](Workspace& workspace) {
        GEMM(workspace.a, workspace.b, workspace.c, workspace.d, alpha, beta);
    });

    const float cublas_ms = TimeMs(workspaces, [&](Workspace& workspace) {
        CublasGemm(handle, workspace.a, workspace.b, workspace.d, alpha, beta);
    });

    CUDA_CHECK(cudaGetLastError());

    const double flops = 2.0 * m * n * k;
    const double mine_tflops = flops / (mine_ms * 1e-3) / 1e12;
    const double cublas_tflops = flops / (cublas_ms * 1e-3) / 1e12;

    std::printf("%5zu %5zu %5zu | %5zu | %8.3f %8.2f | %8.3f %8.2f | %6.1f%%\n", m, n, k,
                workspaces.size(), mine_ms, mine_tflops, cublas_ms, cublas_tflops,
                100.0 * mine_tflops / cublas_tflops);

    ReleaseWorkspaces(workspaces);
}

}  // namespace

int main() {
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf("GPU: %s (SM %d.%d), L2 %.1f MB\n", properties.name, properties.major,
                properties.minor, GetL2CacheSizeBytes() / (1024.0 * 1024.0));
    std::printf("Warmup %d iters, timed %d iters, rotating over L2-sized working sets.\n\n",
                kWarmupIters, kTimedIters);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::printf("    M     N     K |  sets |  mine ms   TFLOP/s | cublas ms   TFLOP/s | of cuBLAS\n");
    std::printf("------------------+-------+--------------------+---------------------+----------\n");

    for (size_t size : {512UL, 1024UL, 2048UL, 4096UL}) {
        RunCase(size, size, size, 1.0f, 0.0f, handle);
    }
    RunCase(1024, 4096, 1024, 1.0f, 0.5f, handle);
    RunCase(4096, 1024, 4096, 1.0f, 0.0f, handle);
    // Not a multiple of the tile size, to show the cost of the boundary path.
    RunCase(1000, 1000, 1000, 1.0f, 0.0f, handle);

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
