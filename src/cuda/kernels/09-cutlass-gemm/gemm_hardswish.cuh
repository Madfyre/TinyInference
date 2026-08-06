#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <string>
#include <cuda_helpers.h>
#include <stdexcept>
#include <vector>
#include <random>

inline void CheckCutlassResult(cutlass::Status result, const std::string& message) {
    if (result != cutlass::Status::kSuccess) {
        cudaGetLastError();  // clean error state
        throw std::runtime_error(message + cutlass::cutlassGetStatusString(result));
    }
}

struct THardSwishGemmParams {
    int M;
    int N;
    int K;

    const void* APtr;
    int64_t AStride;  // in elements

    const void* BPtr;
    int64_t BStride;  // in elements

    void* CPtr;
    int64_t CStride;  // in elements
};

template <typename T>
struct Matrix {
    size_t stride;
    T* pointer;
};

static inline float HardSwishFn(float a) {
    // formula from https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html
    return a <= -3.0f ? 0.0f : (a >= 3.0f ? a : a * (a + 3.0f) / 6.0f);
}

template <typename T>
static inline Matrix<T> AllocDeviceMatrix(size_t lines, size_t lineSize) {
    uint8_t* device_ptr = nullptr;
    size_t stride = 0;

    CheckStatus(cudaMallocPitch(reinterpret_cast<void**>(&device_ptr), &stride,
                                lineSize * sizeof(T), lines));

    return {.stride = stride / sizeof(T), .pointer = reinterpret_cast<T*>(device_ptr)};
}

static inline void FillMatrix(std::vector<cutlass::half_t>& matrix) {
    std::seed_seq seed{48151623};
    std::mt19937 mt{seed};

    std::normal_distribution<> normal_dist{-0.02f, 0.02f};
    for (size_t i = 0; i < matrix.size(); i++) {
        matrix[i] = normal_dist(mt);
    }
}

cutlass::Status DoGemm(const THardSwishGemmParams& params, cudaStream_t stream);
