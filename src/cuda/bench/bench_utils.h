#pragma once

#include "gemm.cuh"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        cudaError_t status = (expr);                                                           \
        if (status != cudaSuccess) {                                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorString(status));                                          \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (false)

#define CUBLAS_CHECK(expr)                                                                     \
    do {                                                                                       \
        cublasStatus_t status = (expr);                                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                 \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,            \
                         static_cast<int>(status));                                            \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (false)

// Host-side matrix that knows its layout and can move itself to and from the device.
// Device allocations are pitched, so strides generally exceed the logical width -- this
// deliberately exercises the stride handling in the kernel under test.
struct HostMatrix {
    HostMatrix(size_t rows, size_t cols, MatrixLayout layout)
        : data(rows * cols), rows{rows}, cols{cols}, layout{layout} {
    }

    HostMatrix(std::vector<__half> data, size_t rows, size_t cols, MatrixLayout layout)
        : data(std::move(data)), rows{rows}, cols{cols}, layout{layout} {
    }

    size_t Width() const {
        return layout == MatrixLayout::RowMajor ? cols : rows;
    }

    size_t Height() const {
        return layout == MatrixLayout::RowMajor ? rows : cols;
    }

    __half& At(size_t row, size_t col) {
        return data[layout == MatrixLayout::RowMajor ? row * cols + col : col * rows + row];
    }

    const __half& At(size_t row, size_t col) const {
        return data[layout == MatrixLayout::RowMajor ? row * cols + col : col * rows + row];
    }

    DeviceMatrix ToGPU() const {
        __half* device_ptr = nullptr;
        size_t device_pitch = 0;
        CUDA_CHECK(cudaMallocPitch(&device_ptr, &device_pitch, Width() * sizeof(__half),
                                   Height()));
        CUDA_CHECK(cudaMemcpy2D(device_ptr, device_pitch, data.data(), Width() * sizeof(__half),
                                Width() * sizeof(__half), Height(), cudaMemcpyHostToDevice));

        return DeviceMatrix{.data = device_ptr,
                            .rows = rows,
                            .cols = cols,
                            .stride = device_pitch / sizeof(__half),
                            .layout = layout};
    }

    static HostMatrix FromGPU(const DeviceMatrix& source) {
        HostMatrix result(source.rows, source.cols, source.layout);
        CUDA_CHECK(cudaMemcpy2D(result.data.data(), result.Width() * sizeof(__half), source.data,
                                source.stride * sizeof(__half), result.Width() * sizeof(__half),
                                result.Height(), cudaMemcpyDeviceToHost));
        return result;
    }

    std::vector<__half> data;
    size_t rows;
    size_t cols;
    MatrixLayout layout;
};

// Allocates an uninitialised device matrix with the same shape and layout as `other`.
inline DeviceMatrix AllocLike(const DeviceMatrix& other) {
    const size_t width = other.layout == MatrixLayout::RowMajor ? other.cols : other.rows;
    const size_t height = other.layout == MatrixLayout::RowMajor ? other.rows : other.cols;

    __half* device_ptr = nullptr;
    size_t device_pitch = 0;
    CUDA_CHECK(cudaMallocPitch(&device_ptr, &device_pitch, width * sizeof(__half), height));

    return DeviceMatrix{.data = device_ptr,
                        .rows = other.rows,
                        .cols = other.cols,
                        .stride = device_pitch / sizeof(__half),
                        .layout = other.layout};
}

inline void FreeDeviceMatrix(const DeviceMatrix& matrix) {
    CUDA_CHECK(cudaFree(matrix.data));
}

template <typename Generator>
HostMatrix GenerateRandomMatrix(size_t rows, size_t cols, MatrixLayout layout, Generator& gen) {
    HostMatrix result(rows, cols, layout);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t index = 0; index < result.data.size(); ++index) {
        result.data[index] = __float2half(distribution(gen));
    }
    return result;
}

// Wraps the cuBLAS call matching the layouts used throughout: A is RowMajor (M,K), which is
// the same buffer as a ColMajor (K,M) matrix, so it enters the column-major API transposed.
inline void CublasGemm(cublasHandle_t handle, const DeviceMatrix& a, const DeviceMatrix& b,
                       DeviceMatrix& d, float alpha, float beta) {
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(d.rows),
                              static_cast<int>(d.cols), static_cast<int>(a.cols), &alpha, a.data,
                              CUDA_R_16F, static_cast<int>(a.stride), b.data, CUDA_R_16F,
                              static_cast<int>(b.stride), &beta, d.data, CUDA_R_16F,
                              static_cast<int>(d.stride), CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// Copies `source` into `destination` on device, respecting both pitches.
inline void CopyDeviceMatrix(const DeviceMatrix& source, DeviceMatrix& destination) {
    const size_t width = source.layout == MatrixLayout::RowMajor ? source.cols : source.rows;
    const size_t height = source.layout == MatrixLayout::RowMajor ? source.rows : source.cols;
    CUDA_CHECK(cudaMemcpy2D(destination.data, destination.stride * sizeof(__half), source.data,
                            source.stride * sizeof(__half), width * sizeof(__half), height,
                            cudaMemcpyDeviceToDevice));
}
