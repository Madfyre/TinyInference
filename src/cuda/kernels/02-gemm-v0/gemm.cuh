#pragma once

#include <cuda_fp16.h>

#include <cuda_helpers.h>

enum class MatrixLayout { RowMajor, ColMajor };

struct DeviceMatrix {
    __half* data;
    size_t rows;
    size_t cols;
    size_t stride;  // Distance in elements between first values of consecutive rows/columns
    MatrixLayout layout;
};

__global__ void Multiply(float alpha, float beta, DeviceMatrix a, DeviceMatrix b, DeviceMatrix c,
                         DeviceMatrix d) {
    size_t idx = threadIdx.x + static_cast<size_t>(blockDim.x) * blockIdx.x;
    if (idx / a.rows > a.rows - 1) {
        return;
    }
    size_t row_index = idx / a.rows;
    size_t col_index = idx % a.rows;
    float result = 0.0f;
    if (a.layout == MatrixLayout::RowMajor && b.layout == MatrixLayout::RowMajor) {
        for (size_t length = 0; length < a.cols; ++length) {
            result += static_cast<float>(a.data[a.stride * row_index + length] *
                                         b.data[b.stride * length + col_index]);
        }
    }
    if (a.layout == MatrixLayout::RowMajor && b.layout == MatrixLayout::ColMajor) {
        for (size_t length = 0; length < a.cols; ++length) {
            result += static_cast<float>(a.data[a.stride * row_index + length] *
                                         b.data[b.stride * col_index + length]);
        }
    }
    if (a.layout == MatrixLayout::ColMajor && b.layout == MatrixLayout::RowMajor) {
        for (size_t length = 0; length < a.rows; ++length) {
            result += static_cast<float>(a.data[a.stride * length + row_index] *
                                         b.data[b.stride * length + col_index]);
        }
    }
    if (a.layout == MatrixLayout::ColMajor && b.layout == MatrixLayout::ColMajor) {
        for (size_t length = 0; length < a.rows; ++length) {
            result += static_cast<float>(a.data[a.stride * length + row_index] *
                                         b.data[b.stride * col_index + length]);
        }
    }

    result *= alpha;

    if (c.layout == MatrixLayout::RowMajor) {
        result += beta * static_cast<float>(c.data[row_index * c.stride + col_index]);
    } else {
        result += beta * static_cast<float>(c.data[col_index * c.stride + row_index]);
    }

    if (d.layout == MatrixLayout::RowMajor) {
        d.data[row_index * d.stride + col_index] = static_cast<__half>(result);
    } else {
        d.data[col_index * d.stride + row_index] = static_cast<__half>(result);
    }
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {

    if (d.rows == 0 || d.cols == 0) {
        return;
    }

    size_t threads = 1024;
    size_t length = d.rows * d.cols;
    size_t blocks = (length + threads - 1) / threads;
    Multiply<<<blocks, threads>>>(alpha, beta, a, b, c, d);
}
