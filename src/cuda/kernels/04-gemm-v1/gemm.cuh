#pragma once

#include <cassert>
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

constexpr int TILE = 32;
constexpr int TM = 8;

__global__ void GEMMKernel(float alpha, float beta, DeviceMatrix a, DeviceMatrix b, DeviceMatrix c,
                           DeviceMatrix d) {

    __shared__ __half a_tile[TILE][TILE + 1];
    __shared__ __half b_tile[TILE][TILE + 1];

    int thread_row = threadIdx.x % (TILE / TM);
    int thread_col = threadIdx.x / (TILE / TM);

    int d_row = blockIdx.x * TILE + thread_row * TM;
    int d_col = blockIdx.y * TILE + thread_col;

    auto a_matrix = a.data + blockIdx.x * TILE * a.stride;
    auto b_matrix = b.data + blockIdx.y * TILE * b.stride;

    float thread_results[TM] = {0.0};

    for (int block = 0; block < a.cols; block += TILE) {

        int load_row = threadIdx.x % (TILE / TM);
        int load_col = threadIdx.x / (TILE / TM);

#pragma unroll
        for (int i = 0; i < TM; ++i) {

            int row = load_row * TM + i;
            int col = load_col;

            int a_r = blockIdx.x * TILE + row;
            int a_c = block + col;

            int b_r = block + row;
            int b_c = blockIdx.y * TILE + col;

            a_tile[row][col] = (a_r < a.rows && a_c < a.cols)
                                   ? static_cast<float>(a_matrix[row * a.stride + col])
                                   : 0.0f;

            b_tile[row][col] = (b_r < b.rows && b_c < b.cols)
                                   ? static_cast<float>(b_matrix[col * b.stride + row])
                                   : 0.0f;
        }

        __syncthreads();

        a_matrix += TILE;
        b_matrix += TILE;

        for (int i = 0; i < TILE; i++) {
            __half b_value = b_tile[i][thread_col];
#pragma unroll
            for (int j = 0; j < TM; ++j) {
                thread_results[j] += __half2float(b_value * a_tile[thread_row * TM + j][i]);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < TM; ++j) {
        if (d_row + j < d.rows && d_col < d.cols) {
            d.data[d_col * d.stride + d_row + j] =
                alpha * thread_results[j] +
                beta * static_cast<float>(c.data[d_col * c.stride + d_row + j]);
        }
    }
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    assert(a.layout == MatrixLayout::RowMajor);
    assert(b.layout == MatrixLayout::ColMajor);
    assert(c.layout == MatrixLayout::ColMajor);
    assert(d.layout == MatrixLayout::ColMajor);

    dim3 block(TILE * TILE / TM);
    dim3 grid((d.rows + TILE - 1) / TILE, (d.cols + TILE - 1) / TILE);
    GEMMKernel<<<grid, block>>>(alpha, beta, a, b, c, d);
}
