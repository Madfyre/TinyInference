#include "quantization.cuh"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>

static constexpr size_t FACTOR = 8;
static constexpr size_t BLOCK_SIZE = 256;

__global__ void QuantizationDevice(size_t rows, size_t cols, const float* d_input_matrix,
                                   const float* d_balance_factors, size_t input_stride,
                                   size_t out_stride, int8_t* d_out, float* d_out_scales) {

    extern __shared__ char smem[];
    float* reduce_array = reinterpret_cast<float*>(smem);

    d_input_matrix += blockIdx.x * input_stride;

    float max_value = 1e-5f;
    for (size_t idx = threadIdx.x; idx < cols; idx += FACTOR * blockDim.x) {
#pragma unroll
        for (int iter = 0; iter < FACTOR; ++iter) {
            if (idx + iter * blockDim.x < cols) {
                max_value = max(max_value, fabsf(d_input_matrix[idx + iter * blockDim.x] +
                                                 d_balance_factors[idx + iter * blockDim.x]));
            }
        }
    }

    reduce_array[threadIdx.x] = max_value;

    __syncthreads();

    for (size_t shift = blockDim.x / 2; shift >= 32; shift /= 2) {
        if (threadIdx.x < shift) {
            reduce_array[threadIdx.x] =
                max(reduce_array[threadIdx.x], reduce_array[threadIdx.x + shift]);
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        float val = reduce_array[threadIdx.x];

#pragma unroll
        for (int shift = 16; shift > 0; shift /= 2) {
            val = max(val, __shfl_down_sync(0xffffffff, val, shift));
        }

        if (threadIdx.x == 0) {
            reduce_array[0] = val;
        }

        __syncwarp();
    }

    float scale = 127.0f / reduce_array[0];

    if (threadIdx.x == 0) {
        d_out_scales[blockIdx.x] = scale;
    }

    __syncthreads();

    d_out += blockIdx.x * out_stride;

    for (size_t idx = threadIdx.x; idx < cols; idx += FACTOR * blockDim.x) {
#pragma unroll
        for (int iter = 0; iter < FACTOR; ++iter) {
            if (idx + iter * blockDim.x < cols) {
                d_out[idx + iter * blockDim.x] =
                    static_cast<int8_t>(roundf((d_input_matrix[idx + iter * blockDim.x] +
                                                d_balance_factors[idx + iter * blockDim.x]) *
                                               (127.0f / reduce_array[0])));
            }
        }
    }
}

void Quantization(size_t rows, size_t cols, const float* d_input_matrix,
                  const float* d_balance_factors, size_t input_stride, size_t out_stride,
                  int8_t* d_out, float* d_out_scales) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);
    size_t shmem_size = block.x * sizeof(float);
    QuantizationDevice<<<grid, block, shmem_size>>>(rows, cols, d_input_matrix, d_balance_factors,
                                                    input_stride, out_stride, d_out, d_out_scales);
}
