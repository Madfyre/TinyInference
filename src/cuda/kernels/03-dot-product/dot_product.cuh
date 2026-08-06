#pragma once
#include <cuda_helpers.h>

constexpr int DOT_BLOCK = 256;
constexpr int ELEMS_PER_THREAD = 4;

__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduce(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduce(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp reduces across all warps
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduce(val);
    }
    return val;
}

__global__ void DotPartialKernel(const float* __restrict__ lhs, const float* __restrict__ rhs,
                                 size_t n, float* workspace) {

    float sum = 0.0f;
    size_t vec_n = n / 4;
    const float4* lhs4 = reinterpret_cast<const float4*>(lhs);
    const float4* rhs4 = reinterpret_cast<const float4*>(rhs);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_n;
         i += gridDim.x * (size_t)blockDim.x) {
        float4 a = lhs4[i];
        float4 b = rhs4[i];
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // Handle tail elements
    size_t tail_start = vec_n * 4;
    for (size_t i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * (size_t)blockDim.x) {
        sum += lhs[i] * rhs[i];
    }

    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        workspace[blockIdx.x] = sum;
    }
}

__global__ void ReduceKernel(const float* workspace, size_t n, float* out) {
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += workspace[i];
    }
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        *out = sum;
    }
}

inline size_t getNumBlocks(size_t num_elements) {
    size_t vec_elements = num_elements / 4;
    size_t blocks = (vec_elements + DOT_BLOCK - 1) / DOT_BLOCK;
    if (blocks > 256) {
        blocks = 256;
    }
    if (blocks < 1) {
        blocks = 1;
    }
    return blocks;
}

size_t EstimateDotProductWorkspaceSizeBytes(size_t num_elements) {
    return getNumBlocks(num_elements) * sizeof(float);
}

void DotProduct(const float* lhs_device, const float* rhs_device, size_t num_elements,
                float* workspace_device, float* out_device) {
    size_t num_blocks = getNumBlocks(num_elements);
    DotPartialKernel<<<num_blocks, DOT_BLOCK>>>(lhs_device, rhs_device, num_elements,
                                                workspace_device);
    ReduceKernel<<<1, DOT_BLOCK>>>(workspace_device, num_blocks, out_device);
}
