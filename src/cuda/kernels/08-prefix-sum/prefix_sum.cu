#include "prefix_sum.cuh"

#include <cuda_runtime.h>
#include <cstddef>

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr int kItemsPerThread = 8;
constexpr int kTileSize = kBlockThreads * kItemsPerThread;

size_t DivUp(size_t x, size_t y) {
    return (x + y - 1) / y;
}

size_t Align4(size_t x) {
    return (x + 3) & ~size_t{3};
}

__device__ __forceinline__ int WarpExclusiveScan(int value) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    int sum = value;

#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, sum, offset);
        if (lane >= offset) {
            sum += other;
        }
    }

    return sum - value;
}

__device__ __forceinline__ int BlockExclusiveScan(int value, int* warpSums) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;

    const int warpPrefix = WarpExclusiveScan(value);
    const int warpTotal = warpPrefix + value;

    if (lane == kWarpSize - 1) {
        warpSums[warp] = warpTotal;
    }

    __syncthreads();

    if (warp == 0) {
        const int x = lane < kWarpsPerBlock ? warpSums[lane] : 0;
        const int scanned = WarpExclusiveScan(x);
        if (lane < kWarpsPerBlock) {
            warpSums[lane] = scanned;
        }
    }

    __syncthreads();

    return warpSums[warp] + warpPrefix;
}

__global__ void ScanSingleBlockKernel(const int* __restrict__ input, int* __restrict__ output,
                                      size_t n) {
    __shared__ int warpSums[kWarpsPerBlock];

    const int tid = threadIdx.x;
    const size_t base = tid * size_t(kItemsPerThread);

    int vals[kItemsPerThread];

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t idx = base + i;
        vals[i] = idx < n ? input[idx] : 0;
    }

    int localSum = 0;

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const int v = vals[i];
        vals[i] = localSum;
        localSum += v;
    }

    const int threadOffset = BlockExclusiveScan(localSum, warpSums);

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t idx = base + i;
        if (idx < n) {
            output[idx] = threadOffset + vals[i];
        }
    }

    if (tid == kBlockThreads - 1) {
        output[n] = threadOffset + localSum;
    }
}

__global__ void ScanBlocksKernel(const int* __restrict__ input, int* __restrict__ output,
                                 int* __restrict__ blockSums, size_t n) {
    __shared__ int warpSums[kWarpsPerBlock];

    const int tid = threadIdx.x;
    const size_t base = blockIdx.x * size_t(kTileSize) + tid * size_t(kItemsPerThread);

    int vals[kItemsPerThread];

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t idx = base + i;
        vals[i] = idx < n ? input[idx] : 0;
    }

    int localSum = 0;

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const int v = vals[i];
        vals[i] = localSum;
        localSum += v;
    }

    const int threadOffset = BlockExclusiveScan(localSum, warpSums);

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t idx = base + i;
        if (idx < n) {
            output[idx] = threadOffset + vals[i];
        }
    }

    if (tid == kBlockThreads - 1) {
        blockSums[blockIdx.x] = threadOffset + localSum;
    }
}

__global__ void AddOffsetsKernel(int* __restrict__ data, const int* __restrict__ blockOffsets,
                                 size_t n) {
    const int tid = threadIdx.x;
    const size_t block = blockIdx.x;
    const size_t base = block * size_t(kTileSize) + tid * size_t(kItemsPerThread);
    const int offset = blockOffsets[block];

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const size_t idx = base + i;
        if (idx < n) {
            data[idx] += offset;
        }
    }

    if (block == gridDim.x - 1 && tid == 0) {
        data[n] = blockOffsets[gridDim.x];
    }
}

__global__ void SetZeroKernel(int* output) {
    output[0] = 0;
}

void PrefixSumRecursive(const int* input, int* output, int* workspace, size_t n) {
    if (n <= kTileSize) {
        ScanSingleBlockKernel<<<1, kBlockThreads>>>(input, output, n);
        return;
    }

    const size_t numBlocks = DivUp(n, kTileSize);

    int* blockSums = workspace;
    int* nextWorkspace = workspace + Align4(numBlocks + 1);

    ScanBlocksKernel<<<numBlocks, kBlockThreads>>>(input, output, blockSums, n);
    PrefixSumRecursive(blockSums, blockSums, nextWorkspace, numBlocks);
    AddOffsetsKernel<<<numBlocks, kBlockThreads>>>(output, blockSums, n);
}

size_t EstimatePrefixSumWorkspaceSizeBytes(size_t num_elements) {
    size_t result = 0;
    size_t n = num_elements;

    while (n > kTileSize) {
        const size_t blocks = DivUp(n, kTileSize);
        result += Align4(blocks + 1);
        n = blocks;
    }

    return result * sizeof(int);
}

void PrefixSumDevice(const int* input, int* output, int* workspace, size_t num_elements) {
    if (num_elements == 0) {
        SetZeroKernel<<<1, 1>>>(output);
        return;
    }

    PrefixSumRecursive(input, output, workspace, num_elements);
}
