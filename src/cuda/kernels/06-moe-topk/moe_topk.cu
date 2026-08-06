// #include "moe_topk.cuh"
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <stdio.h>
// #include <iostream>

// // MoE router: for each token take Top-K experts by probability.
// //
// // logits:        [batchSize, numExperts], row-major, stride in elements (inputStride)
// // outIdxs:  [batchSize, topK], int32 expert ids, stride idxsStride
// // topkWeights:  [batchSize, topK], __half, stride outStride
// //
// // Tie-breaking when probabilities are equal: smaller expert index is preferred.

// inline __device__ void swap(float* a, float* b) {
//     float temp = *a;
//     *a = *b;
//     *b = temp;
// }

// inline __device__ bool should_swap(float* smem, float* idxes, bool desc, int i, int j) {
//     if (smem[i] != smem[j]) {
//         return desc ? smem[i] < smem[j] : smem[i] > smem[j];
//     }
//     return desc ? (idxes[i] > idxes[j]) : (idxes[i] < idxes[j]);
// };

// inline __device__ bool should_swap(float* smem, float* idxes, int i, int j) {
//     if (smem[i] != smem[j]) {
//         return smem[i] < smem[j];
//     }
//     return idxes[i] > idxes[j];
// };

// __global__ void MoeTopKDevice(size_t batchSize, size_t numExperts, size_t bigger_power, size_t
// topK,
//                               const __half* logits, size_t inputStride, int32_t* outIdxs,
//                               size_t idxsStride, __half* topkWeights, size_t outStride) {

//     extern __shared__ float total_memory[];

//     auto smem = total_memory;
//     auto idxes = total_memory + bigger_power;

//     smem[threadIdx.x] = (threadIdx.x < numExperts)
//                             ? __half2float(logits[blockIdx.x * inputStride + threadIdx.x])
//                             : -1.0f / 0.0f;

//     // smem[threadIdx.x] = static_cast<float>(logits[blockIdx.x * inputStride + threadIdx.x]);
//     // smem[threadIdx.x + batchSize / 2] = logits[blockIdx.x * inputStride + threadIdx.x +
//     batchSize
//     // / 2];

//     idxes[threadIdx.x] = (threadIdx.x < numExperts) ? static_cast<float>(threadIdx.x) : -1.0f;
//     // idxes[threadIdx.x + batchSize / 2] = static_cast<__half>(threadIdx.x + batchSize / 2);

//     __syncthreads();

//     for (int max_len = 2; max_len <= bigger_power; max_len *= 2) {
//         for (int dist = max_len / 2; dist > 0; dist /= 2) {
//             bool descending = ((threadIdx.x / max_len) % 2 == 0);
//             bool to_swap = ((threadIdx.x / dist) % 2 == 0);

//             if (to_swap && should_swap(smem, idxes, descending, threadIdx.x, threadIdx.x + dist))
//             {
//                 swap(&smem[threadIdx.x], &smem[threadIdx.x + dist]);
//                 swap(&idxes[threadIdx.x], &idxes[threadIdx.x + dist]);
//             }
//             __syncthreads();
//         }
//     }

//     if (threadIdx.x < topK) {
//         topkWeights[blockIdx.x * outStride + threadIdx.x] = __float2half(smem[threadIdx.x]);
//         outIdxs[blockIdx.x * idxsStride + threadIdx.x] =
//         static_cast<int32_t>(idxes[threadIdx.x]);
//     }
// }

// static constexpr size_t FACTOR = 8;
// static constexpr size_t BLOCK_SIZE = 256;

// __global__ void MoeSmallTopK(size_t batchSize, size_t numExperts, size_t bigger_power, size_t
// topK,
//                               const __half* logits, size_t inputStride, int32_t* outIdxs,
//                               size_t idxsStride, __half* topkWeights, size_t outStride) {

//     extern __shared__ float total_memory[];

//     auto smem = total_memory;
//     auto idxes = total_memory + bigger_power;

//     logits += blockIdx.x * inputStride;

//     float max_value = 1e-15f;
//     float index = 11111;
//     for (size_t idx = threadIdx.x; idx < numExperts; idx += FACTOR * blockDim.x) {
// #pragma unroll
//         for (int iter = 0; iter < FACTOR; ++iter) {
//             if (idx + iter * blockDim.x < numExperts) {
//                 max_value = max(max_value, __half2float(logits[idx + iter * blockDim.x]));
//                 index = idx + iter * blockDim.x;
//             }
//         }
//     }

//     smem[threadIdx.x] = max_value;

//     __syncthreads();

//     for (size_t shift = blockDim.x / 2; shift >= 32; shift /= 2) {
//         if (threadIdx.x < shift && (smem[threadIdx.x] > smem[threadIdx.x + shift] ||
//         (smem[threadIdx.x] == smem[threadIdx.x + shift] && threadIdx.x < threadIdx.x + shift))) {
//             smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + shift]);
//             idxes[threadIdx.x] = threadIdx.x;
//         }
//         __syncthreads();
//     }

//     if (threadIdx.x < 32) {
//         float val = smem[threadIdx.x];

// #pragma unroll
//         for (int shift = 16; shift > 0; shift /= 2) {
//             val = max(val, __shfl_down_sync(0xffffffff, val, shift));
//         }

//         if (threadIdx.x == 0) {
//             smem[0] = val;
//         }

//         __syncwarp();
//     }

//     float scale = 127.0f / smem[0];

// }

// void MoeTopK(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
//              size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
//              size_t outStride) {

//     size_t bigger_power = 1;
//     while (numExperts > bigger_power) {
//         bigger_power *= 2;
//     }

//     dim3 block(BLOCK_SIZE);
//     dim3 grid(batchSize);
//     size_t shmem_size = 2 * block.x * sizeof(float);

//     MoeSmallTopK<<<grid, block, shmem_size>>>(batchSize, numExperts, bigger_power, topK, logits,
//                                                 inputStride, outIdxs, idxsStride, topkWeights,
//                                                 outStride);

//     // dim3 block(bigger_power);
//     // dim3 grid(batchSize);

//     // size_t memory_size = 2 * bigger_power * sizeof(float);

//     // MoeTopKDevice<<<grid, block, memory_size>>>(batchSize, numExperts, bigger_power, topK,
//     logits,
//     //                                             inputStride, outIdxs, idxsStride, topkWeights,
//     //                                             outStride);
// }

#include "moe_topk.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

template <int kSlots>
__global__ void MoeTopKKernel(size_t batchSize, size_t numExperts, size_t topK,
                              const __half* logits, size_t inputStride, int32_t* outIdxs,
                              size_t idxsStride, __half* topkWeights, size_t outStride) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const size_t token = blockIdx.x * kWarpsPerBlock + warp_id;
    if (token >= batchSize) {
        return;
    }

    const __half* row = logits + token * inputStride;
    float vals[kSlots];
    int eidx[kSlots];

#pragma unroll
    for (int s = 0; s < kSlots; ++s) {
        int e = lane + s * kWarpSize;
        eidx[s] = e;
        vals[s] = (e < numExperts) ? __half2float(row[e]) : -CUDART_INF_F;
    }

    for (int k = 0; k < topK; ++k) {
        float lv = vals[0];
        int li = eidx[0];
        int ls = 0;

#pragma unroll
        for (int s = 1; s < kSlots; ++s) {
            bool better = vals[s] > lv || (vals[s] == lv && eidx[s] < li);
            if (better) {
                lv = vals[s];
                li = eidx[s];
                ls = s;
            }
        }

        float bv = lv;
        int bi = li;

#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off);
            int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, off);
            bool take = ov > bv || (ov == bv && oi < bi);
            if (take) {
                bv = ov;
                bi = oi;
            }
        }

        if (lane == 0) {
            topkWeights[token * outStride + k] = __float2half(bv);
            outIdxs[token * idxsStride + k] = bi;
        }

        if (li == bi) {
            vals[ls] = -CUDART_INF_F;
        }
    }
}

template <int kSlots>
inline void Launch(size_t batchSize, int numExperts, int topK, const __half* logits,
                   size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
                   size_t outStride) {
    dim3 block(kWarpsPerBlock * kWarpSize);
    dim3 grid(static_cast<unsigned>((batchSize + kWarpsPerBlock - 1) / kWarpsPerBlock));
    MoeTopKKernel<kSlots><<<grid, block>>>(batchSize, numExperts, topK, logits, inputStride,
                                           outIdxs, idxsStride, topkWeights, outStride);
}

}  // namespace

void MoeTopK(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
             size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
             size_t outStride) {

    if (numExperts <= 32) {
        Launch<1>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                  topkWeights, outStride);
    } else if (numExperts <= 64) {
        Launch<2>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                  topkWeights, outStride);
    } else if (numExperts <= 128) {
        Launch<4>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                  topkWeights, outStride);
    } else if (numExperts <= 256) {
        Launch<8>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                  topkWeights, outStride);
    } else {
        Launch<16>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                   topkWeights, outStride);
    }
}
