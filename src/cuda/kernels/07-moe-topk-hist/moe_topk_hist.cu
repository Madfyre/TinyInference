#include "moe_topk_hist.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <climits>

// Same as MoeTopK plus expertHistogram[e] = count of assignments to expert e.
// Zero expertHistogram (length numExperts) before calling.

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

template <int kSlots>
__global__ void MoeTopKKernel(size_t batchSize, size_t numExperts, size_t topK,
                              const __half* logits, size_t inputStride, int32_t* outIdxs,
                              size_t idxsStride, __half* topkWeights, size_t outStride,
                              unsigned int* expertHistogram) {

    size_t warp_id = threadIdx.x / kWarpSize;
    size_t lane = threadIdx.x & (kWarpSize - 1);
    size_t token = blockIdx.x * kWarpsPerBlock + warp_id;

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
            atomicAdd(&expertHistogram[bi], 1u);
        }

        if (li == bi) {
            eidx[ls] = INT_MAX;
            vals[ls] = -CUDART_INF_F;
        }
    }
}

template <int kSlots>
inline void Launch(size_t batchSize, int numExperts, int topK, const __half* logits,
                   size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
                   size_t outStride, unsigned int* expertHistogram) {
    dim3 block(kWarpsPerBlock * kWarpSize);
    dim3 grid((batchSize + kWarpsPerBlock - 1) / kWarpsPerBlock);
    MoeTopKKernel<kSlots><<<grid, block>>>(batchSize, numExperts, topK, logits, inputStride,
                                           outIdxs, idxsStride, topkWeights, outStride,
                                           expertHistogram);
}

void MoeTopKHist(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
                 size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
                 size_t outStride, unsigned int* expertHistogram) {

    if (numExperts <= 32) {
        Launch<1>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                  topkWeights, outStride, expertHistogram);
    } else if (numExperts <= 64) {
        Launch<2>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                  topkWeights, outStride, expertHistogram);
    } else if (numExperts <= 128) {
        Launch<4>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                  topkWeights, outStride, expertHistogram);
    } else if (numExperts <= 256) {
        Launch<8>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                  topkWeights, outStride, expertHistogram);
    } else {
        Launch<16>(batchSize, numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                   topkWeights, outStride, expertHistogram);
    }
}
