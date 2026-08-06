#include "norm.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ float Silu(float x) {
    return x / (1.0f + __expf(-x));
}

template <int HeadSize>
__global__ void RmsNormGatedKernel(size_t numTokens, size_t numHeads, __half* __restrict__ inOut,
                                   size_t inOutStride, const __half* __restrict__ gate,
                                   size_t gateStride, const __half* __restrict__ gamma,
                                   float epsilon) {
    constexpr int kVec = 2;
    constexpr int kStep = kWarpSize * kVec;
    constexpr int kIters = (HeadSize + kStep - 1) / kStep;

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);

    const size_t token = blockIdx.x;
    const size_t head = blockIdx.y * kWarpsPerBlock + warp;

    if (token >= numTokens || head >= numHeads) {
        return;
    }

    __half* x = inOut + token * inOutStride + head * HeadSize;
    const __half* g = gate + token * gateStride + head * HeadSize;

    __half2 xCache[kIters];
    float sumSq = 0.0f;

#pragma unroll
    for (int it = 0; it < kIters; ++it) {
        const int i = lane * kVec + it * kStep;
        if (i < HeadSize) {
            const __half2 hx = *reinterpret_cast<const __half2*>(x + i);
            xCache[it] = hx;

            const float2 fx = __half22float2(hx);
            sumSq += fx.x * fx.x + fx.y * fx.y;
        }
    }

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sumSq += __shfl_down_sync(0xFFFFFFFFu, sumSq, offset);
    }

    const float totalSumSq = __shfl_sync(0xFFFFFFFFu, sumSq, 0);
    const float rstd = rsqrtf(totalSumSq / static_cast<float>(HeadSize) + epsilon);

#pragma unroll
    for (int it = 0; it < kIters; ++it) {
        const int i = lane * kVec + it * kStep;
        if (i < HeadSize) {
            const __half2 hx = xCache[it];
            const __half2 hg = *reinterpret_cast<const __half2*>(g + i);
            const __half2 hw = *reinterpret_cast<const __half2*>(gamma + i);

            const float2 fx = __half22float2(hx);
            const float2 fg = __half22float2(hg);
            const float2 fw = __half22float2(hw);

            const float y0 = fx.x * rstd * fw.x * Silu(fg.x);
            const float y1 = fx.y * rstd * fw.y * Silu(fg.y);

            *reinterpret_cast<__half2*>(x + i) = __floats2half2_rn(y0, y1);
        }
    }
}

template <int HeadSize>
void LaunchRmsNormGated(size_t numTokens, size_t numHeads, __half* inOut, size_t inOutStride,
                        const __half* gate, size_t gateStride, const __half* gamma, float epsilon) {
    dim3 block(kWarpsPerBlock * kWarpSize);
    dim3 grid(numTokens, (numHeads + kWarpsPerBlock - 1) / kWarpsPerBlock);

    RmsNormGatedKernel<HeadSize><<<grid, block>>>(numTokens, numHeads, inOut, inOutStride, gate,
                                                  gateStride, gamma, epsilon);
}

}  // namespace

void RmsNormGated(const size_t numTokens, const size_t numHeads, const size_t headSize,
                  __half* inOut, const size_t inOutStride, const __half* gate,
                  const size_t gateStride, const __half* gamma, const float epsilon) {
    switch (headSize) {
        case 8:
            LaunchRmsNormGated<8>(numTokens, numHeads, inOut, inOutStride, gate, gateStride, gamma,
                                  epsilon);
            break;
        case 16:
            LaunchRmsNormGated<16>(numTokens, numHeads, inOut, inOutStride, gate, gateStride, gamma,
                                   epsilon);
            break;
        case 128:
            LaunchRmsNormGated<128>(numTokens, numHeads, inOut, inOutStride, gate, gateStride,
                                    gamma, epsilon);
            break;
        case 256:
            LaunchRmsNormGated<256>(numTokens, numHeads, inOut, inOutStride, gate, gateStride,
                                    gamma, epsilon);
            break;
    }
}
