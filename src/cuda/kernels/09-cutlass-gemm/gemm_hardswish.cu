#include "gemm_hardswish.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/layout/matrix.h>

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ float HardSwishDevice(float x) {
    if (x <= -3.0f) {
        return 0.0f;
    }
    if (x >= 3.0f) {
        return x;
    }
    return x * (x + 3.0f) * (1.0f / 6.0f);
}

__global__ void HardswishGemvKernel(int m, int k, const cutlass::half_t* __restrict__ a,
                                    int64_t aStride, const cutlass::half_t* __restrict__ b,
                                    int64_t bStride, cutlass::half_t* __restrict__ c,
                                    int64_t cStride) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int row = blockIdx.x * kWarpsPerBlock + warp;

    if (row >= m) {
        return;
    }

    const half* aRow = reinterpret_cast<const half*>(a + row * aStride);
    const half* bCol = reinterpret_cast<const half*>(b);
    half* cCol = reinterpret_cast<half*>(c);

    float acc = 0.0f;

    for (int i = lane * 2; i < k; i += kWarpSize * 2) {
        const half2 av = *reinterpret_cast<const half2*>(aRow + i);
        const half2 bv = *reinterpret_cast<const half2*>(bCol + i);

        const float2 af = __half22float2(av);
        const float2 bf = __half22float2(bv);

        acc += af.x * bf.x + af.y * bf.y;
    }

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }

    if (lane == 0) {
        cCol[row] = __float2half(HardSwishDevice(acc));
    }
}

cutlass::Status DoGemvN1(const THardSwishGemmParams& params, cudaStream_t stream) {
    dim3 block(kWarpsPerBlock * kWarpSize);
    dim3 grid((params.M + kWarpsPerBlock - 1) / kWarpsPerBlock);

    HardswishGemvKernel<<<grid, block, 0, stream>>>(
        params.M, params.K, static_cast<const cutlass::half_t*>(params.APtr), params.AStride,
        static_cast<const cutlass::half_t*>(params.BPtr), params.BStride,
        static_cast<cutlass::half_t*>(params.CPtr), params.CStride);

    return cutlass::Status::kSuccess;
}

cutlass::Status DoCutlassGemm(const THardSwishGemmParams& params, cudaStream_t stream) {
    using ElementInput = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGeneric<
        cutlass::epilogue::thread::HardSwish, ElementOutput, 8, ElementAccumulator, ElementCompute,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput, LayoutA, ElementInput, LayoutB, ElementOutput, LayoutC, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 8, 8>;

    typename Gemm::Arguments args({params.M, params.N, params.K},
                                  {static_cast<ElementInput const*>(params.APtr), params.AStride},
                                  {static_cast<ElementInput const*>(params.BPtr), params.BStride},
                                  {static_cast<ElementOutput const*>(params.CPtr), params.CStride},
                                  {static_cast<ElementOutput*>(params.CPtr), params.CStride},
                                  {ElementCompute(1.0f), ElementCompute(0.0f)});

    Gemm gemm;
    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    return gemm(args, nullptr, stream);
}

cutlass::Status DoGemm(const THardSwishGemmParams& params, cudaStream_t stream) {
    if (params.N == 1) {
        return DoGemvN1(params, stream);
    }

    return DoCutlassGemm(params, stream);
}
