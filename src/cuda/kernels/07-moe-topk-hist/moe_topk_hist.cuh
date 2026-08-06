#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

// Same routing as MoeTopK, plus expertHistogram[e] = number of (token, slot) assignments
// to expert e across all tokens (each token contributes topK counts). Caller must zero
// expertHistogram (length numExperts) before the call.

void MoeTopKHist(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
                 size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
                 size_t outStride, unsigned int* expertHistogram);
