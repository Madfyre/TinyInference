#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_fp16.h>

// Apply per-head rms_norm and add gated activation with activ_func=silu.
// Reference code:
// https://github.com/huggingface/transformers/blob/96c41be562d43785d07744924c2f1e96bc7d6884/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L69-L84
// inOut: [NUM_TOKENS, NUM_HEADS * HEAD_SIZE], dtype=fp16
// rmsNormWeight: [HEAD_SIZE], dtype=fp16
// gate: [NUM_TOKENS, NUM_HEADS * HEAD_SIZE], dtype=fp16

void RmsNormGated(const size_t numTokens, const size_t numHeads, const size_t headSize,
                  __half* inOut, const size_t inOutStride, const __half* gate,
                  const size_t gateStride, const __half* gamma, const float epsilon);
