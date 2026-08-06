#pragma once

#include <cstdint>
#include <cstddef>

void Quantization(size_t rows, size_t cols, const float* d_input_matrix,
                  const float* d_balance_factors, size_t input_stride, size_t out_stride,
                  int8_t* d_out, float* d_out_scales);
