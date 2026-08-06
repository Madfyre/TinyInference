#pragma once

#include <cuda_helpers.h>

size_t EstimatePrefixSumWorkspaceSizeBytes(size_t num_elements);

void PrefixSumDevice(const int* input,  // [num_elements]
                     int* output,       // [num_elements] + 1
                     int* workspace, size_t num_elements);
