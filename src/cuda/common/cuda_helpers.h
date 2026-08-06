#pragma once

#include <cuda_runtime_api.h>

void CheckStatus(const cudaError_t& status);
size_t GetL2CacheSizeBytes();
