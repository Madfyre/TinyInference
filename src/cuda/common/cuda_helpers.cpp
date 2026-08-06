#include "cuda_helpers.h"

#include <stdexcept>
#include <string>

void CheckStatus(const cudaError_t& status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status));
    }
}

size_t GetL2CacheSizeBytes() {
    cudaDeviceProp device_prop;
    CheckStatus(cudaGetDeviceProperties(&device_prop, 0));

    return device_prop.l2CacheSize;
}
