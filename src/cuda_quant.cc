#include "include/cuda_quant.h"

#include <stdexcept>
#include <vector>

namespace yan_lamma {
namespace {
#ifndef MINI_LLAMA_USE_CUDA
std::runtime_error CudaQuantNotBuiltError() {
    return std::runtime_error(
        "CUDA quant kernels were not built. Reconfigure with "
        "-DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine.");
}
#endif
} // namespace

bool CudaQuantBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA
Tensor CudaQ40Linear(const Tensor& x, const std::vector<Block_Q40>& weight,
                    const std::vector<int>& shape, const Tensor* bias, int device_id) {
    (void)x;
    (void)weight;
    (void)shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}

Tensor CudaQ40DeviceLinear(const Tensor& x, const void* device_weight,
                          const std::vector<int>& shape, const size_t block_size,
                          const Tensor* bias, int device_id) {
    (void)x;
    (void)device_weight;
    (void)shape;
    (void)block_size;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}

CudaTensor CudaQ40DeviceInputLinear(const CudaTensor& x, const void* device_weight,
                                   const std::vector<int>& shape,
                                   const size_t block_size, int device_id) {
    (void)x;
    (void)device_weight;
    (void)shape;
    (void)block_size;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}

Tensor CudaQ41Linear(const Tensor& x, const std::vector<Block_Q41>& weight,
                    const std::vector<int>& shape, const Tensor* bias, int device_id) {
    (void)x;
    (void)weight;
    (void)shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}

Tensor CudaQ41DeviceLinear(const Tensor& x, const void* device_weight,
                          const std::vector<int>& shape, const size_t block_size,
                          const Tensor* bias, int device_id) {
    (void)x;
    (void)device_weight;
    (void)shape;
    (void)block_size;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}

CudaTensor CudaQ41DeviceInputLinear(const CudaTensor& x, const void* device_weight,
                                   const std::vector<int>& shape,
                                   const size_t block_size, int device_id) {
    (void)x;
    (void)device_weight;
    (void)shape;
    (void)block_size;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}

Tensor CudaQ80Linear(const Tensor& x, const std::vector<Block_Q80>& weight,
                    const std::vector<int>& shape, const Tensor* bias, int device_id) {
    (void)x;
    (void)weight;
    (void)shape;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}

Tensor CudaQ80DeviceLinear(const Tensor& x, const void* device_weight,
                          const std::vector<int>& shape, const size_t block_size,
                          const Tensor* bias, int device_id) {
    (void)x;
    (void)device_weight;
    (void)shape;
    (void)block_size;
    (void)bias;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}

CudaTensor CudaQ80DeviceInputLinear(const CudaTensor& x, const void* device_weight,
                                   const std::vector<int>& shape,
                                   const size_t block_size, int device_id) {
    (void)x;
    (void)device_weight;
    (void)shape;
    (void)block_size;
    (void)device_id;
    throw CudaQuantNotBuiltError();
}
#endif
} // namespace yan_lamma
