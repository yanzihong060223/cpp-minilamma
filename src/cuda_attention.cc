#include "include/cuda_attention.h"

#include <stdexcept>
namespace yan_lamma {
namespace {
#ifndef MINI_LLAMA_USE_CUDA
std::runtime_error CudaOpsNotBuiltError() {
    return std::runtime_error(
        "CUDA ops were not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a "
        "NVIDIA CUDA machine.");
}
#endif
}// namespace
bool CudaAttentionBuilt() {
    #ifdef MINI_LLAMA_USE_CUDA
        return true;
    #else
    return false;
    #endif
}
#ifndef MINI_LLAMA_USE_CUDA
Tensor CudaAttentionDecode (const Tensor& q, const CudaCache& cache, int layer, int pos, int n_heads, int n_kv_head, int head_dim, int device_id) {
    (void)q;
    (void)cache;
    (void)layer;
    (void)pos;
    (void)n_heads;
    (void)n_kv_head;
    (void)head_dim;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}
CudaTensor CudaAttentionDevice(const CudaTensor& q, const CudaCache& cache, int layer, int pos, int n_heads, int n_kv_head, int head_dim, int device_id) {
    (void)q;
    (void)cache;
    (void)layer;
    (void)pos;
    (void)n_heads;
    (void)n_kv_head;
    (void)head_dim;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}
#endif
}// namespace yan_lamma
