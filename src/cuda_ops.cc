#include "include/cuda_ops.h"

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
} // namespace

bool CudaopsBuild() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA
CudaTensor CudaEmbeddingDevice(const void* embedding_data,
                               const std::vector<int>& shape, int token_id,
                               int device_id) {
    (void)embedding_data;
    (void)shape;
    (void)token_id;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

Tensor CudaRmsNormHost(const Tensor& x, const Tensor& weight, float eps,
                      int device_id) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaRmsNromDevice(const CudaTensor& x, const void* weight_data,
                            const std::vector<int>& shape, float eps,
                            int device_id) {
    (void)x;
    (void)weight_data;
    (void)shape;
    (void)eps;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

Tensor CudaSilu(const Tensor& a, int device_id) {
    (void)a;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaSiluDevice(const CudaTensor& a, int device_id) {
    (void)a;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaElementwiseMulDevice(const CudaTensor& a, const CudaTensor& b,
                                   int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor CudaElementAddDevice(const CudaTensor& a, const CudaTensor& b,
                               int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

CudaTensor OCudaElementAddDevice(const CudaTensor& a, const void* b,
                                const std::vector<int>& shape, int device_id) {
    (void)a;
    (void)b;
    (void)shape;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

Tensor CudaSoftmax(Tensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

void CudaRopeHost(Tensor& q, Tensor& k, int pos, float theta,
                  RopeType rope_type, int device_id) {
    (void)q;
    (void)k;
    (void)pos;
    (void)theta;
    (void)rope_type;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}

void CudaRopeDevice(CudaTensor& q, CudaTensor& k, int pos,
                    int n_heads, int n_kv_heads, int head_dim, float theta,
                    RopeType rope_type, int device_id) {
    (void)q;
    (void)k;
    (void)pos;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)theta;
    (void)rope_type;
    (void)device_id;
    throw CudaOpsNotBuiltError();
}
#endif
} // namespace yan_lamma
