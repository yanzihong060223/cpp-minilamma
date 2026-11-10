#pragma once
// Internal seams for CPU/CUDA differential tests; not part of the public API.
#include "include/forward.h"
#include "include/cuda_tensor.h"
namespace yan_lamma {
Tensor ForwardAdd(const Model& model, const Tensor& a, const Tensor& b);
Tensor Embedding(const Model& model, int token_id);
Tensor ForwardAttention(const Model& model, const Tensor& q, const Tensor& k,
    const Tensor& v, int layer, int pos, KvCache& cache,
    int n_heads, int n_kv_heads, int head_dim);
CudaTensor ForwardLinearDevice(const Model& model, const CudaTensor& x,
    const std::string name, const QuanTensor& tensor);
}
