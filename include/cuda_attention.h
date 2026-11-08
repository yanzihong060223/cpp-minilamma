#pragma once
#include "include/cuda_kvcache.h"
#include "include/cuda_matmul.h"
#include "include/cuda_runtime.h"
#include "include/cuda_tensor.h"

namespace yan_lamma {
bool CudaAttentionBuilt();
// q and output: [n_heads, head_dim]. Reads the populated cache prefix [0, pos].
// Host wrapper: uploads q, computes on CUDA, and downloads the output.
Tensor CudaAttentionDecode (const Tensor& q, const CudaCache& cache, int layer, int pos, int n_heads, int n_kv_head, int head_dim, int device_id = 0);
// q and cache must both belong to device_id; checked before kernel launch.
CudaTensor CudaAttentionDevice(const CudaTensor& q, const CudaCache& cache, int layer, int pos, int n_heads,  int n_kv_head, int head_dim, int device_id = 0);

}// namespace yan_lamma
