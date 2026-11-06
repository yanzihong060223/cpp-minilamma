#pragma once
#include "include/ops.h"
#include "include/cuda_tensor.h"
#include "include/cuda_matmul.h"
#include "include/tensor.h"

#include <vector>

namespace  yan_lamma {
    // stub拦截
    bool CudaopsBuild();
    // Embedding查表
    CudaTensor CudaEmbeddingDevice(const void* embedding_data, const std::vector<int>& shape, int token_id, int device_id = 0);
    //归一化的两种操作
    Tensor CudaRmsNormHost(const Tensor& x, const Tensor& weight, float eps, int device_id = 0);
    CudaTensor CudaRmsNromDevice(const CudaTensor& x, const void* weight_data, const std::vector<int>& shape, float eps, int device_id = 0);
    // SwigLU
    //激活gate
    Tensor CudaSilu(const Tensor& a, int device_id = 0);
    // 处理特征
    Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id = 0);
    // 处理残差
    CudaTensor CudaElementAddDevice(const CudaTensor& a, const CudaTensor& b, int device_id = 0);
    // 处理bias
    CudaTensor OCudaElementAddDevice (const CudaTensor&a, const void* b, const std::vector<int>& shape,  int device_id = 0);
    //softmax 单算子的处理
    Tensor CudaSoftmax(Tensor& x, int device_id = 0);
    void CudaRopeHost(Tensor& q, Tensor& k, int pos, float theta,
       RopeType rope_type = RopeType::kNormal, int device_id = 0);
    void CudaRopeDevice(CudaTensor&q, CudaTensor& k, int pos,
       int n_heads, int n_kv_heads, int headdim, float theta,  RopeType rope_type = RopeType::kNormal, int device_id = 0);
} // namespace yan_lamma
