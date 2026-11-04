#pragma once

#include "include/cuda_runtime.h"
#include "include/cuda_tensor.h"
#include "include/quantized_tensor.h"
#include "include/quant.h"

#include <vector>
namespace yan_lamma {
bool CudaQuantBuilt();
// cpu cpu cpu
// 输入 权重 都需要上传
Tensor CudaQ40Linear(const Tensor& x, const std::vector<Block_Q40>&weight, const std::vector<int>& shape, const Tensor* bias = nullptr, int device_id = 0);
// 权重在显存中 无需反复
Tensor CudaQ40DeviceLinear(const Tensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, const Tensor* bias = nullptr, int device_id = 0);
CudaTensor CudaQ40DeviceInputLinear (const CudaTensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, int device_id = 0);
Tensor CudaQ41Linear(const Tensor& x, const std::vector<Block_Q41>&weight, const std::vector<int>& shape, const Tensor* bias = nullptr, int device_id = 0);
Tensor CudaQ41DeviceLinear(const Tensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, const Tensor* bias = nullptr, int device_id = 0);
CudaTensor CudaQ41DeviceInputLinear (const CudaTensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, int device_id = 0);
Tensor CudaQ80Linear(const Tensor& x, const std::vector<Block_Q80>&weight, const std::vector<int>& shape, const Tensor* bias = nullptr, int device_id = 0);
Tensor CudaQ80DeviceLinear(const Tensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, const Tensor* bias = nullptr, int device_id = 0);
CudaTensor CudaQ80DeviceInputLinear (const CudaTensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, int device_id = 0);





}// namespace yan_lamma
