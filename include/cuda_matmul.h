#pragma  once
#include "include/cuda_runtime.h"
#include "include/cuda_tensor.h"
#include "include/tensor.h"

namespace yan_lamma {
bool cuBLASBuilt();
Tensor CudaMatmul(const Tensor&a, const Tensor& b, int device_id);
// cpu cpu cpu 
Tensor CudaLinear(const Tensor& x, const Tensor&weight, const Tensor* bias = nullptr, int device_id = 0);
// cpu gpu cpu。权重为 device_id 上连续行主序 float32 [out_features, in_features]。
// 显存容量和生命周期由调用方保证，函数只借用指针。
Tensor CudaLinearHostDevice(const Tensor& x, const void* w_data,
                            const std::vector<int>& w_shape,
                            const Tensor* bias = nullptr, int device_id = 0);
// gpu gpu gpu。x 和权重均须位于 device_id，权重的布局及所有权约定同上。
CudaTensor CudaLinearDevice(const CudaTensor& x, const void* w_data,
                            const std::vector<int>& w_shape,
                            int device_id = 0);
}// namespace yan_lamma
