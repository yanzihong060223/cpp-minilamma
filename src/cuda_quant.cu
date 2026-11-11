#include "include/cuda_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "include/cuda_runtime.h"
#include "include/quantized_tensor.h"
#include "include/quant.h"
#include <stdexcept>

namespace yan_lamma {
namespace {
constexpr int kBlockSize = 128;
void CheckedCudaQuant(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
         throw std::runtime_error("CUDA quant error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
    }
}
void CheckKernel(const char* expr) {
    CheckedCudaQuant(cudaGetLastError(), expr);
}
//16位编码读成float
__device__ float HalfBitsToFloat(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(raw);
}
__global__ void Q80LinearKernel(const float* x, float* y, const Block_Q80* weight, int rows, int in_features, int out_features, int blocks_per_row) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (out >= out_features || row >= rows) {
        return;
    }
    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const Block_Q80* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; block_index++) {
        const Block_Q80& block = w_row[block_index];
        int base_k = block_index * kQuantBlockSize;
        int k_end = base_k + kQuantBlockSize < static_cast<size_t>(in_features) ? base_k + kQuantBlockSize :  in_features;
        float d = HalfBitsToFloat(block.d);
        for (int j = base_k; j < k_end; j++) {
            sum += d * static_cast<float>(block.qs[j - base_k]) * x_row[j];
        }
    }
    y[static_cast<size_t>(row) * out_features + out] = sum;
}
__device__ float Q40Value(const Block_Q40& block, int flat_index) {
  int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0F)
                          : static_cast<int>(block.qs[flat_index - 16] >> 4);
  return HalfBitsToFloat(block.d) * static_cast<float>(q - 8);
}
__device__ float Q41Value(const Block_Q41& block, int flat_index) {
  int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0F)
                          : static_cast<int>(block.qs[flat_index - 16] >> 4);
  return HalfBitsToFloat(block.d) * static_cast<float>(q) +
         HalfBitsToFloat(block.m);
}
__global__ void Q40LinearKernel(const float* x, float* y, const Block_Q40* weight, int rows, int in_features, int out_features, int blocks_per_row) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (out >= out_features || row >= rows) {
        return;
    }
    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const Block_Q40* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; block_index++) {
        const Block_Q40& block = w_row[block_index];
        int base_k = block_index * kQuantBlockSize;
        int k_end = base_k + kQuantBlockSize < static_cast<size_t>(in_features) ? base_k + kQuantBlockSize :  in_features;
        for (int j = base_k; j < k_end; j++) {
            sum += Q40Value(block, j - base_k) * x_row[j];
        }
    }
    y[static_cast<size_t>(row) * out_features + out] = sum;
}
__global__ void Q41LinearKernel(const float* x, float* y, const Block_Q41* weight, int rows, int in_features, int out_features, int blocks_per_row) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (out >= out_features || row >= rows) {
        return;
    }
    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const Block_Q41* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; block_index++) {
        const Block_Q41& block = w_row[block_index];
        int base_k = block_index * kQuantBlockSize;
        int k_end = base_k + kQuantBlockSize < static_cast<size_t>(in_features) ? base_k + kQuantBlockSize :  in_features;
        for (int j = base_k; j < k_end; j++) {
            sum += Q41Value(block, j - base_k) * x_row[j];
        }
    }
    y[static_cast<size_t>(row) * out_features + out] = sum;
}
//加bias操作
void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols,
                    const char* caller) {
  if (bias.GetNumDims() != 1 || bias.shape[0] != cols) {
    throw std::runtime_error(std::string(caller) +
                             ": expected bias shape [out_features]");
  }
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      y.data[static_cast<size_t>(row) * cols + col] += bias.data[col];
    }
  }
}
//通用检查，block_count 是权重块总数，block_size 是每块元素数。
int QuantInFeaturesFromShape(const std::vector<int>& shape, const char* str) {
    if (shape.size() == 1) {
        return shape[0];
    }
    if (shape.size() == 2) {
        return shape[1];
    }
    throw std::runtime_error("Quant linear: expected 1D or 2D input, got " + std::string(str));
}
void ValidateQuantLinearShape(const std::vector<int>& x_shape, const std::string& str,
                              size_t block_count, const std::vector<int>& w_shape,
                              size_t block_size) {
    if (w_shape.size() != 2) {
        throw std::runtime_error("Quant linear: expected weight shape [out_features, in_features]");
    }
    int in_features = QuantInFeaturesFromShape(x_shape, str.c_str());
    if (in_features <= 0 || (x_shape.size() == 2 && x_shape[0] <= 0) ||
        w_shape[0] <= 0 || w_shape[1] <= 0 || block_size == 0) {
        throw std::runtime_error("Quant linear: dimensions and block size must be positive");
    }
    if (in_features != w_shape[1]) {
        throw std::runtime_error("Quant linear: input and weight in_features do not match");
    }
    size_t expected_blocks_per_row = (static_cast<size_t>(in_features) + block_size - 1) / block_size;
    size_t expected_blocks = expected_blocks_per_row * w_shape[0];
    if (expected_blocks != block_count) {
        throw std::runtime_error("Quant linear: weight block count does not match shape");
    }
}
void VailedQuantLinear(const Tensor& tensor, const std::vector<int>& w_shape,
                       size_t block_count, size_t block_size) {
    ValidateQuantLinearShape(tensor.shape, tensor.ShapeToString(), block_count, w_shape, block_size);
}
int Q80InFeaturesFromShape(const std::vector<int>& shape, const char* str) {
    return QuantInFeaturesFromShape(shape, str);
}
void ValidateQ80LinearShape(const std::vector<int>& x_shape, const std::string& str,
                            size_t block_count, const std::vector<int>& w_shape) {
    ValidateQuantLinearShape(x_shape, str, block_count, w_shape, kQuantBlockSize);
}
void VailedQ80Linear(const Tensor& tensor, const std::vector<int>& w_shape, size_t block_count) {
    ValidateQ80LinearShape(tensor.shape, tensor.ShapeToString(), block_count, w_shape);
}
Tensor CudaQ40Device(const Tensor& x, const void* device_weight,
                     const std::vector<int>& shape, const size_t block_size,
                     const Tensor* bias = nullptr, int device_id = 0) {
    if (device_weight == nullptr) {
        throw std::runtime_error("CudaQ40Device: device weight is null");
    }
    VailedQuantLinear(x, shape, block_size, kQuantBlockSize);
    bool is_1d = x.GetNumDims() == 1;
    int rows = is_1d ? 1 : x.shape[0];
    int in_features = is_1d ? x.shape[0] : x.shape[1];
    int out_features = shape[0];
    int blocks_per_row = (static_cast<size_t>(in_features) + kQuantBlockSize - 1) / kQuantBlockSize;
    Tensor y(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features});
    CudaDeviceBuffer x_dev(x.GetDataSize() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.GetDataSize() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.GetDataSize() * sizeof(float));
    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.GetData()),
                                   static_cast<float*>(y_dev.GetData()),
                                   static_cast<const Block_Q40*>(device_weight), rows,
                                   in_features, out_features, blocks_per_row);
    CheckKernel("CudaQ40Device");
    y_dev.download(y.data.data(), y.GetDataSize() * sizeof(float));
    if (bias != nullptr) {
        AddBiasInPlace(y, *bias, rows, out_features, "CudaQ40Device");
    }
    return y;
}
CudaTensor CudaQ40DeviceInput(const CudaTensor& x, const void* device_weight,
                             const std::vector<int>& shape, const size_t block_size,
                             int device_id) {
    if (x.DeviceId() != device_id) {
        throw std::runtime_error("CudaQ40DeviceInput: input device does not match device_id");
    }
    if (device_weight == nullptr) {
        throw std::runtime_error("CudaQ40DeviceInput: device weight is null");
    }
    ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), block_size, shape, kQuantBlockSize);
    bool is_1d = x.num_dims() == 1;
    int rows = is_1d ? 1 : x.shape()[0];
    int in_features = is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = shape[0];
    int blocks_per_row = (static_cast<size_t>(in_features) + kQuantBlockSize - 1) / kQuantBlockSize;
    CudaTensor y(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);
    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x.GetData()),
                                   static_cast<float*>(y.GetData()),
                                   static_cast<const Block_Q40*>(device_weight), rows,
                                   in_features, out_features, blocks_per_row);
    CheckKernel("CudaQ40DeviceInput");
    return y;
}
Tensor CudaQ80Device(const Tensor& x, const void* device_weight,
                     const std::vector<int>& shape, const size_t block_size,
                     const Tensor* bias = nullptr, int device_id = 0) {
    if (device_weight == nullptr) {
        throw std::runtime_error("CudaQ80Device: device weight is null");
    }
    VailedQuantLinear(x, shape, block_size, kQuantBlockSize);
    bool is_1d = x.GetNumDims() == 1;
    int rows = is_1d ? 1 : x.shape[0];
    int in_features = is_1d ? x.shape[0] : x.shape[1];
    int out_features = shape[0];
    int blocks_per_row = (static_cast<size_t>(in_features) + kQuantBlockSize - 1) / kQuantBlockSize;
    Tensor y(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features});
    CudaDeviceBuffer x_dev(x.GetDataSize() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.GetDataSize() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.GetDataSize() * sizeof(float));
    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q80LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.GetData()),
                                   static_cast<float*>(y_dev.GetData()),
                                   static_cast<const Block_Q80*>(device_weight), rows,
                                   in_features, out_features, blocks_per_row);
    CheckKernel("CudaQ80Device");
    y_dev.download(y.data.data(), y.GetDataSize() * sizeof(float));
    if (bias != nullptr) {
        AddBiasInPlace(y, *bias, rows, out_features, "CudaQ80Device");
    }
    return y;
}
CudaTensor CudaQ80DeviceInput(const CudaTensor& x, const void* device_weight,
                             const std::vector<int>& shape, const size_t block_size,
                             int device_id) {
    if (x.DeviceId() != device_id) {
        throw std::runtime_error("CudaQ80DeviceInput: input device does not match device_id");
    }
    if (device_weight == nullptr) {
        throw std::runtime_error("CudaQ80DeviceInput: device weight is null");
    }
    ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), block_size, shape, kQuantBlockSize);
    bool is_1d = x.num_dims() == 1;
    int rows = is_1d ? 1 : x.shape()[0];
    int in_features = is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = shape[0];
    int blocks_per_row = (static_cast<size_t>(in_features) + kQuantBlockSize - 1) / kQuantBlockSize;
    CudaTensor y(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);
    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q80LinearKernel<<<grid, block>>>(static_cast<const float*>(x.GetData()),
                                   static_cast<float*>(y.GetData()),
                                   static_cast<const Block_Q80*>(device_weight), rows,
                                   in_features, out_features, blocks_per_row);
    CheckKernel("CudaQ80DeviceInput");
    return y;
}
Tensor CudaQ41Device(const Tensor& x, const void* device_weight,
                     const std::vector<int>& shape, const size_t block_size,
                     const Tensor* bias = nullptr, int device_id = 0) {
    if (device_weight == nullptr) {
        throw std::runtime_error("CudaQ41Device: device weight is null");
    }
    VailedQuantLinear(x, shape, block_size, kQuantBlockSize);
    bool is_1d = x.GetNumDims() == 1;
    int rows = is_1d ? 1 : x.shape[0];
    int in_features = is_1d ? x.shape[0] : x.shape[1];
    int out_features = shape[0];
    int blocks_per_row = (static_cast<size_t>(in_features) + kQuantBlockSize - 1) / kQuantBlockSize;
    Tensor y(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features});
    CudaDeviceBuffer x_dev(x.GetDataSize() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.GetDataSize() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.GetDataSize() * sizeof(float));
    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.GetData()),
                                   static_cast<float*>(y_dev.GetData()),
                                   static_cast<const Block_Q41*>(device_weight), rows,
                                   in_features, out_features, blocks_per_row);
    CheckKernel("CudaQ41Device");
    y_dev.download(y.data.data(), y.GetDataSize() * sizeof(float));
    if (bias != nullptr) {
        AddBiasInPlace(y, *bias, rows, out_features, "CudaQ41Device");
    }
    return y;
}
CudaTensor CudaQ41DeviceInput(const CudaTensor& x, const void* device_weight,
                             const std::vector<int>& shape, const size_t block_size,
                             int device_id) {
    if (x.DeviceId() != device_id) {
        throw std::runtime_error("CudaQ41DeviceInput: input device does not match device_id");
    }
    if (device_weight == nullptr) {
        throw std::runtime_error("CudaQ41DeviceInput: device weight is null");
    }
    ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), block_size, shape, kQuantBlockSize);
    bool is_1d = x.num_dims() == 1;
    int rows = is_1d ? 1 : x.shape()[0];
    int in_features = is_1d ? x.shape()[0] : x.shape()[1];
    int out_features = shape[0];
    int blocks_per_row = (static_cast<size_t>(in_features) + kQuantBlockSize - 1) / kQuantBlockSize;
    CudaTensor y(is_1d ? std::vector<int>{out_features} : std::vector<int>{rows, out_features}, device_id);
    dim3 block(kBlockSize);
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x.GetData()),
                                   static_cast<float*>(y.GetData()),
                                   static_cast<const Block_Q41*>(device_weight), rows,
                                   in_features, out_features, blocks_per_row);
    CheckKernel("CudaQ41DeviceInput");
    return y;
}
} // namespace
Tensor CudaQ40Linear(const Tensor& x, const std::vector<Block_Q40>&weight, const std::vector<int>& shape, const Tensor* bias, int device_id) {
    CudaDeviceBuffer w_ev(kBytesPerQ40Block * weight.size(), device_id);
    w_ev.upload(weight.data(), kBytesPerQ40Block * weight.size());
    return CudaQ40Device(x, w_ev.GetData(), shape, weight.size(), bias, device_id);
}
Tensor CudaQ40DeviceLinear(const Tensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, const Tensor* bias, int device_id) {
    return CudaQ40Device(x, device_weight, shape, block_size, bias, device_id);
}
CudaTensor CudaQ40DeviceInputLinear (const CudaTensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, int device_id) {
    return CudaQ40DeviceInput(x, device_weight, shape,  block_size, device_id);
}

Tensor CudaQ80Linear(const Tensor& x, const std::vector<Block_Q80>&weight, const std::vector<int>& shape, const Tensor* bias, int device_id) {
    CudaDeviceBuffer w_ev(kBytesPerQ80Block * weight.size(), device_id);
    w_ev.upload(weight.data(), kBytesPerQ80Block * weight.size());
    return CudaQ80Device(x, w_ev.GetData(), shape, weight.size(), bias, device_id);
}
Tensor CudaQ80DeviceLinear(const Tensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, const Tensor* bias, int device_id) {
    return CudaQ80Device(x, device_weight, shape, block_size, bias, device_id);
}
CudaTensor CudaQ80DeviceInputLinear (const CudaTensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, int device_id) {
    return CudaQ80DeviceInput(x, device_weight, shape,  block_size, device_id);
}

Tensor CudaQ41Linear(const Tensor& x, const std::vector<Block_Q41>&weight, const std::vector<int>& shape, const Tensor* bias, int device_id) {
    CudaDeviceBuffer w_ev(kBytesPerQ41Block * weight.size(), device_id);
    w_ev.upload(weight.data(), kBytesPerQ41Block * weight.size());
    return CudaQ41Device(x, w_ev.GetData(), shape, weight.size(), bias, device_id);
}
Tensor CudaQ41DeviceLinear(const Tensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, const Tensor* bias, int device_id) {
    return CudaQ41Device(x, device_weight, shape, block_size, bias, device_id);
}
CudaTensor CudaQ41DeviceInputLinear (const CudaTensor& x, const void* device_weight, const std::vector<int>& shape, const size_t block_size, int device_id) {
    return CudaQ41DeviceInput(x, device_weight, shape,  block_size, device_id);
}

} // namespace yan_lamma
