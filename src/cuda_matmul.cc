#include "include/cuda_tensor.h"
#include "include/cuda_matmul.h"

#include <stdexcept>
#include <vector>
#include <string>

#ifdef  MINI_LLAMA_USE_CUDA
#include <cublas_v2.h>
#endif
namespace yan_lamma {
namespace {
#ifndef MINI_LLAMA_USE_CUDA
std::runtime_error CudaMatmulNotBuiltError() {
    return std::runtime_error(
      "CUDA Matmul was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a "
      "NVIDIA CUDA machine.");
}
#endif
#ifdef MINI_LLAMA_USE_CUDA
const char* CublasStatusName(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN";
}
void CheckCublas(cublasStatus_t status, const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS error in " + std::string(expr) + ": " +
                             CublasStatusName(status));
  }
}
//RAII 管理cublas句柄
class MyCublas {
 public:
    explicit MyCublas(int device_id) {
        SetCudaService(device_id);
        CheckCublas(cublasCreate(&handle_), "cublasCreate");
    }
    ~MyCublas() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }
    MyCublas(const MyCublas&) = delete;
    MyCublas& operator=(const MyCublas&) = delete;
    cublasHandle_t GetHandle () const {return handle_;}
 private:
    cublasHandle_t handle_ = nullptr;
};
void VailedMatmulTensor(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::runtime_error("CudaMatmul: expected two 2D tensors");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("CudaMatmul: expected a.shape[1] == b.shape[0]");
    }
}
// 加bias
void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols) {
  if (bias.GetNumDims() != 1 || bias.shape[0] != cols) {
    throw std::runtime_error("CudaLinear: expected bias shape [out_features]");
  }
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(cols) +
                            static_cast<size_t>(col);
      y.data[offset] += bias.data[col];
    }
  }
}
//提取出的 in_features 后续用于检查：输入特征数是否等于权重形状 [out_features, in_features] 的第二维
int LinearInFeaturesFromShape(const std::vector<int>& x_shape,
                              const std::string& x_shape_str,
                              const char* caller) {
  if (x_shape.size() == 1) {
    return x_shape[0];
  }
  if (x_shape.size() == 2) {
    return x_shape[1];
  }
  throw std::runtime_error(
      std::string(caller) +
      ": expected x shape [in_features] or [batch, in_features], got " +
      x_shape_str);
}
void VailedLinearTensor(const std::vector<int>& x_shape, const std::vector<int>& w_shape,  const std::string& x_shape_str, const char* caller) {
    if (w_shape.size() != 2) {
         throw std::runtime_error(
        std::string(caller) +
        ": expected weight shape [out_features, in_features]");
    }
    const int in_feature = LinearInFeaturesFromShape(x_shape, x_shape_str, caller);
    if (in_feature != w_shape[1]) {
        throw std::runtime_error(std::string(caller) +
                                 ": input features do not match weight");
    }
}
void ValidateLinearInputs(const Tensor& x, const std::vector<int>& w_shape) {
  VailedLinearTensor(x.shape, w_shape, x.ShapeToString(), "CudaLinear");
}
// cpu gpu cpu
Tensor CudaLinearHostToDevice(const Tensor& x, const void* w_data,
                              const std::vector<int>& w_shape,
                              const Tensor* bias, int device_id) {
    if (w_data == nullptr) {
        throw std::runtime_error("CudaLinear: device weight pointer is null");
    }
    ValidateLinearInputs(x, w_shape);
    bool input_is_1d = x.GetNumDims() == 1;
    int rows = input_is_1d ? 1 : x.shape[0];
    int in_features = input_is_1d ? x.shape[0] : x.shape[1];
    int out_features = w_shape[0];
    Tensor y(input_is_1d ? std::vector<int>{out_features}
             : std::vector<int>{rows, out_features});
    CudaDeviceBuffer x_dev(x.GetDataSize() * sizeof(float), device_id);
    CudaDeviceBuffer y_dev(y.GetDataSize() * sizeof(float), device_id);
    x_dev.upload(x.data.data(), x.GetDataSize() * sizeof(float));
    MyCublas handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cublasSgemm 达到转置相乘
    CheckCublas(
    cublasSgemm(handle.GetHandle(), CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows,
                  in_features, &alpha, static_cast<const float*>(w_data),
                  in_features, static_cast<const float*>(x_dev.GetData()),
                  in_features, &beta, static_cast<float*>(y_dev.GetData()),
                  out_features),
      "cublasSgemm(CudaLinear)");
     y_dev.download(y.data.data(), y.GetDataSize() * sizeof(float));
    if (bias != nullptr) {
         AddBiasInPlace(y, *bias, rows, out_features);
    }
    return y;    
}
CudaTensor CudaLinearDeviceToDevice(const CudaTensor& x,
                                    const void* w_data,
                                    const std::vector<int>& w_shape,
                                    int device_id) {
    if (x.IsEmpty() || w_data == nullptr) {
        throw std::runtime_error("CudaLinearDevice: device input or weight pointer is null");
    }
    if (x.DeviceId() != device_id) {
        throw std::runtime_error("CudaLinearDevice: input tensor is on a different CUDA device");
    }
    const auto& x_shape = x.shape();
    VailedLinearTensor(x_shape, w_shape, x.ShapeStringShort(),
                       "CudaLinearDevice");
    bool input_is_1d = x_shape.size() == 1;
    int rows = input_is_1d ? 1 : x_shape[0];
    int in_features = input_is_1d ? x_shape[0] : x_shape[1];
    int out_features = w_shape[0];

    CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                           : std::vector<int>{rows, out_features},
               device_id);

    MyCublas handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CheckCublas(
    cublasSgemm(handle.GetHandle(), CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows,
                  in_features, &alpha, static_cast<const float*>(w_data),
                  in_features, static_cast<const float*>(x.GetData()),
                  in_features, &beta, static_cast<float*>(y.GetData()),
                  out_features),
      "cublasSgemm(CudaLinearDevice)");
       
      return y;
}
#endif
} //namespace
bool cuBLASBuilt() {
    #ifdef MINI_LLAMA_USE_CUDA
        return true;
    #else
        return false;
    #endif
}
Tensor CudaMatmul(const Tensor&a, const Tensor& b, int device_id) {
    #ifdef MINI_LLAMA_USE_CUDA
        VailedMatmulTensor(a, b);
        int m = a.shape[0];
    int k = a.shape[1];
    int n = b.shape[1];
    Tensor c({m, n}, 0.0f);

    CudaDeviceBuffer a_dev(a.GetDataSize() * sizeof(float), device_id);
     CudaDeviceBuffer b_dev(b.GetDataSize() * sizeof(float), device_id);
    CudaDeviceBuffer c_dev(c.GetDataSize() * sizeof(float), device_id);
    a_dev.upload(a.data.data(), a.GetDataSize() * sizeof(float));
     b_dev.upload(b.data.data(), b.GetDataSize() * sizeof(float));

    MyCublas handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    //转置相乘
    CheckCublas(cublasSgemm(handle.GetHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                          &alpha, static_cast<const float*>(b_dev.GetData()), n,
                          static_cast<const float*>(a_dev.GetData()), k, &beta,
                          static_cast<float*>(c_dev.GetData()), n),
              "cublasSgemm(CudaMatmul)");

    c_dev.download(c.data.data(), c.GetDataSize() * sizeof(float));
    return c;
#else
  (void)a;
  (void)b;
  (void)device_id;
  throw CudaMatmulNotBuiltError();
#endif
}
Tensor CudaLinear(const Tensor& x, const Tensor&weight, const Tensor* bias , int device_id) {
    #ifdef MINI_LLAMA_USE_CUDA
        ValidateLinearInputs(x, weight.shape);
        CudaDeviceBuffer w_dev(weight.GetDataSize() * sizeof(float), device_id);
        w_dev.upload(weight.data.data(), weight.GetDataSize() * sizeof(float));
        return CudaLinearHostToDevice(x, w_dev.GetData(), weight.shape, bias, device_id);
    #else
        (void)x;
        (void)weight;
        (void)bias;
        (void)device_id;
        throw CudaMatmulNotBuiltError();
    #endif
}

Tensor CudaLinearHostDevice(const Tensor& x, const void* w_data,
                            const std::vector<int>& w_shape,
                            const Tensor* bias, int device_id) {
    #ifdef MINI_LLAMA_USE_CUDA
        return CudaLinearHostToDevice(x, w_data, w_shape, bias, device_id);
    #else
        (void)x;
        (void)w_data;
        (void)w_shape;
        (void)bias;
        (void)device_id;
        throw CudaMatmulNotBuiltError();
    #endif
}

CudaTensor CudaLinearDevice(const CudaTensor& x, const void* w_data,
                            const std::vector<int>& w_shape,
                            int device_id) {
    #ifdef MINI_LLAMA_USE_CUDA
        return CudaLinearDeviceToDevice(x, w_data, w_shape, device_id);
    #else
        (void)x;
        (void)w_data;
        (void)w_shape;
        (void)device_id;
        throw CudaMatmulNotBuiltError();
    #endif
}
}// namespace yan_lamma
