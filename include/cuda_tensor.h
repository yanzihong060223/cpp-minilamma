#pragma once
#include "include/cuda_runtime.h"
#include "include/tensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace yan_lamma {
class CudaTensor {
 public:
    CudaTensor() = default;
    explicit CudaTensor(const std::vector<int>& shape, int device_id = 0);
    ~CudaTensor() = default;
    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;
    CudaTensor(CudaTensor&& other) noexcept;
    CudaTensor& operator=(CudaTensor&& other) noexcept;

    void Reset();
    // 显式形状必须非空，且每一维都为正数。
    void Reset(const std::vector<int>& shape, int device_id);
    void UploadFromHost(const Tensor& src);
    void DownloadToHost(Tensor& dst) const;
    Tensor Download() const;
    void* GetData();
    const void* GetData() const;
    bool IsEmpty() const;
    size_t NumEle() const;
    int num_dims() const;
    size_t Bytes() const;
    int DeviceId() const;
    std::string ShapeStringShort() const;
    const CudaDeviceBuffer& buffer() const;
    const std::vector<int>& shape() const;
 private:
    std::vector<int> shape_;
    int device_id_ = 0;
    size_t numel_ = 0;
    CudaDeviceBuffer cuda_buffer_;
};

// 测试和调试用的宿主到设备转换；频繁拷贝会增加推理开销。
CudaTensor TensorHostToDevice(const Tensor& src, int device_id = 0);

} // namespace yan_lamma
