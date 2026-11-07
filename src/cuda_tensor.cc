#include "include/cuda_tensor.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace yan_lamma {
namespace {

std::string ShapeToString(const std::vector<int>& shape) {
    std::string result = "[";
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (axis != 0) {
            result += ", ";
        }
        result += std::to_string(shape[axis]);
    }
    return result + "]";
}

size_t CheckNumel(const std::vector<int>& shape, const std::string& caller) {
    if (shape.empty()) {
        throw std::runtime_error(caller + ": shape must not be empty");
    }
    size_t total = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] <= 0) {
            throw std::runtime_error(
                caller + ": dimension at axis " + std::to_string(axis) +
                " must be positive, got " + std::to_string(shape[axis]) +
                " in shape " + ShapeToString(shape));
        }
        const size_t dimension = static_cast<size_t>(shape[axis]);
        if (total > std::numeric_limits<size_t>::max() / dimension) {
            throw std::runtime_error(caller + ": shape element count overflow for " +
                                     ShapeToString(shape));
        }
        total *= dimension;
    }
    return total;
}

void CheckHostTensor(const Tensor& host, const std::vector<int>& shape,
                     const std::string& caller) {
    if (host.shape != shape) {
        throw std::runtime_error(caller + ": expected host shape " +
                                 ShapeToString(shape) + ", got " +
                                 ShapeToString(host.shape));
    }
}

} // namespace

CudaTensor::CudaTensor(const std::vector<int>& shape, int device_id) {
    Reset(shape, device_id);
}

CudaTensor::CudaTensor(CudaTensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      device_id_(std::exchange(other.device_id_, 0)),
      numel_(std::exchange(other.numel_, 0)),
      cuda_buffer_(std::move(other.cuda_buffer_)) {
    other.shape_.clear();
}

CudaTensor& CudaTensor::operator=(CudaTensor&& other) noexcept {
    if (this != &other) {
        cuda_buffer_ = std::move(other.cuda_buffer_);
        shape_ = std::move(other.shape_);
        device_id_ = std::exchange(other.device_id_, 0);
        numel_ = std::exchange(other.numel_, 0);
        other.shape_.clear();
    }
    return *this;
}

void CudaTensor::Reset() {
    cuda_buffer_.ReSet();
    shape_.clear();
    device_id_ = 0;
    numel_ = 0;
}

void CudaTensor::Reset(const std::vector<int>& shape, int device_id) {
    const size_t numel = CheckNumel(shape, "CudaTensor::Reset");

    // 所有可能失败的准备工作完成后，再替换当前张量。
    std::vector<int> new_shape(shape);
    CudaDeviceBuffer new_buffer(numel * sizeof(float), device_id);
    cuda_buffer_ = std::move(new_buffer);
    shape_ = std::move(new_shape);
    device_id_ = device_id;
    numel_ = numel;
}

void CudaTensor::UploadFromHost(const Tensor& src) {
    if (IsEmpty()) {
        throw std::runtime_error("CudaTensor::UploadFromHost: cannot upload to an empty tensor");
    }
    CheckHostTensor(src, shape_, "CudaTensor::UploadFromHost");
    if (src.data.size() != NumEle()) {
        throw std::runtime_error(
            "CudaTensor::UploadFromHost: expected " + std::to_string(NumEle()) +
            " host elements, got " + std::to_string(src.data.size()));
    }
    cuda_buffer_.upload(src.data.data(), Bytes());
}

void CudaTensor::DownloadToHost(Tensor& dst) const {
    if (IsEmpty()) {
        throw std::runtime_error("CudaTensor::DownloadToHost: cannot download an empty tensor");
    }
    CheckHostTensor(dst, shape_, "CudaTensor::DownloadToHost");
    if (dst.data.size() != NumEle()) {
        throw std::runtime_error(
            "CudaTensor::DownloadToHost: expected " + std::to_string(NumEle()) +
            " host elements, got " + std::to_string(dst.data.size()));
    }
    cuda_buffer_.download(dst.data.data(), Bytes());
}

Tensor CudaTensor::Download() const {
    if (IsEmpty()) {
        throw std::runtime_error("CudaTensor::Download: cannot download an empty tensor");
    }
    Tensor dst(shape_);
    DownloadToHost(dst);
    return dst;
}

const void* CudaTensor::GetData() const {
    return cuda_buffer_.GetData();
}

void* CudaTensor::GetData() {
    return cuda_buffer_.GetData();
}

bool CudaTensor::IsEmpty() const {
    return cuda_buffer_.IsEmpty();
}

size_t CudaTensor::NumEle() const {
    return numel_;
}

int CudaTensor::num_dims() const {
    return static_cast<int>(shape_.size());
}

size_t CudaTensor::Bytes() const {
    return cuda_buffer_.GetByteSize();
}

int CudaTensor::DeviceId() const {
    return device_id_;
}
const CudaDeviceBuffer& CudaTensor::buffer() const {
    return cuda_buffer_;
}
const std::vector<int>& CudaTensor::shape() const {
    return shape_;
}
std::string CudaTensor::ShapeStringShort() const {
    return ShapeToString(shape_);
}

CudaTensor TensorHostToDevice(const Tensor& src, int device_id) {
    CudaTensor dst(src.shape, device_id);
    dst.UploadFromHost(src);
    return dst;
}

} // namespace yan_lamma
