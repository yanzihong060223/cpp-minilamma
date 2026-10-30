#include "include/cuda_runtime.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>
#ifdef MINI_LLAMA_USE_CUDA
#include <cuda_runtime_api.h>
#endif
namespace yan_lamma {
namespace {
#ifndef MINI_LLAMA_USE_CUDA
std::runtime_error CudaNotBuiltError() {
       return std::runtime_error(
      "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a "
      "NVIDIA CUDA machine.");
}
#endif
#ifdef MINI_LLAMA_USE_CUDA
void CheckCuda (cudaError_t err, const char*exp) {
    if (err == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(exp) + ": " + cudaGetErrorString(err));
}
cudaMemcpyKind ToCudaMemcpyKind (CudaMemoryKind kind) {
    switch (kind) {
        case CudaMemoryKind::kHostDevice:
            return cudaMemcpyHostToDevice;
        case CudaMemoryKind::kDeviceDevice:
            return cudaMemcpyDeviceToDevice;
        case CudaMemoryKind::kDeviceHost:
            return cudaMemcpyDeviceToHost;
    }
    throw std::runtime_error("Invalid CUDA memory copy kind");
}
#endif
}// namespace
bool CudaRuntimeBuilt() {
    #ifdef  MINI_LLAMA_USE_CUDA
    return true;
    #else
    return false;
    #endif
}
int DeviceCount() {
    #ifdef  MINI_LLAMA_USE_CUDA
    int count = 0;
    CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
    #else 
    throw  CudaNotBuiltError();
    #endif
}
void SetCudaService(int device_id) {
     #ifdef  MINI_LLAMA_USE_CUDA
     CheckCuda(cudaSetDevice(device_id), "cudaSetDevice");
     #else 
     //显式不使用这个参数
     (void) device_id;
     throw  CudaNotBuiltError();
     #endif
}
struct CudaInfo GetCudaInfo (int device_id) {
     #ifdef  MINI_LLAMA_USE_CUDA
     int count =  DeviceCount();
      if (device_id < 0 || device_id >= count) {
        throw std::runtime_error("CUDA device " + std::to_string(device_id) +
                             " is not available; detected " +
                             std::to_string(count) + " device(s)");
    }
    cudaDeviceProp prop{};
    CheckCuda(cudaGetDeviceProperties(&prop, device_id),
            "cudaGetDeviceProperties");
    struct CudaInfo info;
    info.device_id = device_id;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_bytes = prop.totalGlobalMem;
    CheckCuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
    CheckCuda(cudaRuntimeGetVersion(&info.runtime_version),
            "cudaRuntimeGetVersion");
    return info;
    #else
    (void)device_id;
    throw CudaNotBuiltError();
    #endif
}
std::string CudaInfo (const struct CudaInfo& info) {
    std::ostringstream out;
    out << "[CUDA] device_id=" << info.device_id
        << " name=\"" << info.name << "\""
        << " compute=" << info.compute_major << '.' << info.compute_minor
        << " total_bytes=" << info.total_bytes
        << " runtime_version=" << info.runtime_version
        << " driver_version=" << info.driver_version;
    return out.str();
}
void * CudaMemory(size_t byte_size) {
  #ifdef MINI_LLAMA_USE_CUDA
  if (byte_size == 0) {
    return nullptr;
  }
  void* ptr = nullptr;
  CheckCuda(cudaMalloc(&ptr, byte_size), "cudaMalloc");
  return ptr;
#else
  (void)byte_size;
  throw CudaNotBuiltError();
#endif
}
void CudaFreeMemory(void* ptr) {
    #ifdef MINI_LLAMA_USE_CUDA 
    if (ptr == nullptr) {
    return;
    }
    CheckCuda(cudaFree(ptr), "cudaFree");
    #else
    (void) ptr;
    throw CudaNotBuiltError();
    #endif
}
void CudaMemcpy(void* des, void* src, size_t bytes,  CudaMemoryKind kind) {
     #ifdef MINI_LLAMA_USE_CUDA 
     if (bytes == 0) {
        return;
     }
     if (des == nullptr || src == nullptr) {
        throw std::runtime_error(
        "cudaMemcpy requires non-null src and dst for non-empty copies");
     }
      CheckCuda(cudaMemcpy(des, src, bytes, ToCudaMemcpyKind(kind)), "cudaMemcpy");
    #else
    (void) des;
    (void) src;
    (void) bytes;
    (void) kind;
    throw CudaNotBuiltError();
    #endif
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      byte_size_(std::exchange(other.byte_size_, 0)),
      device_id_(std::exchange(other.device_id_, 0)) {}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
  if (this != &other) {
    ReleaseDevice();
    data_ = std::exchange(other.data_, nullptr);
    byte_size_ = std::exchange(other.byte_size_, 0);
    device_id_ = std::exchange(other.device_id_, 0);
  }
  return *this;
}
void CudaDeviceBuffer::ReSet() {
  if (data_ == nullptr) {
    return;
  }
  SetCudaService(device_id_);
  CudaFreeMemory(data_);
  data_ = nullptr;
  byte_size_ = 0;
  device_id_ = 0;
}
void CudaDeviceBuffer::Reset(size_t byte_size, int device_id) {
  ReSet();
  if (byte_size == 0) {
    device_id_ = device_id;
    return;
  }
   SetCudaService(device_id);
   data_ = CudaMemory(byte_size);
   byte_size_ = byte_size;
   device_id_ = device_id;

}
CudaDeviceBuffer:: CudaDeviceBuffer(size_t byte_size, int device_id) {
   Reset(byte_size, device_id);
}
void CudaDeviceBuffer::upload(void* src, size_t byte_size) {
  if (byte_size == 0) {
    return;
  }
  if (byte_size > byte_size_) {
    throw std::runtime_error("CUDA upload exceeds device buffer size");
  }
  if (src == nullptr || data_ == nullptr) {
    throw std::runtime_error(
        "CUDA upload requires non-null host and device pointers");
  }
  SetCudaService(device_id_);
  CudaMemcpy(data_, src, byte_size, CudaMemoryKind::kHostDevice);
}
void CudaDeviceBuffer:: download(void* des, size_t byte_size) {
  if (byte_size == 0) {
    return;
  }
  if (byte_size > byte_size_) {
    throw std::runtime_error("CUDA download exceeds device buffer size");
  }
  if (des == nullptr || data_ == nullptr) {
    throw std::runtime_error(
        "CUDA download requires non-null host and device pointers");
  }
  SetCudaService(device_id_);
  CudaMemcpy(des, data_, byte_size, CudaMemoryKind::kDeviceHost);
}
void CudaDeviceBuffer::ReleaseDevice() noexcept {
  try {
    ReSet();
  } catch (...) {
    // 清理失败时不让异常逃出 noexcept 函数。
  }
  data_ = nullptr;
  byte_size_ = 0;
  device_id_ = 0;
}
 CudaDeviceBuffer::~CudaDeviceBuffer() {
    ReleaseDevice();
 }
}// namespace yan_lamma
