#pragma once
#include <cstddef>
#include <string>

namespace yan_lamma {
// 判断是否启用 Cuda 通过底层宏编译
bool CudaRuntimeBuilt();
struct CudaInfo {
    int device_id = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    size_t total_bytes = 0;
    int runtime_version = 0;
    int driver_version = 0;
};
enum class CudaMemoryKind {
    kHostDevice,
    kDeviceDevice,
    kDeviceHost
};
CudaInfo GetCudaInfo (int device_id);
std::string CudaInfo (const CudaInfo& info);
int DeviceCount();
void SetCudaService(int device_id);
// 获得显存
void * CudaMemory(size_t byte_size);
//释放
void CudaFreeMemory(void* ptr);
//host device之间的数据拷贝
void CudaMemcpy(void* des, void* src, size_t bytes,  CudaMemoryKind kind);
class CudaDeviceBuffer {
 public:
    CudaDeviceBuffer() = default;
    explicit CudaDeviceBuffer(size_t byte_size, int device_id = 0);
    void ReSet();
    void Reset(size_t byte_size, int device_id);
    ~CudaDeviceBuffer();
    CudaDeviceBuffer (const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer operator= (const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer (CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator= (CudaDeviceBuffer&& other) noexcept;
    // CPU 内存中的数据拷贝到显存
    void upload(void* src, size_t byte_size);
    // 显存到内存
    void download(void* des, size_t byte_size);
    bool IsEmpty() {return data_ == nullptr;}
    int GetDeviceId() {return device_id_;}
    const int GetDeviceId() const {return device_id_;}
    void* GetData() {return data_;}
    const void* GetData() const{return data_;}
    size_t GetByteSize() {return byte_size_;}
    const size_t GetByteSize() const {return byte_size_;}
 private:
 void ReleaseDevice() noexcept;
 private:
 void* data_ = nullptr;
 size_t byte_size_ = 0;
 int device_id_ = 0;
};
}//namespace yan_lamma
