#include "include/cuda_kvcache.h"
#include "include/cuda_runtime.h"
#include <stdexcept>
#include <limits>
#include <string>
#include <utility>
namespace yan_lamma {

bool CudaCacheBuild() {
    #ifdef MINI_LLAMA_USE_CUDA
        return true;
    #else
        return false;
    #endif
}
namespace {
void RequireCacheBuffers(const CudaDeviceBuffer& k, const CudaDeviceBuffer& v,
                         const char* caller) {
    if (k.IsEmpty() || v.IsEmpty()) {
        throw std::runtime_error(std::string(caller) +
            ": key and value cache buffers must both be initialized");
    }
}

void CheckCudaBuilt(const char* caller) {
    throw std::runtime_error(std::string(caller) +
        ": CUDA cache support was not built. Reconfigure with "
        "-DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine.");
}
void RequirePositive (int ele, const char* caller) {
    if (ele <= 0) {
        throw std::runtime_error(std::string(caller) +
            ": value must be positive, got " + std::to_string(ele));
    }
}
void CheckMul(size_t a, size_t b, const char* caller) {
    if (a == 0 || b == 0) {
        throw std::runtime_error(std::string(caller) +
            ": multiplication dimensions must be positive, got " +
            std::to_string(a) + " and " + std::to_string(b));
    }
    if (b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error(std::string(caller) +
            ": size multiplication overflow for " + std::to_string(a) +
            " * " + std::to_string(b));
    }
}
void VailedIndices (int index, int limit, const char* caller) {
    if (index < 0 || index >= limit) {
        throw std::runtime_error(std::string(caller) + ": index " +
            std::to_string(index) + " out of range [0, " +
            std::to_string(limit) + ")");
    }
}
void VailedHead(int head, int n_kv_heads, const char* caller) {
    if (head < 0 || head >= n_kv_heads) {
        throw std::runtime_error(std::string(caller) + ": KV head " +
            std::to_string(head) + " out of range [0, " +
            std::to_string(n_kv_heads) + ")");
    }
}
std::string ShapeToString(const std::vector<int>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    return result + "]";
}

// Host Tensor 和 CudaTensor 共用的 KV 形状检查。
void CheckKVShape(const std::vector<int>& shape, int n_kv_heads,
                  int head_dim, const char* caller) {
    RequirePositive(n_kv_heads, caller);
    RequirePositive(head_dim, caller);
    const std::string expected = ShapeToString({n_kv_heads, head_dim});
    if (shape.size() != 2) {
        throw std::runtime_error(std::string(caller) +
            ": expected 2D KV tensor with shape " + expected +
            ", got " + ShapeToString(shape));
    }
    if (shape[0] != n_kv_heads || shape[1] != head_dim) {
        throw std::runtime_error(std::string(caller) +
            ": expected shape " + expected + ", got " + ShapeToString(shape));
    }
}
}//namespace
CudaCache::CudaCache(int n_layers, int max_seq_len, int n_kv_heads,
                     int head_dim, int device_id) {
    Reset(n_layers, max_seq_len, n_kv_heads, head_dim, device_id);
}

CudaCache::~CudaCache() = default;

void CudaCache::Clear() {
    if (k_.IsEmpty() || v_.IsEmpty()) {
        return;
    }
    SetCudaService(device_id_);
    std::vector<float> temp (k_.GetByteSize() /sizeof (float), 0.0f);
    k_.upload(temp.data(), k_.GetByteSize());
    v_.upload(temp.data(), k_.GetByteSize());

}
void CudaCache::Reset() {
    k_.ReSet();
    v_.ReSet();
    n_layers_ = 0;
    max_seq_len_ = 0;
    n_kv_heads_ = 0;
    head_dim_ = 0;
    device_id_ = 0;
}
void CudaCache::Reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id) {
    #ifdef MINI_LLAMA_USE_CUDA
    RequirePositive(n_layers, "CudaCache::Reset n_layers");
    RequirePositive(max_seq_len, "CudaCache::Reset max_seq_len");
    RequirePositive(n_kv_heads, "CudaCache::Reset n_kv_heads");
    RequirePositive(head_dim, "CudaCache::Reset head_dim");
    if (device_id < 0) {
        throw std::runtime_error(
            "CudaCache::Reset: device_id must be non-negative, got " +
            std::to_string(device_id));
    }

    size_t bytes = static_cast<size_t>(n_layers);
    for (const int dimension : {max_seq_len, n_kv_heads, head_dim}) {
        CheckMul(bytes, static_cast<size_t>(dimension), "CudaCache::Reset");
        bytes *= static_cast<size_t>(dimension);
    }
    CheckMul(bytes, sizeof(float), "CudaCache::Reset bytes");
    bytes *= sizeof(float);
    CheckMul(bytes, 2, "CudaCache::Reset total K/V bytes");

    Reset();
    try {
        k_.Reset(bytes, device_id);
        v_.Reset(bytes, device_id);
        n_layers_ = n_layers;
        max_seq_len_ = max_seq_len;
        n_kv_heads_ = n_kv_heads;
        head_dim_ = head_dim;
        device_id_ = device_id;
        Clear();
    } catch (...) {
        try {
            Reset();
        } catch (...) {
            // 清理失败时保留原始的分配或清零异常。
        }
        throw;
    }
    #else
    (void) n_layers;
    (void) max_seq_len;
    (void) n_kv_heads;
    (void) head_dim;
    (void) device_id;
    CheckCudaBuilt("CudaCache::Reset");
    #endif
}
size_t CudaCache::SlotOffsetBytes(int layer, int pos) const {
    RequireCacheBuffers(k_, v_, "CudaCache::SlotOffsetBytes");
    VailedIndices(layer, n_layers_, "CudaCache::SlotOffsetBytes layer");
    VailedIndices(pos, max_seq_len_, "CudaCache::SlotOffsetBytes pos");

    // 返回该位置所有 KV head 的起始字节偏移。
    return ((static_cast<size_t>(layer) * max_seq_len_ + pos) *
            n_kv_heads_ * head_dim_) * sizeof(float);
}

size_t CudaCache::SlotOffsetBytes(int layer, int pos, int head) const {
    const size_t slot_offset = SlotOffsetBytes(layer, pos);
    VailedHead(head, n_kv_heads_, "CudaCache::SlotOffsetBytes head");

    return slot_offset + static_cast<size_t>(head) * head_dim_ * sizeof(float);
}

void CudaCache::WriteHost(int layer, int pos, const yan_lamma::Tensor& k, const Tensor& v) const {
    #ifdef MINI_LLAMA_USE_CUDA
    RequireCacheBuffers(k_, v_, "CudaCache::WriteHost");
    VailedIndices(layer, n_layers_, "CudaCache::WriteHost layer");
    VailedIndices(pos, max_seq_len_, "CudaCache::WriteHost pos");
    CheckKVShape(k.shape, n_kv_heads_, head_dim_, "CudaCache::WriteHost k");
    CheckKVShape(v.shape, n_kv_heads_, head_dim_, "CudaCache::WriteHost v");
    const size_t expected_size = static_cast<size_t>(n_kv_heads_) * head_dim_;
    if (expected_size != k.GetDataSize() || expected_size != v.GetDataSize()) {
        throw std::runtime_error(
            "CudaCache::WriteHost: expected " + std::to_string(expected_size) +
            " elements in each K/V tensor, got K " +
            std::to_string(k.GetDataSize()) + " and V " +
            std::to_string(v.GetDataSize()));
    }
    size_t byte_size = expected_size * sizeof(float);
    size_t offset = SlotOffsetBytes(layer, pos);
    char* k_data = static_cast<char*>(const_cast<void*>(k_.GetData())) + offset;
    char* v_data = static_cast<char*>(const_cast<void*>(v_.GetData())) + offset;
    SetCudaService(device_id_);
    CudaMemcpy(k_data, k.data.data(), byte_size, CudaMemoryKind::kHostDevice);
    CudaMemcpy(v_data, v.data.data(), byte_size, CudaMemoryKind::kHostDevice);
    #else
    (void) layer;
    (void) pos;
    CheckCudaBuilt("CudaCache::WriteHost");
    #endif
}
void CudaCache::WriteDevice(int layer, int pos, const CudaTensor& k, const CudaTensor& v) const {
    #ifdef MINI_LLAMA_USE_CUDA
    RequireCacheBuffers(k_, v_, "CudaCache::WriteDevice");
    VailedIndices(layer, n_layers_, "CudaCache::WriteDevice layer");
    VailedIndices(pos, max_seq_len_, "CudaCache::WriteDevice pos");
    CheckKVShape(k.shape(), n_kv_heads_, head_dim_, "CudaCache::WriteDevice k");
    CheckKVShape(v.shape(), n_kv_heads_, head_dim_, "CudaCache::WriteDevice v");
    if (k.DeviceId() != device_id_ || v.DeviceId() != device_id_) {
        throw std::runtime_error(
            "CudaCache::WriteDevice: K and V must be on cache device " +
            std::to_string(device_id_) + ", got K device " +
            std::to_string(k.DeviceId()) +
            " and V device " + std::to_string(v.DeviceId()));
    }
    const size_t expected_size = static_cast<size_t>(n_kv_heads_) * head_dim_;
    if (expected_size != k.NumEle() || expected_size != v.NumEle()) {
        throw std::runtime_error(
            "CudaCache::WriteDevice: expected " + std::to_string(expected_size) +
            " elements in each K/V tensor, got K " +
            std::to_string(k.NumEle()) + " and V " +
            std::to_string(v.NumEle()));
    }
    size_t byte_size = sizeof(float) * expected_size;
    size_t offset = SlotOffsetBytes(layer, pos);
    char* k_data = static_cast<char*>(const_cast<void*>(k_.GetData())) + offset;
    char* v_data = static_cast<char*>(const_cast<void*>(v_.GetData())) + offset;
    SetCudaService(device_id_);
    CudaMemcpy(k_data, k.GetData(), byte_size, CudaMemoryKind::kDeviceDevice);
    CudaMemcpy(v_data, v.GetData(), byte_size, CudaMemoryKind::kDeviceDevice);
    #else
    (void) layer;
    (void) pos;
    CheckCudaBuilt("CudaCache::WriteDevice");
    #endif
}
Tensor CudaCache::ReadKey(int layer, int pos) const {
    #ifdef MINI_LLAMA_USE_CUDA
    RequireCacheBuffers(k_, v_, "CudaCache::ReadKey");
    VailedIndices(layer, n_layers_, "CudaCache::ReadKey layer");
    VailedIndices(pos, max_seq_len_, "CudaCache::ReadKey pos");
    Tensor out({n_kv_heads_, head_dim_});
    size_t offset = SlotOffsetBytes(layer, pos);
    const char* src = static_cast<const char*>(k_.GetData()) + offset;
    SetCudaService(device_id_);
    CudaMemcpy(out.data.data(), src, out.GetDataSize() * sizeof(float),
               CudaMemoryKind::kDeviceHost);
    return out;
    #else
    (void) layer;
    (void) pos;
    CheckCudaBuilt("CudaCache::ReadKey");
    #endif
}
Tensor CudaCache::ReadVal(int layer, int pos) const {
    #ifdef MINI_LLAMA_USE_CUDA
    RequireCacheBuffers(k_, v_, "CudaCache::ReadVal");
    VailedIndices(layer, n_layers_, "CudaCache::ReadVal layer");
    VailedIndices(pos, max_seq_len_, "CudaCache::ReadVal pos");
    Tensor out({n_kv_heads_, head_dim_});
    size_t offset = SlotOffsetBytes(layer, pos);
    const char* src = static_cast<const char*>(v_.GetData()) + offset;
    SetCudaService(device_id_);
    CudaMemcpy(out.data.data(), src, out.GetDataSize() * sizeof(float),
               CudaMemoryKind::kDeviceHost);
    return out;
    #else
    (void) layer;
    (void) pos;
    CheckCudaBuilt("CudaCache::ReadVal");
    #endif
}
Tensor CudaCache:: ReadHeadKey(int layer, int pos, int head) const {
    #ifdef MINI_LLAMA_USE_CUDA
    RequireCacheBuffers(k_, v_, "CudaCache::ReadHeadKey");
    VailedHead(head, n_kv_heads_, "CudaCache::ReadHeadKey head");
    VailedIndices(layer, n_layers_, "CudaCache::ReadHeadKey layer");
    VailedIndices(pos, max_seq_len_, "CudaCache::ReadHeadKey pos");
    Tensor out({head_dim_});
    size_t offset = SlotOffsetBytes(layer, pos, head);
    const char* src = static_cast<const char*>(k_.GetData()) + offset;
    SetCudaService(device_id_);
    CudaMemcpy(out.data.data(), src, out.GetDataSize() * sizeof(float),
               CudaMemoryKind::kDeviceHost);
    return out;
    #else 
    (void) layer;
    (void) pos;
    (void) head;
    CheckCudaBuilt("CudaCache::ReadHeadKey");
    #endif
}
Tensor CudaCache::ReadHeadVal(int layer, int pos, int head) const {
     #ifdef MINI_LLAMA_USE_CUDA
    RequireCacheBuffers(k_, v_, "CudaCache::ReadHeadVal");
    VailedHead(head, n_kv_heads_, "CudaCache::ReadHeadVal head");
    VailedIndices(layer, n_layers_, "CudaCache::ReadHeadVal layer");
    VailedIndices(pos, max_seq_len_, "CudaCache::ReadHeadVal pos");
    Tensor out({head_dim_});
    size_t offset = SlotOffsetBytes(layer, pos, head);
    const char* src = static_cast<const char*>(v_.GetData()) + offset;
    SetCudaService(device_id_);
    CudaMemcpy(out.data.data(), src, out.GetDataSize() * sizeof(float),
               CudaMemoryKind::kDeviceHost);
    return out;
    #else 
    (void) layer;
    (void) pos;
    (void) head;
    CheckCudaBuilt("CudaCache::ReadHeadVal");
    #endif 
}
const void* CudaCache::KData () const {
    if (k_.IsEmpty()) {
        throw std::runtime_error("CudaCache::KData: key cache is empty");
    }
    return k_.GetData();
}
const void* CudaCache::vData () const {
    if (v_.IsEmpty()) {
         throw std::runtime_error("CudaCache::vData: value cache is empty");
    }  
    return v_.GetData();  
}
size_t CudaCache::Bytes_ () const {
    return k_.GetByteSize() + v_.GetByteSize();
}
}// namespace yan_lamma
