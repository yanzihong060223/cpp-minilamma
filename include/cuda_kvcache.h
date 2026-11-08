#pragma once
#include "include/cuda_tensor.h"
#include "include/cuda_runtime.h"
#include "include/tensor.h"
// keys:   [n_layers, max_seq_len, n_kv_heads, head_dim]
// values: [n_layers, max_seq_len, n_kv_heads, head_dim]
namespace yan_lamma {
bool CudaCacheBuild();
class CudaCache {
 public:
    CudaCache () = default;
    CudaCache (int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id = 0);
    ~CudaCache();
    CudaCache (const CudaCache&) = delete;
    CudaCache& operator= (const CudaCache&) = delete;
    CudaCache (CudaCache&&) noexcept = default;
    CudaCache& operator= (CudaCache&&) noexcept  = default;
    void Reset();
    void Reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim, int device_id = 0);
    //host端写入
    void WriteHost(int layer, int pos, const yan_lamma::Tensor& k, const Tensor& v) const;
    void WriteDevice(int layer, int pos, const CudaTensor& k, const CudaTensor& v) const;
    // 回显
    Tensor ReadKey(int layer, int pos) const;
    Tensor ReadVal(int layer, int pos) const;

    Tensor ReadHeadKey(int layer, int pos, int head) const;
    Tensor ReadHeadVal(int layer, int pos, int head) const;
    // 直接获得kv张量的显存
    const void* KData () const;
    const void* vData () const;
    //
    size_t Bytes_ () const;
    int n_layers() const { return n_layers_; }

    int max_seq_len() const { return max_seq_len_; }

    int n_kv_heads() const { return n_kv_heads_; }

    int head_dim() const { return head_dim_; }

    int device_id() const { return device_id_; }
    
    void Clear();
    //四维转为一维
    size_t SlotOffsetBytes(int layer, int pos) const;
    size_t SlotOffsetBytes(int layer, int pos, int head) const;
 private:
    int n_layers_ = 0;
    int max_seq_len_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int device_id_ = 0;
    yan_lamma::CudaDeviceBuffer k_;
    yan_lamma::CudaDeviceBuffer v_;

};
} // namespace yan_lamma