#include "include/kvcache.h"
#include "include/tensor.h"

#include <stdexcept>
#include <cstring>
#include <string>
namespace yan_lamma {
namespace {
    //判断向量是否是4维
    void  Require4D(const Tensor& t) {
        if (t.GetNumDims() != 4) {
            throw std::runtime_error("No Vailed Shape");
        }
    }
}
KvCache :: KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
    key_ = key_.MakeFouthD(n_layers, max_seq_len,  n_kv_heads, head_dim, 0.0f);
    value_ = value_.MakeFouthD(n_layers, max_seq_len,  n_kv_heads, head_dim, 0.0f);
}
void KvCache::Write(int layers, int pos, const Tensor& k, const Tensor& v) {
    Require4D(key_);
    Require4D(value_);
    if (key_.shape != value_.shape) {
        throw std::runtime_error("KV Cache Shapes Do Not Match");
    }
    if (k.GetNumDims() != 2 || v.GetNumDims() !=2 || k.shape != v.shape) {
        throw std::out_of_range("Can Not Match K,V");
    }
    int n_kv_heads = k.shape[0]; //两个kvhead
    int head_dim  =  k.shape[1]; //每个head向量包含的float
    if(layers < 0 || layers >= key_.shape[0]) {
        throw std::out_of_range("Layers Out Of Range");
    }
    if (pos < 0 || pos >= key_.shape[1]) {
        throw std::out_of_range("Pos Out Of Range");
    }
    if (n_kv_heads != key_.shape[2] || head_dim != key_.shape[3]) {
        throw std::out_of_range("Write Shape Not Match KV Shape");
    }
    const size_t head_stride = static_cast<size_t>(head_dim);
    const size_t pos_stride = static_cast<size_t>(n_kv_heads) * head_stride;
    const size_t layer_stride = static_cast<size_t>(key_.shape[1]) * pos_stride;
    const size_t base = static_cast<size_t>(layers) * layer_stride +
                        static_cast<size_t>(pos) * pos_stride;
    for (int i = 0; i < n_kv_heads; i++) {
        const size_t src_offset = static_cast<size_t>(i) * head_stride;
        const size_t offset = base + src_offset;
        std::memcpy(key_.data.data() + offset, k.data.data() + src_offset,
                    head_stride * sizeof(float));
        std::memcpy(value_.data.data() + offset, v.data.data() + src_offset,
                    head_stride * sizeof(float));

    }
}
float*  KvCache::KeyPtr(int layers, int pos, int kv_heads) {
    Require4D(key_);
     if(layers < 0 || layers >= key_.shape[0]) {
        throw std::out_of_range("Layers Out Of Range");
    }
    if (pos < 0 || pos >= key_.shape[1]) {
        throw std::out_of_range("Pos Out Of Range");
    }
    if (kv_heads < 0 || kv_heads >= key_.shape[2]) {
        throw std::out_of_range("KV Head Out Of Range");
    }
    const size_t head_stride = static_cast<size_t>(key_.shape[3]);
    const size_t pos_stride = static_cast<size_t>(key_.shape[2]) * head_stride;
    const size_t layer_stride = static_cast<size_t>(key_.shape[1]) * pos_stride;
    return &key_.data[layers * layer_stride + pos * pos_stride + kv_heads * head_stride];
}
float* KvCache::ValPtr(int layers, int pos, int kv_heads) {
    Require4D(value_);
      if(layers < 0 || layers >= value_.shape[0]) {
        throw std::out_of_range("Layers Out Of Range");
    }
    if (pos < 0 || pos >= value_.shape[1]) {
        throw std::out_of_range("Pos Out Of Range");
    }
    if (kv_heads < 0 || kv_heads >= value_.shape[2]) {
        throw std::out_of_range("KV Head Out Of Range");
    }
    const size_t head_stride = static_cast<size_t>(value_.shape[3]);
    const size_t pos_stride = static_cast<size_t>(value_.shape[2]) * head_stride;
    const size_t layer_stride = static_cast<size_t>(value_.shape[1]) * pos_stride;
    return &value_.data[layers * layer_stride + pos * pos_stride + kv_heads * head_stride];
}
}// namespace yan_lamma