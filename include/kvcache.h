#pragma once
#include "include/tensor.h"

#include <vector>
#include <string>
namespace yan_lamma {
class KvCache {
//KV cache stores keys and values for all layers and positions
// keys:   [n_layers, max_seq_len, n_kv_heads, head_dim]
// values: [n_layers, max_seq_len, n_kv_heads, head_dim]
public:
KvCache() = default;
KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);
Tensor key_;
Tensor value_;
// write
void Write(int layers, int pos, const Tensor& k, const Tensor& v);
// Read
float* KeyPtr(int layers, int pos, int kv_heads);
float* ValPtr(int layers, int pos, int kv_heads);
};
} // namespace yan_lamma
