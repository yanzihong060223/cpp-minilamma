#pragma once
#include "include/tensor.h"
#include "tensor.h"
#include "include/kvcache.h"
#include <vector>
#include <string>
class LammaModel;
class CudaKvCache;
namespace yan_lamma {
class Context {
public:
Context() = default;
explicit Context(const LammaModel* model);
Context (const Context&) = delete;
Context& operator= (const Context&) = delete;
Context (Context &&) = delete;
Context& operator= (Context &&) = delete;
const LammaModel* model_ptr_ = nullptr;
// token历史
std::vector<int> token_id_;
// cpu kvcache
KvCache cpu_cache_;
//GPU KvCache
//CudaKvCache cuda_cache_;
int n_decode_size = 0; //decode计数
int n_profill_size = 0; //预填充计数
size_t pos =0; //当前token位置


};



}// namespace yam_lamma