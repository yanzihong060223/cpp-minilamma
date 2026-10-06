#include "include/context.h"
#include "include/kvcache.h"

namespace yan_lamma {
Context :: Context(const LammaModel* model) :  model_ptr_(model) {
    if (model_ptr_) {
        const auto a = model_ptr_->config;
         cpu_cache_ = KvCache(a.n_layers, a.max_seq_len, a.n_kv_heads, a. head_dim);
    }
}

}; //namespace yan_lamma