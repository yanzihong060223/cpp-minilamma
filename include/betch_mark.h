#pragma once
#include "include/sampler.h"
#include "include/kvcache.h"
#include "batch.h"
#include "include/sampler.h"
#include "include/tensor.h"
#include "include/model.h"

#include <string>
#include <vector>

namespace yan_lamma {
struct BetchMark {
// n_prompt_tokens
int n_prompt_tokens = 0;
int n_generate_tokens = 0;
int n_decode_tokens = 0;
double prefile_ms = 0;
double decode_ms = 0;
double TokenPerSec () {
    double total_time = prefile_ms + decode_ms;
    if (total_time <= 0) {
        return 0.0;
    }
    return n_generate_tokens * 1000 / total_time;
}
double DeTokenPerSrc() {
    if ( decode_ms <= 0.0 ||  n_decode_tokens == 0) {
        return 0;
    }
    return n_decode_tokens * 1000 / decode_ms;
}
};
//获得张量信息
void DumpTensor(const Tensor& tensor, const std::string& name);
// 获得logits的前k个
void DumpLogitsK(const Tensor& logits, int k);
void DumpKvcacheInfo(const KvCache& cache, int current_pos = -1);
BetchMark RunBetchMrk(const Model& model, int seed, std::vector<int>prompt_tokens, bool verbose, int limits);
}// namespace yan_lamma