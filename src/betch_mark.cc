#include "include/betch_mark.h"
#include "include/context.h"
#include "include/forward.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <chrono>

namespace yan_lamma {

void DumpTensor(const Tensor& tensor, const std::string& name) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<float>::max_digits10);

    out << "Tensor: " << name << "\n"
        << "  shape: " << tensor.ShapeToString() << "\n"
        << "  dims: " << tensor.GetNumDims() << "\n"
        << "  elements: " << tensor.GetDataSize() << "\n"
        << "  data_bytes: "
        << tensor.GetDataSize() * sizeof(float) << "\n"
        << "  data: [";

    for (std::size_t i = 0; i < tensor.data.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << tensor.data[i];
    }

    out << "]\n";
    std::cout << out.str();
}

void DumpLogitsK(const Tensor& logits, int k) {
    // first 保存权重，second 保存 token ID。
    std::vector<std::pair<float, std::size_t>> candidates;
    candidates.reserve(logits.GetDataSize());

    for (std::size_t id = 0; id < logits.GetDataSize(); ++id) {
        const float weight = logits.data[id];
        if (!std::isfinite(weight)) {
            throw std::runtime_error(
                "DumpLogitsK: logits must be finite");
        }
        candidates.emplace_back(weight, id);
    }

    if (k <= 0 || candidates.empty()) {
        std::cout << "Top-0 logits: []\n";
        return;
    }

    const std::size_t count = std::min(
        static_cast<std::size_t>(k), candidates.size());

    std::partial_sort(
        candidates.begin(),
        candidates.begin() + count,
        candidates.end(),
        [](const auto& a, const auto& b) {
            if (a.first == b.first) {
                return a.second < b.second;
            }
            return a.first > b.first;
        });

    candidates.resize(count);

    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "Top-" << count << " logits:\n";

    for (const auto& [weight, token_id] : candidates) {
        out << "  token_id=" << token_id
            << " logit=" << weight << "\n";
    }

    std::cout << out.str();
}

void DumpKvcacheInfo(const KvCache& cache, int current_pos) {
    std::cout << "KV cache:\n"
              << "  layout: "
              << "[n_layers, max_seq_len, n_kv_heads, head_dim]\n";

    if (current_pos >= 0) {
        std::cout << "  current_pos: " << current_pos << "\n";
    }

    DumpTensor(cache.key_, "key");
    DumpTensor(cache.value_, "value");
}
BetchMark RunBetchMrk(const Model& model, int seed, std::vector<int> prompt_tokens, bool verbose, int limits) {
    BetchMark result;
    Context ctx(&model);
    SamplerParmas parmas;
    parmas.seed = seed;
    Sampler sampler(seed);
    result.n_prompt_tokens = prompt_tokens.size();
    if (verbose) {
        std::cout << "prompt tokens size: " << prompt_tokens.size() << "\n";
    }
    auto prefile_start = std::chrono::steady_clock::now();
    Batch prefile = Batch::FromTokens(prompt_tokens, 0);
    Tensor logits = ForwardBatch(model, ctx, prefile);
    auto end =  std::chrono::steady_clock::now();
    result.prefile_ms = std::chrono::duration<double, std::milli>(end - prefile_start).count();
    if (verbose) {
         DumpTensor(logits, "logits");
         DumpLogitsK(logits, 5);
         DumpKvcacheInfo(ctx.cpu_cache_, ctx.pos);
    }
    // decode time
    int generate_tokens = 0;
    if (limits > 0 && static_cast<int>(prompt_tokens.size()) < model.config.max_seq_len) {
        const int max_generate = std::min(
            limits, model.config.max_seq_len - result.n_prompt_tokens);
        std::vector<int> tokens = prompt_tokens;
        int next_token =  sampler.SamplerO(logits,  parmas);
        generate_tokens++;
        tokens.push_back(next_token);
        auto decode_time = std::chrono::steady_clock::now();
        for (int i = 1; i < max_generate; i++) {
            Batch decode = Batch::Single(tokens.back(), tokens.size() - 1);
            Tensor Logits = ForwardBatch(model, ctx, decode);
            if (verbose) {
                DumpTensor(Logits, "logits");
            }
            int next_token = sampler.SamplerO(Logits,  parmas);
            tokens.push_back(next_token);
            generate_tokens++;
        }
        auto decode_time_end = std::chrono::steady_clock::now();
        result.decode_ms = std::chrono::duration<double, std::milli>(decode_time_end - decode_time).count();
    }
    result.n_generate_tokens = generate_tokens;
    result.n_decode_tokens = std::max(generate_tokens - 1, 0);
    if (verbose) {
        DumpKvcacheInfo(ctx.cpu_cache_, ctx.pos);
    }
    return result;

}
}  // namespace yan_lamma
