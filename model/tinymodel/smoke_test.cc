// Standalone integration check using the project's actual loader and forward code.
#include "include/batch.h"
#include "include/context.h"
#include "include/forward.h"
#include "include/load.h"
#include "include/ops.h"
#include "include/sampler.h"
#include "include/tonizer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckLogits(const yan_lamma::Tensor& logits, int vocab_size) {
    Require(logits.shape == std::vector<int>{vocab_size},
            "logits shape does not match vocab_size");
    Require(logits.data.size() == static_cast<std::size_t>(vocab_size),
            "logits storage does not match vocab_size");
    for (float value : logits.data) {
        Require(std::isfinite(value), "logits contain NaN/Inf");
    }
}

float Compare(const std::vector<float>& a, const std::vector<float>& b) {
    Require(a.size() == b.size(), "comparison size mismatch");
    float error = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        Require(std::isfinite(a[i]) && std::isfinite(b[i]),
                "comparison contains NaN/Inf");
        error = std::max(error, std::abs(a[i] - b[i]));
    }
    Require(error <= 1e-6f, "batch and token forward disagree");
    return error;
}

std::string Escape(const std::string& text) {
    std::ostringstream out;
    for (unsigned char byte : text) {
        if (byte >= 32 && byte < 127 && byte != '\\' && byte != '"') {
            out << static_cast<char>(byte);
        } else {
            out << "\\x" << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned>(byte) << std::dec;
        }
    }
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Require(argc >= 2 && argc <= 4,
                "usage: tiny-model-smoke <model-dir> [prompt] [n-predict]");
        const std::filesystem::path directory = argv[1];
        const std::string prompt = argc >= 3 ? argv[2] : "hello";
        int n_predict = 16;
        if (argc >= 4) {
            std::size_t parsed = 0;
            const std::string value = argv[3];
            n_predict = std::stoi(value, &parsed);
            Require(parsed == value.size(), "invalid n-predict");
        }
        Require(n_predict >= 0, "n-predict must be non-negative");

        const auto manifest = yan_lamma::ParseModelJson(
            (directory / "model.json").string());
        auto model = yan_lamma::ParseModel(
            (directory / "model.json").string(),
            (directory / "model.bin").string());
        Require(model.is_loaded, model.load_error_);
        Require(manifest.tensor_json.size() ==
                    3 + 9 * static_cast<std::size_t>(model.config.layers),
                "unexpected tensor count");
        auto tokenizer = yan_lamma::GetTonizer("");
        const auto& tokenizer_info = manifest.toninzer_json;
        Require(tokenizer_info.type == "ASCII" &&
                    tokenizer_info.path_name.empty(),
                "this smoke runner requires the ASCII tiny manifest");
        Require(model.config.vocab_size == tokenizer->GetVocabSize() &&
                    tokenizer_info.bos_id == tokenizer->GetBOSId() &&
                    tokenizer_info.eos_id == tokenizer->GetEOSId() &&
                    tokenizer_info.unk_id == tokenizer->GetUNKId(),
                "model/tokenizer vocabulary mismatch");
        auto tokens = tokenizer->Encode(prompt);
        for (int token : tokens) {
            Require(token >= 0 && token < model.config.vocab_size,
                    "prompt token out of range");
        }
        Require(tokens.size() + static_cast<std::size_t>(n_predict) <=
                    static_cast<std::size_t>(model.config.max_seq_len),
                "prompt + n-predict exceeds max_seq_len");
        const std::string decoded_prompt = tokenizer->Decode(tokens);
        std::cout << "Prompt tokens:";
        for (int token : tokens) std::cout << ' ' << token;
        std::cout << "\nDecoded prompt: \"" << Escape(decoded_prompt) << "\"\n";

        yan_lamma::Context batch_context(&model);
        yan_lamma::Context token_context(&model);
        const std::vector<int> kv_shape{
            model.config.layers, model.config.max_seq_len,
            model.config.n_kv_heads, model.config.head_dim};
        Require(batch_context.cpu_cache_.key_.shape == kv_shape &&
                    batch_context.cpu_cache_.value_.shape == kv_shape,
                "KV cache shape mismatch");
        yan_lamma::Batch builder;
        auto logits = yan_lamma::ForwardBatch(
            model, batch_context, builder.FromTokens(tokens, 0));
        CheckLogits(logits, model.config.vocab_size);
        yan_lamma::Tensor token_logits;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            token_context.pos = i;  // ForwardToken does not advance the position.
            token_logits = yan_lamma::ForwardToken(model, token_context, tokens[i]);
            CheckLogits(token_logits, model.config.vocab_size);
        }
        float max_error = Compare(logits.data, token_logits.data);
        Require(batch_context.token_id_ == tokens, "prefill token history mismatch");
        Require(batch_context.pos == tokens.size() - 1, "prefill position mismatch");

        yan_lamma::SamplerParmas params;
        params.temperature = 0.0f;
        params.seed = 42;
        yan_lamma::Sampler sampler(params);
        std::vector<int> generated;
        std::size_t decode_steps = 0;
        for (int i = 0; i < n_predict; ++i) {
            const int next = sampler.SamplerO(logits, params);
            Require(next == static_cast<int>(yan_lamma::ArgMax(logits)),
                    "greedy sampler disagrees with ArgMax");
            Require(next >= 0 && next < model.config.vocab_size,
                    "generated token out of range");
            generated.push_back(next);
            if (next == tokenizer->GetEOSId()) break;
            const int position = static_cast<int>(tokens.size() + decode_steps);
            logits = yan_lamma::ForwardBatch(
                model, batch_context, builder.Single(next, position));
            token_context.pos = static_cast<std::size_t>(position);
            token_logits = yan_lamma::ForwardToken(model, token_context, next);
            CheckLogits(logits, model.config.vocab_size);
            CheckLogits(token_logits, model.config.vocab_size);
            max_error = std::max(max_error, Compare(logits.data, token_logits.data));
            ++decode_steps;
            Require(batch_context.token_id_.size() == tokens.size() + decode_steps,
                    "decode token history mismatch");
            Require(batch_context.token_id_.back() == next &&
                        batch_context.pos == static_cast<std::size_t>(position),
                    "decode token or position mismatch");
        }
        max_error = std::max(max_error,
            Compare(batch_context.cpu_cache_.key_.data, token_context.cpu_cache_.key_.data));
        max_error = std::max(max_error,
            Compare(batch_context.cpu_cache_.value_.data, token_context.cpu_cache_.value_.data));

        std::cout << "Generated tokens:";
        for (int token : generated) std::cout << ' ' << token;
        std::cout << "\nDecoded output: \"" << Escape(tokenizer->Decode(generated)) << "\"\n";
        std::cout << "PASS: " << manifest.tensor_json.size()
                  << " tensors; logits=[" << model.config.vocab_size
                  << "]; prefill=" << tokens.size()
                  << "; decode=" << decode_steps
                  << "; batch/token max_error=" << max_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
