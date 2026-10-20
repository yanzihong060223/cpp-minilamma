// Manual fixed-length performance runner. A real GGUF model is supplied by the caller.
#include "include/betch_mark.h"
#include "include/gguf_load.h"
#include "include/gguf_tokenizer.h"
#include "include/prompt_build.h"
#include "test/gguf_smoke_utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using yan_lamma::smoke::Require;

void Usage(std::ostream& out) {
    out << "usage: betch_mark_test <model.gguf> [prompt] [limits] [--verbose]\n"
           "  prompt: default is '你好'\n"
           "  limits: total generated tokens, default 16; 0 measures prefill only\n"
           "  generation is capped by remaining context capacity; EOS does not stop it\n"
           "  sampling: greedy (temperature=0, seed=42)\n"
           "  --verbose: dump logits/KV cache; decode dumps affect decode_ms\n"
           "  linear weights: F32, Q8_0, Q4_0, Q4_1; converted by forward\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        Usage(std::cout);
        return 0;
    }
    const bool verbose = argc > 1 && std::string(argv[argc - 1]) == "--verbose";
    const int positional_argc = argc - (verbose ? 1 : 0);
    if (positional_argc < 2 || positional_argc > 4) {
        Usage(std::cerr);
        return 1;
    }

    const char* stage = "arguments";
    try {
        const std::string path = argv[1];
        const std::string text = positional_argc >= 3 ? argv[2] : "你好";
        int requested_limit = 16;
        if (positional_argc >= 4) {
            const std::string value = argv[3];
            std::size_t parsed = 0;
            requested_limit = std::stoi(value, &parsed);
            Require(parsed == value.size() && requested_limit >= 0,
                    "limits must be a nonnegative integer");
        }

        stage = "model load";
        std::cerr << "Loading model: " << path << std::endl;
        auto model = yan_lamma::ModelLoad(path);
        Require(model.is_loaded, model.load_error_.empty()
                    ? "ModelLoad failed; see loader diagnostics above"
                    : model.load_error_);

        stage = "tokenizer load";
        auto tokenizer = yan_lamma::GetGgufTokenizer(path);
        Require(tokenizer != nullptr, "GetGgufTokenizer returned null");
        Require(model.config.vocab_size == tokenizer->GetVocabSize(),
                "model/tokenizer vocabulary mismatch");

        stage = "prompt build";
        yan_lamma::PromptBuild prompt_builder;
        prompt_builder.SetChatTemplate(yan_lamma::GetChatTemplate(path));
        const auto prompt = prompt_builder.Build({{"user", text}});

        stage = "tokenization";
        auto tokens = tokenizer->Encode(prompt);
        // Remove only the EOS automatically appended by Encode.
        if (tokenizer->GetEOSId() >= 0 && !tokens.empty() &&
            tokens.back() == tokenizer->GetEOSId()) {
            tokens.pop_back();
        }
        Require(!tokens.empty(), "prompt encoded to an empty token sequence");
        for (int token : tokens) {
            Require(token >= 0 && token < model.config.vocab_size,
                    "prompt token ID out of range");
        }
        Require(model.config.max_seq_len > 0, "model context length must be positive");
        Require(tokens.size() <= static_cast<std::size_t>(model.config.max_seq_len),
                "prompt exceeds model context length");
        const int prompt_size = static_cast<int>(tokens.size());
        const int generation_budget = std::min(
            requested_limit, model.config.max_seq_len - prompt_size);
        // Allocate only the KV positions needed by this run.
        model.config.max_seq_len = prompt_size + generation_budget;
        std::cerr << "Prompt tokens: " << prompt_size
                  << "; requested limit: " << requested_limit
                  << "; generation budget: " << generation_budget << '\n';
        if (verbose) {
            std::cerr << "Verbose decode dumps are included in decode_ms.\n";
        }

        stage = "benchmark";
        std::cerr << "Running benchmark..." << std::endl;
        auto result = yan_lamma::RunBetchMrk(model, 42, tokens, verbose, requested_limit);
        Require(result.n_prompt_tokens == prompt_size, "prompt token count mismatch");
        Require(result.n_generate_tokens == generation_budget,
                "generated token count mismatch");
        Require(result.n_decode_tokens == std::max(0, generation_budget - 1),
                "decode token count mismatch");
        Require(std::isfinite(result.prefile_ms) && result.prefile_ms >= 0.0 &&
                    std::isfinite(result.decode_ms) && result.decode_ms >= 0.0,
                "benchmark durations must be finite and nonnegative");

        const double prefill_tokens_per_sec = result.prefile_ms > 0.0
            ? result.n_prompt_tokens * 1000.0 / result.prefile_ms : 0.0;
        std::cout << std::fixed << std::setprecision(3)
                  << "requested_limit=" << requested_limit << '\n'
                  << "n_prompt_tokens=" << result.n_prompt_tokens << '\n'
                  << "n_generate_tokens=" << result.n_generate_tokens << '\n'
                  << "n_decode_tokens=" << result.n_decode_tokens << '\n'
                  << "prefile_ms=" << result.prefile_ms << '\n'
                  << "decode_ms=" << result.decode_ms << '\n'
                  << "prefill_tokens_per_sec=" << prefill_tokens_per_sec << '\n'
                  << "total_tokens_per_sec=" << result.TokenPerSec() << '\n'
                  << "decode_tokens_per_sec=" << result.DeTokenPerSrc() << '\n';
        std::cerr << "PASS: benchmark completed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nFAIL [" << stage << "]: " << error.what() << '\n';
        return 1;
    }
}
