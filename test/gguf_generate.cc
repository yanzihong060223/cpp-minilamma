// Manual end-to-end runner. A real GGUF model is supplied by the caller.
#include "include/batch.h"
#include "include/context.h"
#include "include/forward.h"
#include "include/gguf_load.h"
#include "include/gguf_tokenizer.h"
#include "include/prompt_build.h"
#include "include/sampler.h"
#include "test/gguf_smoke_utils.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using yan_lamma::smoke::CheckLogits;
using yan_lamma::smoke::Require;

void Usage(std::ostream& out) {
    out << "usage: gguf_generate <model.gguf> [prompt] [n-predict]\n"
           "  prompt: default is '你好'\n"
           "  n-predict: default is 16; 0 checks prefill only\n"
           "  sampling: greedy (temperature=0, seed=42)\n"
           "  linear weights: F32, Q8_0, Q4_0, Q4_1; converted by forward\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        Usage(std::cout);
        return 0;
    }
    if (argc < 2 || argc > 4) {
        Usage(std::cerr);
        return 1;
    }

    const char* stage = "arguments";
    try {
        const std::string path = argv[1];
        const std::string text = argc >= 3 ? argv[2] : "你好";
        int n_predict = 16;
        if (argc >= 4) {
            const std::string value = argv[3];
            std::size_t parsed = 0;
            n_predict = std::stoi(value, &parsed);
            Require(parsed == value.size() && n_predict >= 0,
                    "n-predict must be a nonnegative integer");
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
        std::cerr << "Prompt:\n" << prompt << std::endl;

        stage = "tokenization";
        auto tokens = tokenizer->Encode(prompt);
        // Encode appends EOS whenever configured. The assistant prefix must
        // remain open for generation; remove only that appended token.
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
        const auto max_context = static_cast<std::size_t>(model.config.max_seq_len);
        Require(tokens.size() <= max_context &&
                    static_cast<std::size_t>(n_predict) <= max_context - tokens.size(),
                "prompt + n-predict exceeds model context length");
        // Allocate only the KV positions needed by this single test run.
        model.config.max_seq_len = static_cast<int>(tokens.size() + n_predict);
        std::cerr << "Prompt tokens: " << tokens.size()
                  << "; generation limit: " << n_predict << std::endl;

        stage = "prefill";
        std::cerr << "Prefill..." << std::endl;
        yan_lamma::Context context(&model);
        yan_lamma::Batch batch;
        auto logits = yan_lamma::ForwardBatch(model, context, batch.FromTokens(tokens, 0));
        CheckLogits(logits, model.config.vocab_size);
        Require(context.token_id_ == tokens && context.pos == tokens.size() - 1,
                "prefill history/position mismatch");

        yan_lamma::SamplerParmas params;
        params.temperature = 0.0f;
        params.seed = 42;
        yan_lamma::Sampler sampler(params);
        std::vector<int> generated;
        const char* stop_reason = "token limit";
        stage = "generation";
        std::cerr << "Generated text:\n" << std::flush;
        for (int step = 0; step < n_predict; ++step) {
            const int next = sampler.SamplerO(logits, params);
            Require(next >= 0 && next < model.config.vocab_size,
                    "generated token ID out of range");
            if (next == tokenizer->GetEOSId()) {
                stop_reason = "EOS";
                break;
            }
            generated.push_back(next);
            // Write raw bytes: one BPE token may be only part of a UTF-8 character.
            std::cout << tokenizer->DecodeToken(next) << std::flush;
            if (step + 1 == n_predict) break;

            const int position = static_cast<int>(tokens.size()) + step;
            logits = yan_lamma::ForwardBatch(model, context, batch.Single(next, position));
            CheckLogits(logits, model.config.vocab_size);
            Require(context.pos == static_cast<std::size_t>(position) &&
                        context.token_id_.size() == tokens.size() + step + 1 &&
                        context.token_id_.back() == next,
                    "decode history/position mismatch");
        }
        std::cout << '\n' << std::flush;
        std::cerr << "PASS: prompt_tokens=" << tokens.size()
                  << "; generated_tokens=" << generated.size()
                  << "; stop=" << (n_predict == 0 ? "prefill only" : stop_reason) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nFAIL [" << stage << "]: " << error.what() << '\n';
        return 1;
    }
}
