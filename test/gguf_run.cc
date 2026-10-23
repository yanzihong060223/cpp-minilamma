// Manual interactive smoke runner: complete chat history is prefilled every turn.
#include "include/batch.h"
#include "include/chatsession.h"
#include "include/context.h"
#include "include/forward.h"
#include "include/gguf_load.h"
#include "include/gguf_tokenizer.h"
#include "include/prompt_build.h"
#include "include/terminal.h"
#include "test/gguf_smoke_utils.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace yan_lamma;
using smoke::CheckLogits;
using smoke::Require;

void Usage(std::ostream& out) {
    out << "usage: gguf_run <model.gguf> [n-predict]\n"
           "  n-predict: positive per-turn token limit, default 16\n"
           "  sampling: greedy (temperature=0, seed=42)\n"
           "  full history is prefilled on every turn; /help lists commands\n"
           "  linear weights: F32, Q8_0, Q4_0, Q4_1; converted by forward\n";
}

void RunTurn(Model& model, const GgufTokenizer& tokenizer, PromptBuild& builder,
             ChatSession& session, Sampler& sampler, Terminal& terminal,
             const std::string& input, int n_predict, int max_context) {
    const char* stage = "prompt build";
    bool assistant_started = false;
    try {
        const auto started = std::chrono::steady_clock::now();
        auto messages = session.messages_;
        messages.push_back({"user", input});
        const auto prompt = builder.Build(messages);

        stage = "tokenization";
        auto tokens = tokenizer.Encode(prompt);
        // Remove only the EOS automatically appended by the existing Encode API.
        if (tokenizer.GetEOSId() >= 0 && !tokens.empty() &&
            tokens.back() == tokenizer.GetEOSId()) {
            tokens.pop_back();
        }
        Require(!tokens.empty(), "prompt encoded to an empty token sequence");
        for (int token : tokens) {
            Require(token >= 0 && token < model.config.vocab_size,
                    "prompt token ID out of range");
        }
        const auto limit = static_cast<std::size_t>(max_context);
        Require(tokens.size() <= limit &&
                    static_cast<std::size_t>(n_predict) <= limit - tokens.size(),
                "prompt + n-predict exceeds model context length; use /clear or a shorter message");

        // max_context is the original model limit, independent of prior allocations.
        model.config.max_seq_len = static_cast<int>(tokens.size() + n_predict);
        Context context(&model);
        Batch batch;
        Tensor logits;
        stage = "prefill";
        std::cerr << "Prompt tokens: " << tokens.size()
                  << "; generation limit: " << n_predict << '\n'
                  << "Prefill 0/" << tokens.size() << std::flush;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            logits = ForwardBatch(model, context, batch.Single(tokens[i], static_cast<int>(i)));
            CheckLogits(logits, model.config.vocab_size);
            std::cerr << "\rPrefill " << i + 1 << '/' << tokens.size() << std::flush;
        }
        std::cerr << '\n';
        Require(context.token_id_ == tokens && context.pos == tokens.size() - 1,
                "prefill history/position mismatch");

        stage = "generation";
        terminal.PrintAssistantPrefix();
        assistant_started = true;
        std::vector<int> generated;
        const char* stop_reason = "token limit";
        for (int step = 0; step < n_predict; ++step) {
            const int next = sampler.SamplerO(logits, session.parma_);
            Require(next >= 0 && next < model.config.vocab_size,
                    "generated token ID out of range");
            if (next == tokenizer.GetEOSId()) {
                stop_reason = "EOS";
                break;
            }
            generated.push_back(next);
            terminal.PrintTokenText(tokenizer.DecodeToken(next));
            terminal.flush();
            if (step + 1 == n_predict) break;

            const int position = static_cast<int>(tokens.size()) + step;
            logits = ForwardBatch(model, context, batch.Single(next, position));
            CheckLogits(logits, model.config.vocab_size);
            Require(context.pos == static_cast<std::size_t>(position) &&
                        context.token_id_.size() == tokens.size() + step + 1 &&
                        context.token_id_.back() == next,
                    "decode history/position mismatch");
        }

        // Prepare both updates before committing, so failed turns leave no user entry.
        messages.push_back({"assistant", tokenizer.Decode(generated)});
        auto history = tokens;
        history.insert(history.end(), generated.begin(), generated.end());
        const float elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - started).count();
        session.messages_.swap(messages);
        session.token_history_.swap(history);
        session.RecordTurn(elapsed, static_cast<int>(tokens.size()),
                           static_cast<int>(generated.size()));
        terminal.NewLine();
        terminal.flush();
        std::cerr << "Turn complete: generated_tokens=" << generated.size()
                  << "; stop=" << stop_reason << "; time=" << elapsed << "s\n";
    } catch (const std::exception& error) {
        if (assistant_started) {
            terminal.NewLine();
            terminal.flush();
        }
        std::cerr << "\nFAIL [" << stage << "]: " << error.what()
                  << "\nTurn not saved; chat history unchanged.\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        Usage(std::cout);
        return 0;
    }
    if (argc < 2 || argc > 3) {
        Usage(std::cerr);
        return 1;
    }

    const char* stage = "arguments";
    try {
        int n_predict = 16;
        if (argc == 3) {
            const std::string value = argv[2];
            std::size_t parsed = 0;
            n_predict = std::stoi(value, &parsed);
            Require(parsed == value.size() && n_predict > 0,
                    "n-predict must be a positive integer");
        }
        const std::string path = argv[1];
        stage = "model load";
        std::cerr << "Loading model: " << path << std::endl;
        auto model = ModelLoad(path);
        Require(model.is_loaded, model.load_error_.empty()
                    ? "ModelLoad failed; see loader diagnostics above" : model.load_error_);
        const int max_context = model.config.max_seq_len;
        Require(max_context > 0, "model context length must be positive");

        stage = "tokenizer load";
        auto tokenizer = GetGgufTokenizer(path);
        Require(tokenizer != nullptr, "GetGgufTokenizer returned null");
        Require(tokenizer->GetVocabSize() == model.config.vocab_size,
                "model/tokenizer vocabulary mismatch");
        stage = "chat template load";
        PromptBuild builder;
        builder.SetChatTemplate(GetChatTemplate(path));
        ChatSession session;
        session.parma_.temperature = 0.0f;
        session.parma_.seed = 42;
        Sampler sampler(session.parma_);
        Terminal terminal;
        terminal.PrintMessage("Chat ready. Type /help for commands; /exit or EOF to quit.");
        stage = "terminal input";
        while (true) {
            terminal.PrintUserPrompt();
            const auto input = terminal.ReadLine();
            if (!std::cin) {
                Require(std::cin.eof(), "failed to read terminal input");
                terminal.NewLine();
                return 0;
            }
            const auto first = input.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            if (input[first] == '/') {
                const auto last = input.find_last_not_of(" \t\r\n");
                const auto command = input.substr(first, last - first + 1);
                if (command == "/exit") return 0;
                if (command == "/help") terminal.PrintHelp();
                else if (command == "/clear") {
                    session.Clear();
                    terminal.PrintMessage("Chat history and stats cleared.");
                } else if (command == "/stats") terminal.PrintStats(session);
                else if (command == "/params" || command == "/parmas") {
                    terminal.PrintParams(session.parma_);
                } else terminal.PrintMessage("Unknown command; type /help.");
                continue;
            }
            RunTurn(model, *tokenizer, builder, session, sampler, terminal,
                    input, n_predict, max_context);
        }
    } catch (const std::exception& error) {
        std::cerr << "\nFAIL [" << stage << "]: " << error.what() << '\n';
        return 1;
    }
}
