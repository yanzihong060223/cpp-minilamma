#include "include/betch_mark.h"
#include "include/forward.h"
#include "include/load.h"
#include "include/sampler.h"
#include "include/tonizer.h"
#include "matumal/matuml_ops.h"
#include "thread_pool/thread_pool.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

int main(int argc, char** argv) {
    using namespace yan_lamma;
    namespace fs = std::filesystem;

    const char* usage =
        "usage: mini-llama bench <model-dir>\n"
        "  [--tokenizer vocab.json] [--prompt hello]\n"
        "  [--n-predict 4] [--threads 0] [--verbose]\n"
        "  [--linear-mode native|threaded|simd|threaded-simd]\n"
        "  mini-llama --config model.json --model model.bin\n"
        "    --dump-logits <fresh-dir> [--prompt hello] [--n-predict 16]\n"
        "    [--threads 0] [--linear-mode native|threaded|simd|threaded-simd]\n"
        "  --threads 0 selects the automatic thread count; -threads is an alias\n"
        "  generation is capped by context capacity; EOS does not stop it\n"
        "  --verbose dumps logits/KV cache; decode dumps affect decode_ms\n";

    if ((argc == 2 && std::string(argv[1]) == "--help") ||
        (argc == 3 && std::string(argv[1]) == "bench" &&
                      std::string(argv[2]) == "--help")) {
        std::cout << usage;
        return 0;
    }
    const bool is_bench = argc >= 2 && std::string(argv[1]) == "bench";
    if (argc < 2 || (is_bench && argc < 3) ||
        (!is_bench && std::string(argv[1]).rfind("--", 0) != 0)) {
        std::cerr << usage;
        return 1;
    }

    try {
        const fs::path model_dir = is_bench ? argv[2] : "models/tiny";
        std::string config_path = (model_dir / "model.json").string();
        std::string weight_path = (model_dir / "model.bin").string();
        fs::path dump_dir;
        std::string tokenizer_path;
        std::string prompt = "hello";
        std::string mode_name = "native";
        MatulMode linear_mode = MatulMode::Native;
        int n_predict = is_bench ? 4 : 16;
        int threads = 0;
        bool verbose = false;

        for (int i = is_bench ? 3 : 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << usage;
                return 0;
            }
            if (option == "--verbose") {
                verbose = true;
                continue;
            }
            if (option != "--tokenizer" && option != "--prompt" &&
                option != "--n-predict" && option != "--threads" &&
                option != "-threads" && option != "--linear-mode" &&
                (is_bench || (option != "--config" && option != "--model" &&
                              option != "--dump-logits"))) {
                throw std::runtime_error("Unknown option: " + option);
            }
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + option);
            }

            const std::string value = argv[i];
            if (option == "--linear-mode") {
                mode_name = value;
                if (value == "native") linear_mode = MatulMode::Native;
                else if (value == "threaded") linear_mode = MatulMode::Threaded;
                else if (value == "simd") linear_mode = MatulMode::SIMD;
                else if (value == "threaded-simd") linear_mode = MatulMode::ThreadedSIMD;
                else throw std::runtime_error("Unknown linear mode: " + value);
            } else if (option == "--config" || option == "--model" ||
                       option == "--dump-logits") {
                if (value.empty()) {
                    throw std::runtime_error("Empty path for " + option);
                }
                if (option == "--config") config_path = value;
                else if (option == "--model") weight_path = value;
                else dump_dir = value;
            } else if (option == "--tokenizer") {
                if (!is_bench) {
                    throw std::runtime_error("Logits dump uses ASCII prompt tokens and manifest BOS");
                }
                if (value.empty()) {
                    throw std::runtime_error("Tokenizer path must not be empty");
                }
                tokenizer_path = value;
            } else if (option == "--prompt") {
                prompt = value;
            } else {
                int number = 0;
                const auto parsed = std::from_chars(
                    value.data(), value.data() + value.size(), number);
                if (parsed.ec != std::errc{} ||
                    parsed.ptr != value.data() + value.size() || number < 0) {
                    throw std::runtime_error(
                        option + " requires a nonnegative integer");
                }
                if (option == "--n-predict") {
                    n_predict = number;
                } else {
                    threads = number;
                }
            }
        }

        if (!is_bench && dump_dir.empty()) {
            throw std::runtime_error("--dump-logits is required for the F32 golden entry");
        }
        const auto metadata = ParseModelJson(config_path);
        auto model = ParseModel(config_path, weight_path);
        if (!model.is_loaded) {
            throw std::runtime_error(model.load_error_.empty()
                ? "Model load failed" : model.load_error_);
        }

        SetThreadNum(threads);
        SetDefaultMatulMode(linear_mode);

        if (!is_bench) {
            const int bos_id = metadata.toninzer_json.bos_id;
            if (bos_id < 0 || bos_id >= model.config.vocab_size) {
                throw std::runtime_error("BOS token ID out of range");
            }
            std::vector<int> tokens{bos_id};
            for (unsigned char byte : prompt) {
                if (byte > 127) {
                    throw std::runtime_error("Golden prompt must contain only ASCII characters");
                }
                if (byte >= model.config.vocab_size) {
                    throw std::runtime_error("Prompt token ID out of range");
                }
                tokens.push_back(byte);
            }
            const int max_seq_len = model.config.max_seq_len;
            if (tokens.size() > static_cast<size_t>(max_seq_len)) {
                throw std::runtime_error("Prompt exceeds model context length");
            }
            const int prompt_size = static_cast<int>(tokens.size());
            const int generated_count = std::min(n_predict, max_seq_len - prompt_size);
            const int expected_steps = prompt_size + std::max(generated_count - 1, 0);

            fs::create_directories(dump_dir);
            for (const auto& entry : fs::directory_iterator(dump_dir)) {
                const std::string name = entry.path().filename().string();
                if ((name.rfind("logits_step", 0) == 0 &&
                     entry.path().extension() == ".bin") ||
                    name == "generation_tokens.txt" || name == "cpp_run.json") {
                    throw std::runtime_error("Output already contains C++ golden artifacts: " + name);
                }
            }

            static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                          "Golden logits require IEEE-754 float32");
            int step = 0;
            const auto write_logits = [&](const Tensor& logits) {
                if (logits.shape != std::vector<int>{model.config.vocab_size} ||
                    logits.data.size() != static_cast<size_t>(model.config.vocab_size)) {
                    throw std::runtime_error("Golden logits shape mismatch");
                }
                const auto path = dump_dir / ("logits_step" + std::to_string(step) + ".bin");
                std::ofstream output(path, std::ios::binary);
                if (!output) throw std::runtime_error("Cannot create " + path.string());
                for (float value : logits.data) {
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("Golden logits contain NaN/Inf at step " + std::to_string(step));
                    }
                    uint32_t bits;
                    std::memcpy(&bits, &value, sizeof(bits));
                    const uint8_t bytes[4] = {
                        static_cast<uint8_t>(bits), static_cast<uint8_t>(bits >> 8),
                        static_cast<uint8_t>(bits >> 16), static_cast<uint8_t>(bits >> 24)};
                    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
                }
                output.close();
                if (!output) throw std::runtime_error("Failed to write " + path.string());
                ++step;
            };

            Context context(&model);
            Tensor logits;
            for (int pos = 0; pos < prompt_size; ++pos) {
                logits = ForwardBatch(model, context, Batch::Single(tokens[pos], pos));
                write_logits(logits);
            }
            SamplerParmas parameters;
            parameters.temperature = 0.0f;
            parameters.seed = 42;
            Sampler sampler(parameters);
            std::vector<int> generated;
            for (int i = 0; i < generated_count; ++i) {
                const int next = sampler.SamplerO(logits, parameters);
                generated.push_back(next);
                if (i + 1 < generated_count) {
                    logits = ForwardBatch(model, context,
                        Batch::Single(next, prompt_size + i));
                    write_logits(logits);
                }
            }
            if (step != expected_steps) {
                throw std::runtime_error("Golden step count mismatch");
            }

            const auto tokens_path = dump_dir / "generation_tokens.txt";
            std::ofstream token_output(tokens_path);
            for (int token : generated) token_output << token << '\n';
            token_output.close();
            if (!token_output) throw std::runtime_error("Failed to write " + tokens_path.string());

            // The metadata is written last and marks a complete export.
            const auto run_path = dump_dir / "cpp_run.json";
            std::ofstream run_output(run_path);
            run_output << "{\n  \"format_version\": 1,\n  \"vocab_size\": "
                       << model.config.vocab_size << ",\n  \"bos_id\": " << bos_id
                       << ",\n  \"prompt_tokens\": [";
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i != 0) run_output << ", ";
                run_output << tokens[i];
            }
            run_output << "],\n  \"requested_generation\": " << n_predict
                       << ",\n  \"max_seq_len\": " << max_seq_len
                       << ",\n  \"generated_count\": " << generated_count
                       << ",\n  \"step_count\": " << step
                       << ",\n  \"linear_mode\": \"" << mode_name
                       << "\",\n  \"threads\": " << GetThreadNum() << "\n}\n";
            run_output.close();
            if (!run_output) throw std::runtime_error("Failed to write " + run_path.string());
            std::cout << "linear_mode:" << mode_name << "\nthreads:" << GetThreadNum() << '\n';
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            std::cout << "compiled_simd:NEON\n";
#elif defined(__AVX2__) && defined(__FMA__)
            std::cout << "compiled_simd:AVX2/FMA\n";
#else
            std::cout << "compiled_simd:scalar fallback\n";
#endif
            std::cout << "logits_steps:" << step << "\ngenerated_tokens:"
                      << generated.size() << '\n';
            return 0;
        }

        // 显式路径相对于工作目录，模型元数据路径相对于模型目录。
        const auto& tokenizer_info = metadata.toninzer_json;
        if (tokenizer_path.empty() && tokenizer_info.type == "JsonVocab") {
            tokenizer_path = (model_dir / tokenizer_info.path_name).string();
        }
        auto tokenizer = GetTonizer(tokenizer_path);
        if (tokenizer->GetVocabSize() != model.config.vocab_size ||
            tokenizer->GetBOSId() != tokenizer_info.bos_id ||
            tokenizer->GetEOSId() != tokenizer_info.eos_id ||
            tokenizer->GetUNKId() != tokenizer_info.unk_id) {
            throw std::runtime_error("Model/tokenizer vocabulary mismatch");
        }

        auto tokens = tokenizer->Encode(prompt);
        // 只去掉 Encode 自动追加的 EOS。
        if (!tokens.empty() && tokens.back() == tokenizer->GetEOSId()) {
            tokens.pop_back();
        }
        if (tokens.empty()) {
            throw std::runtime_error("Prompt encoded to an empty sequence");
        }
        if (model.config.max_seq_len <= 0 ||
            tokens.size() > static_cast<std::size_t>(model.config.max_seq_len)) {
            throw std::runtime_error("Prompt exceeds model context length");
        }
        for (int token : tokens) {
            if (token < 0 || token >= model.config.vocab_size) {
                throw std::runtime_error("Prompt token ID out of range");
            }
        }

        SetThreadNum(threads);
        const int generation_budget = std::min(
            n_predict,
            model.config.max_seq_len - static_cast<int>(tokens.size()));
        // 只分配本次运行需要的 KV 位置，与现有 GGUF benchmark 一致。
        model.config.max_seq_len =
            static_cast<int>(tokens.size()) + generation_budget;
        auto result = RunBetchMrk(model, 42, tokens, verbose, n_predict);
        if (result.n_prompt_tokens != static_cast<int>(tokens.size()) ||
            result.n_generate_tokens != generation_budget ||
            result.n_decode_tokens != std::max(0, generation_budget - 1)) {
            throw std::runtime_error("Benchmark token count mismatch");
        }
        if (!std::isfinite(result.prefile_ms) || result.prefile_ms < 0 ||
            !std::isfinite(result.decode_ms) || result.decode_ms < 0) {
            throw std::runtime_error("Invalid benchmark timing");
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "threads:" << GetThreadNum() << '\n'
                  << "linear_mode:";
        switch (DefaultMatulMode()) {
            case MatulMode::Native: std::cout << "Native\n"; break;
            case MatulMode::Threaded: std::cout << "Threaded\n"; break;
            case MatulMode::SIMD: std::cout << "SIMD\n"; break;
            case MatulMode::ThreadedSIMD: std::cout << "ThreadedSIMD\n"; break;
        }
        // 编译后端与实际模式分开报告；未显式选择模式时保持 Native。
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        std::cout << "compiled_simd:NEON\n";
#elif defined(__AVX2__) && defined(__FMA__)
        std::cout << "compiled_simd:AVX2/FMA\n";
#else
        std::cout << "compiled_simd:scalar fallback\n";
#endif
        std::cout << "n_prompt_tokens:" << result.n_prompt_tokens << '\n'
                  << "n_generate_tokens:" << result.n_generate_tokens << '\n'
                  << "n_decode_tokens:" << result.n_decode_tokens << '\n'
                  << "prefill_ms:" << result.prefile_ms << '\n'
                  << "decode_ms:" << result.decode_ms << '\n'
                  << "tokens/s(total):" << result.TokenPerSec() << '\n'
                  << "tokens/s(Decode):" << result.DeTokenPerSrc() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << (is_bench ? "bench: " : "golden: ") << error.what() << '\n';
        return 1;
    }
}
