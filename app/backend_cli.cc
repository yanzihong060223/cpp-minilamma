#include "app/backend_cli.h"
#include "include/backend.h"
#include "include/betch_mark.h"
#include "include/forward.h"
#include "include/gguf_load.h"
#include "include/gguf_tokenizer.h"
#include "include/load.h"
#include "include/sampler.h"
#include "include/tonizer.h"
#include "matumal/matuml_ops.h"
#include "thread_pool/thread_pool.h"

#include <atomic>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using namespace yan_lamma;
namespace fs = std::filesystem;
int Nonnegative(const std::string& value, const std::string& option) {
    int number = 0;
    auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number < 0)
        throw std::runtime_error("Invalid " + option + " value: " + value);
    return number;
}
std::string TraceId() {
    static std::atomic<unsigned long> serial{0};
    return std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
           "-" + std::to_string(++serial);
}
void DeviceInfo(int device) {
    const auto info = GetCudaInfo(device);
    std::cout << "cuda runtime: built\ncuda device count: " << DeviceCount()
              << "\ndevice name: " << info.name
              << "\ncompute capability: " << info.compute_major << '.' << info.compute_minor
              << "\ntotal memory: " << info.total_bytes
              << "\ndriver version: " << info.driver_version
              << "\nruntime version: " << info.runtime_version << '\n';
}
}

bool UsesBackendCli(int argc, char** argv) {
    if (argc >= 2 && (std::string(argv[1]) == "generate" || std::string(argv[1]) == "inspect")) return true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend" || arg == "--device" || arg == "--cuda-info" ||
            arg == "--seed" || arg == "-p" || arg == "-n") return true;
    }
    return false;
}

int RunBackendCli(int argc, char** argv) {
    const std::string trace_id = TraceId();
    Model model;
    BackendConfig backend;
    bool ready = false;
    try {
        std::string command = argc > 1 ? argv[1] : "";
        bool info = command == "--cuda-info";
        std::string path;
        int begin = 2;
        if (!info) {
            if (command != "generate" && command != "bench" && command != "inspect")
                throw std::runtime_error("Expected generate, bench, inspect or --cuda-info");
            if (argc < 3) throw std::runtime_error("Missing model path");
            path = argv[2];
            begin = 3;
        }
        std::string prompt = "hello", tokenizer_path;
        int count = 4, seed = 1, threads = 0;
        bool verbose = false;
        MatulMode linear_mode = MatulMode::Native;
        for (int i = begin; i < argc; ++i) {
            std::string option = argv[i];
            if (option == "--verbose") { verbose = true; continue; }
            if (option == "--cuda-info") { info = true; continue; }
            if (option != "--backend" && option != "--device" && option != "--prompt" &&
                option != "-p" && option != "--n-predict" && option != "-n" &&
                option != "--seed" && option != "--threads" && option != "-threads" &&
                option != "--linear-mode" && option != "--tokenizer")
                throw std::runtime_error("Unknown option: " + option);
            if (++i >= argc) throw std::runtime_error("Missing value for " + option);
            std::string value = argv[i];
            if (option == "--backend") {
                if (!ParseBackendKind(value, backend.kind))
                    throw std::runtime_error("Invalid --backend value: " + value);
            } else if (option == "--device") {
                backend.device_set = true;
                backend.device_id = Nonnegative(value, "--device");
            } else if (option == "--prompt" || option == "-p") prompt = value;
            else if (option == "--tokenizer") tokenizer_path = value;
            else if (option == "--n-predict" || option == "-n") count = Nonnegative(value, "--n-predict");
            else if (option == "--seed") seed = Nonnegative(value, "--seed");
            else if (option == "--linear-mode") {
                if (value == "native") linear_mode = MatulMode::Native;
                else if (value == "threaded") linear_mode = MatulMode::Threaded;
                else if (value == "simd") linear_mode = MatulMode::SIMD;
                else if (value == "threaded-simd") linear_mode = MatulMode::ThreadedSIMD;
                else throw std::runtime_error("Unknown linear mode: " + value);
            } else threads = Nonnegative(value, "--threads");
        }
        if (info) backend.kind = BackendKind::Kgpu;
        try {
            VailedBackend(backend);
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("Backend setup failed: ") + error.what());
        }
        ready = true;
        if (info) {
            DeviceInfo(backend.device_id);
        } else {
            fs::path location(path);
            std::unique_ptr<Tonizer> tokenizer;
            if (location.extension() == ".gguf") {
                model = ModelLoad(path);
                if (model.is_loaded && command != "inspect") tokenizer = GetGgufTokenizer(path);
                if (model.is_loaded) std::cout << "GGUF model loaded successfully\n";
            } else {
                model = ParseModel((location / "model.json").string(), (location / "model.bin").string());
                if (model.is_loaded && command != "inspect") {
                    const auto metadata = ParseModelJson((location / "model.json").string());
                    if (!tokenizer_path.empty()) tokenizer = std::make_unique<JsonVocabTokenizer>(tokenizer_path);
                    else if (metadata.toninzer_json.type == "ASCII") tokenizer = std::make_unique<AsciiTokenizer>();
                    else tokenizer = std::make_unique<JsonVocabTokenizer>((location / metadata.toninzer_json.path_name).string());
                }
            }
            if (!model.is_loaded) throw std::runtime_error("Model load failed: " + model.load_error_);
            SetThreadNum(threads);
            SetDefaultMatulMode(linear_mode);
            std::vector<int> tokens;
            if (command != "inspect") {
                if (!tokenizer || tokenizer->GetVocabSize() != model.config.vocab_size)
                    throw std::runtime_error("Model/tokenizer vocabulary mismatch");
                tokens = tokenizer->Encode(prompt);
                if (!tokens.empty() && tokens.back() == tokenizer->GetEOSId()) tokens.pop_back();
                if (tokens.empty()) throw std::runtime_error("Prompt encoded to an empty sequence");
                const auto capacity = static_cast<size_t>(model.config.max_seq_len);
                if (model.config.max_seq_len <= 0 || tokens.size() > capacity ||
                    static_cast<size_t>(count) > capacity - tokens.size())
                    throw std::runtime_error("prompt + n-predict exceeds context window");
                // Allocate only the required cache prefix; keep the loaded model's numerical parameters.
                model.config.max_seq_len = static_cast<int>(tokens.size()) + count;
            }
            if (backend.kind == BackendKind::Kgpu) UploadModelWeightToCuda(model, backend.device_id);
            ResetModelCudaRuntimeStats(model);
            if (command == "bench") {
                const auto result = RunBetchMrk(model, seed, tokens, verbose, count);
                auto metrics = result;
                std::cout << "Benchmark: " << path << '\n' << std::fixed << std::setprecision(3)
                          << "tokens/s (total): " << metrics.TokenPerSec() << '\n'
                          << "tokens/s(total):" << metrics.TokenPerSec() << '\n'
                          << "n_prompt_tokens:" << metrics.n_prompt_tokens << '\n'
                          << "n_generate_tokens:" << metrics.n_generate_tokens << '\n';
            } else if (command == "generate") {
                Context context(&model);
                ResetModelCudaRuntimeStats(model);
                Tensor logits = ForwardBatch(model, context, Batch::FromTokens(tokens, 0));
                SamplerParmas params;
                params.seed = seed;
                Sampler sampler(params);
                for (int i = 0; i < count; ++i) {
                    const int token = sampler.SamplerO(logits, params);
                    std::cout << tokenizer->DecodeToken(token);
                    if (i + 1 < count) logits = ForwardBatch(model, context,
                        Batch::Single(token, static_cast<int>(tokens.size()) + i));
                }
                std::cout << '\n';
            }
            PrintBackendSummary(backend, model, std::cout);
        }
        std::cout << "trace summary: trace_id=" << trace_id << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        // Successful setup guarantees the device query in the summary was valid.
        if (ready && model.is_loaded) PrintBackendSummary(backend, model, std::cout);
        std::cerr << "trace summary: trace_id=" << trace_id << " status=error\n";
        return 1;
    }
}
