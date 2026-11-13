#include "include/mini_llama/bench_strict.h"
#include "include/backend.h"
#include "include/context.h"
#include "include/forward.h"
#include "include/gguf_load.h"
#include "include/gguf_tokenizer.h"
#include "include/load.h"
#include "include/sampler.h"
#include "include/tonizer.h"
#include "matumal/matuml_ops.h"
#include "thread_pool/thread_pool.h"

#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace yan_lamma;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
struct CapacityError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Args {
    std::string path, prompt = "hello", tokenizer, quant;
    BackendConfig backend;
    int n = 32, warmup = 2, seed = 1, threads = 0;
    MatulMode mode = MatulMode::Native;
};
int Number(const std::string& text, const std::string& option) {
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < 0)
        throw std::runtime_error("Invalid " + option + " value: " + text);
    return value;
}
Args ParseStrictArgs(int argc, char** argv) {
    if (argc < 3 || std::string(argv[2]).empty() || std::string(argv[2])[0] == '-')
        throw std::runtime_error("缺模型目录 (missing model path)");
    Args args;
    args.path = argv[2];
    bool backend_seen = false;
    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--verbose") continue; // Strict stdout never contains debug dumps.
        if (option != "--backend" && option != "--device" && option != "--prompt" &&
            option != "-p" && option != "--n-predict" && option != "-n" &&
            option != "--warmup" && option != "--seed" && option != "--threads" &&
            option != "-threads" && option != "--tokenizer" && option != "--quant" &&
            option != "--linear-mode") throw std::runtime_error("Unknown option: " + option);
        if (option == "--backend" && backend_seen)
            throw std::runtime_error("Repeated --backend");
        if (++i >= argc) throw std::runtime_error("Missing value for " + option);
        const std::string value = argv[i];
        if (option == "--backend") {
            backend_seen = true;
            if (value != "cpu" && value != "cuda")
                throw std::runtime_error("Invalid --backend value: " + value);
            ParseBackendKind(value, args.backend.kind);
        } else if (option == "--device") {
            args.backend.device_set = true;
            args.backend.device_id = Number(value, option);
        } else if (option == "--prompt" || option == "-p") args.prompt = value;
        else if (option == "--tokenizer") {
            if (value.empty()) throw std::runtime_error("Empty tokenizer path");
            args.tokenizer = value;
        } else if (option == "--quant") {
            if (value != "q4_0") throw std::runtime_error("Invalid --quant value: " + value);
            args.quant = value;
        } else if (option == "--linear-mode") {
            if (value == "native") args.mode = MatulMode::Native;
            else if (value == "threaded") args.mode = MatulMode::Threaded;
            else if (value == "simd") args.mode = MatulMode::SIMD;
            else if (value == "threaded-simd") args.mode = MatulMode::ThreadedSIMD;
            else throw std::runtime_error("Invalid --linear-mode value: " + value);
        } else {
            const int number = Number(value, option);
            if (option == "-n" || option == "--n-predict") args.n = number;
            else if (option == "--warmup") args.warmup = number;
            else if (option == "--seed") args.seed = number;
            else args.threads = number;
        }
    }
    if (!backend_seen) throw std::runtime_error("Missing --backend");
    if (args.n < 1) throw std::runtime_error("n-predict must be at least 1");
    return args;
}
double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct Prepared {
    Model model;
    std::vector<int> tokens;
    double load_ms = 0;
};
Prepared LoadAndPrepare(const Args& args) {
    const auto start = Clock::now();
    try { VailedBackend(args.backend); }
    catch (const std::exception& error) {
        throw std::runtime_error(std::string("Backend setup failed: ") + error.what());
    }
    Prepared prepared;
    std::unique_ptr<Tonizer> tokenizer;
    const fs::path path(args.path);
    if (path.extension() == ".gguf") {
        prepared.model = ModelLoad(args.path);
        if (prepared.model.is_loaded) tokenizer = GetGgufTokenizer(args.path);
    } else {
        prepared.model = ParseModel((path / "model.json").string(), (path / "model.bin").string());
        if (prepared.model.is_loaded) {
            const auto metadata = ParseModelJson((path / "model.json").string());
            if (!args.tokenizer.empty()) tokenizer = std::make_unique<JsonVocabTokenizer>(args.tokenizer);
            else if (metadata.toninzer_json.type == "ASCII") tokenizer = std::make_unique<AsciiTokenizer>();
            else tokenizer = std::make_unique<JsonVocabTokenizer>((path / metadata.toninzer_json.path_name).string());
        }
    }
    auto& model = prepared.model;
    if (!model.is_loaded) throw std::runtime_error("Model load failed: " + model.load_error_);
    if (!tokenizer || tokenizer->GetVocabSize() != model.config.vocab_size)
        throw std::runtime_error("Model/tokenizer vocabulary mismatch");
    prepared.tokens = tokenizer->Encode(args.prompt);
    if (!prepared.tokens.empty() && prepared.tokens.back() == tokenizer->GetEOSId()) prepared.tokens.pop_back();
    if (prepared.tokens.empty()) throw std::runtime_error("Prompt encoded to an empty sequence");
    const size_t prompt_size = prepared.tokens.size();
    if (model.config.max_seq_len <= 0 || prompt_size > static_cast<size_t>(model.config.max_seq_len) ||
        static_cast<size_t>(args.n) > static_cast<size_t>(model.config.max_seq_len) - prompt_size)
        throw CapacityError("prompt tokens + n-predict exceeds context window");
    for (int token : prepared.tokens)
        if (token < 0 || token >= model.config.vocab_size) throw std::runtime_error("Prompt token out of range");
    // Allocate the required prefix without changing mathematical model parameters.
    model.config.max_seq_len = static_cast<int>(prompt_size) + args.n;
    SetThreadNum(args.threads);
    SetDefaultMatulMode(args.mode);
    if (args.quant == "q4_0") ModeloQuanTensorQ40(model);
    if (args.backend.kind == BackendKind::Kgpu) UploadModelWeightToCuda(model, args.backend.device_id);
    prepared.load_ms = Milliseconds(start);
    return prepared;
}
struct Measurement { double prefill_ms = 0, decode_ms = 0, sampling_ms = 0; };
Measurement Measure(const Model& model, Context& context, Sampler& sampler,
                    const SamplerParmas& params, const std::vector<int>& prompt, int n) {
    Measurement result;
    auto prompt_copy = prompt;
    const Batch prefill = Batch::FromTokens(prompt_copy, 0);
    auto start = Clock::now();
    Tensor logits = ForwardBatch(model, context, prefill);
    result.prefill_ms = Milliseconds(start);
    for (int i = 0; i < n; ++i) {
        start = Clock::now();
        const int token = sampler.SamplerO(logits, params);
        result.sampling_ms += Milliseconds(start);
        if (i + 1 == n) break;
        // EOS follows exactly the same path as every other generated token.
        const Batch batch = Batch::Single(token, static_cast<int>(prompt.size()) + i);
        start = Clock::now();
        Tensor next = ForwardBatch(model, context, batch);
        result.decode_ms += Milliseconds(start);
        logits = std::move(next);
    }
    return result;
}
void Warmup(const Prepared& prepared, const Args& args) {
    for (int i = 0; i < args.warmup; ++i) {
        Context context(&prepared.model);
        SamplerParmas params;
        params.seed = args.seed;
        Sampler sampler(params);
        Measure(prepared.model, context, sampler, params, prepared.tokens, args.n);
    }
}
constexpr std::array<QuantType, 4> types{QuantType::kF32, QuantType::kQ80, QuantType::kQ40, QuantType::kQ41};
struct Snapshot {
    size_t uploaded = 0;
    std::array<size_t, 4> numerators{}, denominators{};
};
Snapshot ResetAndSnapshot(Model& model) {
    Snapshot snapshot;
    if (model.cuda_weights) {
        snapshot.uploaded = model.cuda_weights->uploaded_weight_count;
        for (const auto& weight : model.cuda_weights->weights)
            for (size_t i = 0; i < types.size(); ++i)
                if (weight.linear && weight.type == types[i]) ++snapshot.numerators[i];
    }
    for (size_t i = 0; i < types.size(); ++i)
        snapshot.denominators[i] = ModelLinearWeightTypeTotals(model, types[i]);
    ResetModelCudaRuntimeStats(model);
    return snapshot;
}
std::string BuildStrictReport(const Prepared& prepared, const Args& args,
                              const Measurement& measured, const Snapshot& snapshot, size_t peak) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(2);
    const double total = measured.prefill_ms + measured.decode_ms;
    out << "prompt tokens: " << prepared.tokens.size() << '\n'
        << "generated tokens: " << args.n << '\n'
        << "prefill ms: " << measured.prefill_ms << '\n'
        << "decode ms: " << measured.decode_ms << '\n'
        << "total ms: " << total << '\n' << "tokens/s (total): ";
    if (total > 0) out << 1000.0 * args.n / total; else out << "unavailable";
    out << "\ntokens/s (decode): ";
    if (args.n > 1 && measured.decode_ms > 0) out << 1000.0 * (args.n - 1) / measured.decode_ms;
    else out << "unavailable";
    out << '\n';
    const bool built = CudaRuntimeBuilt();
    auto integer = [&](const char* key, size_t value, bool bytes = false) {
        out << key << ": ";
        if (!built) out << "unavailable";
        else { out << value; if (bytes) out << " (" << value << " bytes)"; }
        out << '\n';
    };
    integer("uploaded weights", snapshot.uploaded);
    out << "gpu memory used: ";
    if (built) out << static_cast<double>(peak) / 1000000.0 << " MB\n";
    else out << "unavailable\n";
    const std::array<const char*, 4> names{"f32", "q8_0", "q4_0", "q4_1"};
    for (size_t i = 0; i < types.size(); ++i) {
        out << "cuda " << names[i] << " linear weights: ";
        if (built) out << snapshot.numerators[i] << '/' << snapshot.denominators[i];
        else out << "unavailable";
        out << '\n';
    }
    const auto& model = prepared.model;
    integer("cuda linear calls", ModelCudaLinearCalls(model));
    integer("cuda activation calls", ModelCudaActivationCalls(model));
    integer("cuda attention calls", ModelCudaAttentionCalls(model));
    integer("cpu attention fallback calls", ModelCudaAttentionCpuFallbacks(model));
    integer("cuda kv write bytes", ModelCudaKvCacheWriteBytes(model), true);
    integer("cuda kv read bytes", ModelCudaKvCacheReadBytes(model), true);
    integer("host->device copies", ModelCudaHostToDeviceCopies(model));
    integer("device->host copies", ModelCudaDeviceToHostCopies(model));
    out << std::setprecision(3) << "load ms: " << prepared.load_ms << '\n'
        << "sampling ms: " << measured.sampling_ms << '\n';
    return out.str();
}
void PrintStrictReport(const std::string& report) {
    std::cout << report;
}
// Existing loaders may log to cout. Redirect their output, including unwinding,
// until the complete successful report has been constructed.
class Diagnostics {
 public:
    Diagnostics() : previous_(std::cout.rdbuf(std::cerr.rdbuf())) {}
    ~Diagnostics() { std::cout.rdbuf(previous_); }
 private:
    std::streambuf* previous_;
};
}
int RunBenchStrict(int argc, char** argv) {
    std::string report;
    try {
        Diagnostics diagnostics;
        const auto args = ParseStrictArgs(argc, argv);
        auto allocation = std::make_shared<CudaAllocationStats>();
        CudaAllocationScope scope(allocation);
        auto prepared = LoadAndPrepare(args);
        Warmup(prepared, args);
        const auto start = Clock::now();
        Context context(&prepared.model);
        SamplerParmas params;
        params.seed = args.seed;
        Sampler sampler(params);
        prepared.load_ms += Milliseconds(start);
        const auto snapshot = ResetAndSnapshot(prepared.model);
        const auto measured = Measure(prepared.model, context, sampler, params, prepared.tokens, args.n);
        report = BuildStrictReport(prepared, args, measured, snapshot, allocation->peak_bytes);
    } catch (const CapacityError& error) {
        std::cerr << "bench: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "bench: " << error.what() << '\n';
        return 1;
    }
    PrintStrictReport(report);
    return 0;
}
