#pragma once
#include "tests/test_utils.h"
#include "include/backend.h"
#include "include/cuda_attention.h"
#include "include/cuda_matmul.h"
#include "include/cuda_ops.h"
#include "include/cuda_quant.h"
#include "include/forward.h"
#include "include/load.h"
#include "include/quant.h"
#include "src/internal/forward.h"
#include "src/internal/cuda_stats.h"
#include <functional>
#include <initializer_list>
#include <iostream>
#include <utility>

namespace yan_lamma::test {
using Case = std::pair<const char*, std::function<void()>>;
inline int Run(int argc, char** argv, std::initializer_list<Case> cases) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    try {
        Require(argc <= 2, "expected at most one mode");
        bool valid = mode == "all";
        for (const auto& entry : cases) valid = valid || mode == entry.first;
        Require(valid, "unknown mode: " + mode);
        if (CudaRuntimeBuilt()) {
            Require(DeviceCount() > 0, "CUDA build requires an available device");
            SetCudaService(0);
        }
        for (const auto& entry : cases) {
            if (mode != "all" && mode != entry.first) continue;
            entry.second();
            std::cout << "PASS " << entry.first << (CudaRuntimeBuilt() ? " CUDA" : " CPU branch") << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << mode << ": " << error.what() << '\n';
        return 1;
    }
}
inline Tensor Values(std::vector<int> shape, float phase = 0.0f) {
    Tensor t(shape);
    for (size_t i = 0; i < t.data.size(); ++i)
        t.data[i] = std::sin(float(i) * 0.19f + phase) * 0.3f;
    return t;
}
inline Model Tiny() {
    Model model = ParseModel("models/tiny/model.json", "models/tiny/model.bin");
    Require(model.is_loaded, "tiny load: " + model.load_error_);
    Require(model.config.vocab_size == 131 && model.config.hidden_dim == 86 &&
            model.config.dim == 32 && model.config.layers == 2 && model.config.n_heads == 4 &&
            model.config.n_kv_heads == 4 && model.config.head_dim == 8, "unexpected tiny parameters");
    return model;
}
template<class F> void EachLinear(Model& model, F operation) {
    for (size_t i = 0; i < model.n_layers_.size(); ++i) {
        auto& layer = model.n_layers_[i];
        const std::string prefix = "blk." + std::to_string(i) + ".";
        operation(prefix + "wq", layer.wq); operation(prefix + "wk", layer.wk);
        operation(prefix + "wv", layer.wv); operation(prefix + "wo", layer.wo);
        operation(prefix + "w_gate", layer.w_gate); operation(prefix + "w_up", layer.w_up);
        operation(prefix + "w_down", layer.w_down);
    }
    operation("lm_head", model.lm_head_);
}
inline void Quantize(Model& model, QuantType type) {
    if (type == QuantType::kF32) return;
    if (type == QuantType::kQ80) ModeloQuanTensorQ80(model);
    else if (type == QuantType::kQ40) ModeloQuanTensorQ40(model);
    else {
        EachLinear(model, [](const std::string&, QuanTensor& q) {
            const Tensor values = ToTensor(q);
            q.kQ41_data = TensorToQ41(values);
            q.kF32_data.clear();
            q.type = QuantType::kQ41;
        });
    }
}
inline void RequireCpuStats(const Model& model) {
    Require(!model.cuda_weights && ModelCudaLinearCalls(model) == 0 &&
        ModelCudaActivationCalls(model) == 0 && ModelCudaAttentionCalls(model) == 0 &&
        ModelCudaAttentionCpuFallbacks(model) == 0 && ModelCudaLinearCpuFallbacks(model) == 0 &&
        ModelCudaHostToDeviceCopies(model) == 0 && ModelCudaDeviceToHostCopies(model) == 0 &&
        ModelCudaHostToDeviceBytes(model) == 0 && ModelCudaDeviceToHostBytes(model) == 0,
        "CPU model must have zero CUDA statistics");
}
inline void ForwardCase(QuantType type, bool buffer = false) {
    auto cpu = Tiny();
    auto gpu = Tiny();
    Quantize(cpu, type); Quantize(gpu, type);
    Context cpu_context(&cpu);
    if (!CudaRuntimeBuilt()) {
        const auto logits = ForwardBatch(cpu, cpu_context, Batch::Single(1, 0));
        Require(logits.shape == std::vector<int>{131}, "CPU logits shape");
        RequireCloseTensor(logits, logits, "CPU finite logits");
        ResetModelCudaRuntimeStats(cpu);
        RequireCpuStats(cpu);
        RequireError([&] { UploadModelWeightToCuda(gpu, 0); }, "not built");
        return;
    }
    UploadModelWeightToCuda(gpu, 0);
    Context gpu_context(&gpu);
    const size_t uploaded = gpu.cuda_weights->uploaded_weight_count;
    Require(uploaded == 21, "tiny must upload 15 linears and 6 non-linear tensors");
    size_t linear_count = 0;
    for (const auto& weight : gpu.cuda_weights->weights)
        if (weight.linear) { ++linear_count; Require(weight.type == type, "uploaded linear type"); }
    Require(linear_count == 15, "15 uploaded linears");
    const std::vector<int> tokens = buffer ? std::vector<int>{1} : std::vector<int>{128, 'h', 'e', 'l', 'l', 'o', 1};
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        ResetModelCudaRuntimeStats(gpu);
        const auto batch = Batch::Single(tokens[pos], int(pos));
        Tensor expected = ForwardBatch(cpu, cpu_context, batch);
        Tensor actual = ForwardBatch(gpu, gpu_context, batch);
        RequireCloseTensor(actual, expected, "final logits pos=" + std::to_string(pos));
        Require(ModelCudaLinearCalls(gpu) == 15, "linear calls=15");
        Require(ModelCudaActivationCalls(gpu) == 15, "activation calls=15");
        Require(ModelCudaAttentionCalls(gpu) == 2, "attention calls=2");
        Require(ModelCudaAttentionCpuFallbacks(gpu) == 0 && ModelCudaLinearCpuFallbacks(gpu) == 0, "no fallback");
        Require(ModelCudaHostToDeviceCopies(gpu) == 0 && ModelCudaHostToDeviceBytes(gpu) == 0, "H2D copies/bytes=0");
        Require(ModelCudaDeviceToHostCopies(gpu) == 1 && ModelCudaDeviceToHostBytes(gpu) == 524, "one 524-byte logits download");
        const size_t kv_bytes = 2 * size_t(cpu.config.dim) * sizeof(float) * 2;
        Require(ModelCudaKvCacheWriteBytes(gpu) == kv_bytes, "KV writes");
        Require(ModelCudaKvCacheReadBytes(gpu) == kv_bytes * (pos + 1), "KV reads");
    }
    BackendConfig config; config.kind = BackendKind::Kgpu;
    PrintBackendSummary(config, gpu, std::cout);
    ResetModelCudaRuntimeStats(gpu);
    Require(ModelCudaLinearCalls(gpu) == 0 && ModelCudaActivationCalls(gpu) == 0 &&
        ModelCudaAttentionCalls(gpu) == 0 && ModelCudaAttentionCpuFallbacks(gpu) == 0 &&
        ModelCudaLinearCpuFallbacks(gpu) == 0 && ModelCudaHostToDeviceCopies(gpu) == 0 &&
        ModelCudaDeviceToHostCopies(gpu) == 0 && ModelCudaHostToDeviceBytes(gpu) == 0 &&
        ModelCudaDeviceToHostBytes(gpu) == 0 && ModelCudaKvCacheWriteBytes(gpu) == 0 &&
        ModelCudaKvCacheReadBytes(gpu) == 0, "reset all runtime statistics");
    Require(gpu.cuda_weights->uploaded_weight_count == uploaded, "reset preserves upload statistics");
}
}
