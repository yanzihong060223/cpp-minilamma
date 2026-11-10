#include "tests/cuda_test_support.h"
#include <algorithm>
using namespace yan_lamma;
using namespace yan_lamma::test;
void LayerLinears() {
    for (QuantType type : {QuantType::kF32, QuantType::kQ80, QuantType::kQ40, QuantType::kQ41}) {
        auto model = Tiny(); Quantize(model, type);
        if (CudaRuntimeBuilt()) UploadModelWeightToCuda(model, 0);
        EachLinear(model, [&](const std::string& name, QuanTensor& weight) {
            const Tensor x = Values({weight.shape[1]}, 0.5f);
            const Tensor expected = Linear(x, weight);
            if (!CudaRuntimeBuilt()) {
                RequireCloseTensor(expected, expected, name + " CPU");
                return;
            }
            auto dx = TensorHostToDevice(x);
            ResetModelCudaRuntimeStats(model);
            RequireCloseTensor(ForwardLinearDevice(model, dx, name, weight).Download(), expected, name);
            Require(ModelCudaLinearCalls(model) == 1 && ModelCudaLinearCpuFallbacks(model) == 0, name + " dispatch counters");
        });
        if (!CudaRuntimeBuilt()) { RequireCpuStats(model); continue; }
        // Remove only the CUDA copy, keeping exactly the same CPU model and weights.
        auto cpu = Tiny(); Quantize(cpu, type);
        auto& uploaded = model.cuda_weights->weights;
        uploaded.erase(std::remove_if(uploaded.begin(), uploaded.end(), [](const CudaloadWeight& w) {
            return w.name == "lm_head";
        }), uploaded.end());
        Context cpu_ctx(&cpu), gpu_ctx(&model);
        for (int pos : {0, 1}) {
            ResetModelCudaRuntimeStats(model);
            auto expected = ForwardBatch(cpu, cpu_ctx, Batch::Single(1, pos));
            auto actual = ForwardBatch(model, gpu_ctx, Batch::Single(1, pos));
            RequireCloseTensor(actual, expected, "missing CUDA lm_head logits");
            Require(ModelCudaLinearCalls(model) == 14 && ModelCudaLinearCpuFallbacks(model) == 1, "one linear fallback");
            Require(ModelCudaAttentionCpuFallbacks(model) == 0, "linear fallback retains CUDA attention");
            Require(ModelCudaHostToDeviceCopies(model) == 1 && ModelCudaHostToDeviceBytes(model) == 524 &&
                    ModelCudaDeviceToHostCopies(model) == 2 && ModelCudaDeviceToHostBytes(model) == 128+524,
                    "fallback actual transfers");
        }
        ResetModelCudaRuntimeStats(model);
        Require(ModelCudaLinearCpuFallbacks(model) == 0, "fallback reset");
    }
    if (!CudaRuntimeBuilt()) {
        auto model = Tiny();
        RequireError([&] { (void)ForwardLinearDevice(model, CudaTensor(), "lm_head", model.lm_head_); }, "not loaded");
        return;
    }
}
int main(int argc, char** argv) { return Run(argc, argv, {{"linear", LayerLinears}}); }
