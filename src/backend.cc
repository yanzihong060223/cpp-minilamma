#include "include/backend.h"
#include "include/model.h"
#include <ostream>

#include <exception>
#include <stdexcept>
#include <string>

namespace yan_lamma {

bool ParseBackendKind(const std::string& text, BackendKind& kind) {
    if (text == "cpu") {
        kind = BackendKind::Kcpu;
        return true;
    }
    if (text == "cuda" || text == "gpu") {
        kind = BackendKind::Kgpu;
        return true;
    }
    return false;
}

std::string BackendKindName(BackendKind kind) {
    switch (kind) {
        case BackendKind::Kcpu:
            return "cpu";
        case BackendKind::Kgpu:
            return "cuda";
    }
    return "unknown";
}

bool CudaBackendBuilt() {
    return CudaRuntimeBuilt();
}

bool CudaBackendAvailable(int device_id) {
    try {
        int count = DeviceCount();
        return device_id >= 0 && device_id < count;
    } catch (const std::exception&) {
        return false;
    }
}

std::string CudaDeviceSummary(int device_id) {
    try {
        return CudaInfo(GetCudaInfo(device_id));
    } catch (const std::exception& e) {
        return e.what();
    }
}

void VailedBackend(const BackendConfig& config) {
    switch (config.kind) {
        case BackendKind::Kcpu:
            if (config.device_set) {
                throw std::runtime_error(
                    "--device can only be used with --backend cuda");
            }
            return;
        case BackendKind::Kgpu:
            if (!CudaBackendBuilt()) {
                throw std::runtime_error(
                    "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON "
                    "on a NVIDIA CUDA machine.");
            }
            try {
                (void)GetCudaInfo(config.device_id);
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    "Failed to validate CUDA device " + std::to_string(config.device_id) +
                    ": " + e.what());
            }
            return;
    }
    throw std::runtime_error("unknown backend");
}

std::string BackendActivePath(const BackendConfig& config) {
    switch (config.kind) {
        case BackendKind::Kcpu:
            return "compute: cpu";
        case BackendKind::Kgpu:
            return "compute: cuda linear; CUDA attention over GPU KV cache; CPU sampler";
    }
    return "unknown backend";
}

void PrintBackendSummary(const BackendConfig& config, const Model& model, std::ostream& out) {
    out << "backend: " << BackendKindName(config.kind) << '\n'
        << BackendActivePath(config) << '\n'
        << "cuda runtime: " << (CudaRuntimeBuilt() ? "built" : "not built") << '\n'
        << "cuda device count: " << (config.kind == BackendKind::Kgpu ? DeviceCount() : 0) << '\n';
    size_t f32 = 0, q8 = 0, q40 = 0, q41 = 0;
    if (model.cuda_weights) {
        for (const auto& weight : model.cuda_weights->weights) {
            if (!weight.linear) continue;
            switch (weight.type) {
                case QuantType::kF32: ++f32; break;
                case QuantType::kQ80: ++q8; break;
                case QuantType::kQ40: ++q40; break;
                case QuantType::kQ41: ++q41; break;
                default: break;
            }
        }
    }
    out << "uploaded weights: " << (model.cuda_weights ? model.cuda_weights->uploaded_weight_count : 0) << '\n'
        << "cuda f32 linear weights: " << f32 << '\n'
        << "cuda q8_0 linear weights: " << q8 << '\n'
        << "cuda q4_0 linear weights: " << q40 << '\n'
        << "cuda q4_1 linear weights: " << q41 << '\n'
        << "cuda linear calls: " << ModelCudaLinearCalls(model) << '\n'
        << "cuda activation calls: " << ModelCudaActivationCalls(model) << '\n'
        << "cuda attention calls: " << ModelCudaAttentionCalls(model) << '\n'
        << "cpu linear fallback calls: " << ModelCudaLinearCpuFallbacks(model) << '\n'
        << "cpu attention fallback calls: " << ModelCudaAttentionCpuFallbacks(model) << '\n'
        << "host->device copies: " << ModelCudaHostToDeviceCopies(model) << '\n'
        << "host->device bytes: " << ModelCudaHostToDeviceBytes(model) << '\n'
        << "device->host copies: " << ModelCudaDeviceToHostCopies(model) << '\n'
        << "device->host bytes: " << ModelCudaDeviceToHostBytes(model) << '\n';
}

}// namespace yan_lamma
