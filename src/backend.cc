#include "include/backend.h"

#include <exception>
#include <stdexcept>
#include <string>

namespace yan_lamma {

bool ParseBackendKind(const std::string& text, BackendKind& kind) {
    if (text == "cpu") {
        kind = BackendKind::Kcpu;
        return true;
    }
    if (text == "gpu") {
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
            return "gpu";
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
                    "--device can only be used with --backend gpu.");
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
            return "backend: gpu (CUDA runtime support only; CUDA compute path is not implemented)";
    }
    return "unknown backend";
}

}// namespace yan_lamma
