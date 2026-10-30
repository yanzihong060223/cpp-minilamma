#include "include/backend.h"
#include "include/cuda_runtime.h"

#include <gtest/gtest.h>

#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace yan_lamma {
namespace {

TEST(BackendTest, ParsesCpuAndGpu) {
    BackendKind kind = BackendKind::Kgpu;
    EXPECT_TRUE(ParseBackendKind("cpu", kind));
    EXPECT_EQ(kind, BackendKind::Kcpu);
    EXPECT_TRUE(ParseBackendKind("gpu", kind));
    EXPECT_EQ(kind, BackendKind::Kgpu);
}

TEST(BackendTest, InvalidTextLeavesKindUnchanged) {
    for (const char* text : {"", "cuda", "auto", "CPU", "GPU", " cpu", "gpu ", "gpu\n"}) {
        SCOPED_TRACE(text);
        for (BackendKind original : {BackendKind::Kcpu, BackendKind::Kgpu}) {
            SCOPED_TRACE(BackendKindName(original));
            BackendKind kind = original;
            EXPECT_FALSE(ParseBackendKind(text, kind));
            EXPECT_EQ(kind, original);
        }
    }
}

TEST(BackendTest, NamesKnownAndUnknownKinds) {
    EXPECT_EQ(BackendKindName(BackendKind::Kcpu), "cpu");
    EXPECT_EQ(BackendKindName(BackendKind::Kgpu), "gpu");
    EXPECT_EQ(BackendKindName(static_cast<BackendKind>(-1)), "unknown");
}

TEST(BackendTest, DefaultCpuConfigPassesValidation) {
    const BackendConfig config;
    EXPECT_EQ(config.kind, BackendKind::Kcpu);
    EXPECT_NO_THROW(VailedBackend(config));
}

TEST(BackendTest, CpuRejectsExplicitDeviceEvenWhenDeviceIsZero) {
    BackendConfig config;
    config.device_set = true;
    config.device_id = 0;

    try {
        VailedBackend(config);
        FAIL() << "CPU configuration with an explicit device was accepted";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("--device"), std::string::npos);
        EXPECT_NE(message.find("--backend gpu"), std::string::npos);
    }
}

TEST(BackendTest, RejectsUnknownBackendKind) {
    BackendConfig config;
    config.kind = static_cast<BackendKind>(-1);
    EXPECT_THROW(VailedBackend(config), std::runtime_error);
}

TEST(BackendTest, NegativeCudaDeviceIsUnavailableWithoutThrowing) {
    bool available = true;
    ASSERT_NO_THROW(available = CudaBackendAvailable(-1));
    EXPECT_FALSE(available);
}

TEST(BackendTest, InvalidDeviceSummaryReturnsErrorTextWithoutThrowing) {
    std::string summary;
    ASSERT_NO_THROW(summary = CudaDeviceSummary(-1));
    EXPECT_FALSE(summary.empty());
    if (!CudaRuntimeBuilt()) {
        EXPECT_NE(summary.find("not built"), std::string::npos);
    }
}

TEST(BackendTest, GpuValidationReportsBuildFailureOrDeviceErrorContext) {
    BackendConfig config;
    config.kind = BackendKind::Kgpu;
    config.device_id = -1;
    config.device_set = true;

    try {
        VailedBackend(config);
        FAIL() << "GPU configuration with a negative device was accepted";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (!CudaRuntimeBuilt()) {
            EXPECT_NE(message.find("not built"), std::string::npos);
            EXPECT_NE(message.find("MINI_LLAMA_CUDA=ON"), std::string::npos);
        } else {
            const std::string context = "device -1: ";
            const auto position = message.find(context);
            ASSERT_NE(position, std::string::npos) << message;
            // The original error must accompany the device context.
            EXPECT_GT(message.size(), position + context.size());
        }
    }
}

TEST(BackendTest, ExecutionNotesDescribeCurrentSupport) {
    BackendConfig config;
    EXPECT_EQ(BackendActivePath(config), "compute: cpu");

    config.kind = BackendKind::Kgpu;
    const std::string gpu_note = BackendActivePath(config);
    EXPECT_NE(gpu_note.find("gpu"), std::string::npos);
    EXPECT_NE(gpu_note.find("runtime"), std::string::npos);
    EXPECT_NE(gpu_note.find("not implemented"), std::string::npos);

    config.kind = static_cast<BackendKind>(-1);
    EXPECT_EQ(BackendActivePath(config), "unknown backend");
}

TEST(BackendTest, AvailableCudaDevicePassesValidationAndHasSummary) {
    struct CudaInfo info;
    try {
        // Probe the runtime directly so backend regressions cannot cause a skip.
        info = GetCudaInfo(0);
    } catch (const std::exception& error) {
        GTEST_SKIP() << "CUDA device 0 is unavailable: " << error.what();
    }

    EXPECT_TRUE(CudaBackendBuilt());
    EXPECT_TRUE(CudaBackendAvailable(0));

    BackendConfig config;
    config.kind = BackendKind::Kgpu;
    config.device_id = 0;
    config.device_set = true;
    EXPECT_NO_THROW(VailedBackend(config));

    std::string summary;
    ASSERT_NO_THROW(summary = CudaDeviceSummary(0));
    EXPECT_NE(summary.find("device_id=0"), std::string::npos);
    EXPECT_NE(summary.find("name=\"" + info.name + "\""), std::string::npos);
}

}  // namespace
}  // namespace yan_lamma
