#include "include/cuda_quant.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {

// Share scenarios while exercising every public entry point for each format.
#define QUANT_ROUTES(Q)                                                        \
    struct Q##Routes {                                                        \
        using Block = Block_##Q;                                              \
        static auto Quantize(const Tensor& x) { return TensorTo##Q(x); }       \
        static Tensor Decode(const std::vector<Block>& w,                      \
                             const std::vector<int>& shape) {                 \
            return Q##ToTensor(w, shape);                                      \
        }                                                                     \
        static constexpr auto Host = &Cuda##Q##Linear;                         \
        static constexpr auto WeightDevice = &Cuda##Q##DeviceLinear;           \
        static constexpr auto Device = &Cuda##Q##DeviceInputLinear;            \
    }
QUANT_ROUTES(Q40);
QUANT_ROUTES(Q41);
QUANT_ROUTES(Q80);
#undef QUANT_ROUTES

Tensor Values(const std::vector<int>& shape) {
    Tensor result(shape);
    for (size_t i = 0; i < result.data.size(); ++i) {
        result.data[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 43) - 21) / 16.0f;
    }
    return result;
}

// Compare against the actual quantized weights, excluding quantization loss.
Tensor Reference(const Tensor& x, const Tensor& w, const Tensor* bias = nullptr) {
    const bool vector = x.shape.size() == 1;
    const int rows = vector ? 1 : x.shape[0];
    const int in = w.shape[1];
    const int out = w.shape[0];
    Tensor y(vector ? std::vector<int>{out} : std::vector<int>{rows, out});
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out; ++o) {
            double sum = bias ? bias->data[o] : 0.0;
            for (int k = 0; k < in; ++k) {
                sum += static_cast<double>(x.data[r * in + k]) * w.data[o * in + k];
            }
            y.data[r * out + o] = static_cast<float>(sum);
        }
    }
    return y;
}

void ExpectNear(const Tensor& actual, const Tensor& expected) {
    ASSERT_EQ(actual.shape, expected.shape);
    ASSERT_EQ(actual.data.size(), expected.data.size());
    for (size_t i = 0; i < expected.data.size(); ++i) {
        EXPECT_NEAR(actual.data[i], expected.data[i],
                    2e-5f * std::max(1.0f, std::abs(expected.data[i]))) << "element " << i;
    }
}

using Formats = ::testing::Types<Q40Routes, Q41Routes, Q80Routes>;
template <typename T>
class CudaQuantTest : public ::testing::Test {};
TYPED_TEST_SUITE(CudaQuantTest, Formats);

TYPED_TEST(CudaQuantTest, CpuBuildReportsUnavailableForAllRoutes) {
    EXPECT_EQ(CudaQuantBuilt(), CudaRuntimeBuilt());
    if (CudaQuantBuilt()) {
        GTEST_SKIP() << "This test checks the build without CUDA support";
    }
    const Tensor x({32});
    const auto w = TypeParam::Quantize(Tensor({2, 32}));
    auto expect_unavailable = [](auto operation) {
        try {
            operation();
            FAIL() << "Expected an unavailable CUDA backend error";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("MINI_LLAMA_CUDA=ON"), std::string::npos);
        }
    };
    expect_unavailable([&] { (void)TypeParam::Host(x, w, {2, 32}, nullptr, 0); });
    expect_unavailable([&] { (void)TypeParam::WeightDevice(x, nullptr, {2, 32}, w.size(), nullptr, 0); });
    expect_unavailable([&] { (void)TypeParam::Device(CudaTensor(), nullptr, {2, 32}, w.size(), 0); });
}

template <typename T>
class CudaQuantGpuTest : public ::testing::Test {
 protected:
    void SetUp() override {
        if (!CudaQuantBuilt()) {
            GTEST_SKIP() << "CUDA quant kernels were not built";
        }
        try {
            (void)GetCudaInfo(0);
        } catch (const std::exception& error) {
            GTEST_SKIP() << "CUDA device 0 is unavailable: " << error.what();
        }
    }
};
TYPED_TEST_SUITE(CudaQuantGpuTest, Formats);

TYPED_TEST(CudaQuantGpuTest, AllRoutesMatchReferenceForVectorsBatchesAndTailBlocks) {
    for (int in : {1, 17, 32, 33, 65}) {
        SCOPED_TRACE(in);
        // 131 outputs also exercises a partial CUDA thread block.
        const std::vector<int> shape{131, in};
        const auto w = TypeParam::Quantize(Values(shape));
        const Tensor decoded = TypeParam::Decode(w, shape);
        CudaDeviceBuffer w_dev(w.size() * sizeof(typename TypeParam::Block));
        w_dev.upload(w.data(), w_dev.GetByteSize());
        for (bool batch : {false, true}) {
            SCOPED_TRACE(batch);
            const Tensor x = Values(batch ? std::vector<int>{3, in} : std::vector<int>{in});
            const Tensor expected = Reference(x, decoded);
            ExpectNear(TypeParam::Host(x, w, shape, nullptr, 0), expected);
            // Reuse the same uploaded weights across input shapes and calls.
            ExpectNear(TypeParam::WeightDevice(x, w_dev.GetData(), shape, w.size(), nullptr, 0), expected);
            const CudaTensor x_dev = TensorHostToDevice(x);
            const CudaTensor y = TypeParam::Device(x_dev, w_dev.GetData(), shape, w.size(), 0);
            EXPECT_EQ(y.DeviceId(), 0);
            ExpectNear(y.Download(), expected);
            const Tensor bias = Values({shape[0]});
            const Tensor biased = Reference(x, decoded, &bias);
            ExpectNear(TypeParam::Host(x, w, shape, &bias, 0), biased);
            ExpectNear(TypeParam::WeightDevice(x, w_dev.GetData(), shape, w.size(), &bias, 0), biased);
        }
    }
}

TYPED_TEST(CudaQuantGpuTest, DeviceOutputFeedsNextLinearWithoutHostIntermediate) {
    const std::vector<int> first_shape{33, 65};
    const std::vector<int> second_shape{7, 33};
    const auto first = TypeParam::Quantize(Values(first_shape));
    const auto second = TypeParam::Quantize(Values(second_shape));
    CudaDeviceBuffer first_dev(first.size() * sizeof(typename TypeParam::Block));
    CudaDeviceBuffer second_dev(second.size() * sizeof(typename TypeParam::Block));
    first_dev.upload(first.data(), first_dev.GetByteSize());
    second_dev.upload(second.data(), second_dev.GetByteSize());
    const Tensor x = Values({3, 65});
    const Tensor intermediate_ref = Reference(x, TypeParam::Decode(first, first_shape));
    const Tensor expected = Reference(intermediate_ref, TypeParam::Decode(second, second_shape));
    const CudaTensor x_dev = TensorHostToDevice(x);
    const CudaTensor intermediate = TypeParam::Device(x_dev, first_dev.GetData(), first_shape, first.size(), 0);
    const CudaTensor result = TypeParam::Device(intermediate, second_dev.GetData(), second_shape, second.size(), 0);
    ExpectNear(result.Download(), expected);
}

TYPED_TEST(CudaQuantGpuTest, RejectsInvalidShapeBlockCountBiasAndDevice) {
    const std::vector<int> shape{4, 33};
    const auto w = TypeParam::Quantize(Values(shape));
    CudaDeviceBuffer w_dev(w.size() * sizeof(typename TypeParam::Block));
    w_dev.upload(w.data(), w_dev.GetByteSize());
    const Tensor x = Values({2, 33});
    const CudaTensor x_dev = TensorHostToDevice(x);
    for (const auto& bad_shape : std::vector<std::vector<int>>{{4}, {4, 32}, {0, 33}}) {
        EXPECT_THROW((void)TypeParam::Host(x, w, bad_shape, nullptr, 0), std::runtime_error);
        EXPECT_THROW((void)TypeParam::WeightDevice(x, w_dev.GetData(), bad_shape, w.size(), nullptr, 0), std::runtime_error);
        EXPECT_THROW((void)TypeParam::Device(x_dev, w_dev.GetData(), bad_shape, w.size(), 0), std::runtime_error);
    }
    EXPECT_THROW((void)TypeParam::Host(Tensor({1, 2, 33}), w, shape, nullptr, 0), std::runtime_error);
    EXPECT_THROW((void)TypeParam::WeightDevice(x, nullptr, shape, w.size(), nullptr, 0), std::runtime_error);
    EXPECT_THROW((void)TypeParam::Device(x_dev, nullptr, shape, w.size(), 0), std::runtime_error);
    EXPECT_THROW((void)TypeParam::WeightDevice(x, w_dev.GetData(), shape, w.size() - 1, nullptr, 0), std::runtime_error);
    EXPECT_THROW((void)TypeParam::Device(x_dev, w_dev.GetData(), shape, w.size() + 1, 0), std::runtime_error);
    EXPECT_THROW((void)TypeParam::Device(x_dev, w_dev.GetData(), shape, w.size(), 1), std::runtime_error);
    const Tensor bad_bias({2, 2});
    EXPECT_THROW((void)TypeParam::Host(x, w, shape, &bad_bias, 0), std::runtime_error);
    EXPECT_THROW((void)TypeParam::WeightDevice(x, w_dev.GetData(), shape, w.size(), &bad_bias, 0), std::runtime_error);
}

} // namespace
} // namespace yan_lamma
