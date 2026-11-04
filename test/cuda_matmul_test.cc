#include "include/cuda_matmul.h"
#include "include/cuda_runtime.h"
#include "include/cuda_tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {

class CudaMatmulTest : public ::testing::Test {
 protected:
    template <typename Operation>
    static void ExpectNotBuilt(Operation operation) {
        try {
            operation();
            FAIL() << "Expected an unavailable CUDA backend error";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("MINI_LLAMA_CUDA=ON"),
                      std::string::npos);
        }
    }
};

TEST_F(CudaMatmulTest, BuildPredicateMatchesCudaRuntime) {
    EXPECT_EQ(cuBLASBuilt(), CudaRuntimeBuilt());
}

TEST_F(CudaMatmulTest, CpuBuildReportsUnavailableBackendForEveryEntryPoint) {
    if (CudaRuntimeBuilt()) {
        GTEST_SKIP() << "This test checks the build without CUDA support";
    }
    const Tensor a({2, 3});
    const Tensor b({3, 4});
    const Tensor x({3});
    const Tensor weight({4, 3});
    ExpectNotBuilt([&] { (void)CudaMatmul(a, b, 0); });
    ExpectNotBuilt([&] { (void)CudaLinear(x, weight); });
    ExpectNotBuilt([&] {
        (void)CudaLinearHostDevice(x, nullptr, weight.shape);
    });
    ExpectNotBuilt([&] {
        (void)CudaLinearDevice(CudaTensor(), nullptr, weight.shape);
    });
}

class CudaMatmulGpuTest : public ::testing::Test {
 protected:
    void SetUp() override {
        try {
            (void)GetCudaInfo(0);
        } catch (const std::exception& error) {
            GTEST_SKIP() << "CUDA device 0 is unavailable: " << error.what();
        }
    }

    static Tensor WeightValues() {
        Tensor weight({4, 3});
        weight.data = {1.0f, -2.0f, 0.5f, -0.5f, 3.0f, 2.0f,
                       4.0f, 0.25f, -1.0f, -3.0f, 2.0f, 1.5f};
        return weight;
    }

    static Tensor InputValues(bool batched) {
        Tensor x(batched ? std::vector<int>{2, 3} : std::vector<int>{3});
        x.data = batched ? std::vector<float>{1.5f, -2.0f, 0.25f,
                                             -1.0f, 3.0f, 2.0f}
                         : std::vector<float>{1.5f, -2.0f, 0.25f};
        return x;
    }

    static Tensor ReferenceLinear(const Tensor& x, const Tensor& weight,
                                  const Tensor* bias = nullptr) {
        const bool vector_input = x.shape.size() == 1;
        const int rows = vector_input ? 1 : x.shape[0];
        const int in_features = weight.shape[1];
        const int out_features = weight.shape[0];
        Tensor result(vector_input ? std::vector<int>{out_features}
                                   : std::vector<int>{rows, out_features});
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < out_features; ++col) {
                double sum = bias == nullptr ? 0.0 : bias->data[col];
                for (int feature = 0; feature < in_features; ++feature) {
                    sum += static_cast<double>(x.data[row * in_features + feature]) *
                           weight.data[col * in_features + feature];
                }
                result.data[row * out_features + col] = static_cast<float>(sum);
            }
        }
        return result;
    }

    static void ExpectNear(const Tensor& actual, const Tensor& expected) {
        ASSERT_EQ(actual.shape, expected.shape);
        ASSERT_EQ(actual.data.size(), expected.data.size());
        for (size_t index = 0; index < expected.data.size(); ++index) {
            EXPECT_NEAR(actual.data[index], expected.data[index],
                        1e-5f * std::max(1.0f, std::abs(expected.data[index])))
                << "element " << index;
        }
    }
};

TEST_F(CudaMatmulGpuTest, NonSquareMatmulMatchesIndependentReference) {
    Tensor a({2, 3});
    a.data = {1.0f, -2.0f, 0.5f, 3.0f, 1.0f, -1.0f};
    Tensor b({3, 4});
    b.data = {1.0f, 2.0f, -1.0f, 0.5f, -2.0f, 3.0f,
              4.0f, -1.0f, 0.25f, -0.5f, 2.0f, 3.0f};
    Tensor expected({2, 4});
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 4; ++col) {
            double sum = 0.0;
            for (int inner = 0; inner < 3; ++inner) {
                sum += static_cast<double>(a.data[row * 3 + inner]) *
                       b.data[inner * 4 + col];
            }
            expected.data[row * 4 + col] = static_cast<float>(sum);
        }
    }
    ExpectNear(CudaMatmul(a, b, 0), expected);
}

TEST_F(CudaMatmulGpuTest, AllLinearRoutesPreserveVectorAndBatchShapes) {
    const Tensor weight = WeightValues();
    const CudaTensor weight_dev = TensorHostToDevice(weight);
    for (bool batched : {false, true}) {
        SCOPED_TRACE(batched);
        const Tensor x = InputValues(batched);
        const Tensor expected = ReferenceLinear(x, weight);
        const CudaTensor x_dev = TensorHostToDevice(x);
        ExpectNear(CudaLinear(x, weight), expected);
        ExpectNear(CudaLinearHostDevice(x, weight_dev.GetData(), weight.shape),
                   expected);
        const CudaTensor result = CudaLinearDevice(
            x_dev, weight_dev.GetData(), weight.shape);
        EXPECT_EQ(result.DeviceId(), 0);
        ExpectNear(result.Download(), expected);
    }
}

TEST_F(CudaMatmulGpuTest, HostOutputsBroadcastBiasAcrossVectorAndBatch) {
    const Tensor weight = WeightValues();
    const CudaTensor weight_dev = TensorHostToDevice(weight);
    Tensor bias({4});
    bias.data = {0.5f, -1.5f, 2.0f, -3.0f};
    for (bool batched : {false, true}) {
        SCOPED_TRACE(batched);
        const Tensor x = InputValues(batched);
        const Tensor expected = ReferenceLinear(x, weight, &bias);
        ExpectNear(CudaLinear(x, weight, &bias), expected);
        ExpectNear(CudaLinearHostDevice(
                       x, weight_dev.GetData(), weight.shape, &bias), expected);
    }
}

TEST_F(CudaMatmulGpuTest, RawWeightBufferIsBorrowedAndReusable) {
    const Tensor weight = WeightValues();
    const size_t bytes = weight.data.size() * sizeof(float);
    CudaDeviceBuffer weight_dev(bytes);
    weight_dev.upload(weight.data.data(), bytes);
    const void* original_pointer = weight_dev.GetData();
    for (bool batched : {false, true}) {
        const Tensor x = InputValues(batched);
        ExpectNear(CudaLinearHostDevice(x, weight_dev.GetData(), weight.shape),
                   ReferenceLinear(x, weight));
        EXPECT_EQ(weight_dev.GetData(), original_pointer);
    }
    Tensor downloaded(weight.shape);
    weight_dev.download(downloaded.data.data(), bytes);
    EXPECT_EQ(downloaded.data, weight.data);
}

TEST_F(CudaMatmulGpuTest, DeviceOutputFeedsTheNextLinearWithoutHostIntermediate) {
    const Tensor x = InputValues(true);
    const Tensor first_weight = WeightValues();
    Tensor second_weight({2, 4});
    second_weight.data = {1.0f, -0.5f, 2.0f, -1.0f,
                          -2.0f, 3.0f, 0.25f, 1.5f};
    const Tensor expected = ReferenceLinear(
        ReferenceLinear(x, first_weight), second_weight);
    const CudaTensor x_dev = TensorHostToDevice(x);
    const CudaTensor first_weight_dev = TensorHostToDevice(first_weight);
    const CudaTensor second_weight_dev = TensorHostToDevice(second_weight);
    const CudaTensor intermediate = CudaLinearDevice(
        x_dev, first_weight_dev.GetData(), first_weight.shape);
    const CudaTensor result = CudaLinearDevice(
        intermediate, second_weight_dev.GetData(), second_weight.shape);
    ExpectNear(result.Download(), expected);
}

TEST_F(CudaMatmulGpuTest, MatmulRejectsInvalidRanksAndInnerDimensions) {
    const Tensor a({2, 3});
    const Tensor b({3, 4});
    EXPECT_THROW(CudaMatmul(Tensor({3}), b, 0), std::runtime_error);
    EXPECT_THROW(CudaMatmul(a, Tensor({4, 2}), 0), std::runtime_error);
}

TEST_F(CudaMatmulGpuTest, LinearRejectsInvalidRanksAndInnerDimensions) {
    const Tensor x = InputValues(false);
    const Tensor weight = WeightValues();
    EXPECT_THROW(CudaLinear(Tensor(), weight), std::runtime_error);
    EXPECT_THROW(CudaLinear(Tensor({1, 1, 3}), weight), std::runtime_error);
    EXPECT_THROW(CudaLinear(x, Tensor({3})), std::runtime_error);
    EXPECT_THROW(CudaLinear(x, Tensor({4, 2})), std::runtime_error);
}

TEST_F(CudaMatmulGpuTest, DeviceInputAndWeightPointerMustBeValid) {
    const Tensor x = InputValues(false);
    const Tensor weight = WeightValues();
    const CudaTensor x_dev = TensorHostToDevice(x);
    const CudaTensor weight_dev = TensorHostToDevice(weight);
    EXPECT_THROW(CudaLinearHostDevice(x, nullptr, weight.shape),
                 std::runtime_error);
    EXPECT_THROW(CudaLinearHostDevice(x, weight_dev.GetData(), {4, 2}),
                 std::runtime_error);
    EXPECT_THROW(CudaLinearDevice(
                     CudaTensor(), weight_dev.GetData(), weight.shape),
                 std::runtime_error);
    EXPECT_THROW(CudaLinearDevice(x_dev, nullptr, weight.shape),
                 std::runtime_error);
    const CudaTensor wrong_features({2});
    EXPECT_THROW(CudaLinearDevice(
                     wrong_features, weight_dev.GetData(), weight.shape),
                 std::runtime_error);
    const CudaTensor wrong_rank({1, 1, 3});
    EXPECT_THROW(CudaLinearDevice(
                     wrong_rank, weight_dev.GetData(), weight.shape),
                 std::runtime_error);
    EXPECT_THROW(CudaLinearDevice(x_dev, weight_dev.GetData(), {4}),
                 std::runtime_error);
    EXPECT_THROW(CudaLinearDevice(
                     x_dev, weight_dev.GetData(), weight.shape, -1),
                 std::runtime_error);
}

TEST_F(CudaMatmulGpuTest, HostOutputRoutesRejectInvalidBiasShape) {
    const Tensor x = InputValues(true);
    const Tensor weight = WeightValues();
    const CudaTensor weight_dev = TensorHostToDevice(weight);
    for (const Tensor& bias : {Tensor({3}), Tensor({1, 4})}) {
        EXPECT_THROW(CudaLinear(x, weight, &bias), std::runtime_error);
        EXPECT_THROW(CudaLinearHostDevice(
                         x, weight_dev.GetData(), weight.shape, &bias),
                     std::runtime_error);
    }
}

}  // namespace yan_lamma
