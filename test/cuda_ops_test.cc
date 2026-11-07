#include "include/cuda_ops.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {
template <typename Operation>
void ExpectOpsNotBuilt(Operation operation) {
    try {
        operation();
        FAIL() << "Expected an unavailable CUDA ops error";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("MINI_LLAMA_CUDA=ON"),
                  std::string::npos);
    }
}
} // namespace

TEST(CudaOpsTest, BuildPredicateMatchesCudaRuntime) {
    EXPECT_EQ(CudaopsBuild(), CudaRuntimeBuilt());
}

TEST(CudaOpsTest, CpuBuildReportsUnavailableBackendForEveryEntryPoint) {
    if (CudaopsBuild()) {
        GTEST_SKIP() << "This test checks the build without CUDA support";
    }
    Tensor x({2});
    CudaTensor device_x;
    const std::vector<int> shape{2};
    ExpectOpsNotBuilt([&] { (void)CudaEmbeddingDevice(nullptr, shape, 0); });
    ExpectOpsNotBuilt([&] { (void)CudaRmsNormHost(x, x, 1e-5f); });
    ExpectOpsNotBuilt([&] {
        (void)CudaRmsNromDevice(device_x, nullptr, shape, 1e-5f);
    });
    ExpectOpsNotBuilt([&] { (void)CudaSilu(x); });
    ExpectOpsNotBuilt([&] { (void)CudaElementwiseMul(x, x); });
    ExpectOpsNotBuilt([&] { (void)CudaElementAddDevice(device_x, device_x); });
    ExpectOpsNotBuilt([&] {
        (void)OCudaElementAddDevice(device_x, nullptr, shape);
    });
    ExpectOpsNotBuilt([&] { (void)CudaSoftmax(x); });
    ExpectOpsNotBuilt([&] { CudaRopeHost(x, x, 0, 10000.0f); });
    ExpectOpsNotBuilt([&] {
        CudaRopeDevice(device_x, device_x, 0, 1, 1, 2, 10000.0f);
    });
}

class CudaOpsGpuTest : public ::testing::Test {
 protected:
    void SetUp() override {
        if (!CudaopsBuild()) {
            GTEST_SKIP() << "CUDA ops were not built";
        }
        try {
            (void)GetCudaInfo(0);
        } catch (const std::exception& error) {
            GTEST_SKIP() << "CUDA device 0 is unavailable: " << error.what();
        }
    }

    static Tensor Values(const std::vector<int>& shape) {
        Tensor x(shape);
        for (size_t i = 0; i < x.data.size(); ++i) {
            x.data[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 7.0f;
        }
        return x;
    }

    static void ExpectNear(const Tensor& actual, const Tensor& expected,
                           float tolerance = 1e-5f) {
        ASSERT_EQ(actual.shape, expected.shape);
        ASSERT_EQ(actual.data.size(), expected.data.size());
        for (size_t i = 0; i < expected.data.size(); ++i) {
            EXPECT_TRUE(std::isfinite(actual.data[i])) << "element " << i;
            EXPECT_NEAR(actual.data[i], expected.data[i],
                        tolerance * (1.0f + std::abs(expected.data[i])))
                << "element " << i;
        }
    }
};

TEST_F(CudaOpsGpuTest, EmbeddingSelectsFirstAndLastRows) {
    const Tensor weight = Values({3, 259});
    CudaDeviceBuffer buffer(weight.data.size() * sizeof(float));
    buffer.upload(weight.data.data(), weight.data.size() * sizeof(float));
    for (int token : {0, 2}) {
        SCOPED_TRACE(token);
        Tensor expected({259});
        std::copy_n(weight.data.begin() + token * 259, 259, expected.data.begin());
        const CudaTensor y = CudaEmbeddingDevice(buffer.GetData(), weight.shape, token);
        EXPECT_EQ(y.DeviceId(), 0);
        ExpectNear(y.Download(), expected);
    }
}

TEST_F(CudaOpsGpuTest, RmsNormHostAndDeviceMatchCpu) {
    for (int n : {1, 7, 259, 1025}) {
        for (bool zero : {false, true}) {
            SCOPED_TRACE(n);
            SCOPED_TRACE(zero);
            Tensor x = Values({n});
            if (zero) std::fill(x.data.begin(), x.data.end(), 0.0f);
            const Tensor weight = Values({n});
            const auto original = x.data;
            const Tensor expected = RmsNorm(x, weight, 1e-5f);
            ExpectNear(CudaRmsNormHost(x, weight, 1e-5f), expected, 1e-4f);
            const CudaTensor device_x = TensorHostToDevice(x);
            const CudaTensor device_weight = TensorHostToDevice(weight);
            const CudaTensor y = CudaRmsNromDevice(
                device_x, device_weight.GetData(), weight.shape, 1e-5f);
            EXPECT_EQ(y.DeviceId(), 0);
            ExpectNear(y.Download(), expected, 1e-4f);
            EXPECT_EQ(x.data, original);
            ExpectNear(device_x.Download(), x);
        }
    }
}

TEST_F(CudaOpsGpuTest, ElementwiseOperationsPreserveShapeAndInputs) {
    for (const auto& shape : {std::vector<int>{1}, std::vector<int>{259},
                              std::vector<int>{3, 173}}) {
        Tensor a = Values(shape);
        Tensor b = Values(shape);
        for (float& value : b.data) value = 0.5f - value;
        const auto original_a = a.data;
        const auto original_b = b.data;
        ExpectNear(CudaSilu(a), Silu(a));
        ExpectNear(CudaElementwiseMul(a, b), ElementwiseMul(a, b));
        Tensor expected(shape);
        for (size_t i = 0; i < a.data.size(); ++i) expected.data[i] = a.data[i] + b.data[i];
        const CudaTensor device_a = TensorHostToDevice(a);
        const CudaTensor device_b = TensorHostToDevice(b);
        const CudaTensor sum = CudaElementAddDevice(device_a, device_b);
        const CudaTensor raw_sum = OCudaElementAddDevice(device_a, device_b.GetData(), shape);
        EXPECT_EQ(sum.DeviceId(), 0);
        EXPECT_EQ(raw_sum.DeviceId(), 0);
        ExpectNear(sum.Download(), expected);
        ExpectNear(raw_sum.Download(), expected);
        ExpectNear(device_a.Download(), a);
        ExpectNear(device_b.Download(), b);
        EXPECT_EQ(a.data, original_a);
        EXPECT_EQ(b.data, original_b);
    }
}

TEST_F(CudaOpsGpuTest, SoftmaxIsStableAndNormalized) {
    std::vector<Tensor> cases;
    for (const auto& values : {std::vector<float>{3.0f},
                               std::vector<float>{2.0f, 2.0f, 2.0f},
                               std::vector<float>{-5.0f, -2.0f, -9.0f},
                               std::vector<float>{10000.0f, 10001.0f, -10000.0f}}) {
        Tensor x({static_cast<int>(values.size())});
        x.data = values;
        cases.push_back(x);
    }
    cases.push_back(Values({259}));
    cases.push_back(Values({1025}));
    for (Tensor& x : cases) {
        SCOPED_TRACE(x.data.size());
        const auto original = x.data;
        Tensor expected(x.shape);
        double maximum = *std::max_element(x.data.begin(), x.data.end());
        double sum = 0.0;
        for (float value : x.data) sum += std::exp(static_cast<double>(value) - maximum);
        for (size_t i = 0; i < x.data.size(); ++i) {
            expected.data[i] = static_cast<float>(std::exp(x.data[i] - maximum) / sum);
        }
        const Tensor y = CudaSoftmax(x);
        ExpectNear(y, expected);
        for (float value : y.data) EXPECT_GE(value, 0.0f);
        EXPECT_NEAR(std::accumulate(y.data.begin(), y.data.end(), 0.0), 1.0, 1e-5);
        EXPECT_EQ(x.data, original);
    }
}

TEST_F(CudaOpsGpuTest, RopeHostAndDeviceMatchBothCpuLayouts) {
    for (RopeType type : {RopeType::kNormal, RopeType::kNeoX}) {
        for (int head_dim : {2, 8, 130}) {
            for (int pos : {0, 7}) {
                SCOPED_TRACE(static_cast<int>(type));
                SCOPED_TRACE(head_dim);
                SCOPED_TRACE(pos);
                Tensor q = Values({5, head_dim});
                Tensor k = Values({2, head_dim});
                Tensor expected_q = q;
                Tensor expected_k = k;
                CudaTensor device_q = TensorHostToDevice(q);
                CudaTensor device_k = TensorHostToDevice(k);
                Rope(expected_q, expected_k, pos, 10000.0f, type);
                CudaRopeHost(q, k, pos, 10000.0f, type);
                ExpectNear(q, expected_q, 1e-4f);
                ExpectNear(k, expected_k, 1e-4f);
                CudaRopeDevice(device_q, device_k, pos, 5, 2, head_dim, 10000.0f, type);
                EXPECT_EQ(device_q.DeviceId(), 0);
                EXPECT_EQ(device_k.DeviceId(), 0);
                ExpectNear(device_q.Download(), expected_q, 1e-4f);
                ExpectNear(device_k.Download(), expected_k, 1e-4f);
            }
        }
    }
}

TEST_F(CudaOpsGpuTest, RejectsInvalidEmbeddingInputs) {
    const Tensor weight = Values({3, 4});
    const CudaTensor device_weight = TensorHostToDevice(weight);
    const void* ptr = device_weight.GetData();
    EXPECT_THROW(CudaEmbeddingDevice(nullptr, weight.shape, 0), std::runtime_error);
    EXPECT_THROW(CudaEmbeddingDevice(ptr, {12}, 0), std::runtime_error);
    EXPECT_THROW(CudaEmbeddingDevice(ptr, {0, 4}, 0), std::runtime_error);
    EXPECT_THROW(CudaEmbeddingDevice(ptr, {3, 0}, 0), std::runtime_error);
    EXPECT_THROW(CudaEmbeddingDevice(ptr, weight.shape, -1), std::runtime_error);
    EXPECT_THROW(CudaEmbeddingDevice(ptr, weight.shape, 3), std::runtime_error);
}

TEST_F(CudaOpsGpuTest, RejectsInvalidNormAndSoftmaxInputs) {
    Tensor x = Values({4});
    Tensor weight = Values({4});
    Tensor other = Values({3});
    Tensor matrix = Values({2, 2});
    Tensor empty;
    // 明确构造形状为一维、但没有元素的输入，覆盖已有的非空检查。
    empty.shape = {0};
    const CudaTensor device_x = TensorHostToDevice(x);
    const CudaTensor device_weight = TensorHostToDevice(weight);
    const CudaTensor device_matrix = TensorHostToDevice(matrix);
    const CudaTensor empty_device;
    const void* ptr = device_weight.GetData();
    for (float eps : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()}) {
        EXPECT_THROW(CudaRmsNormHost(x, weight, eps), std::runtime_error);
        EXPECT_THROW(CudaRmsNromDevice(device_x, ptr, weight.shape, eps), std::runtime_error);
    }
    EXPECT_THROW(CudaRmsNormHost(matrix, weight, 1e-5f), std::runtime_error);
    EXPECT_THROW(CudaRmsNormHost(x, matrix, 1e-5f), std::runtime_error);
    EXPECT_THROW(CudaRmsNormHost(x, other, 1e-5f), std::runtime_error);
    EXPECT_THROW(CudaRmsNormHost(empty, empty, 1e-5f), std::runtime_error);
    EXPECT_THROW(CudaRmsNromDevice(empty_device, ptr, weight.shape, 1e-5f), std::runtime_error);
    EXPECT_THROW(CudaRmsNromDevice(device_matrix, ptr, matrix.shape, 1e-5f), std::runtime_error);
    EXPECT_THROW(CudaRmsNromDevice(device_x, nullptr, weight.shape, 1e-5f), std::runtime_error);
    EXPECT_THROW(CudaRmsNromDevice(device_x, ptr, other.shape, 1e-5f), std::runtime_error);
    EXPECT_THROW(CudaRmsNromDevice(device_x, ptr, weight.shape, 1e-5f, 1), std::runtime_error);
    EXPECT_THROW(CudaSoftmax(matrix), std::runtime_error);
    EXPECT_THROW(CudaSoftmax(empty), std::runtime_error);
}

TEST_F(CudaOpsGpuTest, RejectsMismatchedElementwiseInputs) {
    const Tensor a = Values({4});
    const Tensor b = Values({2, 2});
    const CudaTensor device_a = TensorHostToDevice(a);
    const CudaTensor device_b = TensorHostToDevice(b);
    EXPECT_THROW(CudaElementwiseMul(a, b), std::runtime_error);
    EXPECT_THROW(CudaElementAddDevice(device_a, device_b), std::runtime_error);
    EXPECT_THROW(OCudaElementAddDevice(device_a, nullptr, a.shape), std::runtime_error);
    EXPECT_THROW(OCudaElementAddDevice(device_a, device_b.GetData(), b.shape), std::runtime_error);
    EXPECT_THROW(CudaElementAddDevice(device_a, device_a, 1), std::runtime_error);
}

TEST_F(CudaOpsGpuTest, RejectsInvalidRopeInputs) {
    Tensor q = Values({2, 4});
    Tensor k = Values({1, 4});
    Tensor vector = Values({4});
    Tensor odd = Values({1, 3});
    Tensor mismatch = Values({1, 6});
    CudaTensor device_q = TensorHostToDevice(q);
    CudaTensor device_k = TensorHostToDevice(k);
    for (RopeType type : {RopeType::kNormal, RopeType::kNeoX}) {
        EXPECT_THROW(CudaRopeHost(vector, k, 0, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeHost(q, vector, 0, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeHost(odd, odd, 0, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeHost(q, mismatch, 0, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeHost(q, k, -1, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeDevice(device_q, device_k, -1, 2, 1, 4, 10000.0f, type), std::runtime_error);
        for (float theta : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
            EXPECT_THROW(CudaRopeHost(q, k, 0, theta, type), std::runtime_error);
            EXPECT_THROW(CudaRopeDevice(device_q, device_k, 0, 2, 1, 4, theta, type), std::runtime_error);
        }
        EXPECT_THROW(CudaRopeDevice(device_q, device_k, 0, 0, 1, 4, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeDevice(device_q, device_k, 0, 2, 0, 4, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeDevice(device_q, device_k, 0, 2, 1, 0, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeDevice(device_q, device_k, 0, 2, 1, 3, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeDevice(device_q, device_k, 0, 3, 1, 4, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeDevice(device_q, device_k, 0, 2, 2, 4, 10000.0f, type), std::runtime_error);
        EXPECT_THROW(CudaRopeDevice(device_q, device_k, 0, 2, 1, 4, 10000.0f, type, 1), std::runtime_error);
    }
}
} // namespace yan_lamma
