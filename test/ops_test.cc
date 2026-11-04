#include "include/ops.h"
#include "matumal/matuml_ops.h"
#include "thread_pool/thread_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using yan_lamma::MatulMode;
using yan_lamma::Tensor;

struct RestoreAutomaticThreadCount {
    ~RestoreAutomaticThreadCount() { yan_lamma::SetThreadNum(0); }
};

Tensor MakeTensor(std::vector<int> shape, std::vector<float> values) {
    Tensor tensor(shape);
    if (tensor.data.size() != values.size()) {
        throw std::invalid_argument("Test fixture data does not match shape");
    }
    tensor.data = std::move(values);
    return tensor;
}

// Public fields can be corrupted after construction. Checked operators must
// reject these fixtures before accessing storage.
Tensor Malformed(std::vector<int> shape, std::vector<float> values = {}) {
    Tensor tensor;
    tensor.shape = std::move(shape);
    tensor.data = std::move(values);
    return tensor;
}

void ExpectNearDouble(double actual, double expected) {
    EXPECT_TRUE(std::isfinite(actual));
    EXPECT_NEAR(actual, expected, 2e-5 * std::max(1.0, std::abs(expected)));
}

void ExpectTensor(const Tensor& actual, const std::vector<int>& shape,
                  const std::vector<double>& expected) {
    EXPECT_EQ(actual.shape, shape);
    ASSERT_EQ(actual.data.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        SCOPED_TRACE("element " + std::to_string(index));
        ExpectNearDouble(actual.data[index], expected[index]);
    }
}

std::vector<double> RmsReference(const std::vector<float>& values,
                                 const std::vector<float>& weights, float eps) {
    double squared_sum = 0.0;
    for (float value : values) {
        const double promoted = static_cast<double>(value);
        squared_sum += promoted * promoted;
    }
    const double denominator =
        std::sqrt(squared_sum / static_cast<double>(values.size()) + eps);
    std::vector<double> expected(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        expected[index] = static_cast<double>(values[index]) / denominator *
                          static_cast<double>(weights[index]);
    }
    return expected;
}

using NamedTensor = std::pair<const char*, Tensor>;

std::vector<NamedTensor> InvalidMatrices() {
    return {
        {"default empty", Tensor{}},
        {"scalar rank", Malformed({}, {1})},
        {"vector rank", MakeTensor({2}, {1, 2})},
        {"three-dimensional rank", MakeTensor({1, 1, 2}, {1, 2})},
        {"zero rows", Malformed({0, 2})},
        {"zero columns", Malformed({2, 0})},
        {"negative rows", Malformed({-1, 2})},
        {"negative columns", Malformed({2, -1})},
        {"short storage", Malformed({2, 2}, {1, 2, 3})},
        {"long storage", Malformed({2, 2}, {1, 2, 3, 4, 5})},
    };
}

TEST(ArgMaxTest, UpdatesRunningMaximum) {
    EXPECT_EQ(yan_lamma::ArgMax(MakeTensor({3}, {1, 5, 3})), 1u);
}

TEST(LinearTest, QuantizedWeightsUseIndependentRowTailBlocks) {
    using namespace yan_lamma;
    for (int width : {1, 16, 31, 32, 33, 64, 86}) {
        for (QuantType type : {QuantType::kQ80, QuantType::kQ40,
                               QuantType::kQ41}) {
            SCOPED_TRACE(width);
            SCOPED_TRACE(static_cast<int>(type));
            QuanTensor weight;
            weight.type = type;
            weight.shape = {2, width};
            const std::size_t blocks = 2 * (1 + (width - 1) / 32);
            const std::size_t bytes = type == QuantType::kQ80 ? 34 :
                                      type == QuantType::kQ40 ? 18 : 20;
            auto& data = type == QuantType::kQ80 ? weight.kQ80_data :
                         type == QuantType::kQ40 ? weight.kQ40_data :
                                                  weight.kQ41_data;
            data.assign(blocks * bytes, 0);
            for (std::size_t b = 0; b < blocks; ++b) {
                // FP16 scale = 1; all valid quantized values decode to 1.
                data[b * bytes + 1] = 0x3c;
                const std::size_t header = type == QuantType::kQ41 ? 4 : 2;
                const uint8_t code = type == QuantType::kQ80 ? 1 :
                                     type == QuantType::kQ40 ? 0x99 : 0x11;
                std::fill(data.begin() + b * bytes + header,
                          data.begin() + (b + 1) * bytes, code);
            }
            for (const auto& shape : {std::vector<int>{width},
                                      std::vector<int>{2, width}}) {
                Tensor input(shape);
                std::fill(input.data.begin(), input.data.end(), 1.0f);
                const Tensor result = Linear(input, weight);
                for (float value : result.data) {
                    EXPECT_FLOAT_EQ(value, static_cast<float>(width));
                }
                const auto valid = data;
                data.pop_back();
                EXPECT_THROW(Linear(input, weight), std::runtime_error);
                data = valid;
                data.resize(valid.size() - bytes);
                EXPECT_THROW(Linear(input, weight), std::runtime_error);
                data = valid;
                data.resize(valid.size() + bytes);
                EXPECT_THROW(Linear(input, weight), std::runtime_error);
                data = valid;
            }
        }
    }
}

TEST(ArgMaxTest, KeepsFirstIndexWhenMaximumIsRepeated) {
    EXPECT_EQ(yan_lamma::ArgMax(MakeTensor({4}, {1, 5, 5, 3})), 1u);
    EXPECT_EQ(yan_lamma::ArgMax(MakeTensor({3}, {7, 7, 7})), 0u);
}

TEST(ArgMaxTest, HandlesNegativeValues) {
    EXPECT_EQ(yan_lamma::ArgMax(MakeTensor({3}, {-9, -2, -4})), 1u);
}

TEST(ArgMaxTest, RejectsEmptyInput) {
    EXPECT_THROW(yan_lamma::ArgMax(Tensor{}), std::invalid_argument);
}

TEST(RmsNormTest, NormalizesByMeanSquareAndAppliesWeights) {
    ExpectTensor(yan_lamma::RmsNorm(MakeTensor({2}, {3, 4}),
                                   MakeTensor({2}, {1, 1}), 0.0f),
                 {2}, {0.8485281374, 1.1313708499});
    const std::vector<float> values{-2, 0.25f, 3.5f};
    const std::vector<float> weights{0.5f, 2, -1};
    ExpectTensor(yan_lamma::RmsNorm(MakeTensor({3}, values),
                                   MakeTensor({3}, weights), 1e-5f),
                 {3}, RmsReference(values, weights, 1e-5f));
}

TEST(RmsNormTest, PromotesBeforeSquaringExtremeFiniteValues) {
    const float largest = std::numeric_limits<float>::max();
    const float smallest = std::numeric_limits<float>::denorm_min();
    const std::vector<NamedTensor> cases{
        {"float square would overflow", MakeTensor({2}, {largest, -largest})},
        {"float square would underflow", MakeTensor({2}, {smallest, 2 * smallest})},
    };
    const std::vector<float> weights{1, 1};
    for (const auto& named : cases) {
        SCOPED_TRACE(named.first);
        ExpectTensor(yan_lamma::RmsNorm(named.second, MakeTensor({2}, weights), 0),
                     {2}, RmsReference(named.second.data, weights, 0));
    }
}

TEST(RmsNormTest, AllowsZeroInputWithPositiveEpsilon) {
    ExpectTensor(yan_lamma::RmsNorm(MakeTensor({2}, {0, 0}),
                                   MakeTensor({2}, {1, 1}), 1e-5f),
                 {2}, {0, 0});
}

TEST(RmsNormTest, RejectsNegativeOrNonfiniteEpsilon) {
    const Tensor x = MakeTensor({2}, {3, 4});
    const Tensor weight = MakeTensor({2}, {1, 1});
    const std::vector<std::pair<const char*, float>> cases{
        {"negative", -1.0f},
        {"NaN", std::numeric_limits<float>::quiet_NaN()},
        {"infinity", std::numeric_limits<float>::infinity()},
    };
    for (const auto& named : cases) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::RmsNorm(x, weight, named.second),
                     std::invalid_argument);
    }
}

TEST(RmsNormTest, RejectsZeroOrNonfiniteDenominator) {
    const Tensor weight = MakeTensor({2}, {1, 1});
    const std::vector<NamedTensor> cases{
        {"zero denominator", MakeTensor({2}, {0, 0})},
        {"NaN denominator", MakeTensor({2}, {std::numeric_limits<float>::quiet_NaN(), 1})},
        {"infinite denominator", MakeTensor({2}, {std::numeric_limits<float>::infinity(), 1})},
    };
    for (const auto& named : cases) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::RmsNorm(named.second, weight, 0),
                     std::invalid_argument);
    }
}

TEST(RmsNormTest, RejectsInvalidRanksDimensionsAndMismatches) {
    const Tensor valid = MakeTensor({2}, {3, 4});
    const std::vector<NamedTensor> cases{
        {"empty", Tensor{}},
        {"two-dimensional", MakeTensor({1, 2}, {1, 2})},
        {"zero dimension", Malformed({0})},
        {"negative dimension", Malformed({-2})},
        {"dimension mismatch", MakeTensor({3}, {1, 2, 3})},
    };
    for (const auto& named : cases) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::RmsNorm(named.second, valid, 1e-5f), std::exception);
        EXPECT_THROW(yan_lamma::RmsNorm(valid, named.second, 1e-5f), std::exception);
    }
}

TEST(RmsNormTest, RejectsShapeStorageMismatchOnEitherInput) {
    const Tensor valid = MakeTensor({2}, {3, 4});
    const std::vector<NamedTensor> cases{
        {"short storage", Malformed({2}, {1})},
        {"long storage", Malformed({2}, {1, 2, 3})},
    };
    for (const auto& named : cases) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::RmsNorm(named.second, valid, 1e-5f), std::invalid_argument);
        EXPECT_THROW(yan_lamma::RmsNorm(valid, named.second, 1e-5f), std::invalid_argument);
    }
}

TEST(LinearTest, VectorInputProducesWeightedDotProducts) {
    const Tensor x = MakeTensor({3}, {1, 2, 3});
    const Tensor weight = MakeTensor({2, 3}, {1, 0, 1, 0, 1, 1});
    ExpectTensor(yan_lamma::Linear(x, weight), {2}, {4, 5});
}

TEST(LinearTest, SingleRowInputPreservesTwoDimensionalShape) {
    const Tensor x = MakeTensor({1, 3}, {1, 2, 3});
    const Tensor weight = MakeTensor({2, 3}, {1, 0, 1, 0, 1, 1});
    ExpectTensor(yan_lamma::Linear(x, weight), {1, 2}, {4, 5});
}

TEST(LinearTest, RejectsInvalidInputAndMultirowBatch) {
    const Tensor weight = MakeTensor({2, 2}, {1, 2, 3, 4});
    const std::vector<NamedTensor> cases{
        {"empty", Tensor{}},
        {"scalar rank", Malformed({}, {1})},
        {"three-dimensional", MakeTensor({1, 1, 2}, {1, 2})},
        {"zero vector dimension", Malformed({0})},
        {"negative vector dimension", Malformed({-2})},
        {"zero rows", Malformed({0, 2})},
        {"zero columns", Malformed({1, 0})},
        {"negative rows", Malformed({-1, 2})},
        {"negative columns", Malformed({1, -2})},
        {"short vector storage", Malformed({2}, {1})},
        {"long vector storage", Malformed({2}, {1, 2, 3})},
        {"short row storage", Malformed({1, 2}, {1})},
        {"long row storage", Malformed({1, 2}, {1, 2, 3})},
        {"input size mismatch", MakeTensor({3}, {1, 2, 3})},
        {"row size mismatch", MakeTensor({1, 3}, {1, 2, 3})},
        {"multirow batch is unsupported", MakeTensor({2, 2}, {1, 2, 3, 4})},
    };
    for (const auto& named : cases) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::Linear(named.second, weight), std::invalid_argument);
        for (const auto mode : {MatulMode::Native, MatulMode::Threaded}) {
            SCOPED_TRACE(static_cast<int>(mode));
            EXPECT_THROW(yan_lamma::LinearDispatch(named.second, weight, mode),
                         std::invalid_argument);
        }
    }
}

TEST(LinearTest, RejectsInvalidWeights) {
    const Tensor x = MakeTensor({2}, {1, 2});
    for (const auto& named : InvalidMatrices()) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::Linear(x, named.second), std::invalid_argument);
        for (const auto mode : {MatulMode::Native, MatulMode::Threaded}) {
            SCOPED_TRACE(static_cast<int>(mode));
            EXPECT_THROW(yan_lamma::LinearDispatch(x, named.second, mode),
                         std::invalid_argument);
        }
    }
    const Tensor wrong_inner_dimension = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(yan_lamma::Linear(x, wrong_inner_dimension), std::invalid_argument);
    for (const auto mode : {MatulMode::Native, MatulMode::Threaded}) {
        EXPECT_THROW(yan_lamma::LinearDispatch(x, wrong_inner_dimension, mode),
                     std::invalid_argument);
    }
}

TEST(MatmulTest, MultipliesNonSquareMatrices) {
    const Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b = MakeTensor({3, 2}, {1, 0, 0, 1, 1, 1});
    ExpectTensor(yan_lamma::Matmul(a, b), {2, 2}, {4, 5, 10, 11});
}

TEST(MatmulTest, RejectsInvalidShapeOrStorageOnEitherInput) {
    const Tensor valid = MakeTensor({2, 2}, {1, 2, 3, 4});
    for (const auto& named : InvalidMatrices()) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::Matmul(named.second, valid), std::exception);
        EXPECT_THROW(yan_lamma::Matmul(valid, named.second), std::exception);
        for (const auto mode : {MatulMode::Native, MatulMode::Threaded}) {
            SCOPED_TRACE(static_cast<int>(mode));
            EXPECT_THROW(yan_lamma::MatmulDispatch(named.second, valid, mode),
                         std::invalid_argument);
            EXPECT_THROW(yan_lamma::MatmulDispatch(valid, named.second, mode),
                         std::invalid_argument);
        }
    }
}

TEST(MatmulTest, RejectsMismatchedInnerDimensions) {
    const Tensor a = MakeTensor({2, 2}, {1, 2, 3, 4});
    const Tensor b = MakeTensor({3, 2}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(yan_lamma::Matmul(a, b), std::invalid_argument);
    for (const auto mode : {MatulMode::Native, MatulMode::Threaded}) {
        EXPECT_THROW(yan_lamma::MatmulDispatch(a, b, mode), std::invalid_argument);
    }
}

TEST(MatmulDispatchTest, ExplicitNativeModeReturnsExpectedResults) {
    EXPECT_EQ(yan_lamma::DefaultMatulMode(), MatulMode::Native);
    const Tensor x = MakeTensor({3}, {1, 2, 3});
    const Tensor row = MakeTensor({1, 3}, {1, 2, 3});
    const Tensor weight = MakeTensor({2, 3}, {1, 0, 1, 0, 1, 1});
    ExpectTensor(yan_lamma::LinearDispatch(x, weight, MatulMode::Native), {2}, {4, 5});
    ExpectTensor(yan_lamma::LinearDispatch(row, weight, MatulMode::Native), {1, 2}, {4, 5});
    const Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b = MakeTensor({3, 2}, {1, 0, 0, 1, 1, 1});
    ExpectTensor(yan_lamma::MatmulDispatch(a, b, MatulMode::Native),
                 {2, 2}, {4, 5, 10, 11});
}

TEST(MatmulDispatchTest, ThreadedLinearMatchesNativeAndDoubleReference) {
    RestoreAutomaticThreadCount restore_threads;
    for (const int out_features : {3, 65}) {
        SCOPED_TRACE(out_features);
        Tensor weight({out_features, 64});
        for (std::size_t i = 0; i < weight.data.size(); ++i) {
            weight.data[i] = 0.8f * std::sin(static_cast<float>(i) * 0.13f) - 0.2f;
        }
        for (const auto& shape : {std::vector<int>{64}, std::vector<int>{1, 64}}) {
            SCOPED_TRACE(::testing::PrintToString(shape));
            Tensor input(shape);
            for (std::size_t i = 0; i < input.data.size(); ++i) {
                input.data[i] = 0.7f * std::cos(static_cast<float>(i) * 0.23f);
            }
            const Tensor native =
                yan_lamma::LinearDispatch(input, weight, MatulMode::Native);
            const Tensor default_result = yan_lamma::Linear(input, weight);
            ASSERT_EQ(default_result.shape, native.shape);
            EXPECT_EQ(default_result.data, native.data);
            for (const int threads : {1, 3}) {
                SCOPED_TRACE(threads);
                yan_lamma::SetThreadNum(threads);
                const Tensor actual =
                    yan_lamma::LinearDispatch(input, weight, MatulMode::Threaded);
                ASSERT_EQ(actual.shape, native.shape);
                ASSERT_EQ(actual.data.size(), native.data.size());
                EXPECT_EQ(actual.data, native.data);
                for (int out = 0; out < out_features; ++out) {
                    double expected = 0.0;
                    for (int k = 0; k < 64; ++k) {
                        expected += static_cast<double>(input.data[k]) *
                                    weight.At2(out, k);
                    }
                    ExpectNearDouble(actual.data[out], expected);
                }
            }
        }
    }
}

TEST(MatmulDispatchTest, SimdLinearMatchesDoubleReferenceAcrossVectorTails) {
    RestoreAutomaticThreadCount restore_threads;
    for (const int in_features : {1, 4, 8, 9, 16, 17, 33}) {
        SCOPED_TRACE(in_features);
        for (const int out_features : {3, 65}) {
            SCOPED_TRACE(out_features);
            Tensor weight({out_features, in_features});
            for (int out = 0; out < out_features; ++out) {
                for (int k = 0; k < in_features; ++k) {
                    weight.At2(out, k) =
                        0.2f + 0.01f * out + 0.03f * (k % 5);
                }
            }
            for (const auto& shape : {std::vector<int>{in_features},
                                      std::vector<int>{1, in_features}}) {
                SCOPED_TRACE(::testing::PrintToString(shape));
                Tensor input(shape);
                for (int k = 0; k < in_features; ++k) {
                    input.data[k] = 0.25f + 0.125f * (k % 7);
                }
                // 尾部也保留非零贡献，检查不足一个向量的标量收尾。
                input.data.back() = 3.0f;

                std::vector<double> expected(out_features, 0.0);
                for (int out = 0; out < out_features; ++out) {
                    for (int k = 0; k < in_features; ++k) {
                        expected[out] += static_cast<double>(input.data[k]) *
                                         weight.At2(out, k);
                    }
                }
                const std::vector<int> output_shape = shape.size() == 1
                    ? std::vector<int>{out_features}
                    : std::vector<int>{1, out_features};

                const Tensor simd =
                    yan_lamma::LinearDispatch(input, weight, MatulMode::SIMD);
                ExpectTensor(simd, output_shape, expected);
                for (const int threads : {1, 3}) {
                    SCOPED_TRACE(threads);
                    yan_lamma::SetThreadNum(threads);
                    const Tensor threaded = yan_lamma::LinearDispatch(
                        input, weight, MatulMode::ThreadedSIMD);
                    ExpectTensor(threaded, output_shape, expected);
                    EXPECT_EQ(threaded.data, simd.data);
                }
            }
        }
    }
}

TEST(MatmulDispatchTest, linear_all_modes_match) {
    RestoreAutomaticThreadCount restore_threads;
    EXPECT_EQ(yan_lamma::DefaultMatulMode(), MatulMode::Native);

    for (const int in_features : {1, 3, 4, 5, 7, 8, 9, 16, 17, 33, 64, 86}) {
        SCOPED_TRACE(in_features);
        for (const int out_features : {3, 65}) {
            SCOPED_TRACE(out_features);
            Tensor weight({out_features, in_features});
            for (int out = 0; out < out_features; ++out) {
                for (int k = 0; k < in_features; ++k) {
                    const float magnitude =
                        0.1f + 0.03f * ((k * 5 + out) % 17);
                    weight.At2(out, k) = (out + k) % 2 == 0
                        ? magnitude : -magnitude;
                }
            }

            for (const auto& shape : {std::vector<int>{in_features},
                                      std::vector<int>{1, in_features}}) {
                SCOPED_TRACE(::testing::PrintToString(shape));
                Tensor input(shape);
                for (int k = 0; k < in_features; ++k) {
                    input.data[k] = 0.9f * std::sin(k * 0.31f) -
                                    0.4f * std::cos(k * 0.11f);
                }
                // A nonzero final input and weight make a skipped tail visible.
                input.data.back() = -1.75f;

                std::vector<double> expected(out_features, 0.0);
                std::vector<double> sum_abs(out_features, 0.0);
                for (int out = 0; out < out_features; ++out) {
                    for (int k = 0; k < in_features; ++k) {
                        const double product =
                            static_cast<double>(input.data[k]) *
                            static_cast<double>(weight.At2(out, k));
                        expected[out] += product;
                        sum_abs[out] += std::abs(product);
                    }
                }
                const std::vector<int> output_shape = shape.size() == 1
                    ? std::vector<int>{out_features}
                    : std::vector<int>{1, out_features};

                for (const int threads : {1, 4}) {
                    SCOPED_TRACE(threads);
                    yan_lamma::SetThreadNum(threads);
                    // 65 outputs exceed ParallelFor's four-thread threshold.
                    const Tensor native = yan_lamma::LinearDispatch(
                        input, weight, MatulMode::Native);
                    const Tensor threaded = yan_lamma::LinearDispatch(
                        input, weight, MatulMode::Threaded);
                    const Tensor simd = yan_lamma::LinearDispatch(
                        input, weight, MatulMode::SIMD);
                    const Tensor threaded_simd = yan_lamma::LinearDispatch(
                        input, weight, MatulMode::ThreadedSIMD);
                    const Tensor public_result = yan_lamma::Linear(input, weight);

                    EXPECT_EQ(threaded.data, native.data);
                    EXPECT_EQ(threaded_simd.data, simd.data);
                    EXPECT_EQ(public_result.data, native.data);

                    const std::pair<const char*, const Tensor*> results[] = {
                        {"Native", &native},
                        {"Threaded", &threaded},
                        {"SIMD", &simd},
                        {"ThreadedSIMD", &threaded_simd},
                        {"public Linear", &public_result},
                    };
                    for (const auto& result : results) {
                        SCOPED_TRACE(result.first);
                        ASSERT_EQ(result.second->shape, output_shape);
                        ASSERT_EQ(result.second->data.size(), expected.size());
                        for (int out = 0; out < out_features; ++out) {
                            SCOPED_TRACE(out);
                            EXPECT_TRUE(std::isfinite(result.second->data[out]));
                            EXPECT_NEAR(result.second->data[out], expected[out],
                                        1e-5 * std::max(1.0, sum_abs[out]));
                        }
                    }
                }
            }
        }
    }
}

TEST(MatmulDispatchTest, ThreadedMatmulMatchesNativeAndDoubleReference) {
    RestoreAutomaticThreadCount restore_threads;
    struct Shape {
        int m;
        int k;
        int n;
    };
    for (const auto& shape : {Shape{5, 17, 13}, Shape{1, 17, 65}, Shape{2, 3, 2}}) {
        SCOPED_TRACE("dimensions " + std::to_string(shape.m) + ", " +
                     std::to_string(shape.k) + ", " + std::to_string(shape.n));
        Tensor a({shape.m, shape.k});
        Tensor b({shape.k, shape.n});
        for (std::size_t i = 0; i < a.data.size(); ++i) {
            a.data[i] = 0.8f * std::sin(static_cast<float>(i) * 0.13f) - 0.2f;
        }
        for (std::size_t i = 0; i < b.data.size(); ++i) {
            b.data[i] = 0.7f * std::cos(static_cast<float>(i) * 0.23f);
        }
        const Tensor native = yan_lamma::MatmulDispatch(a, b, MatulMode::Native);
        for (const int threads : {1, 3}) {
            SCOPED_TRACE(threads);
            yan_lamma::SetThreadNum(threads);
            const Tensor actual =
                yan_lamma::MatmulDispatch(a, b, MatulMode::Threaded);
            const Tensor default_result = yan_lamma::Matmul(a, b);
            ASSERT_EQ(actual.shape, native.shape);
            ASSERT_EQ(default_result.shape, actual.shape);
            ASSERT_EQ(actual.data.size(), native.data.size());
            EXPECT_EQ(actual.data, native.data);
            EXPECT_EQ(default_result.data, actual.data);
            for (int row = 0; row < shape.m; ++row) {
                for (int col = 0; col < shape.n; ++col) {
                    double expected = 0.0;
                    for (int k = 0; k < shape.k; ++k) {
                        expected += static_cast<double>(a.At2(row, k)) * b.At2(k, col);
                    }
                    ExpectNearDouble(actual.At2(row, col), expected);
                }
            }
        }
    }
}

TEST(MatmulDispatchTest, ThreadedMatmulRejectsOutputTaskCountOverflow) {
    const Tensor a({46341, 1}, 1.0f);
    const Tensor b({1, 46341}, 1.0f);
    for (const bool dispatch : {false, true}) {
        SCOPED_TRACE(dispatch);
        try {
            if (dispatch) {
                yan_lamma::MatmulDispatch(a, b, MatulMode::Threaded);
            } else {
                yan_lamma::Matmul(a, b);
            }
            FAIL() << "Expected an output task count overflow error";
        } catch (const std::overflow_error& error) {
            EXPECT_NE(std::string(error.what()).find("too many output elements"),
                      std::string::npos);
        }
    }
}

TEST(MatmulDispatchTest, RejectsEveryUnimplementedMode) {
    const Tensor matrix = MakeTensor({2, 2}, {1, 2, 3, 4});
    const std::vector<std::pair<const char*, MatulMode>> modes{
        {"SIMD", MatulMode::SIMD},
        {"ThreadedSIMD", MatulMode::ThreadedSIMD},
    };
    for (const auto& named : modes) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::MatmulDispatch(matrix, matrix, named.second), std::runtime_error);
    }
}

TEST(MatmulDispatchTest, RejectsInvalidEnumValues) {
    const Tensor x = MakeTensor({2}, {1, 2});
    const Tensor matrix = MakeTensor({2, 2}, {1, 2, 3, 4});
    for (int raw_mode : {-1, 999}) {
        SCOPED_TRACE("mode " + std::to_string(raw_mode));
        const MatulMode mode = static_cast<MatulMode>(raw_mode);
        EXPECT_THROW(yan_lamma::LinearDispatch(x, matrix, mode), std::invalid_argument);
        EXPECT_THROW(yan_lamma::MatmulDispatch(matrix, matrix, mode), std::invalid_argument);
    }
}

TEST(NumericalRegressionTest, NativeOperatorsMatchFixedSeedDoubleReferences) {
    std::mt19937 random(20261003);
    std::uniform_int_distribution<int> dimension(1, 8);
    std::uniform_real_distribution<float> value(-3.0f, 3.0f);
    for (int trial = 0; trial < 60; ++trial) {
        const int m = dimension(random);
        const int k = dimension(random);
        const int n = dimension(random);
        SCOPED_TRACE("trial " + std::to_string(trial) + " dimensions " +
                     std::to_string(m) + ", " + std::to_string(k) + ", " +
                     std::to_string(n));
        Tensor a({m, k});
        Tensor b({k, n});
        for (float& element : a.data) { element = value(random); }
        for (float& element : b.data) { element = value(random); }
        const Tensor product = yan_lamma::MatmulDispatch(a, b, MatulMode::Native);
        ASSERT_EQ(product.shape, std::vector<int>({m, n}));
        ASSERT_EQ(product.data.size(), static_cast<std::size_t>(m * n));
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                SCOPED_TRACE("Matmul coordinate " + std::to_string(row) + ", " +
                             std::to_string(col));
                double expected = 0;
                for (int inner = 0; inner < k; ++inner) {
                    expected += static_cast<double>(a.At2(row, inner)) *
                                static_cast<double>(b.At2(inner, col));
                }
                ExpectNearDouble(product.At2(row, col), expected);
            }
        }
        Tensor weights({n, k});
        for (float& element : weights.data) { element = value(random); }
        Tensor row_input({1, k});
        Tensor vector_input({k});
        for (int inner = 0; inner < k; ++inner) {
            row_input.At2(0, inner) = a.At2(0, inner);
            vector_input.At1(inner) = a.At2(0, inner);
        }
        const Tensor row_result =
            yan_lamma::LinearDispatch(row_input, weights, MatulMode::Native);
        const Tensor vector_result =
            yan_lamma::LinearDispatch(vector_input, weights, MatulMode::Native);
        ASSERT_EQ(row_result.shape, std::vector<int>({1, n}));
        ASSERT_EQ(vector_result.shape, std::vector<int>({n}));
        ASSERT_EQ(row_result.data.size(), static_cast<std::size_t>(n));
        ASSERT_EQ(vector_result.data.size(), static_cast<std::size_t>(n));
        for (int output = 0; output < n; ++output) {
            SCOPED_TRACE("Linear output " + std::to_string(output));
            double expected = 0;
            for (int inner = 0; inner < k; ++inner) {
                expected += static_cast<double>(row_input.At2(0, inner)) *
                            static_cast<double>(weights.At2(output, inner));
            }
            ExpectNearDouble(row_result.At2(0, output), expected);
            ExpectNearDouble(vector_result.At1(output), expected);
        }
    }
}

TEST(ElementwiseMulTest, MultipliesCorrespondingElementsAndPreservesShape) {
    ExpectTensor(yan_lamma::ElementwiseMul(MakeTensor({2, 2}, {2, -3, 0, 0.5f}),
                                          MakeTensor({2, 2}, {4, 5, 7, -2})),
                 {2, 2}, {8, -15, 0, -1});
}

TEST(ElementwiseMulTest, RejectsDifferentShapesEvenWithEqualElementCount) {
    EXPECT_THROW(yan_lamma::ElementwiseMul(MakeTensor({2}, {1, 2}),
                                          MakeTensor({1, 2}, {1, 2})),
                 std::runtime_error);
}

TEST(SoftMaxTest, ConvertsScoresToPositiveWeightsSummingToOne) {
    const Tensor result = yan_lamma::SoftMax(MakeTensor({3}, {2, 1, 0}));
    const double denominator = std::exp(2.0) + std::exp(1.0) + 1.0;
    ExpectTensor(result, {3}, {std::exp(2.0) / denominator,
                             std::exp(1.0) / denominator, 1.0 / denominator});
    double sum = 0.0;
    for (float element : result.data) {
        EXPECT_GT(element, 0.0f);
        sum += element;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST(SoftMaxTest, RemainsFiniteWhenScoresHaveLargeCommonOffset) {
    const double denominator = 1.0 + std::exp(-1.0) + std::exp(-2.0);
    ExpectTensor(yan_lamma::SoftMax(MakeTensor({3}, {10000, 9999, 9998})),
                 {3}, {1.0 / denominator, std::exp(-1.0) / denominator,
                       std::exp(-2.0) / denominator});
    ExpectTensor(yan_lamma::SoftMax(MakeTensor({3}, {10000, 10000, 10000})),
                 {3}, {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
}

TEST(SoftMaxTest, RejectsTwoDimensionalInput) {
    EXPECT_THROW(yan_lamma::SoftMax(MakeTensor({1, 3}, {2, 1, 0})),
                 std::runtime_error);
}

TEST(SiluTest, AppliesActivationElementwiseAndPreservesTwoDimensionalShape) {
    ExpectTensor(yan_lamma::Silu(MakeTensor({2, 2}, {-2, 0, 2, 1})),
                 {2, 2}, {-2.0 / (1.0 + std::exp(2.0)), 0,
                          2.0 / (1.0 + std::exp(-2.0)),
                          1.0 / (1.0 + std::exp(-1.0))});
}

TEST(SwiGluTest, MultipliesActivatedGateByUpVector) {
    ExpectTensor(yan_lamma::SwiGlu(MakeTensor({3}, {-2, 0, 2}),
                                   MakeTensor({3}, {3, 4, 5})),
                 {3}, {-6.0 / (1.0 + std::exp(2.0)), 0,
                       10.0 / (1.0 + std::exp(-2.0))});
}

TEST(SwiGluTest, RejectsDifferentGateAndUpShapes) {
    EXPECT_THROW(yan_lamma::SwiGlu(MakeTensor({3}, {-2, 0, 2}),
                                  MakeTensor({1, 3}, {3, 4, 5})),
                 std::runtime_error);
}


TEST(RopeTest, PositionZeroLeavesBothLayoutsUnchanged) {
    for (auto type : {yan_lamma::RopeType::kNormal, yan_lamma::RopeType::kNeoX}) {
        SCOPED_TRACE(static_cast<int>(type));
        Tensor q = MakeTensor({2, 4}, {1, 2, 3, 4, -2, 0.5f, 1, -3});
        Tensor k = MakeTensor({1, 4}, {5, -6, 7, -8});
        const Tensor original_q = q;
        const Tensor original_k = k;
        yan_lamma::Rope(q, k, 0, 10000.0f, type);
        EXPECT_EQ(q.shape, original_q.shape);
        EXPECT_EQ(k.shape, original_k.shape);
        EXPECT_EQ(q.data, original_q.data);
        EXPECT_EQ(k.data, original_k.data);
    }
}

TEST(RopeTest, NormalMatchesFixedMathematicalValues) {
    Tensor q = MakeTensor({1, 4}, {1, 2, 3, 4});
    Tensor k = MakeTensor({1, 4}, {-2, 0.5f, 1, -3});
    // pos=1, theta=100: adjacent pairs rotate by 1 and 0.1 radians.
    yan_lamma::Rope(q, k, 1, 100.0f, yan_lamma::RopeType::kNormal);
    ExpectTensor(q, {1, 4}, {-1.1426396637476532, 1.922075596544176,
                             2.585678829246765, 4.279516911052588});
    ExpectTensor(k, {1, 4}, {-1.501340104140228, -1.4127908166817231,
                             1.2945044152185103, -2.8851790791872496});
}

TEST(RopeTest, NeoXMatchesFixedMathematicalValues) {
    Tensor q = MakeTensor({1, 4}, {1, 2, 3, 4});
    Tensor k = MakeTensor({1, 4}, {-2, 0.5f, 1, -3});
    // Half-region pairs (0,2) and (1,3) rotate by 1 and 0.1 radians.
    yan_lamma::Rope(q, k, 1, 100.0f, yan_lamma::RopeType::kNeoX);
    ExpectTensor(q, {1, 4}, {-1.9841106485555495, 1.590674663968739,
                             2.4623779024123156, 4.17968349440576});
    ExpectTensor(k, {1, 4}, {-1.922075596544176, 0.7970023325794974,
                             -1.1426396637476532, -2.9350957875106634});
}

TEST(RopeTest, DifferentQueryAndKeyHeadCountsMatchIndependentHeads) {
    const Tensor original_q = MakeTensor(
        {3, 4}, {1, 2, 3, 4, -2, 0.5f, 1, -3, 4, -1, 2, 0.25f});
    const Tensor original_k = MakeTensor({2, 4}, {3, 2, 1, 0, -1, 2, -3, 4});
    for (auto type : {yan_lamma::RopeType::kNormal, yan_lamma::RopeType::kNeoX}) {
        SCOPED_TRACE(static_cast<int>(type));
        Tensor q = original_q;
        Tensor k = original_k;
        yan_lamma::Rope(q, k, 7, 10000.0f, type);
        std::vector<double> expected_q;
        std::vector<double> expected_k;
        for (int head = 0; head < original_q.shape[0]; ++head) {
            const auto first = original_q.data.begin() + head * 4;
            Tensor one_q = MakeTensor({1, 4}, std::vector<float>(first, first + 4));
            Tensor one_k = MakeTensor({1, 4}, {0, 0, 0, 0});
            yan_lamma::Rope(one_q, one_k, 7, 10000.0f, type);
            expected_q.insert(expected_q.end(), one_q.data.begin(), one_q.data.end());
        }
        for (int head = 0; head < original_k.shape[0]; ++head) {
            const auto first = original_k.data.begin() + head * 4;
            Tensor one_q = MakeTensor({1, 4}, {0, 0, 0, 0});
            Tensor one_k = MakeTensor({1, 4}, std::vector<float>(first, first + 4));
            yan_lamma::Rope(one_q, one_k, 7, 10000.0f, type);
            expected_k.insert(expected_k.end(), one_k.data.begin(), one_k.data.end());
        }
        ExpectTensor(q, {3, 4}, expected_q);
        ExpectTensor(k, {2, 4}, expected_k);
    }
}

TEST(RopeTest, RotationsComposeAndPreserveEachHeadNorm) {
    const Tensor original_q = MakeTensor({2, 4}, {1, 2, -3, 4, -2, 0.5f, 1, -3});
    const Tensor original_k = MakeTensor({1, 4}, {3, -2, 1, 0.25f});
    for (auto type : {yan_lamma::RopeType::kNormal, yan_lamma::RopeType::kNeoX}) {
        SCOPED_TRACE(static_cast<int>(type));
        Tensor q = original_q;
        Tensor k = original_k;
        Tensor combined_q = original_q;
        Tensor combined_k = original_k;
        yan_lamma::Rope(q, k, 3, 100.0f, type);
        yan_lamma::Rope(q, k, 4, 100.0f, type);
        yan_lamma::Rope(combined_q, combined_k, 7, 100.0f, type);
        ExpectTensor(q, original_q.shape,
                     std::vector<double>(combined_q.data.begin(), combined_q.data.end()));
        ExpectTensor(k, original_k.shape,
                     std::vector<double>(combined_k.data.begin(), combined_k.data.end()));
        for (const auto& tensors : {std::make_pair(original_q, q),
                                   std::make_pair(original_k, k)}) {
            for (int head = 0; head < tensors.first.shape[0]; ++head) {
                SCOPED_TRACE("head " + std::to_string(head));
                double before = 0.0;
                double after = 0.0;
                for (int dim = 0; dim < 4; ++dim) {
                    const double a = tensors.first.At2(head, dim);
                    const double b = tensors.second.At2(head, dim);
                    before += a * a;
                    after += b * b;
                }
                ExpectNearDouble(after, before);
            }
        }
    }
}

TEST(RopeTest, RejectsInvalidArgumentsBeforeModifyingEitherTensor) {
    const Tensor valid_q = MakeTensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    const Tensor valid_k = MakeTensor({1, 4}, {-1, 2, -3, 4});
    const auto check_rejected = [](Tensor q, Tensor k, int pos, float theta,
                                   yan_lamma::RopeType type) {
        const Tensor before_q = q;
        const Tensor before_k = k;
        EXPECT_THROW(yan_lamma::Rope(q, k, pos, theta, type), std::exception);
        EXPECT_EQ(q.shape, before_q.shape);
        EXPECT_EQ(k.shape, before_k.shape);
        EXPECT_EQ(q.data, before_q.data);
        EXPECT_EQ(k.data, before_k.data);
    };
    const std::vector<NamedTensor> malformed{
        {"empty", Tensor{}},
        {"scalar rank", Malformed({}, {1})},
        {"vector rank", MakeTensor({4}, {1, 2, 3, 4})},
        {"three-dimensional rank", MakeTensor({1, 1, 4}, {1, 2, 3, 4})},
        {"zero heads", Malformed({0, 4})},
        {"negative heads", Malformed({-1, 4})},
        {"zero head dimension", Malformed({1, 0})},
        {"negative head dimension", Malformed({1, -4})},
        {"mismatched head dimension", MakeTensor({1, 2}, {1, 2})},
        {"short storage", Malformed({1, 4}, {1, 2, 3})},
        {"long storage", Malformed({1, 4}, {1, 2, 3, 4, 5})},
    };
    for (const auto& named : malformed) {
        SCOPED_TRACE(named.first);
        check_rejected(named.second, valid_k, 2, 10000.0f, yan_lamma::RopeType::kNormal);
        check_rejected(valid_q, named.second, 2, 10000.0f, yan_lamma::RopeType::kNeoX);
    }
    check_rejected(MakeTensor({1, 3}, {1, 2, 3}), MakeTensor({1, 3}, {4, 5, 6}),
                   2, 10000.0f, yan_lamma::RopeType::kNormal);
    for (int pos : {-1, std::numeric_limits<int>::min()}) {
        SCOPED_TRACE("pos " + std::to_string(pos));
        check_rejected(valid_q, valid_k, pos, 10000.0f, yan_lamma::RopeType::kNormal);
    }
    for (float theta : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity()}) {
        SCOPED_TRACE("theta " + std::to_string(theta));
        check_rejected(valid_q, valid_k, 2, theta, yan_lamma::RopeType::kNeoX);
    }
}

}  // namespace
