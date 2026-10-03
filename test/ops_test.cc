#include "include/ops.h"

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
        EXPECT_THROW(yan_lamma::LinearDispatch(named.second, weight, MatulMode::Native),
                     std::invalid_argument);
    }
}

TEST(LinearTest, RejectsInvalidWeights) {
    const Tensor x = MakeTensor({2}, {1, 2});
    for (const auto& named : InvalidMatrices()) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::Linear(x, named.second), std::invalid_argument);
        EXPECT_THROW(yan_lamma::LinearDispatch(x, named.second, MatulMode::Native),
                     std::invalid_argument);
    }
    const Tensor wrong_inner_dimension = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(yan_lamma::Linear(x, wrong_inner_dimension), std::invalid_argument);
    EXPECT_THROW(yan_lamma::LinearDispatch(x, wrong_inner_dimension, MatulMode::Native),
                 std::invalid_argument);
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
        EXPECT_THROW(yan_lamma::MatmulDispatch(named.second, valid, MatulMode::Native),
                     std::invalid_argument);
        EXPECT_THROW(yan_lamma::MatmulDispatch(valid, named.second, MatulMode::Native),
                     std::invalid_argument);
    }
}

TEST(MatmulTest, RejectsMismatchedInnerDimensions) {
    const Tensor a = MakeTensor({2, 2}, {1, 2, 3, 4});
    const Tensor b = MakeTensor({3, 2}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(yan_lamma::Matmul(a, b), std::invalid_argument);
    EXPECT_THROW(yan_lamma::MatmulDispatch(a, b, MatulMode::Native), std::invalid_argument);
}

TEST(MatmulDispatchTest, DefaultsToNativeAndReturnsNativeResults) {
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

TEST(MatmulDispatchTest, RejectsEveryUnimplementedMode) {
    const Tensor x = MakeTensor({2}, {1, 2});
    const Tensor matrix = MakeTensor({2, 2}, {1, 2, 3, 4});
    const std::vector<std::pair<const char*, MatulMode>> modes{
        {"Threaded", MatulMode::Threaded},
        {"SIMD", MatulMode::SIMD},
        {"ThreadedSIMD", MatulMode::ThreadedSIMD},
    };
    for (const auto& named : modes) {
        SCOPED_TRACE(named.first);
        EXPECT_THROW(yan_lamma::Linear(x, matrix, named.second), std::runtime_error);
        EXPECT_THROW(yan_lamma::LinearDispatch(x, matrix, named.second), std::runtime_error);
        EXPECT_THROW(yan_lamma::Matmul(matrix, matrix, named.second), std::runtime_error);
        EXPECT_THROW(yan_lamma::MatmulDispatch(matrix, matrix, named.second), std::runtime_error);
    }
}

TEST(MatmulDispatchTest, RejectsInvalidEnumValues) {
    const Tensor x = MakeTensor({2}, {1, 2});
    const Tensor matrix = MakeTensor({2, 2}, {1, 2, 3, 4});
    for (int raw_mode : {-1, 999}) {
        SCOPED_TRACE("mode " + std::to_string(raw_mode));
        const MatulMode mode = static_cast<MatulMode>(raw_mode);
        EXPECT_THROW(yan_lamma::Linear(x, matrix, mode), std::invalid_argument);
        EXPECT_THROW(yan_lamma::LinearDispatch(x, matrix, mode), std::invalid_argument);
        EXPECT_THROW(yan_lamma::Matmul(matrix, matrix, mode), std::invalid_argument);
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
        const Tensor product = yan_lamma::Matmul(a, b);
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
        const Tensor row_result = yan_lamma::Linear(row_input, weights);
        const Tensor vector_result = yan_lamma::Linear(vector_input, weights);
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

}  // namespace
