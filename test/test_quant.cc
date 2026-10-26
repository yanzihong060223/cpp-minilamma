#include "include/quant.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {

struct MatmulError {
    float max_abs;
    double mean_abs;
};

// 比较矩阵乘法结果的最大绝对误差和平均绝对误差。
MatmulError CompareMatmulError(const Tensor& expected, const Tensor& actual) {
    if (expected.shape != actual.shape ||
        expected.data.size() != actual.data.size()) {
        throw std::invalid_argument("CompareMatmulError: shape or size mismatch");
    }

    float max_abs = 0.0f;
    double sum_abs = 0.0;
    for (std::size_t i = 0; i < expected.data.size(); ++i) {
        const float error = std::abs(expected.data[i] - actual.data[i]);
        if (!std::isfinite(error)) {
            return {std::numeric_limits<float>::infinity(),
                    std::numeric_limits<double>::infinity()};
        }
        max_abs = std::max(max_abs, error);
        sum_abs += error;
    }

    return {max_abs, expected.data.empty() ? 0.0
                                         : sum_abs / expected.data.size()};
}

TEST(QuantTest, q8_0_block_layout) {
    EXPECT_EQ(kQuantBlockSize, 32u);
    EXPECT_EQ(sizeof(Block_Q80), 34u);
    EXPECT_EQ(sizeof(Block_Q80::d), 2u);
    EXPECT_EQ(sizeof(Block_Q80::qs), 32u);
}

TEST(QuantTest, q8_0_roundtrip_identity) {
    Tensor source({2, 32});
    for (std::size_t i = 0; i < source.data.size(); ++i) {
        source.data[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.3f;
    }
    // 包含首元素就是最大绝对值的情况，以及负数、小数和零。
    source.data[0] = -4.0f;
    source.data[32] = 4.0f;

    const auto blocks = TensorToQ80(source);
    ASSERT_EQ(blocks.size(), 2u);
    const Tensor restored = Q80ToTensor(blocks, source.shape);
    ASSERT_EQ(restored.shape, source.shape);
    ASSERT_EQ(restored.data.size(), source.data.size());
    // 4 / 127 的半个量化步长约为 0.01575，额外留出 FP16 scale 的误差。
    for (std::size_t i = 0; i < source.data.size(); ++i) {
        EXPECT_NEAR(restored.data[i], source.data[i], 0.02f) << "index=" << i;
    }
}

TEST(QuantTest, q8_0_all_zeros) {
    const Tensor source({64}, 0.0f);
    const auto blocks = TensorToQ80(source);
    ASSERT_EQ(blocks.size(), 2u);
    for (const auto& block : blocks) {
        EXPECT_EQ(block.d, 0u);
        for (const auto value : block.qs) {
            EXPECT_EQ(value, 0);
        }
    }

    const Tensor restored = Q80ToTensor(blocks, source.shape);
    EXPECT_EQ(restored.shape, source.shape);
    EXPECT_EQ(restored.data, source.data);
}

TEST(QuantTest, q8_0_multi_block) {
    Tensor source({100});
    for (std::size_t i = 0; i < source.data.size(); ++i) {
        source.data[i] = static_cast<float>(static_cast<int>(i) - 50) / 16.0f;
    }

    const auto blocks = TensorToQ80(source);
    ASSERT_EQ(blocks.size(), 4u);
    // 此用例保留待验证的尾块补零要求；当前实现只接受完整块，会在上面失败。
    for (std::size_t i = 4; i < kQuantBlockSize; ++i) {
        EXPECT_EQ(blocks.back().qs[i], 0) << "padding index=" << i;
    }

    const Tensor restored = Q80ToTensor(blocks, source.shape);
    ASSERT_EQ(restored.shape, source.shape);
    ASSERT_EQ(restored.data.size(), 100u);
    for (std::size_t i = 0; i < source.data.size(); ++i) {
        EXPECT_NEAR(restored.data[i], source.data[i], 0.02f) << "index=" << i;
    }

}

TEST(QuantTest, q8_0_empty_tensor) {
    // Tensor 拒绝零长度维度，默认构造的空对象仍然允许存在。
    EXPECT_THROW(Tensor({0}), std::runtime_error);
    EXPECT_THROW(Tensor({2, 0}), std::runtime_error);
}

TEST(QuantTest, matmul_q8_0_matches_f32) {
    Tensor lhs({3, 64});
    Tensor rhs({64, 5});
    for (std::size_t i = 0; i < lhs.data.size(); ++i) {
        lhs.data[i] = 1.5f * std::sin(static_cast<float>(i) * 0.13f) +
                      0.2f * std::cos(static_cast<float>(i) * 0.31f);
    }
    for (std::size_t i = 0; i < rhs.data.size(); ++i) {
        rhs.data[i] = 0.5f * std::sin(static_cast<float>(i) * 0.23f);
    }

    const auto blocks = TensorToQ80(lhs);
    ASSERT_EQ(blocks.size(), 6u);
    const Tensor expected = Matmul(lhs, rhs);
    const Tensor actual = MatmulQ80(blocks, rhs, lhs.shape);
    ASSERT_EQ(actual.shape, (std::vector<int>{3, 5}));
    ASSERT_EQ(actual.data.size(), expected.data.size());

    const MatmulError error = CompareMatmulError(expected, actual);
    EXPECT_LT(error.max_abs, 0.05f);
    EXPECT_LT(error.mean_abs, 0.02);
}

TEST(QuantTest, compare_matmul_error) {
    Tensor expected({2, 2});
    expected.data = {0.0f, 1.0f, -2.0f, 0.5f};
    Tensor actual({2, 2});
    actual.data = {0.0f, 1.25f, -1.5f, 0.25f};

    const MatmulError error = CompareMatmulError(expected, actual);
    EXPECT_FLOAT_EQ(error.max_abs, 0.5f);
    EXPECT_DOUBLE_EQ(error.mean_abs, 0.25);
    const MatmulError identical = CompareMatmulError(expected, expected);
    EXPECT_FLOAT_EQ(identical.max_abs, 0.0f);
    EXPECT_DOUBLE_EQ(identical.mean_abs, 0.0);

    EXPECT_THROW(CompareMatmulError(expected, Tensor({4})), std::invalid_argument);
    actual.data.pop_back();
    EXPECT_THROW(CompareMatmulError(expected, actual), std::invalid_argument);
}

struct QuantErrorFormat {
    const char* name;
    Tensor (*roundtrip)(const Tensor&);
    float (*compare_matmul)(const Tensor&, const Tensor&);
    float (*compare_linear)(const Tensor&, const Tensor&);
};

const QuantErrorFormat kQuantErrorFormats[] = {
    {"Q80",
     [](const Tensor& source) {
         return Q80ToTensor(TensorToQ80(source), source.shape);
     },
     CompareMatmulErrorQ80, CompareLinearErrorQ80},
    {"Q40",
     [](const Tensor& source) {
         return Q40ToTensor(TensorToQ40(source), source.shape);
     },
     CompareMatmulErrorQ40, CompareLinearErrorQ40},
    {"Q41",
     [](const Tensor& source) {
         return Q41ToTensor(TensorToQ41(source), source.shape);
     },
     CompareMatmulErrorQ41, CompareLinearErrorQ41}
};

TEST(QuantTest, compare_matmul_errors_match_max_abs) {
    Tensor weight({3, 64});
    Tensor input({64, 4});
    for (std::size_t i = 0; i < weight.data.size(); ++i) {
        weight.data[i] = 1.7f * std::sin(static_cast<float>(i) * 0.17f) - 0.3f;
    }
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = 0.6f * std::cos(static_cast<float>(i) * 0.23f);
    }

    const Tensor reference = Matmul(weight, input);
    for (const auto& format : kQuantErrorFormats) {
        SCOPED_TRACE(format.name);
        const Tensor quantized = Matmul(format.roundtrip(weight), input);
        const float expected = CompareMatmulError(reference, quantized).max_abs;
        ASSERT_GT(expected, 0.0f);
        EXPECT_FLOAT_EQ(format.compare_matmul(weight, input), expected);
    }
}

TEST(QuantTest, compare_linear_errors_match_max_abs) {
    for (const int in_features : {32, 64}) {
        SCOPED_TRACE(in_features);
        Tensor weight({3, in_features});
        for (std::size_t i = 0; i < weight.data.size(); ++i) {
            weight.data[i] =
                1.7f * std::sin(static_cast<float>(i) * 0.17f) - 0.3f;
        }

        for (const auto& shape : {std::vector<int>{in_features},
                                  std::vector<int>{1, in_features}}) {
            SCOPED_TRACE(::testing::PrintToString(shape));
            Tensor input(shape);
            for (std::size_t i = 0; i < input.data.size(); ++i) {
                input.data[i] = 0.6f * std::cos(static_cast<float>(i) * 0.23f);
            }

            const Tensor reference = Linear(input, weight);
            for (const auto& format : kQuantErrorFormats) {
                SCOPED_TRACE(format.name);
                const Tensor quantized = Linear(input, format.roundtrip(weight));
                const float expected =
                    CompareMatmulError(reference, quantized).max_abs;
                ASSERT_GT(expected, 0.0f);
                EXPECT_FLOAT_EQ(format.compare_linear(weight, input), expected);
            }
        }
    }
}

TEST(QuantTest, compare_errors_return_zero_for_zero_inputs) {
    Tensor weight({3, 64});
    for (std::size_t i = 0; i < weight.data.size(); ++i) {
        weight.data[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.3f;
    }
    const Tensor matmul_input({64, 4}, 0.0f);
    const Tensor linear_input({64}, 0.0f);
    for (const auto& format : kQuantErrorFormats) {
        SCOPED_TRACE(format.name);
        EXPECT_FLOAT_EQ(format.compare_matmul(weight, matmul_input), 0.0f);
        EXPECT_FLOAT_EQ(format.compare_linear(weight, linear_input), 0.0f);
    }
}

TEST(QuantTest, compare_errors_reject_non_finite_differences) {
    // 输入均有限，但量化 scale 超出 FP16 范围会产生非有限结果。
    Tensor weight({1, 32}, 0.0f);
    weight.data[0] = 1.0e10f;
    const Tensor matmul_input({32, 1}, 1.0f);
    const Tensor linear_input({32}, 1.0f);
    for (const auto& format : kQuantErrorFormats) {
        SCOPED_TRACE(format.name);
        for (const bool is_linear : {false, true}) {
            SCOPED_TRACE(is_linear ? "Linear" : "Matmul");
            try {
                if (is_linear) {
                    format.compare_linear(weight, linear_input);
                } else {
                    format.compare_matmul(weight, matmul_input);
                }
                FAIL() << "Expected a non-finite difference error";
            } catch (const std::runtime_error& error) {
                EXPECT_NE(std::string(error.what()).find("non-finite"),
                          std::string::npos);
            }
        }
    }
}

TEST(QuantTest, dequantize_rejects_bad_shape) {
    const std::vector<Block_Q80> blocks(1);
    const std::vector<std::vector<int>> invalid_shapes = {
        {}, {0}, {-32}, {2, 0}, {2, -16}
    };
    for (const auto& shape : invalid_shapes) {
        SCOPED_TRACE(::testing::PrintToString(shape));
        EXPECT_THROW(Q80ToTensor(blocks, shape), std::runtime_error);
    }
}

TEST(QuantTest, dequantize_rejects_block_count_mismatch) {
    const std::vector<Block_Q80> one_block(1);
    const std::vector<Block_Q80> two_blocks(2);
    const std::vector<Block_Q80> no_blocks;
    EXPECT_THROW(Q80ToTensor(one_block, {64}), std::runtime_error);
    EXPECT_THROW(Q80ToTensor(two_blocks, {32}), std::runtime_error);
    EXPECT_THROW(Q80ToTensor(no_blocks, {32}), std::runtime_error);
}

}  // namespace
}  // namespace yan_lamma
