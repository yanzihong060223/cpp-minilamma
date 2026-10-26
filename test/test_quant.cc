#include "include/quant.h"
#include "thread_pool/thread_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

TEST(QuantTest, q8_0_empty_tensor) {
    // Tensor 拒绝零长度维度，默认构造的空对象仍然允许存在。
    EXPECT_THROW(Tensor({0}), std::runtime_error);
    EXPECT_THROW(Tensor({2, 0}), std::runtime_error);
}

template<typename BlockType>
void CheckPartialQuantization(
    std::vector<BlockType> (*quantize)(const Tensor&),
    Tensor (*dequantize)(const std::vector<BlockType>&, const std::vector<int>&),
    Tensor (*matmul)(const std::vector<BlockType>&, const Tensor&,
                     const std::vector<int>&),
    float tolerance) {
    const std::vector<std::vector<int>> shapes = {
        {100}, {32, 86}, {2, 35}, {2, 48}, {2, 49}, {2, 2, 17}
    };
    const std::vector<std::size_t> expected_blocks = {4, 96, 4, 4, 4, 4};

    for (std::size_t test = 0; test < shapes.size(); ++test) {
        SCOPED_TRACE(::testing::PrintToString(shapes[test]));
        Tensor source(shapes[test]);
        const int cols = source.shape.back();
        const std::size_t rows = source.data.size() / cols;
        for (std::size_t row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                // 相邻行符号不同，避免跨行打包或尾块覆盖被相同数据掩盖。
                const float value = 0.8f + static_cast<float>(row) * 0.005f +
                    0.2f * std::sin(static_cast<float>(col) * 0.17f) +
                    0.1f * std::cos(static_cast<float>(col) * 0.11f);
                source.data[row * cols + col] = row % 2 == 0 ? value : -value;
            }
        }

        const auto blocks = quantize(source);
        ASSERT_EQ(blocks.size(), expected_blocks[test]);
        const Tensor restored = dequantize(blocks, source.shape);
        ASSERT_EQ(restored.shape, source.shape);
        ASSERT_EQ(restored.data.size(), source.data.size());
        for (std::size_t i = 0; i < source.data.size(); ++i) {
            EXPECT_NEAR(restored.data[i], source.data[i], tolerance)
                << "index=" << i;
        }

        auto wrong_blocks = blocks;
        wrong_blocks.pop_back();
        EXPECT_THROW(dequantize(wrong_blocks, source.shape), std::runtime_error);
        wrong_blocks = blocks;
        wrong_blocks.emplace_back();
        EXPECT_THROW(dequantize(wrong_blocks, source.shape), std::runtime_error);

        if (source.shape.size() == 2) {
            Tensor rhs({cols, 3});
            for (std::size_t i = 0; i < rhs.data.size(); ++i) {
                rhs.data[i] = 0.05f * std::sin(static_cast<float>(i) * 0.23f);
            }
            const Tensor actual = matmul(blocks, rhs, source.shape);
            const Tensor expected = Matmul(restored, rhs);
            EXPECT_EQ(actual.shape, (std::vector<int>{source.shape[0], 3}));
            EXPECT_EQ(actual.data, expected.data);
        }
    }

    EXPECT_TRUE(quantize(Tensor()).empty());
    Tensor malformed({2, 35});
    malformed.data.pop_back();
    EXPECT_THROW(quantize(malformed), std::runtime_error);

    Tensor missing_data({35});
    missing_data.data.clear();
    EXPECT_THROW(quantize(missing_data), std::runtime_error);

    const int large = std::numeric_limits<int>::max();
    const std::vector<std::vector<int>> invalid_shapes = {
        {}, {0}, {-1}, {2, 0}, {2, -1}, {large, large, large}
    };
    for (const auto& shape : invalid_shapes) {
        SCOPED_TRACE(::testing::PrintToString(shape));
        // 仅设置元数据，不分配非法或巨大的张量。
        Tensor invalid;
        invalid.shape = shape;
        invalid.data = {1.0f};
        EXPECT_THROW(quantize(invalid), std::runtime_error);
        EXPECT_THROW(dequantize(std::vector<BlockType>{}, shape),
                     std::runtime_error);
    }
}

TEST(QuantTest, q80_partial_blocks_preserve_rows) {
    CheckPartialQuantization<Block_Q80>(
        TensorToQ80, Q80ToTensor, MatmulQ80, 0.01f);

    const auto blocks = TensorToQ80(Tensor({100}, 0.5f));
    ASSERT_EQ(blocks.size(), 4u);
    for (std::size_t i = 4; i < kQuantBlockSize; ++i) {
        EXPECT_EQ(blocks.back().qs[i], 0);
    }
}

TEST(QuantTest, q40_partial_blocks_preserve_rows) {
    CheckPartialQuantization<Block_Q40>(
        TensorToQ40, Q40ToTensor, MatmulQ40, 0.1f);

    const auto blocks = TensorToQ40(Tensor({100}, 0.5f));
    ASSERT_EQ(blocks.size(), 4u);
    for (std::size_t i = 0; i < 16; ++i) {
        // Q40 的浮点零编码为 8；末块只有前四个低半元素有效。
        EXPECT_EQ(blocks.back().qs[i] >> 4, 8);
        if (i >= 4) {
            EXPECT_EQ(blocks.back().qs[i] & 0x0F, 8);
        }
    }
}

TEST(QuantTest, q41_partial_blocks_preserve_rows) {
    CheckPartialQuantization<Block_Q41>(
        TensorToQ41, Q41ToTensor, MatmulQ41, 0.05f);

    const auto blocks = TensorToQ41(Tensor({100}, 0.5f));
    ASSERT_EQ(blocks.size(), 4u);
    // 全正数尾块也包含补入的零，因此最小值为零、scale 非零。
    EXPECT_EQ(blocks.back().m, 0u);
    EXPECT_NE(blocks.back().d, 0u);
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
                const Tensor decoded = format.roundtrip(weight);
                const Tensor quantized = Linear(input, decoded);
                const float expected =
                    CompareMatmulError(reference, quantized).max_abs;
                ASSERT_GT(expected, 0.0f);

                // SIMD 改变累加顺序，最大误差允许对应点积的舍入误差。
                double tolerance = 0.0;
                for (int out = 0; out < weight.shape[0]; ++out) {
                    double sum_abs = 0.0;
                    for (int k = 0; k < in_features; ++k) {
                        sum_abs += std::abs(
                            static_cast<double>(input.data[k]) *
                            decoded.data[out * in_features + k]);
                    }
                    tolerance = std::max(
                        tolerance, 1e-5 * std::max(1.0, sum_abs));
                }
                EXPECT_NEAR(format.compare_linear(weight, input), expected,
                            tolerance);
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

struct RestoreAutomaticThreadCount {
    ~RestoreAutomaticThreadCount() { SetThreadNum(0); }
};

// 使用数学表达式独立解释 FP16，避免参考值经过生产反量化路径。
double ReferenceFp16(uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    double value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<double>(fraction), -24);
    } else if (exponent == 31) {
        value = fraction == 0 ? std::numeric_limits<double>::infinity()
                              : std::numeric_limits<double>::quiet_NaN();
    } else {
        value = std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
    }
    return (bits & 0x8000u) != 0 ? -value : value;
}

double ReferenceQuantWeight(const Block_Q80& block, int index) {
    return ReferenceFp16(block.d) * static_cast<int>(block.qs[index]);
}

double ReferenceQuantWeight(const Block_Q40& block, int index) {
    const int code = index < 16 ? (block.qs[index] & 15)
                                : (block.qs[index - 16] >> 4);
    return ReferenceFp16(block.d) * (code - 8);
}

double ReferenceQuantWeight(const Block_Q41& block, int index) {
    const int code = index < 16 ? (block.qs[index] & 15)
                                : (block.qs[index - 16] >> 4);
    return ReferenceFp16(block.d) * code + ReferenceFp16(block.m);
}

template<typename BlockType>
void ExpectQuantLinearReference(
    const Tensor& actual, const Tensor& input,
    const std::vector<BlockType>& blocks, const std::vector<int>& weight_shape) {
    const int rows = input.shape.size() == 1 ? 1 : input.shape[0];
    const int out_features = weight_shape[0];
    const int in_features = weight_shape[1];
    const std::size_t blocks_per_row =
        1 + (static_cast<std::size_t>(in_features) - 1) / kQuantBlockSize;
    const std::vector<int> expected_shape = input.shape.size() == 1
        ? std::vector<int>{out_features} : std::vector<int>{rows, out_features};
    ASSERT_EQ(actual.shape, expected_shape);
    ASSERT_EQ(actual.data.size(), static_cast<std::size_t>(rows) * out_features);

    for (int row = 0; row < rows; ++row) {
        for (int out = 0; out < out_features; ++out) {
            double expected = 0.0;
            double sum_abs = 0.0;
            for (int k = 0; k < in_features; ++k) {
                const auto& block = blocks[
                    static_cast<std::size_t>(out) * blocks_per_row +
                    static_cast<std::size_t>(k) / kQuantBlockSize];
                const double product =
                    static_cast<double>(input.data[
                        static_cast<std::size_t>(row) * in_features + k]) *
                    ReferenceQuantWeight(
                        block, static_cast<int>(k % kQuantBlockSize));
                expected += product;
                sum_abs += std::abs(product);
            }
            EXPECT_NEAR(actual.data[row * out_features + out], expected,
                        1e-5 * std::max(1.0, sum_abs))
                << "row=" << row << ", out=" << out;
        }
    }
}

template<typename BlockType>
void CheckThreadedQuantLinear(
    std::vector<BlockType> (*quantize)(const Tensor&),
    Tensor (*linear)(const Tensor&, const std::vector<BlockType>&,
                     const std::vector<int>&),
    int in_features) {
    RestoreAutomaticThreadCount restore_threads;
    Tensor weight({65, in_features});
    for (std::size_t i = 0; i < weight.data.size(); ++i) {
        weight.data[i] = 1.5f * std::sin(static_cast<float>(i) * 0.031f) +
                        0.2f * std::cos(static_cast<float>(i) * 0.13f);
    }
    const auto blocks = quantize(weight);

    for (const auto& shape : {std::vector<int>{in_features},
                              std::vector<int>{1, in_features},
                              std::vector<int>{2, in_features}}) {
        SCOPED_TRACE(::testing::PrintToString(shape));
        Tensor input(shape);
        for (std::size_t i = 0; i < input.data.size(); ++i) {
            input.data[i] = 0.7f * std::sin(static_cast<float>(i) * 0.07f) -
                            0.3f * std::cos(static_cast<float>(i) * 0.31f);
        }

        SetThreadNum(1);
        const Tensor serial = linear(input, blocks, weight.shape);
        SetThreadNum(3);
        // 65 或 130 个输出会真正分成三个不等长的任务区间。
        const Tensor parallel = linear(input, blocks, weight.shape);
        const int rows = shape.size() == 1 ? 1 : shape[0];
        const std::vector<int> expected_shape =
            shape.size() == 1 ? std::vector<int>{65} : std::vector<int>{rows, 65};
        ASSERT_EQ(serial.shape, expected_shape);
        ASSERT_EQ(parallel.shape, expected_shape);
        ASSERT_EQ(parallel.data.size(), static_cast<std::size_t>(rows) * 65);
        EXPECT_EQ(parallel.data, serial.data);

        ExpectQuantLinearReference(parallel, input, blocks, weight.shape);
    }

    Tensor small_weight({3, in_features});
    std::copy_n(weight.data.begin(), small_weight.data.size(),
                small_weight.data.begin());
    const auto small_blocks = quantize(small_weight);
    const Tensor input({in_features}, 0.25f);
    SetThreadNum(1);
    const Tensor small_serial = linear(input, small_blocks, small_weight.shape);
    SetThreadNum(3);
    const Tensor small_parallel = linear(input, small_blocks, small_weight.shape);
    EXPECT_EQ(small_parallel.shape, (std::vector<int>{3}));
    EXPECT_EQ(small_parallel.data, small_serial.data);

    EXPECT_THROW(linear(input, blocks, {65}), std::runtime_error);
    EXPECT_THROW(linear(input, blocks, {0, in_features}), std::runtime_error);
    EXPECT_THROW(linear(Tensor({1, 1, in_features}), blocks, weight.shape),
                 std::runtime_error);
    EXPECT_THROW(linear(Tensor({in_features - 1}), blocks, weight.shape),
                 std::runtime_error);
    Tensor short_input({in_features});
    short_input.data.pop_back();
    EXPECT_THROW(linear(short_input, blocks, weight.shape), std::runtime_error);
    auto short_blocks = blocks;
    short_blocks.pop_back();
    EXPECT_THROW(linear(input, short_blocks, weight.shape), std::runtime_error);

    // 只设置元数据，确认任务数溢出在分配巨大结果前被拒绝。
    Tensor huge_input;
    huge_input.shape = {std::numeric_limits<int>::max(), in_features};
    try {
        linear(huge_input, std::vector<BlockType>{}, {2, in_features});
        FAIL() << "Expected an output task count overflow error";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("too many output elements"),
                  std::string::npos);
    }
}

TEST(QuantTest, linear_q80_threads_match_reference) {
    for (int in_features : {64, 86}) {
        SCOPED_TRACE(in_features);
        CheckThreadedQuantLinear<Block_Q80>(
            TensorToQ80, LinearQ80, in_features);
    }
}

TEST(QuantTest, linear_q40_threads_match_reference) {
    for (int in_features : {64, 86}) {
        SCOPED_TRACE(in_features);
        CheckThreadedQuantLinear<Block_Q40>(
            TensorToQ40, LinearQ40, in_features);
    }
}

TEST(QuantTest, linear_q41_threads_match_reference) {
    for (int in_features : {64, 86}) {
        SCOPED_TRACE(in_features);
        CheckThreadedQuantLinear<Block_Q41>(
            TensorToQ41, LinearQ41, in_features);
    }
}

void FillReferenceBlock(Block_Q80& block, int channel, std::size_t b) {
    // 包含零比例和最小 FP16 非规格化数，直接设置编码以覆盖 -128 和 127。
    const uint16_t scales[] = {0x0000, 0x0001, 0x3400, 0x3A00, 0x4000};
    block.d = scales[(channel + b) % 5];
    for (int i = 0; i < 32; ++i) {
        const int code = static_cast<int>((i * 37 + channel * 19 + b * 13) % 256);
        block.qs[i] = static_cast<int8_t>(code - 128);
    }
}

void FillReferenceBlock(Block_Q40& block, int channel, std::size_t b) {
    const uint16_t scales[] = {0x0000, 0x0001, 0x3400, 0x3A00, 0x4000};
    block.d = scales[(channel + b) % 5];
    for (int i = 0; i < 16; ++i) {
        const int low = static_cast<int>((i * 3 + channel + b) % 16);
        const int high = static_cast<int>((i * 5 + channel * 7 + b + 3) % 16);
        block.qs[i] = static_cast<uint8_t>(low | (high << 4));
    }
}

void FillReferenceBlock(Block_Q41& block, int channel, std::size_t b) {
    const uint16_t scales[] = {0x0000, 0x3400, 0x3A00, 0x4000};
    const uint16_t minima[] = {0xC000, 0xB800, 0x0000, 0x3E00};
    block.d = scales[(channel + b) % 4];
    // d=0 的输出行仍覆盖 m=-2、-0.5、0、1.5。
    block.m = minima[(channel / 4 + b) % 4];
    for (int i = 0; i < 16; ++i) {
        const int low = static_cast<int>((i * 3 + channel + b) % 16);
        const int high = static_cast<int>((i * 5 + channel * 7 + b + 3) % 16);
        block.qs[i] = static_cast<uint8_t>(low | (high << 4));
    }
}

template<typename BlockType>
void CheckHandcraftedQuantLinear(
    Tensor (*linear)(const Tensor&, const std::vector<BlockType>&,
                     const std::vector<int>&)) {
    RestoreAutomaticThreadCount restore_threads;
    constexpr int out_features = 65;

    for (const int in_features : {1, 7, 8, 9, 15, 16, 17, 22, 23, 24, 25,
                                 31, 32, 33, 63, 64, 65, 86}) {
        SCOPED_TRACE(in_features);
        const std::size_t blocks_per_row =
            1 + (static_cast<std::size_t>(in_features) - 1) / kQuantBlockSize;
        const std::vector<int> weight_shape = {out_features, in_features};
        std::vector<BlockType> blocks(out_features * blocks_per_row);
        for (int out = 0; out < out_features; ++out) {
            for (std::size_t b = 0; b < blocks_per_row; ++b) {
                // 无效补位保留变化编码，不按浮点零补齐。
                FillReferenceBlock(blocks[out * blocks_per_row + b], out, b);
            }
        }

        for (const auto& shape : {std::vector<int>{in_features},
                                  std::vector<int>{1, in_features},
                                  std::vector<int>{2, in_features}}) {
            SCOPED_TRACE(::testing::PrintToString(shape));
            Tensor input(shape);
            for (std::size_t i = 0; i < input.data.size(); ++i) {
                input.data[i] =
                    0.7f * std::sin(static_cast<float>(i) * 0.17f) +
                    0.3f * std::cos(static_cast<float>(i) * 0.41f);
            }

            SetThreadNum(1);
            const Tensor serial = linear(input, blocks, weight_shape);
            SetThreadNum(3);
            const Tensor parallel = linear(input, blocks, weight_shape);
            EXPECT_EQ(parallel.shape, serial.shape);
            EXPECT_EQ(parallel.data, serial.data);
            ExpectQuantLinearReference(parallel, input, blocks, weight_shape);
        }
    }
}

TEST(QuantTest, linear_q80_handcrafted_blocks_match_double_reference) {
    CheckHandcraftedQuantLinear<Block_Q80>(LinearQ80);
}

TEST(QuantTest, linear_q40_handcrafted_blocks_match_double_reference) {
    CheckHandcraftedQuantLinear<Block_Q40>(LinearQ40);
}

TEST(QuantTest, linear_q41_handcrafted_blocks_match_double_reference) {
    CheckHandcraftedQuantLinear<Block_Q41>(LinearQ41);
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
