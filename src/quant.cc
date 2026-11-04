#include "include/quant.h"
#include "thread_pool/thread_pool.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <functional>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

namespace yan_lamma {
namespace {

static_assert(sizeof(float) == sizeof(uint32_t) &&
                  std::numeric_limits<float>::is_iec559,
              "FP16 conversion requires IEEE-754 binary32 float");

// 返回 IEEE-754 binary16 位模式，可直接存入 Block_Q80::d。
inline uint16_t Fp32ToFp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    const uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent == 0xFFu) {
        if (mantissa == 0) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        // 保留符号和高位 payload，并设为 quiet NaN，避免截断成无穷。
        return static_cast<uint16_t>(sign | 0x7C00u |
                                     (mantissa >> 13) | 0x0200u);
    }

    // FP32 与 FP16 的指数偏置分别为 127 和 15。
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }

        // 非规格化数没有隐含的首位 1，需要将它移入尾数。
        const uint32_t significand = mantissa | 0x800000u;
        const int shift = 14 - half_exponent;
        uint32_t half_mantissa = significand >> shift;
        const uint32_t remainder = significand & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        // 就近舍入，中点取偶数；进位可得到最小规格化数。
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1u) != 0)) {
            ++half_mantissa;
        }
        return static_cast<uint16_t>(sign | half_mantissa);
    }

    // 丢弃 13 位，按保留部分的最低位决定中点向哪边舍入。
    const uint32_t half_mantissa =
        (mantissa + 0xFFFu + ((mantissa >> 13) & 1u)) >> 13;
    // 使用加法将尾数进位传入指数，溢出时自然得到无穷。
    return static_cast<uint16_t>(
        sign | ((static_cast<uint32_t>(half_exponent) << 10) + half_mantissa));
}

inline float Fp16ToFp32(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1Fu;
    uint32_t mantissa = bits & 0x03FFu;
    uint32_t out;

    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            // 将非规格化尾数左移至首位为 1，再恢复 FP32 指数。
            int unbiased_exponent = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiased_exponent;
            }
            mantissa &= 0x03FFu;
            out = sign |
                  (static_cast<uint32_t>(unbiased_exponent + 127) << 23) |
                  (mantissa << 13);
        }
    } else if (exponent == 0x1Fu) {
        // 无穷的尾数为零，NaN 的尾数非零。
        out = sign | 0x7F800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }

    float value;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

size_t GetQuantRowSize(const Tensor& tensor) {
    if (tensor.shape.empty()) {
        throw std::runtime_error("Quantize: shape must not be empty");
    }

    size_t total = 1;
    for (int dim : tensor.shape) {
        if (dim <= 0) {
            throw std::runtime_error("Quantize: dimensions must be positive");
        }
        const size_t size = static_cast<size_t>(dim);
        if (total > std::numeric_limits<size_t>::max() / size) {
            throw std::runtime_error("Quantize: element count overflow");
        }
        total *= size;
    }
    if (total != tensor.data.size()) {
        throw std::runtime_error("Quantize: shape and data size mismatch");
    }
    return static_cast<size_t>(tensor.shape.back());
}
}  // namespace
std::vector<Block_Q80> TensorToQ80(const Tensor& tensor) {
    if (tensor.shape.empty() && tensor.data.empty()) {
        return {};
    }

    const size_t row_size = GetQuantRowSize(tensor);
    const size_t n_rows = tensor.data.size() / row_size;
    const size_t row_blocks =
        (row_size + kQuantBlockSize - 1) / kQuantBlockSize;
    std::vector<Block_Q80> blocks(n_rows * row_blocks);

    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t b = 0; b < row_blocks; ++b) {
            auto& block = blocks[row * row_blocks + b];
            const size_t offset = b * kQuantBlockSize;
            const size_t block_len =
                std::min(kQuantBlockSize, row_size - offset);

            // 只复制有效元素，尾部保持为浮点零。
            float padded[kQuantBlockSize]{};
            std::copy_n(tensor.data.data() + row * row_size + offset,
                        block_len, padded);
            const float* x = padded;

            float amax = std::abs(x[0]);
            for (size_t i = 1; i < kQuantBlockSize; ++i) {
                amax = std::max(amax, std::abs(x[i]));
            }
            const float d = amax / 127.0f;
            block.d = Fp32ToFp16(d);
            for (size_t i = 0; i < kQuantBlockSize; ++i) {
                const float q =
                    d != 0.0f ? std::round(x[i] / d) : 0.0f;
                block.qs[i] = static_cast<int8_t>(q);
            }
        }
    }
    return blocks;
}
Tensor Q80ToTensor(const std::vector<Block_Q80>& blocks, const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("Q80ToTensor: shape must not be empty");
    }
    Tensor result(shape);
    const size_t row_size = static_cast<size_t>(shape.back());
    const size_t n_rows = result.data.size() / row_size;
    const size_t row_blocks =
        (row_size + kQuantBlockSize - 1) / kQuantBlockSize;
    if (blocks.size() != n_rows * row_blocks) {
        throw std::runtime_error("Q80ToTensor: block count mismatch");
    }

    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t b = 0; b < row_blocks; ++b) {
            const auto& block = blocks[row * row_blocks + b];
            const size_t offset = b * kQuantBlockSize;
            const size_t block_len =
                std::min(kQuantBlockSize, row_size - offset);
            float* x = result.data.data() + row * row_size + offset;
            const float d = Fp16ToFp32(block.d);

            // 只还原有效元素，跳过量化块内的补位。
            for (size_t i = 0; i < block_len; ++i) {
                x[i] = d * static_cast<float>(block.qs[i]);
            }
        }
    }
    return result;
}
std::vector<Block_Q40> TensorToQ40(const Tensor& tensor) {
    if (tensor.shape.empty() && tensor.data.empty()) {
        return {};
    }

    const size_t row_size = GetQuantRowSize(tensor);
    const size_t n_rows = tensor.data.size() / row_size;
    const size_t row_blocks =
        (row_size + kQuantBlockSize - 1) / kQuantBlockSize;
    std::vector<Block_Q40> blocks(n_rows * row_blocks);

    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t b = 0; b < row_blocks; ++b) {
            auto& block = blocks[row * row_blocks + b];
            const size_t offset = b * kQuantBlockSize;
            const size_t block_len =
                std::min(kQuantBlockSize, row_size - offset);

            float padded[kQuantBlockSize]{};
            std::copy_n(tensor.data.data() + row * row_size + offset,
                        block_len, padded);
            const float* x = padded;

            float vmax = x[0];
            float amax = std::abs(x[0]);
            for (size_t i = 1; i < kQuantBlockSize; ++i) {
                if (std::abs(x[i]) > amax) {
                    amax = std::abs(x[i]);
                    vmax = x[i];
                }
            }
            const float d = vmax / -8.0f;
            const float inv_d = d != 0.0f ? 1.0f / d : 0.0f;
            block.d = Fp32ToFp16(d);
            for (size_t i = 0; i < 16; ++i) {
                const float v0 = x[i] * inv_d;
                const float v1 = x[i + 16] * inv_d;
                const int u0 = std::min(15, static_cast<int>(v0 + 8.5f));
                const int u1 = std::min(15, static_cast<int>(v1 + 8.5f));
                block.qs[i] = static_cast<uint8_t>(u0 | (u1 << 4));
            }
        }
    }
    return blocks;
}
Tensor Q40ToTensor(const std::vector<Block_Q40>& blocks, const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("Q40ToTensor: shape must not be empty");
    }
    Tensor result(shape);
    const size_t row_size = static_cast<size_t>(shape.back());
    const size_t n_rows = result.data.size() / row_size;
    const size_t row_blocks =
        (row_size + kQuantBlockSize - 1) / kQuantBlockSize;
    if (blocks.size() != n_rows * row_blocks) {
        throw std::runtime_error("Q40ToTensor: block count mismatch");
    }

    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t b = 0; b < row_blocks; ++b) {
            const auto& block = blocks[row * row_blocks + b];
            const size_t offset = b * kQuantBlockSize;
            const size_t block_len =
                std::min(kQuantBlockSize, row_size - offset);
            float* x = result.data.data() + row * row_size + offset;
            const float d = Fp16ToFp32(block.d);

            for (size_t i = 0; i < 16; ++i) {
                if (i < block_len) {
                    const int q0 = static_cast<int>(block.qs[i] & 0x0F) - 8;
                    x[i] = d * static_cast<float>(q0);
                }
                if (i + 16 < block_len) {
                    const int q1 = static_cast<int>(block.qs[i] >> 4) - 8;
                    x[i + 16] = d * static_cast<float>(q1);
                }
            }
        }
    }
    return result;
}
std::vector<Block_Q41> TensorToQ41(const Tensor& tensor) {
    if (tensor.shape.empty() && tensor.data.empty()) {
        return {};
    }

    const size_t row_size = GetQuantRowSize(tensor);
    const size_t n_rows = tensor.data.size() / row_size;
    const size_t row_blocks =
        (row_size + kQuantBlockSize - 1) / kQuantBlockSize;
    std::vector<Block_Q41> blocks(n_rows * row_blocks);

    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t b = 0; b < row_blocks; ++b) {
            auto& block = blocks[row * row_blocks + b];
            const size_t offset = b * kQuantBlockSize;
            const size_t block_len =
                std::min(kQuantBlockSize, row_size - offset);

            float padded[kQuantBlockSize]{};
            std::copy_n(tensor.data.data() + row * row_size + offset,
                        block_len, padded);
            const float* x = padded;

            // 补入的零也参与最小值、最大值计算。
            float xmin = x[0];
            float xmax = x[0];
            for (size_t i = 1; i < kQuantBlockSize; ++i) {
                xmin = std::min(xmin, x[i]);
                xmax = std::max(xmax, x[i]);
            }
            const float m = xmin;
            const float d = (xmax - xmin) / 15.0f;
            const float inv_d = d != 0.0f ? 1.0f / d : 0.0f;
            block.d = Fp32ToFp16(d);
            block.m = Fp32ToFp16(m);
            for (size_t i = 0; i < 16; ++i) {
                const int u0 = std::min(
                    15, static_cast<int>((x[i] - m) * inv_d + 0.5f));
                const int u1 = std::min(
                    15, static_cast<int>((x[i + 16] - m) * inv_d + 0.5f));
                block.qs[i] = static_cast<uint8_t>(u0 | (u1 << 4));
            }
        }
    }
    return blocks;
}
Tensor Q41ToTensor(const std::vector<Block_Q41>& blocks, const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("Q41ToTensor: shape must not be empty");
    }
    Tensor result(shape);
    const size_t row_size = static_cast<size_t>(shape.back());
    const size_t n_rows = result.data.size() / row_size;
    const size_t row_blocks =
        (row_size + kQuantBlockSize - 1) / kQuantBlockSize;
    if (blocks.size() != n_rows * row_blocks) {
        throw std::runtime_error("Q41ToTensor: block count mismatch");
    }

    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t b = 0; b < row_blocks; ++b) {
            const auto& block = blocks[row * row_blocks + b];
            const size_t offset = b * kQuantBlockSize;
            const size_t block_len =
                std::min(kQuantBlockSize, row_size - offset);
            float* x = result.data.data() + row * row_size + offset;
            const float d = Fp16ToFp32(block.d);
            const float m = Fp16ToFp32(block.m);

            for (size_t i = 0; i < 16; ++i) {
                if (i < block_len) {
                    const int q0 = block.qs[i] & 0x0F;
                    x[i] = d * static_cast<float>(q0) + m;
                }
                if (i + 16 < block_len) {
                    const int q1 = block.qs[i] >> 4;
                    x[i + 16] = d * static_cast<float>(q1) + m;
                }
            }
        }
    }
    return result;
}
float DequantQ80(const Block_Q80& block, int i) {
    float d = Fp16ToFp32(block.d);
    float res = block.qs[i] * d;
    return res;
}
float DequantQ40(const Block_Q40& block, int idx) {
    float d = Fp16ToFp32(block.d);
    int q  = 0;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0F) - 8;
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;
    }
    return static_cast<float>(q) * d;
}
float DequantQ41(const Block_Q41& block, int idx) {
    const float d = Fp16ToFp32(block.d);
    const float m = Fp16ToFp32(block.m);
    const int code = idx < 16
        ? (block.qs[idx] & 0x0F)
        : (block.qs[idx - 16] >> 4);
    return d * static_cast<float>(code) + m;
}
template<typename BlockType, size_t block_size = 32>
Tensor LinearQuant(
    const Tensor& x,
    const std::vector<BlockType>& weight,
    const std::vector<int>& shape,
    std::function<float(const BlockType&, int)> f) {
    static_assert(block_size > 0, "LinearQuant: block_size must be positive");

    if (shape.size() != 2 ||
        (x.GetNumDims() != 1 && x.GetNumDims() != 2)) {
        throw std::runtime_error("LinearQuant: invalid dimensions");
    }
    
    // shape 是权重形状：[输出特征数，输入特征数]。
    const bool is_1d = x.GetNumDims() == 1;
    const int rows = is_1d ? 1 : x.shape[0];
    const int in_features = shape[1];
    const int out_features = shape[0];

    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("LinearQuant: dimensions must be positive");
    }
    if (rows > std::numeric_limits<int>::max() / out_features) {
        throw std::runtime_error("LinearQuant: too many output elements");
    }
    const int tasks = rows * out_features;

    if (x.shape.back() != in_features) {
        throw std::runtime_error("LinearQuant: input shape mismatch");
    }

    const size_t blocks_per_row =
        (static_cast<size_t>(in_features) + block_size - 1) / block_size;
    if (weight.size() != static_cast<size_t>(out_features) * blocks_per_row ||
        x.data.size() != static_cast<size_t>(rows) * in_features) {
        throw std::runtime_error("LinearQuant: data size mismatch");
    }

    // 一维输入返回一维结果；[1, in] 输入仍然返回 [1, out]。
    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features});
    ParallelFor(tasks, [&](int start, int end) {
        for (int index = start; index < end; ++index) {
            // 展平的输出下标对应输入第 row 行和权重第 j 行。
            const int row = index / out_features;
            const int j = index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;

            float sum = 0.0f;
            const size_t block_base =
                static_cast<size_t>(j) * blocks_per_row;

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const BlockType& block = weight[block_base + b];
                const size_t base_k = b * block_size;
                const size_t block_len = std::min(
                    block_size, static_cast<size_t>(in_features) - base_k);

                for (size_t i = 0; i < block_len; ++i) {
                    // 回调还原块内第 i 个权重，再与对应输入相乘累加。
                    const float w = f(block, static_cast<int>(i));
                    sum += x_row[base_k + i] * w;
                }
            }

            result.data[index] = sum;
        }
    });

    return result;
}
#ifdef MINI_LLAMA_USE_NEON
// Q80 NEON：每轮处理 8 个量化元素。
Tensor LinearQ80(
    const Tensor& x,
    const std::vector<Block_Q80>& weight,
    const std::vector<int>& shape) {
    if (shape.size() != 2 ||
        (x.GetNumDims() != 1 && x.GetNumDims() != 2)) {
        throw std::runtime_error("LinearQ80: invalid dimensions");
    }

    const bool is_1d = x.GetNumDims() == 1;
    const int rows = is_1d ? 1 : x.shape[0];
    const int in_features = shape[1];
    const int out_features = shape[0];
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("LinearQ80: dimensions must be positive");
    }
    if (rows > std::numeric_limits<int>::max() / out_features) {
        throw std::runtime_error("LinearQ80: too many output elements");
    }
    if (x.shape.back() != in_features) {
        throw std::runtime_error("LinearQ80: input shape mismatch");
    }

    const size_t blocks_per_row =
        1 + (static_cast<size_t>(in_features) - 1) / kQuantBlockSize;
    // 用商和余数校验长度，避免计算预期长度时乘法溢出。
    if (x.data.size() % in_features != 0 ||
        x.data.size() / in_features != static_cast<size_t>(rows) ||
        weight.size() % blocks_per_row != 0 ||
        weight.size() / blocks_per_row != static_cast<size_t>(out_features)) {
        throw std::runtime_error("LinearQ80: data size mismatch");
    }

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features});
    ParallelFor(rows * out_features, [&](int begin, int end) {
        for (int index = begin; index < end; ++index) {
            const int row = index / out_features;
            const int channel = index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            const size_t block_base =
                static_cast<size_t>(channel) * blocks_per_row;
            float sum = 0.0f;

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const Block_Q80& block = weight[block_base + b];
                const size_t offset = b * kQuantBlockSize;
                const int len = static_cast<int>(std::min(
                    kQuantBlockSize,
                    static_cast<size_t>(in_features) - offset));
                const float* xb = x_row + offset;
                const float d = Fp16ToFp32(block.d);
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float tail = 0.0f;

                int i = 0;
                for (; i + 8 <= len; i += 8) {
                    const int8x8_t q8 = vld1_s8(block.qs + i);
                    const int16x8_t q16 = vmovl_s8(q8);
                    const float32x4_t q0 = vcvtq_f32_s32(
                        vmovl_s16(vget_low_s16(q16)));
                    const float32x4_t q1 = vcvtq_f32_s32(
                        vmovl_s16(vget_high_s16(q16)));
                    const float32x4_t w0 = vmulq_n_f32(q0, d);
                    const float32x4_t w1 = vmulq_n_f32(q1, d);

                    sum0 = vmlaq_f32(
                        sum0, w0, vld1q_f32(xb + i));
                    sum1 = vmlaq_f32(
                        sum1, w1, vld1q_f32(xb + i + 4));
                }
                for (; i < len; ++i) {
                    const float w = d * static_cast<float>(block.qs[i]);
                    tail += w * xb[i];
                }

                // 水平求和直接在 Linear 内完成。
                const float32x4_t sum4 = vaddq_f32(sum0, sum1);
                float32x2_t sum2 = vadd_f32(
                    vget_low_f32(sum4), vget_high_f32(sum4));
                sum2 = vpadd_f32(sum2, sum2);
                sum += vget_lane_f32(sum2, 0) + tail;
            }

            result.data[index] = sum;
        }
    });
    return result;
}
#elif defined(MINI_LLAMA_USE_AVX2)
// Q80 AVX2/FMA：每轮处理 8 个量化元素。
Tensor LinearQ80(
    const Tensor& x,
    const std::vector<Block_Q80>& weight,
    const std::vector<int>& shape) {
    if (shape.size() != 2 ||
        (x.GetNumDims() != 1 && x.GetNumDims() != 2)) {
        throw std::runtime_error("LinearQ80: invalid dimensions");
    }

    const bool is_1d = x.GetNumDims() == 1;
    const int rows = is_1d ? 1 : x.shape[0];
    const int in_features = shape[1];
    const int out_features = shape[0];
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("LinearQ80: dimensions must be positive");
    }
    if (rows > std::numeric_limits<int>::max() / out_features) {
        throw std::runtime_error("LinearQ80: too many output elements");
    }
    if (x.shape.back() != in_features) {
        throw std::runtime_error("LinearQ80: input shape mismatch");
    }

    const size_t blocks_per_row =
        1 + (static_cast<size_t>(in_features) - 1) / kQuantBlockSize;
    // 用商和余数校验长度，避免计算预期长度时乘法溢出。
    if (x.data.size() % in_features != 0 ||
        x.data.size() / in_features != static_cast<size_t>(rows) ||
        weight.size() % blocks_per_row != 0 ||
        weight.size() / blocks_per_row != static_cast<size_t>(out_features)) {
        throw std::runtime_error("LinearQ80: data size mismatch");
    }

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features});
    ParallelFor(rows * out_features, [&](int begin, int end) {
        for (int index = begin; index < end; ++index) {
            const int row = index / out_features;
            const int channel = index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            const size_t block_base =
                static_cast<size_t>(channel) * blocks_per_row;
            float sum = 0.0f;

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const Block_Q80& block = weight[block_base + b];
                const size_t offset = b * kQuantBlockSize;
                const int len = static_cast<int>(std::min(
                    kQuantBlockSize,
                    static_cast<size_t>(in_features) - offset));
                const float* xb = x_row + offset;
                const float d = Fp16ToFp32(block.d);
                const __m256 scale = _mm256_set1_ps(d);
                __m256 sum_vec = _mm256_setzero_ps();
                float tail = 0.0f;

                int i = 0;
                for (; i + 8 <= len; i += 8) {
                    const __m128i packed = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(block.qs + i));
                    // 有符号扩展，保留 Q80 的负编码。
                    const __m256i q = _mm256_cvtepi8_epi32(packed);
                    const __m256 w = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(q), scale);
                    const __m256 xv = _mm256_loadu_ps(xb + i);
                    sum_vec = _mm256_fmadd_ps(w, xv, sum_vec);
                }
                for (; i < len; ++i) {
                    const float w = d * static_cast<float>(block.qs[i]);
                    tail += w * xb[i];
                }

                // 水平求和直接在 Linear 内完成。
                __m128 sum4 = _mm_add_ps(
                    _mm256_castps256_ps128(sum_vec),
                    _mm256_extractf128_ps(sum_vec, 1));
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum += _mm_cvtss_f32(sum4) + tail;
            }

            result.data[index] = sum;
        }
    });
    return result;
}
#else
Tensor LinearQ80(const Tensor& tensor, const std::vector<Block_Q80>& weight, const std::vector<int>& shape) {
    std::function<float(const Block_Q80&, int)> f = DequantQ80;
    return LinearQuant(tensor, weight, shape, f);
}
#endif

#ifdef MINI_LLAMA_USE_NEON
// Q40 NEON：每轮处理 8 个量化元素。
Tensor LinearQ40(
    const Tensor& x,
    const std::vector<Block_Q40>& weight,
    const std::vector<int>& shape) {
    if (shape.size() != 2 ||
        (x.GetNumDims() != 1 && x.GetNumDims() != 2)) {
        throw std::runtime_error("LinearQ40: invalid dimensions");
    }

    const bool is_1d = x.GetNumDims() == 1;
    const int rows = is_1d ? 1 : x.shape[0];
    const int in_features = shape[1];
    const int out_features = shape[0];
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("LinearQ40: dimensions must be positive");
    }
    if (rows > std::numeric_limits<int>::max() / out_features) {
        throw std::runtime_error("LinearQ40: too many output elements");
    }
    if (x.shape.back() != in_features) {
        throw std::runtime_error("LinearQ40: input shape mismatch");
    }

    const size_t blocks_per_row =
        1 + (static_cast<size_t>(in_features) - 1) / kQuantBlockSize;
    // 用商和余数校验长度，避免计算预期长度时乘法溢出。
    if (x.data.size() % in_features != 0 ||
        x.data.size() / in_features != static_cast<size_t>(rows) ||
        weight.size() % blocks_per_row != 0 ||
        weight.size() / blocks_per_row != static_cast<size_t>(out_features)) {
        throw std::runtime_error("LinearQ40: data size mismatch");
    }

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features});
    ParallelFor(rows * out_features, [&](int begin, int end) {
        const uint8x8_t mask = vdup_n_u8(15);
        const int16x8_t bias = vdupq_n_s16(8);

        for (int index = begin; index < end; ++index) {
            const int row = index / out_features;
            const int channel = index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            const size_t block_base =
                static_cast<size_t>(channel) * blocks_per_row;
            float sum = 0.0f;

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const Block_Q40& block = weight[block_base + b];
                const size_t offset = b * kQuantBlockSize;
                const int len = static_cast<int>(std::min(
                    kQuantBlockSize,
                    static_cast<size_t>(in_features) - offset));
                const float* xb = x_row + offset;
                const float d = Fp16ToFp32(block.d);
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float tail = 0.0f;

                // 低四位对应前 16 个元素，高四位对应后 16 个元素。
                for (int base = 0; base < len; base += 16) {
                    const int count = std::min(16, len - base);
                    int i = 0;
                    for (; i + 8 <= count; i += 8) {
                        const uint8x8_t packed = vld1_u8(block.qs + i);
                        const uint8x8_t code = base == 0
                            ? vand_u8(packed, mask)
                            : vshr_n_u8(packed, 4);
                        const int16x8_t q = vsubq_s16(
                            vreinterpretq_s16_u16(vmovl_u8(code)), bias);
                        const float32x4_t q0 = vcvtq_f32_s32(
                            vmovl_s16(vget_low_s16(q)));
                        const float32x4_t q1 = vcvtq_f32_s32(
                            vmovl_s16(vget_high_s16(q)));
                        const float32x4_t w0 = vmulq_n_f32(q0, d);
                        const float32x4_t w1 = vmulq_n_f32(q1, d);

                        sum0 = vmlaq_f32(
                            sum0, w0, vld1q_f32(xb + base + i));
                        sum1 = vmlaq_f32(
                            sum1, w1, vld1q_f32(xb + base + i + 4));
                    }
                    // 不足 8 个时逐元素计算，只读取有效输入。
                    for (; i < count; ++i) {
                        const int code = base == 0
                            ? (block.qs[i] & 15)
                            : (block.qs[i] >> 4);
                        const float w = d * static_cast<float>(code - 8);
                        tail += w * xb[base + i];
                    }
                }

                // 水平求和直接在 Linear 内完成。
                const float32x4_t sum4 = vaddq_f32(sum0, sum1);
                float32x2_t sum2 = vadd_f32(
                    vget_low_f32(sum4), vget_high_f32(sum4));
                sum2 = vpadd_f32(sum2, sum2);
                sum += vget_lane_f32(sum2, 0) + tail;
            }

            result.data[index] = sum;
        }
    });
    return result;
}
#elif defined(MINI_LLAMA_USE_AVX2)
// Q40 AVX2/FMA：每轮处理 8 个量化元素。
Tensor LinearQ40(
    const Tensor& x,
    const std::vector<Block_Q40>& weight,
    const std::vector<int>& shape) {
    if (shape.size() != 2 ||
        (x.GetNumDims() != 1 && x.GetNumDims() != 2)) {
        throw std::runtime_error("LinearQ40: invalid dimensions");
    }

    const bool is_1d = x.GetNumDims() == 1;
    const int rows = is_1d ? 1 : x.shape[0];
    const int in_features = shape[1];
    const int out_features = shape[0];
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("LinearQ40: dimensions must be positive");
    }
    if (rows > std::numeric_limits<int>::max() / out_features) {
        throw std::runtime_error("LinearQ40: too many output elements");
    }
    if (x.shape.back() != in_features) {
        throw std::runtime_error("LinearQ40: input shape mismatch");
    }

    const size_t blocks_per_row =
        1 + (static_cast<size_t>(in_features) - 1) / kQuantBlockSize;
    // 用商和余数校验长度，避免计算预期长度时乘法溢出。
    if (x.data.size() % in_features != 0 ||
        x.data.size() / in_features != static_cast<size_t>(rows) ||
        weight.size() % blocks_per_row != 0 ||
        weight.size() / blocks_per_row != static_cast<size_t>(out_features)) {
        throw std::runtime_error("LinearQ40: data size mismatch");
    }

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features});
    ParallelFor(rows * out_features, [&](int begin, int end) {
        const __m256i mask = _mm256_set1_epi32(15);
        const __m256i bias = _mm256_set1_epi32(8);

        for (int index = begin; index < end; ++index) {
            const int row = index / out_features;
            const int channel = index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            const size_t block_base =
                static_cast<size_t>(channel) * blocks_per_row;
            float sum = 0.0f;

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const Block_Q40& block = weight[block_base + b];
                const size_t offset = b * kQuantBlockSize;
                const int len = static_cast<int>(std::min(
                    kQuantBlockSize,
                    static_cast<size_t>(in_features) - offset));
                const float* xb = x_row + offset;
                const float d = Fp16ToFp32(block.d);
                const __m256 scale = _mm256_set1_ps(d);
                __m256 sum_vec = _mm256_setzero_ps();
                float tail = 0.0f;

                // 低四位对应前 16 个元素，高四位对应后 16 个元素。
                for (int base = 0; base < len; base += 16) {
                    const int count = std::min(16, len - base);
                    int i = 0;
                    for (; i + 8 <= count; i += 8) {
                        const __m128i packed = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(block.qs + i));
                        __m256i code = _mm256_cvtepu8_epi32(packed);
                        if (base == 16) {
                            code = _mm256_srli_epi32(code, 4);
                        }
                        code = _mm256_and_si256(code, mask);
                        const __m256i q = _mm256_sub_epi32(code, bias);
                        const __m256 w = _mm256_mul_ps(
                            _mm256_cvtepi32_ps(q), scale);
                        const __m256 xv =
                            _mm256_loadu_ps(xb + base + i);
                        sum_vec = _mm256_fmadd_ps(w, xv, sum_vec);
                    }
                    // 不足 8 个时逐元素计算，只读取有效输入。
                    for (; i < count; ++i) {
                        const int code = base == 0
                            ? (block.qs[i] & 15)
                            : (block.qs[i] >> 4);
                        const float w = d * static_cast<float>(code - 8);
                        tail += w * xb[base + i];
                    }
                }

                // 水平求和直接在 Linear 内完成。
                __m128 sum4 = _mm_add_ps(
                    _mm256_castps256_ps128(sum_vec),
                    _mm256_extractf128_ps(sum_vec, 1));
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum += _mm_cvtss_f32(sum4) + tail;
            }

            result.data[index] = sum;
        }
    });
    return result;
}
#else
Tensor LinearQ40(const Tensor& tensor, const std::vector<Block_Q40>& weight, const std::vector<int>& shape) {
    std::function<float(const Block_Q40&, int)> f = DequantQ40;
    return LinearQuant(tensor, weight, shape, f);
}
#endif

#ifdef MINI_LLAMA_USE_NEON
// Q41 NEON：每轮处理 8 个量化元素。
Tensor LinearQ41(
    const Tensor& x,
    const std::vector<Block_Q41>& weight,
    const std::vector<int>& shape) {
    if (shape.size() != 2 ||
        (x.GetNumDims() != 1 && x.GetNumDims() != 2)) {
        throw std::runtime_error("LinearQ41: invalid dimensions");
    }

    const bool is_1d = x.GetNumDims() == 1;
    const int rows = is_1d ? 1 : x.shape[0];
    const int in_features = shape[1];
    const int out_features = shape[0];
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("LinearQ41: dimensions must be positive");
    }
    if (rows > std::numeric_limits<int>::max() / out_features) {
        throw std::runtime_error("LinearQ41: too many output elements");
    }
    if (x.shape.back() != in_features) {
        throw std::runtime_error("LinearQ41: input shape mismatch");
    }

    const size_t blocks_per_row =
        1 + (static_cast<size_t>(in_features) - 1) / kQuantBlockSize;
    // 用商和余数校验长度，避免计算预期长度时乘法溢出。
    if (x.data.size() % in_features != 0 ||
        x.data.size() / in_features != static_cast<size_t>(rows) ||
        weight.size() % blocks_per_row != 0 ||
        weight.size() / blocks_per_row != static_cast<size_t>(out_features)) {
        throw std::runtime_error("LinearQ41: data size mismatch");
    }

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features});
    ParallelFor(rows * out_features, [&](int begin, int end) {
        const uint8x8_t mask = vdup_n_u8(15);

        for (int index = begin; index < end; ++index) {
            const int row = index / out_features;
            const int channel = index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            const size_t block_base =
                static_cast<size_t>(channel) * blocks_per_row;
            float sum = 0.0f;

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const Block_Q41& block = weight[block_base + b];
                const size_t offset = b * kQuantBlockSize;
                const int len = static_cast<int>(std::min(
                    kQuantBlockSize,
                    static_cast<size_t>(in_features) - offset));
                const float* xb = x_row + offset;
                const float d = Fp16ToFp32(block.d);
                const float m = Fp16ToFp32(block.m);
                const float32x4_t minimum = vdupq_n_f32(m);
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float tail = 0.0f;

                // 低四位对应前 16 个元素，高四位对应后 16 个元素。
                for (int base = 0; base < len; base += 16) {
                    const int count = std::min(16, len - base);
                    int i = 0;
                    for (; i + 8 <= count; i += 8) {
                        const uint8x8_t packed = vld1_u8(block.qs + i);
                        const uint8x8_t code = base == 0
                            ? vand_u8(packed, mask)
                            : vshr_n_u8(packed, 4);
                        const uint16x8_t q = vmovl_u8(code);
                        const float32x4_t q0 = vcvtq_f32_u32(
                            vmovl_u16(vget_low_u16(q)));
                        const float32x4_t q1 = vcvtq_f32_u32(
                            vmovl_u16(vget_high_u16(q)));
                        const float32x4_t w0 = vaddq_f32(
                            vmulq_n_f32(q0, d), minimum);
                        const float32x4_t w1 = vaddq_f32(
                            vmulq_n_f32(q1, d), minimum);

                        sum0 = vmlaq_f32(
                            sum0, w0, vld1q_f32(xb + base + i));
                        sum1 = vmlaq_f32(
                            sum1, w1, vld1q_f32(xb + base + i + 4));
                    }
                    // 不足 8 个时逐元素计算，只读取有效输入。
                    for (; i < count; ++i) {
                        const int code = base == 0
                            ? (block.qs[i] & 15)
                            : (block.qs[i] >> 4);
                        const float w = d * static_cast<float>(code) + m;
                        tail += w * xb[base + i];
                    }
                }

                // 水平求和直接在 Linear 内完成。
                const float32x4_t sum4 = vaddq_f32(sum0, sum1);
                float32x2_t sum2 = vadd_f32(
                    vget_low_f32(sum4), vget_high_f32(sum4));
                sum2 = vpadd_f32(sum2, sum2);
                sum += vget_lane_f32(sum2, 0) + tail;
            }

            result.data[index] = sum;
        }
    });
    return result;
}
#elif defined(MINI_LLAMA_USE_AVX2)
// Q41 AVX2/FMA：每轮处理 8 个量化元素。
Tensor LinearQ41(
    const Tensor& x,
    const std::vector<Block_Q41>& weight,
    const std::vector<int>& shape) {
    if (shape.size() != 2 ||
        (x.GetNumDims() != 1 && x.GetNumDims() != 2)) {
        throw std::runtime_error("LinearQ41: invalid dimensions");
    }

    const bool is_1d = x.GetNumDims() == 1;
    const int rows = is_1d ? 1 : x.shape[0];
    const int in_features = shape[1];
    const int out_features = shape[0];
    if (rows <= 0 || in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("LinearQ41: dimensions must be positive");
    }
    if (rows > std::numeric_limits<int>::max() / out_features) {
        throw std::runtime_error("LinearQ41: too many output elements");
    }
    if (x.shape.back() != in_features) {
        throw std::runtime_error("LinearQ41: input shape mismatch");
    }

    const size_t blocks_per_row =
        1 + (static_cast<size_t>(in_features) - 1) / kQuantBlockSize;
    // 用商和余数校验长度，避免计算预期长度时乘法溢出。
    if (x.data.size() % in_features != 0 ||
        x.data.size() / in_features != static_cast<size_t>(rows) ||
        weight.size() % blocks_per_row != 0 ||
        weight.size() / blocks_per_row != static_cast<size_t>(out_features)) {
        throw std::runtime_error("LinearQ41: data size mismatch");
    }

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features});
    ParallelFor(rows * out_features, [&](int begin, int end) {
        const __m256i mask = _mm256_set1_epi32(15);

        for (int index = begin; index < end; ++index) {
            const int row = index / out_features;
            const int channel = index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            const size_t block_base =
                static_cast<size_t>(channel) * blocks_per_row;
            float sum = 0.0f;

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const Block_Q41& block = weight[block_base + b];
                const size_t offset = b * kQuantBlockSize;
                const int len = static_cast<int>(std::min(
                    kQuantBlockSize,
                    static_cast<size_t>(in_features) - offset));
                const float* xb = x_row + offset;
                const float d = Fp16ToFp32(block.d);
                const float m = Fp16ToFp32(block.m);
                const __m256 scale = _mm256_set1_ps(d);
                const __m256 minimum = _mm256_set1_ps(m);
                __m256 sum_vec = _mm256_setzero_ps();
                float tail = 0.0f;

                // 低四位对应前 16 个元素，高四位对应后 16 个元素。
                for (int base = 0; base < len; base += 16) {
                    const int count = std::min(16, len - base);
                    int i = 0;
                    for (; i + 8 <= count; i += 8) {
                        const __m128i packed = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(block.qs + i));
                        __m256i code = _mm256_cvtepu8_epi32(packed);
                        if (base == 16) {
                            code = _mm256_srli_epi32(code, 4);
                        }
                        code = _mm256_and_si256(code, mask);
                        const __m256 w = _mm256_add_ps(
                            _mm256_mul_ps(
                                _mm256_cvtepi32_ps(code), scale),
                            minimum);
                        const __m256 xv =
                            _mm256_loadu_ps(xb + base + i);
                        sum_vec = _mm256_fmadd_ps(w, xv, sum_vec);
                    }
                    // 不足 8 个时逐元素计算，只读取有效输入。
                    for (; i < count; ++i) {
                        const int code = base == 0
                            ? (block.qs[i] & 15)
                            : (block.qs[i] >> 4);
                        const float w = d * static_cast<float>(code) + m;
                        tail += w * xb[base + i];
                    }
                }

                // 水平求和直接在 Linear 内完成。
                __m128 sum4 = _mm_add_ps(
                    _mm256_castps256_ps128(sum_vec),
                    _mm256_extractf128_ps(sum_vec, 1));
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum += _mm_cvtss_f32(sum4) + tail;
            }

            result.data[index] = sum;
        }
    });
    return result;
}
#else
Tensor LinearQ41(const Tensor& tensor, const std::vector<Block_Q41>& weight, const std::vector<int>& shape) {
    std::function<float(const Block_Q41&, int)> f = DequantQ41;
    return LinearQuant(tensor, weight, shape, f);
}
#endif

Tensor MatmulQ80(const std::vector<Block_Q80>& weight, const Tensor& b, const std::vector<int>& shape) {
    const Tensor& x = Q80ToTensor(weight, shape);
    return Matmul(x, b);
}
Tensor MatmulQ40(const std::vector<Block_Q40>& weight, const Tensor& b, const std::vector<int>& shape) {
    const Tensor& x = Q40ToTensor(weight, shape);
    return Matmul(x, b);
}
Tensor MatmulQ41(const std::vector<Block_Q41>& weight, const Tensor& b, const std::vector<int>& shape) {
    const Tensor& x = Q41ToTensor(weight, shape);
    return Matmul(x, b);
}
float CompareMatmulErrorQ80(const Tensor& tensor, const Tensor& input) {
    Tensor F32 = Matmul(tensor, input);
    std::vector<Block_Q80> weight = TensorToQ80(tensor);
    Tensor Q80 = MatmulQ80(weight, input , tensor.shape);
    if (Q80.shape != F32.shape) {
        throw std::runtime_error("CompareMatmulError: shape mismatch");
    }
    float error = 0.0f;
    for (size_t i = 0; i < F32.GetDataSize(); i++) {
        const float diff = std::abs(F32.data[i] - Q80.data[i]);
        if (!std::isfinite(diff)) {
            throw std::runtime_error("CompareMatmulError: non-finite difference");
        }
        error = std::max(error, diff);
    }
    return error;
}
float CompareMatmulErrorQ40(const Tensor& tensor, const Tensor& input) {
    Tensor F32 = Matmul(tensor, input);
    std::vector<Block_Q40> weight = TensorToQ40(tensor);
    Tensor Q40 = MatmulQ40(weight, input , tensor.shape);
    if (Q40.shape != F32.shape) {
        throw std::runtime_error("CompareMatmulError: shape mismatch");
    }
    float error = 0.0f;
    for (size_t i = 0; i < F32.GetDataSize(); i++) {
        const float diff = std::abs(F32.data[i] - Q40.data[i]);
        if (!std::isfinite(diff)) {
            throw std::runtime_error("CompareMatmulError: non-finite difference");
        }
        error = std::max(error, diff);
    }
    return error;
}
float CompareMatmulErrorQ41(const Tensor& tensor, const Tensor& input) {
    Tensor F32 = Matmul(tensor, input);
    std::vector<Block_Q41> weight = TensorToQ41(tensor);
    Tensor Q41 = MatmulQ41(weight, input , tensor.shape);
    if (Q41.shape != F32.shape) {
        throw std::runtime_error("CompareMatmulError: shape mismatch");
    }
    float error = 0.0f;
    for (size_t i = 0; i < F32.GetDataSize(); i++) {
        const float diff = std::abs(F32.data[i] - Q41.data[i]);
        if (!std::isfinite(diff)) {
            throw std::runtime_error("CompareMatmulError: non-finite difference");
        }
        error = std::max(error, diff);
    }
    return error;
}
float CompareLinearErrorQ80(const Tensor& tensor, const Tensor& input) {
    Tensor F32 = Linear(input, tensor);
    std::vector<Block_Q80> weight = TensorToQ80(tensor);
    Tensor Q80 = LinearQ80(input, weight, tensor.shape);
    if (Q80.shape != F32.shape) {
        throw std::runtime_error("CompareLinearError: shape mismatch");
    }
    float error = 0.0f;
    for (size_t i = 0; i < F32.GetDataSize(); i++) {
        const float diff = std::abs(F32.data[i] - Q80.data[i]);
        if (!std::isfinite(diff)) {
            throw std::runtime_error("CompareLinearError: non-finite difference");
        }
        error = std::max(error, diff);
    }
    return error;
}
float CompareLinearErrorQ40(const Tensor& tensor, const Tensor& input) {
    Tensor F32 = Linear(input, tensor);
    std::vector<Block_Q40> weight = TensorToQ40(tensor);
    Tensor Q40 = LinearQ40(input, weight , tensor.shape);
    if (Q40.shape != F32.shape) {
        throw std::runtime_error("CompareLinearError: shape mismatch");
    }
    float error = 0.0f;
    for (size_t i = 0; i < F32.GetDataSize(); i++) {
        const float diff = std::abs(F32.data[i] - Q40.data[i]);
        if (!std::isfinite(diff)) {
            throw std::runtime_error("CompareLinearError: non-finite difference");
        }
        error = std::max(error, diff);
    }
    return error;
}
float CompareLinearErrorQ41(const Tensor& tensor, const Tensor& input) {
    Tensor F32 = Linear(input, tensor);
    std::vector<Block_Q41> weight = TensorToQ41(tensor);
    Tensor Q41 = LinearQ41(input, weight , tensor.shape);
    if (Q41.shape != F32.shape) {
        throw std::runtime_error("CompareLinearError: shape mismatch");
    }
    float error = 0.0f;
    for (size_t i = 0; i < F32.GetDataSize(); i++) {
        const float diff = std::abs(F32.data[i] - Q41.data[i]);
        if (!std::isfinite(diff)) {
            throw std::runtime_error("CompareLinearError: non-finite difference");
        }
        error = std::max(error, diff);
    }
    return error;
}
}  // namespace yan_lamma
