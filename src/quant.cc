#include "include/quant.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <functional>
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
}  // namespace
std::vector<Block_Q80> TensorToQ80(const Tensor& tensor) {
    if(tensor.data.size() % kQuantBlockSize  != 0) {
        return {};
    }
    std::vector<Block_Q80> blocks(tensor.data.size() / kQuantBlockSize);
    for (size_t i = 0; i < blocks.size(); i++) {
        auto& block = blocks[i];
        const float* x = tensor.data.data() + i * kQuantBlockSize;
        // 找最大绝对值
        auto amax = std::abs(x[0]);
        for (size_t i = 1; i < kQuantBlockSize; i++) {
            if (std::abs(x[i]) > amax) {
                amax = std::abs(x[i]);
            }
        }
        auto d = amax / 127.0f;
        block.d = Fp32ToFp16(d);
        for (size_t i = 0; i < kQuantBlockSize; i++) {
            float q = 0.0f;
            if (d != 0.0f) {
                q = std::round(x[i] / d);
            }
            block.qs[i] = static_cast<int8_t>(q);
        }
    }
    return blocks;
}
 Tensor Q80ToTensor(const std::vector<Block_Q80>& Q80tensor, const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("Invailed shape");
    }
    Tensor result(shape);
    if (result.GetDataSize() != Q80tensor.size() * kQuantBlockSize) {
        throw std::runtime_error("Invailed num size");
    }
    for (size_t i = 0; i < Q80tensor.size(); i++) {
        auto& block = Q80tensor[i];
        float d = Fp16ToFp32(block.d);
        for(size_t j = 0; j < kQuantBlockSize; j++) {
            result.data[j + i * kQuantBlockSize] = d * static_cast<float>(block.qs[j]);
        }
    }
    return result;
 }
std::vector<Block_Q40> TensorToQ40(const Tensor& tensor) {
    if (tensor.data.size() % kQuantBlockSize != 0) {
        return {};
    }
    std::vector<Block_Q40> blocks(tensor.data.size() / kQuantBlockSize);
    for (size_t i = 0; i < blocks.size(); ++i) {
        auto& block = blocks[i];
        const float* x = tensor.data.data() + i * kQuantBlockSize;
        float vmax = x[0];
        float mmax = std::abs(x[0]);
        for (size_t j = 1; j < kQuantBlockSize; j++) {
            if (std::abs(x[j])> mmax) {
                mmax = std::abs(x[j]);
                vmax  = x[j];
            }
        }
        float d = vmax / -8.0f;
        float invd = 0.0f;
        if (d != 0.0f) {
            invd = 1.0f / d;
        }
        block.d = Fp32ToFp16(d);
        for(size_t i = 0; i < 16; i++) {
            float v0 = x[i] * invd;
            float v1 = x[i + 16] * invd;
            int u0 = std::min(15, static_cast<int>(v0 + 8.5f));
            int u1 = std::min(15, static_cast<int>(v1 + 8.5f));
            block.qs[i] = static_cast<uint8_t>(u0 | (u1 << 4));
        }
    }
    return blocks;
}
Tensor Q40ToTensor(const std::vector<Block_Q40>& Q40tensor, const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("Invailed shape");
    }
    Tensor result(shape);
    if (result.data.size() != Q40tensor.size() * kQuantBlockSize) {
        throw std::runtime_error("Invailed shape");
    }
    for (size_t i = 0; i < Q40tensor.size(); i++) {
        const auto& block = Q40tensor[i];
        float d = Fp16ToFp32(block.d);
        for (size_t j = 0; j < 16; j++) {
            const int q0 = static_cast<int>(block.qs[j] & 0x0F) - 8;
            const int q1 = static_cast<int>(block.qs[j] >> 4) - 8;
            result[i * kQuantBlockSize + j] = d * static_cast<float>(q0);
            result[i * kQuantBlockSize + j + 16] = d * static_cast<float>(q1);
        }   
    }
    return result;
}
std::vector<Block_Q41> TensorToQ41 (const Tensor& tensor) {
    if (tensor.GetDataSize() % kQuantBlockSize != 0) {
        throw std::runtime_error("invalied tensor");
    }
    std::vector<Block_Q41> blocks(tensor.data.size() / kQuantBlockSize);
    for (size_t i = 0; i < blocks.size(); i++) {
        auto& block = blocks[i];
        const float * x = tensor.data.data() + i * kQuantBlockSize;
        float xmin = x[0];
        float xmax = x[0];
        for (size_t j = 1; j < kQuantBlockSize; j++) {
            if (x[j] > xmax) {
                xmax = x[j];
            }
            if (x[j] < xmin) {
                xmin = x[j];
            }
        }
        float m = xmin;
        float d = (xmax - xmin) / 15.0f;
        block.d = Fp32ToFp16(d);
        block.m = Fp32ToFp16(m);
        float invd = 0.0f;
        if (d != 0.0f) {
            invd = 1.0f / d;
        }
        for(size_t i = 0; i < 16; i++) {
            int u0 = std::min(15, static_cast<int>((x[i] - m) * invd + 0.5f));
            int u1 = std::min(15, static_cast<int>((x[i + 16] - m) * invd + 0.5f));
            block.qs[i] = static_cast<uint8_t>(u0 | u1 << 4);
        }

    }
    return blocks;
}
 Tensor Q41ToTensor(const std::vector<Block_Q41>& Q41tensor, const std::vector<int>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("invalied shape");
    }
    Tensor result(shape);
    if (result.GetDataSize() != Q41tensor.size() * kQuantBlockSize) {
        throw std::runtime_error("invalied tensor");
    }
    for (size_t i = 0; i < Q41tensor.size(); i++) {
        float * x = result.data.data() + i * kQuantBlockSize;
        auto& block =  Q41tensor[i];
        float m = Fp16ToFp32(block.m);
        float d = Fp16ToFp32(block.d);
        for(size_t i = 0; i < 16; i++) {
            int val1 = block.qs[i] & 0x0F;
            int val2 = block.qs[i] >> 4;
            x[i] = d *val1 + m;
            x[i + 16]  = d * val2 + m;
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
    if (x.shape.back() != in_features ||
        static_cast<size_t>(in_features) % block_size != 0) {
        throw std::runtime_error("LinearQuant: invalid shape or block alignment");
    }

    const size_t blocks_per_row =
        static_cast<size_t>(in_features) / block_size;
    if (weight.size() != static_cast<size_t>(out_features) * blocks_per_row ||
        x.data.size() != static_cast<size_t>(rows) * in_features) {
        throw std::runtime_error("LinearQuant: data size mismatch");
    }

    // 一维输入返回一维结果；[1, in] 输入仍然返回 [1, out]。
    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features});

    for (int row = 0; row < rows; ++row) {
        const float* x_row =
            x.data.data() + static_cast<size_t>(row) * in_features;

        // 每个输出是当前输入行与权重第 j 行的点积。
        for (int j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            const size_t block_base =
                static_cast<size_t>(j) * blocks_per_row;

            for (size_t b = 0; b < blocks_per_row; ++b) {
                const BlockType& block = weight[block_base + b];
                const size_t base_k = b * block_size;

                for (size_t i = 0; i < block_size; ++i) {
                    // 回调还原块内第 i 个权重，再与对应输入相乘累加。
                    const float w = f(block, static_cast<int>(i));
                    sum += x_row[base_k + i] * w;
                }
            }

            result.data[static_cast<size_t>(row) * out_features + j] = sum;
        }
    }

    return result;
}
Tensor LinearQ80(const Tensor& tensor, const std::vector<Block_Q80>& weight, const std::vector<int>& shape) {
    std::function<float(const Block_Q80&, int)> f = DequantQ80;
    return LinearQuant(tensor, weight, shape, f);
}
Tensor LinearQ40(const Tensor& tensor, const std::vector<Block_Q40>& weight, const std::vector<int>& shape) {
    std::function<float(const Block_Q40&, int)> f = DequantQ40;
    return LinearQuant(tensor, weight, shape, f);
}
Tensor LinearQ41(const Tensor& tensor, const std::vector<Block_Q41>& weight, const std::vector<int>& shape) {
    std::function<float(const Block_Q41&, int)> f = DequantQ41;
    return LinearQuant(tensor, weight, shape, f);
}
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
